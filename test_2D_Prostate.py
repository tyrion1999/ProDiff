import argparse
import os
import shutil

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from dataloaders.dataset import MRSEG19Normalization

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/workspace/dataset/PROSTATE_split2', help='Name of Experiment')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=3, help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=2, help='labeled data')
parser.add_argument('--ckpt', type=str, default='', help='checkpoint_name')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        jc = metric.binary.jc(pred, gt)
        return dice, jc, hd95, asd
    else:
        return 0, 0, 50, 10.


def test_single_volume(case, net, FLAGS, save_dir=None):
    """
    对单个 case 做推理和指标计算。
    若 save_dir 不为 None，则将 image / prediction / label 保存为 .nii.gz。
    """
    h5f = h5py.File(os.path.join(FLAGS.root_path, "data/{}.h5".format(case)), 'r')
    image = h5f['image'][:]
    # 若像素值范围大于 1，则做归一化
    if image.max() > 1:
        image = MRSEG19Normalization()(image, mode='Max_Min')
    label = h5f['label'][:]
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice_img = image[ind, :, :]
        x, y = slice_img.shape[0], slice_img.shape[1]
        slice_resized = zoom(slice_img, (256 / x, 256 / y), order=0)
        inp = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()

        net.eval()
        with torch.no_grad():
            out_main = net(inp)
            # 若网络返回的是 (pred, aux, ...) 之类的 tuple/list，则取第一个作为主输出
            if isinstance(out_main, (tuple, list)) and len(out_main) > 1:
                out_main = out_main[0]

            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    # 计算 per-class 指标
    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    # 转成 SimpleITK Image
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))

    # 保存为 .nii.gz
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, f"{case}_img.nii.gz")
        pred_path = os.path.join(save_dir, f"{case}_pred.nii.gz")
        gt_path = os.path.join(save_dir, f"{case}_gt.nii.gz")

        sitk.WriteImage(img_itk, img_path)
        sitk.WriteImage(prd_itk, pred_path)
        sitk.WriteImage(lab_itk, gt_path)

    return first_metric, second_metric, third_metric


def Inference(FLAGS, split_name=None):
    """
    split_name: 'val' 或 'test'
    """
    list_file = os.path.join(FLAGS.root_path, f"{split_name}.list")
    with open(list_file, 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])

    # 保存目录：logs/.../unet/predictions/val/ 或 logs/.../unet/predictions/test/
    base_dir = FLAGS.ckpt.split('/unet/')[0]
    test_save_path = os.path.join(base_dir, 'predictions', split_name)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path, exist_ok=True)

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    ckpt = torch.load(FLAGS.ckpt)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt.keys():
        info = net.load_state_dict(ckpt['state_dict'])
    else:
        info = net.load_state_dict(ckpt)
    print("init weight from {}".format(FLAGS.ckpt))
    print(info)
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0

    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, FLAGS, save_dir=test_save_path
        )
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)

    avg_metric = [
        first_total / len(image_list),
        second_total / len(image_list),
        third_total / len(image_list)
    ]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()

    split_list = ['test','val']

    for split_name in split_list:
        print(f"\n========== Split: {split_name} ==========")
        result_metric = Inference(FLAGS, split_name)
        print("Results per class")
        print(result_metric)
        print(" | Dice | HD95 | Jaccard | ASD |")
        print("Average")
        res = np.zeros_like(result_metric[0])
        for i in range(FLAGS.num_classes - 1):
            res += result_metric[i]
        res /= (FLAGS.num_classes - 1)
        print(res)
