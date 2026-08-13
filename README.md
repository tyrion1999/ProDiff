# ProDiff

PyTorch implementation of ProDiff for semi-supervised medical image segmentation.

**Author**: Yang Zuo  
**Email**: 666zy666@163.com  
**Affiliation**: School of Artificial Intelligence and Computer Science, Jiangnan University, Wuxi, China

## Installation

```bash
conda create -n prodiff python=3.10
conda activate prodiff
pip install -r requirement.txt
```

## Data preparation

All datasets provided below have already been preprocessed. You only need to download and extract each archive, then place the extracted files in the corresponding directories shown below. No additional preprocessing is required.

- **ACDC**: The original dataset is available from [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC). Our preprocessed data are available from [Baidu Netdisk](https://pan.baidu.com/s/1eAast_2kzwuuuipcFeU6Bw?pwd=4p2n) (`ACDC(1).rar`, extraction code: `4p2n`).
- **MS-CMRSEG19**: The original dataset is available from the [official website](https://zmiclab.github.io/zxh/0/mscmrseg19/). Our preprocessed data are available from [Baidu Netdisk](https://pan.baidu.com/s/1qgQ09cEDteujO4uDbzHOFg?pwd=xcdr) (`mscmrseg19.rar`, extraction code: `xcdr`).
- **MSD Prostate**: The original dataset is available from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/). Our preprocessed data are available from [Baidu Netdisk](https://pan.baidu.com/s/1_D1ZOd-9IdujJhtPjrr_uw?pwd=ij5i) (`MSD Prostate.rar`, extraction code: `ij5i`).

Organize the datasets as follows:

```text
dataset/
├── ACDC/
│   ├── data/
│   │   ├── patient001_frame01.h5
│   │   ├── ...
│   │   └── slices/
│   │       ├── patient001_frame01_slice_1.h5
│   │       └── ...
│   ├── all_slices.list
│   ├── test.list
│   ├── train.list
│   ├── train_slices.list
│   └── val.list
├── Prostate/
│   ├── prostate_split1/
│   │   ├── data/
│   │   │   ├── prostate_00.h5
│   │   │   ├── ...
│   │   │   └── slices/
│   │   │       ├── prostate_00_slice_0.h5
│   │   │       └── ...
│   │   ├── test.list
│   │   ├── train_slices.list
│   │   └── val.list
│   └── prostate_split2/
│       ├── data/
│       │   ├── prostate_17.h5
│       │   ├── ...
│       │   └── slices/
│       │       ├── prostate_17_slice_0.h5
│       │       └── ...
│       ├── test.list
│       ├── train_slices.list
│       └── val.list
└── mscmrseg/
    ├── mscmrseg_split1/
    │   ├── data/
    │   │   ├── patient1_LGE.h5
    │   │   ├── ...
    │   │   └── slices/
    │   │       ├── patient1_LGE_slice_0.h5
    │   │       └── ...
    │   ├── test.list
    │   ├── train_slices.list
    │   └── val.list
    └── mscmrseg_split2/
        ├── data/
        │   ├── patient1_LGE.h5
        │   ├── ...
        │   └── slices/
        │       ├── patient1_LGE_slice_0.h5
        │       └── ...
        ├── test.list
        ├── train_slices.list
        └── val.list
```

**Note**: For MS-CMRSEG19 and MSD Prostate, the datasets are split into training and validation only. We report the averaged results on the validation sets of the two random splits.

**Note**: The data must be used for research purposes only and in accordance with the conditions set by the original data owners. We may disable the download links for our preprocessed data if requested by the original data owners.

## Training

You can train ProDiff by specifying the GPU ID, experiment name, number of labeled patients, number of classes, and dataset root path. For example, use the following command to train ProDiff on ACDC with 7 labeled patients and 4 classes:

```bash
CUDA_VISIBLE_DEVICES=0 python train_ACDC.py --exp ACDC/ProDiff --labeled_num 7 --num_classes 4 --root_path ./dataset/ACDC
```

## Testing

Use the following command to evaluate a trained ProDiff model on ACDC:

```bash
CUDA_VISIBLE_DEVICES=0 python test_2D_ACDC.py --root_path ./dataset/ACDC --ckpt /path/to/checkpoint.pth --num_classes 4
```

## Acknowledgement

We sincerely appreciate [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [guided-diffusion](https://github.com/openai/guided-diffusion), [GSS](https://github.com/fudan-zvg/GSS), [DiffUNet](https://github.com/ge-xing/Diff-UNet) for their awesome codebases. If you have any questions, contact 666zy666@163.com or open an issue.
