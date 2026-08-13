import argparse
import logging
import os
import inspect
import random
import sys
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from dataloaders.dataset import (
    BaseDataSets,
    TwoStreamBatchSampler,
    WeakStrongAugment_Ours,
)
from networks.net_factory import net_factory

# Teacher (diffusion rectifier), image-only version: no CLIP/text branch and no text attention.
from networks.prodiff_mri import UNet_LDMV2

from utils import losses, ramps, util
from utils.labeled_cutmix import build_student_labeled_cutmix
from val_2D import test_single_volume_refinev2 as test_single_volume


parser = argparse.ArgumentParser()

# ---- Data / Experiment ----
parser.add_argument("--root_path", type=str, default="/workspace/dataset/ACDC", help="dataset root")
parser.add_argument("--exp", type=str, default="", help="experiment_name")
parser.add_argument("--model", type=str, default="unet", help="student model name for net_factory")
parser.add_argument("--max_iterations", type=int, default=50000, help="maximum iterations to train")
parser.add_argument("--batch_size", type=int, default=8, help="batch size per gpu")
parser.add_argument("--labeled_bs", type=int, default=4, help="labeled samples per batch")
parser.add_argument("--labeled_num", type=int, default=7, help="labeled patients (ACDC protocol)")
parser.add_argument("--deterministic", type=int, default=1, help="deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="base learning rate")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
parser.add_argument("--img_channels", type=int, default=1, help="input image channels")
parser.add_argument("--load", default=False, action="store_true", help="restore checkpoint from snapshot_path")
parser.add_argument("--conf_thresh", type=float, default=0.8, help="confidence threshold for pseudo labels")
parser.add_argument("--refine_start", type=int, default=1000, help="start iter to distill from teacher")

# ---- Consistency / loss weights ----
parser.add_argument("--consistency", type=float, default=0.1, help="max consistency weight")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="rampup length (epochs-ish)")

# ---- Student-only virtual labeled augmentation ----
parser.add_argument("--labeled_cutmix_w", type=float, default=0.2, help="weight for student-only labeled CutMix supervision")
parser.add_argument("--labeled_cutmix_prob", type=float, default=0.25, help="per labeled receiver probability for virtual CutMix")
parser.add_argument("--labeled_cutmix_min_area", type=int, default=200, help="minimum donor foreground area")
parser.add_argument("--labeled_cutmix_max_area", type=int, default=20000, help="maximum donor foreground area")
parser.add_argument(
    "--labeled_cutmix_branch",
    type=str,
    default="both",
    choices=["weak", "strong", "both"],
    help="which virtual labeled branch receives supervised student loss",
)

# ---- Teacher diffusion hyper-params ----
parser.add_argument("--ldm_beta_sch", type=str, default="cosine", help="beta schedule")
parser.add_argument("--ts", type=int, default=10, help="diffusion steps")
parser.add_argument("--ts_sample", type=int, default=4, help="sampling stride")
parser.add_argument("--ref_consistency_weight", type=float, default=-1, help="teacher unsup weight override (-1: follow main)")
parser.add_argument("--bridge_proto_w", type=float, default=0.5, help="weight for labeled->unlabeled prototype bridge KL loss")
parser.add_argument("--bridge_proto_temp", type=float, default=1.0, help="temperature for bridge prototype softmax")
parser.add_argument("--no_color", default=False, action="store_true", help="disable color embedding (if supported)")
parser.add_argument("--no_blur", default=False, action="store_true", help="disable blur aug")
parser.add_argument("--rot", type=int, default=359, help="rotation aug")

# ---- Physics-inspired reverse diffusion guidance (teacher DDIM only) ----
parser.add_argument("--phys_guidance", type=int, default=1,help="Enable edge-aware anisotropic heat conduction during teacher DDIM sampling (1=on, 0=off)")
parser.add_argument("--phys_lambda0", type=float, default=0.0001,help="small initial physical-guidance strength to avoid hurting baseline")
parser.add_argument("--phys_gamma", type=float, default=6.0, help="Fast decay so guidance mainly affects only the earliest DDIM steps")
parser.add_argument("--phys_kappa", type=float, default=0.15, help="Edge sensitivity for anisotropic conduction; smaller means stronger edge stopping")
parser.add_argument("--phys_cmin", type=float, default=0.0,help="No forced diffusion across edges for the lite setting")
parser.add_argument("--phys_iters", type=int, default=1, help="Keep one tiny diffusion substep when physics is triggered")
parser.add_argument("--phys_blend", type=float, default=0.05, help="Residual blend ratio for physics output; smaller means closer to baseline")
parser.add_argument("--phys_steps", type=str, default="2", help="Comma-separated DDIM steps (or ratios in [0,1]) where physics is applied; default only first step")
parser.add_argument("--phys_start", type=int, default=30000,help="start iteration for physics guidance in Step3")

# ---- Distillation (teacher -> student; test is student-only) ----
parser.add_argument("--distill_rect_w", type=float, default=1.0, help="hard pseudo supervision weight")
parser.add_argument("--distill_kl_w", type=float, default=0.5, help="soft KL distillation weight")
parser.add_argument("--distill_T", type=float, default=1.0, help="temperature for KL distillation")

args = parser.parse_args()


def _as_logits(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset or "acdc" in dataset.lower():
        ref_dict = {
            "1": 32,     # 1%
            "3": 68,     # 5%
            "7": 136,    # 10%
            "14": 256,   # 20%
            "21": 396,   # 30%
            "28": 512,   # 40%
            "35": 664,   # 50%
            "140": 1312, # 100%
        }
    else:
        raise NotImplementedError
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch_like):
    return args.consistency * ramps.sigmoid_rampup(epoch_like, args.consistency_rampup)


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    # Min-max normalize along class dim
    min_val = tensor.min(1, keepdim=True)[0]
    max_val = tensor.max(1, keepdim=True)[0]
    out = tensor - min_val
    out = out / (max_val + 1e-6)
    return out

def pl_embed(color_map, mask_np: np.ndarray) -> torch.Tensor:
    """
    mask_np: (B,H,W) int
    return: (B,3,H,W) float tensor (CPU)
    把一批整数标签 mask（(B,H,W)，每个像素是 class_id）按 color_map 映射成对应的 RGB 彩色图，并转成 torch.float32 的张量
    """
    b, h, w = mask_np.shape
    out = torch.zeros((b, 3, h, w), dtype=torch.float32)
    for i in range(b):
        color_data = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[mask_np[i] == class_id] = color
        color_image = Image.fromarray(color_data, mode="RGB")
        out[i] = transforms.ToTensor()(color_image)
    return out


def label_embed(color_map, mask_np: np.ndarray) -> torch.Tensor:
    """
    mask_np: (B,H,W) int
    return: (B,3,H,W) float tensor (CUDA)
    """
    out = pl_embed(color_map, mask_np)
    return out.cuda(non_blocking=True)


def train(args, snapshot_path):
    # Log args
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    # --------- Data ----------
    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([WeakStrongAugment_Ours(args.patch_size, args)]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    db_test = BaseDataSets(base_dir=args.root_path, split="test")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    logging.info(f"Total slices: {total_slices}, labeled slices: {labeled_slice}")

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=lambda wid: random.seed(args.seed + wid),
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    # --------- Models ----------
    model = net_factory(
        net_type=args.model,
        in_chns=args.img_channels,
        class_num=num_classes,
    ).cuda()
    refine_model = UNet_LDMV2(
        in_chns=3 + args.img_channels,
        class_num=num_classes,
        out_chns=num_classes,
        ldm_method="replace",
        ldm_beta_sch=args.ldm_beta_sch,
        ts=args.ts,
        ts_sample=args.ts_sample,
    ).cuda()

    # --------- Teacher kw filtering wrapper ----------
    try:
        _refine_kw = set(inspect.signature(refine_model.forward).parameters.keys())
    except Exception:
        _refine_kw = None

    def refine_forward(*f_args, **f_kwargs):
        if _refine_kw is None:
            return refine_model(*f_args, **f_kwargs)
        filt = {k: v for k, v in f_kwargs.items() if k in _refine_kw}
        return refine_model(*f_args, **filt)

    def parse_refine_out(out):
        """
        Normalize refine_model output to (lat_loss, logits, aux_dict).
        Supports:
          - (lat_loss, logits)
          - (logits, aux)
          - (lat_loss, logits, aux)
          - logits
        """
        lat = torch.tensor(0.0, device="cuda")
        aux = {}
        if torch.is_tensor(out):
            return lat, out, aux
        if isinstance(out, (tuple, list)):
            if len(out) == 2:
                a, b = out
                if torch.is_tensor(a) and torch.is_tensor(b):
                    # guess (lat_loss, logits) if a is scalar
                    if a.dim() == 0 and b.dim() == 4:
                        return a, b, aux
                    # else treat as (logits, something-not-aux)
                    return lat, a, aux
                if torch.is_tensor(a) and isinstance(b, dict):
                    return lat, a, b
                if torch.is_tensor(b) and isinstance(a, dict):
                    return lat, b, a
            if len(out) == 3:
                a, b, c = out
                if torch.is_tensor(a) and torch.is_tensor(b) and isinstance(c, dict):
                    return a, b, c
                if torch.is_tensor(a) and isinstance(b, dict) and torch.is_tensor(c):
                    return lat, a, b
            # fallback
            for item in out:
                if torch.is_tensor(item) and item.dim() == 4:
                    return lat, item, aux
        return lat, torch.tensor(0.0, device="cuda"), aux

    # --------- Optimizers ----------
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    refine_optimizer = optim.SGD(refine_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # --------- Losses ----------
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    def get_comp_loss(weak_prob, strong_prob, bs=None):
        # bs 用真实 batch，别用 args.batch_size（最后一个 batch 可能不足）
        if bs is None:
            bs = strong_prob.shape[0]

        # ---- 1) 概率防呆：去 NaN + 保证每个像素的类别分布和为1 ----
        p = torch.nan_to_num(strong_prob, nan=0.0, posinf=0.0, neginf=0.0)
        p = p.clamp(min=1e-6, max=1.0)  # 避免 log(0)
        p = p / (p.sum(dim=1, keepdim=True) + 1e-6)  # (B,C,H,W) -> pixel-wise simplex

        # ---- 2) 变形为 (B, N, C)，Categorical 的最后一维必须是类别维 ----
        # N = H*W
        p_flat = p.permute(0, 2, 3, 1).contiguous().view(bs, -1, args.num_classes)  # (B,N,C)

        # ---- 3) 熵：每个像素一个 entropy，取均值；最大熵是 log(C) ----
        ent = Categorical(probs=p_flat).entropy()  # (B,N)
        as_weight = 1.0 - ent.mean() / np.log(args.num_classes)  # 标准化到[0,1]附近
        as_weight = as_weight.clamp(0.0, 1.0)

        # ---- 4) 你原来的 comp_loss 逻辑保持不变（只建议同样做 nan_to_num）----
        weak_prob2 = torch.nan_to_num(weak_prob, nan=0.0, posinf=0.0, neginf=0.0)
        strong_prob2 = torch.nan_to_num(strong_prob, nan=0.0, posinf=0.0, neginf=0.0)

        comp_labels = torch.argmin(weak_prob2.detach(), dim=1)  # (B,H,W)
        comp_loss = as_weight * ce_loss(1.0 - strong_prob2, comp_labels)

        return comp_loss, as_weight

    def get_bridge_proto_loss(labeled_logits, unlabeled_logits, temp=1.0):
        if labeled_logits is None or unlabeled_logits is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.tensor(0.0, device=device)
        if labeled_logits.numel() == 0 or unlabeled_logits.numel() == 0:
            return labeled_logits.new_tensor(0.0)

        temp = max(float(temp), 1e-6)

        def _batch_proto(logits):
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            pooled = F.adaptive_avg_pool2d(logits, 1).flatten(1)
            sample_prob = torch.softmax(pooled / temp, dim=1)
            proto = sample_prob.mean(dim=0, keepdim=True)
            proto = proto / (proto.sum(dim=1, keepdim=True) + 1e-6)
            return proto.clamp(min=1e-6, max=1.0)

        # Eq.4: stop-gradient on the labeled prototype so it stays a fixed GT-oriented target.
        p_l = _batch_proto(labeled_logits).detach()
        p_u = _batch_proto(unlabeled_logits)
        return F.kl_div(p_u.log(), p_l, reduction="batchmean")

    # --------- Color map (for teacher input) ----------
    if args.num_classes == 4:
        color_map = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
    elif args.num_classes == 3:
        color_map = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}
    else:
        color_map = {0: (0, 0, 0), 1: (255, 255, 255)}

    # --------- Resume ----------
    iter_num = 0
    start_epoch = 0
    best_performance = 0.0

    # ====================== 恢复训练过程中的中断 ==========================
    if args.load:
        try:
            model_checkpoint = None
            iters = []
            for filename in os.listdir(snapshot_path):
                if "model_iter" in filename:
                    basename, _ = os.path.splitext(filename)
                    iters.append(int(basename.split("_")[2]))
            if iters:
                iter_num = max(iters)
                for filename in os.listdir(snapshot_path):
                    if "model_iter" in filename and str(iter_num) in filename:
                        model_checkpoint = filename
                        break
            if model_checkpoint is not None:
                model, optimizer, start_epoch, _ = util.load_checkpoint(
                    os.path.join(snapshot_path, model_checkpoint), model, optimizer
                )
                logging.info(f"Restored student checkpoint: {model_checkpoint}")
        except Exception as e:
            logging.warning(f"Restore failed: {e}")

    # --------- Train loop ----------
    model.train()
    refine_model.train()
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(start_epoch, max_epoch), ncols=80)

    for epoch_num in iterator:
        for sampled_batch in trainloader:
            weak_batch = sampled_batch["image_weak"].cuda(non_blocking=True)
            strong_batch = sampled_batch["image_strong"].cuda(non_blocking=True)
            label_batch = sampled_batch["label_aug"].cuda(non_blocking=True)

            # safety: zero out unlabeled labels
            label_batch[args.labeled_bs:] = torch.zeros_like(label_batch[args.labeled_bs:])

            # -----------------------------
            # Step1: student SSL
            # -----------------------------
            out_w = _as_logits(model(weak_batch, iter_num=iter_num))
            out_s = _as_logits(model(strong_batch, iter_num=iter_num))
            out_w_soft = torch.softmax(out_w, dim=1)
            out_s_soft = torch.softmax(out_s, dim=1)
            pseudo_mask = (normalize(out_w_soft) > args.conf_thresh).float()
            out_w_masked = out_w_soft * pseudo_mask
            pseudo_outputs = torch.argmax(out_w_masked.detach(), dim=1)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            comp_loss, as_weight = get_comp_loss(out_w_soft, out_s_soft, bs=strong_batch.shape[0])

            sup_loss = (
                ce_loss(out_w[: args.labeled_bs], label_batch[: args.labeled_bs].long())
                + dice_loss(out_w_soft[: args.labeled_bs], label_batch[: args.labeled_bs].unsqueeze(1))
            )

            aug_sup_loss = torch.tensor(0.0, device=weak_batch.device)
            cutmix_labeled = 0
            if args.labeled_cutmix_w > 0 and args.labeled_cutmix_prob > 0 and args.labeled_bs > 1:
                cutmix_weak, cutmix_strong, cutmix_label, cutmix_mask = build_student_labeled_cutmix(
                    weak_batch,
                    strong_batch,
                    label_batch,
                    labeled_bs=args.labeled_bs,
                    prob=args.labeled_cutmix_prob,
                    min_area=args.labeled_cutmix_min_area,
                    max_area=args.labeled_cutmix_max_area,
                )
                if cutmix_mask.any():
                    cutmix_labeled = int(cutmix_mask.sum().item())
                    cutmix_label_sel = cutmix_label[cutmix_mask].long()
                    aug_terms = []

                    if args.labeled_cutmix_branch in ("weak", "both"):
                        cutmix_logits_w = _as_logits(model(cutmix_weak[cutmix_mask], iter_num=iter_num))
                        cutmix_prob_w = torch.softmax(cutmix_logits_w, dim=1)
                        aug_terms.append(
                            ce_loss(cutmix_logits_w, cutmix_label_sel)
                            + dice_loss(cutmix_prob_w, cutmix_label_sel.unsqueeze(1))
                        )

                    if args.labeled_cutmix_branch in ("strong", "both"):
                        cutmix_logits_s = _as_logits(model(cutmix_strong[cutmix_mask], iter_num=iter_num))
                        cutmix_prob_s = torch.softmax(cutmix_logits_s, dim=1)
                        aug_terms.append(
                            ce_loss(cutmix_logits_s, cutmix_label_sel)
                            + dice_loss(cutmix_prob_s, cutmix_label_sel.unsqueeze(1))
                        )

                    if aug_terms:
                        aug_sup_loss = sum(aug_terms) / float(len(aug_terms))

            unsup_loss = (
                ce_loss(out_s[args.labeled_bs :], pseudo_outputs[args.labeled_bs :])
                + dice_loss(out_s_soft[args.labeled_bs :], pseudo_outputs[args.labeled_bs :].unsqueeze(1))
                + as_weight * comp_loss
            )

            loss = sup_loss + args.labeled_cutmix_w * aug_sup_loss + consistency_weight * unsup_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -----------------------------
            # Prepare RGB masks for teacher
            # -----------------------------
            pseudo_mask_s = (normalize(out_s_soft) > args.conf_thresh).float()
            out_s_masked = out_s_soft * pseudo_mask_s
            pseudo_outputs_s = torch.argmax(out_s_masked.detach(), dim=1)

            pseudo_np_w = pseudo_outputs.detach().cpu().numpy()
            pseudo_np_s = pseudo_outputs_s.detach().cpu().numpy()

            pseudo_rgb_w = pl_embed(color_map, pseudo_np_w).cuda(non_blocking=True)  # (B,3,H,W)
            pseudo_rgb_s = pl_embed(color_map, pseudo_np_s).cuda(non_blocking=True)

            lb = args.labeled_bs
            bs = pseudo_rgb_w.shape[0]
            unlb = bs - lb

            label_rgb = None
            if lb > 0:
                label_np = label_batch[:lb].detach().cpu().numpy()
                label_rgb = label_embed(color_map, label_np)  # (lb,3,H,W) cuda

            # -----------------------------
            # Step2: teacher training (sequential cascade)
            #   Labeled: Strong->Weak (stage 1) then midrw->GT (stage 2)
            #   Unlabeled: only Strong->Weak (stage 1)
            #   The extra complementary teacher term is removed.
            # -----------------------------
            # Segment 1 (Strong -> Weak) for labeled
            lat_loss_s2w_lab = torch.tensor(0.0, device="cuda")
            loss_s2w_lab = torch.tensor(0.0, device="cuda")
            midrw_color = None
            midrw_pseudo = None

            if lb > 0:
                t2_lab_scalar = dice_loss(
                    pseudo_outputs_s[:lb].unsqueeze(1),
                    pseudo_outputs[:lb].unsqueeze(1),
                    oh_input=True,
                )
                t2_lab = torch.ones((lb,), dtype=torch.float32, device="cuda") * t2_lab_scalar * 999

                out = refine_forward(
                    pseudo_rgb_s[:lb],
                    t2_lab,
                    strong_batch[:lb],
                    training=True,
                    good=pseudo_rgb_w[:lb],
                )
                lat_loss_s2w_lab, ref_logits_s2w_lab, _aux = parse_refine_out(out)
                ref_soft_s2w_lab = torch.softmax(ref_logits_s2w_lab, dim=1)

                loss_s2w_lab_cedice = (
                    ce_loss(ref_logits_s2w_lab, pseudo_outputs[:lb])
                    + dice_loss(ref_soft_s2w_lab, pseudo_outputs[:lb].unsqueeze(1))
                )
                loss_s2w_lab = loss_s2w_lab_cedice + lat_loss_s2w_lab

                # build midrw from segment1 output
                mid_mask = (normalize(ref_soft_s2w_lab) > args.conf_thresh).float()
                mid_soft_masked = ref_soft_s2w_lab * mid_mask
                midrw_pseudo = torch.argmax(mid_soft_masked.detach(), dim=1,keepdim=False)
                midrw_np = midrw_pseudo.detach().cpu().numpy()
                midrw_color = pl_embed(color_map, midrw_np).cuda(non_blocking=True)

            # Segment 2 (midrw -> GT) for labeled only
            lat_loss_sup = torch.tensor(0.0, device="cuda")
            sup_loss_ref = torch.tensor(0.0, device="cuda")
            if lb > 0 and (midrw_color is not None) and (label_rgb is not None) and (midrw_pseudo is not None):
                t_scalar = dice_loss(midrw_pseudo.unsqueeze(1), label_batch[:lb].unsqueeze(1), oh_input=True)
                t = torch.ones((lb,), dtype=torch.float32, device="cuda") * t_scalar * 999

                out = refine_forward(
                    midrw_color,
                    t,
                    weak_batch[:lb],
                    training=True,
                    good=label_rgb,
                )
                lat_loss_sup, ref_logits_sup, _aux2 = parse_refine_out(out)
                ref_soft_sup = torch.softmax(ref_logits_sup, dim=1)
                sup_loss_cedice = (
                    ce_loss(ref_logits_sup, label_batch[:lb].long())
                    + dice_loss(ref_soft_sup, label_batch[:lb].unsqueeze(1))
                )
                sup_loss_ref = sup_loss_cedice + lat_loss_sup

            # Segment 1 (Strong -> Weak) for unlabeled only.
            lat_loss_unsup = torch.tensor(0.0, device="cuda")
            unsup_loss_ref = torch.tensor(0.0, device="cuda")
            bridge_proto_loss = torch.tensor(0.0, device="cuda")
            t2_unlb_vec = torch.zeros((1,), dtype=torch.float32, device="cuda")

            if unlb > 0:
                t2_scalar = dice_loss(
                    pseudo_outputs_s[lb:].unsqueeze(1),
                    pseudo_outputs[lb:].unsqueeze(1),
                    oh_input=True,
                )
                t2_unlb_vec = torch.ones((unlb,), dtype=torch.float32, device="cuda") * t2_scalar * 999

                out = refine_forward(
                    pseudo_rgb_s[lb:],
                    t2_unlb_vec,
                    strong_batch[lb:],
                    training=True,
                    good=pseudo_rgb_w[lb:],
                )
                lat_loss_unsup, ref_logits_unlb_tr, _aux3 = parse_refine_out(out)
                ref_soft_unlb_tr = torch.softmax(ref_logits_unlb_tr, dim=1)

                unsup_loss_cedice = (
                    ce_loss(ref_logits_unlb_tr, pseudo_outputs[lb:])
                    + dice_loss(ref_soft_unlb_tr, pseudo_outputs[lb:].unsqueeze(1))
                )
                unsup_loss_ref = unsup_loss_cedice + lat_loss_unsup

            # Total teacher loss
            ref_consistency_weight = get_current_consistency_weight(iter_num // 150)
            if args.ref_consistency_weight != -1:
                ref_consistency_weight = args.ref_consistency_weight

            if args.bridge_proto_w > 0 and lb > 0 and unlb > 0:
                bridge_proto_loss = get_bridge_proto_loss(
                    ref_logits_s2w_lab,
                    ref_logits_unlb_tr,
                    temp=args.bridge_proto_temp,
                )

            refine_loss = (
                sup_loss_ref
                + loss_s2w_lab
                + ref_consistency_weight * unsup_loss_ref
                + args.bridge_proto_w * bridge_proto_loss
            )
            refine_optimizer.zero_grad()
            refine_loss.backward()
            refine_optimizer.step()

            # -----------------------------
            # Step3: distill teacher rectification into student (UNLABELED ONLY)
            #   - distillation = hard CE/Dice target + optional soft KL on high-confidence pixels.
            #   Teacher only at train; test stays student-only.
            # -----------------------------
            if iter_num > args.refine_start:
                with torch.no_grad():
                    # teacher inference uses strong->weak path on unlabeled
                    refine_model.eval()
                    out = refine_forward(
                        pseudo_rgb_s[lb:], # 给 teacher 的伪标签/语义彩色编码
                        t2_unlb_vec,
                        strong_batch[lb:], # 只取 batch 里 unlabeled 部分
                        training=False,
                        phys_guidance=bool(args.phys_guidance) and (iter_num >= args.phys_start),
                        phys_lambda0=args.phys_lambda0,
                        phys_gamma=args.phys_gamma,
                        phys_kappa=args.phys_kappa,
                        phys_cmin=args.phys_cmin,
                        phys_iters=args.phys_iters,
                        phys_blend=args.phys_blend,
                        phys_steps=args.phys_steps,
                    )
                    _lat0, tea_logits, _aux = parse_refine_out(out) # teacher 的分类 logits（每像素 C 类）
                    tea_prob = torch.softmax(tea_logits, dim=1) # softmax 后的概率图

                    # 置信度筛选：只相信 teacher 高置信的像素
                    tea_mask = (normalize(tea_prob) > args.conf_thresh).float()
                    tea_prob_masked = tea_prob * tea_mask
                    tea_hard = torch.argmax(tea_prob_masked, dim=1)

                refine_model.train()

                # student forward strong
                stu_logits_all = _as_logits(model(strong_batch, iter_num=iter_num)) # strong_batch 包含 labeled+unlabeled 一起过 student
                stu_logits_unlb = stu_logits_all[lb:]  # 只取 lb: 作为无标签部分来算硬监督损失
                stu_prob_unlb = torch.softmax(stu_logits_unlb, dim=1)
                # 1.硬监督
                rect_loss = ce_loss(stu_logits_unlb, tea_hard) + dice_loss(stu_prob_unlb, tea_hard.unsqueeze(1))

                # 2.软监督 teacher 不仅给答案，还给“每个类别的可能性分布”；student 学这种软信息，但只在 teacher 有把握的地方学。
                kl_loss = torch.tensor(0.0, device="cuda")
                if args.distill_kl_w > 0:
                    T = max(float(args.distill_T), 1e-6) # 温度
                    tea_prob2 = tea_prob_masked / (tea_prob_masked.sum(dim=1, keepdim=True) + 1e-6) # 把 mask 后的 teacher prob 重新归一化
                    logp = F.log_softmax(stu_logits_unlb / T, dim=1)  # (B,C,H,W)
                    conf = tea_prob.max(dim=1, keepdim=True)[0]  # 每个像素 teacher 最相信的那个类的概率
                    mask = (conf > args.conf_thresh).float()  # 只在 teacher真有把握的像素算 KL

                    kl_map = F.kl_div(logp, tea_prob2, reduction="none").sum(dim=1, keepdim=True)  # (B,1,H,W)
                    kl_loss = (kl_map * mask).mean() * (T * T)

                total_distill = args.distill_rect_w * rect_loss + args.distill_kl_w * kl_loss

                optimizer.zero_grad()
                total_distill.backward()
                optimizer.step()

            # learning rate schedule
            lr_ = base_lr * (1.0 - float(iter_num) / float(max_iterations)) ** 0.9
            for pg in optimizer.param_groups:
                pg["lr"] = lr_

            iter_num += 1

            if iter_num % 1 == 0:
                logging.info(
                    "iter %d | seg_loss %.4f | aug_sup %.4f | cutmix_labeled %d | ref_loss %.4f | bridge %.4f | distill %.4f"
                    % (iter_num, loss.item(), float(aug_sup_loss.item()), int(cutmix_labeled), refine_loss.item(),
                       float(bridge_proto_loss.item() if 'bridge_proto_loss' in locals() else 0.0),
                       float(total_distill.item() if 'total_distill' in locals() else 0.0))
                )

            # Validation (STUDENT ONLY)
            if iter_num % 25 == 0 and iter_num> 35000:
                model.eval()
                metric_list = 0.0
                for sampled_val in valloader:
                    metric_i = test_single_volume(
                        sampled_val["image"],
                        sampled_val["label"],
                        model,
                        classes=num_classes,
                    )
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                mean_jaccard = np.mean(metric_list, axis=0)[2]
                logging.info("VALIDATION iter %d: Dice %.6f | Jaccard %.6f | HD95 %.6f"
                        % (iter_num, performance, mean_jaccard, mean_hd95))

                if performance > best_performance:
                    best_performance = performance
                    logging.info( "BEST UPDATED @ iter %d: Dice %.6f | Jaccard %.6f | HD95 %.6f"
                        % (iter_num, performance, mean_jaccard, mean_hd95))
                    save_best = os.path.join(snapshot_path, f"{args.model}_best_model_iter_num{iter_num}_{performance}.pth")
                    util.save_checkpoint(epoch_num, model, optimizer, loss, save_best)

                model.train()

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            break

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/workspace/DiffRect-main/logs/{}_{}_labeled/{}".format(args.exp, args.labeled_num, args.model)
    os.makedirs(snapshot_path, exist_ok=True)

    logging.getLogger("").handlers = []
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.log"),
        level=logging.DEBUG,
        filemode="w",
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # ---------------------------
    # Backup training script & network files to snapshot_path
    # ---------------------------
    try:
        # 2.1 备份当前 train.py（正在运行的脚本）
        train_src = os.path.abspath(__file__)
        shutil.copy2(train_src, os.path.join(snapshot_path, os.path.basename(train_src)))

        # 2.2 备份 teacher 网络和新增强文件，保证实验快照可复现
        net_files = [
            "networks/prodiff_mri.py",
            "utils/labeled_cutmix.py",
        ]

        for rel_path in net_files:
            abs_path = os.path.abspath(rel_path)
            if os.path.isfile(abs_path):
                dst_name = os.path.basename(rel_path)
                shutil.copy2(abs_path, os.path.join(snapshot_path, dst_name))
            else:
                logging.warning(f"[Backup] File not found, skip: {abs_path}")

        logging.info(f"[Backup] Saved train+net files into: {snapshot_path}")

    except Exception as e:
        logging.warning(f"[Backup] Failed to backup source files: {e}")


    train(args, snapshot_path)
