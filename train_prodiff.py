import argparse
import logging
import os
import math
import inspect
import random
import sys
import json
import shutil
from datetime import datetime
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

try:
    import clip  # OpenAI CLIP
except Exception:
    clip = None

from dataloaders.dataset import (
    BaseDataSets,
    TwoStreamBatchSampler,
    WeakStrongAugment_Ours,
)
from networks.net_factory_sam_aux_student import net_factory

# Teacher (diffusion rectifier). Prefer the physical-guidance version if present.
try:
    from networks.ProDiffNet import UNet_LDMV2  # type: ignore
except Exception:
    try:
        from networks.unet_de_512_sequential_TRI_CFG import UNet_LDMV2  # type: ignore
    except Exception:
        try:
            from networks.unet_de_512 import UNet_LDMV2  # type: ignore
        except Exception:
            from networks.unet_de import UNet_LDMV2  # fallback

from utils import losses, ramps, util
from val_2D import test_single_volume_refinev2 as test_single_volume


parser = argparse.ArgumentParser()

# ---- Data / Experiment ----
parser.add_argument("--root_path", type=str, default="/workspace/dataset/ACDC", help="dataset root")
parser.add_argument("--exp", type=str, default="221 ACDC_sequential_VRRCLIP_2026_0502_1643_SAM", help="experiment_name")
parser.add_argument("--model", type=str, default="unet", help="student model name for net_factory")
parser.add_argument("--max_iterations", type=int, default=72000, help="maximum iterations to train")
parser.add_argument("--batch_size", type=int, default=8, help="batch size per gpu")
parser.add_argument("--labeled_bs", type=int, default=4, help="labeled samples per batch")
parser.add_argument("--labeled_num", type=int, default=7, help="labeled patients (ACDC protocol)")
parser.add_argument("--deterministic", type=int, default=1, help="deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="base learning rate")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
parser.add_argument("--img_channels", type=int, default=1, help="input image channels")

# ---- Consistency / loss weights ----
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency type")
parser.add_argument("--consistency", type=float, default=0.1, help="max consistency weight")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="rampup length (epochs-ish)")

# ---- Teacher diffusion hyper-params ----
parser.add_argument("--base_chn_rf", type=int, default=64, help="teacher base channels")
parser.add_argument("--ldm_beta_sch", type=str, default="cosine", help="beta schedule")
parser.add_argument("--ts", type=int, default=10, help="diffusion steps")
parser.add_argument("--ts_sample", type=int, default=4, help="sampling stride")
parser.add_argument("--ref_consistency_weight", type=float, default=-1, help="teacher unsup weight override (-1: follow main)")
parser.add_argument("--no_color", default=False, action="store_true", help="disable color embedding (if supported)")
parser.add_argument("--no_blur", default=False, action="store_true", help="disable blur aug")
parser.add_argument("--rot", type=int, default=359, help="rotation aug")

# ---- Teacher text constraint (CLIP) + VRR (TRAINING ONLY) ----
parser.add_argument("--prompt_bank_path", type=str, default="/workspace/DiffRect-main/MyPromptBank/acdc_promprt_CLIP_full_name.json",help="JSON prompt bank path (ACDC)")
parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP backbone")
parser.add_argument("--vrr_k", type=int, default=2, help="VRR top-k prompts per class")
parser.add_argument("--vrr_temp", type=float, default=0.07, help="VRR softmax temperature")
parser.add_argument("--vrr_steps", type=str, default="0,1,2,3", help="comma-separated steps to refresh text_feat in sampling")
parser.add_argument("--cfg_pos_scale", type=float, default=1.2, help="tri-CFG positive guidance scale (teacher sampling only)")
parser.add_argument("--cfg_neg_scale", type=float, default=0.0, help="tri-CFG negative guidance scale (teacher sampling only); 0 disables neg")# ---- CLIP explicit guidance (teacher inference only; affects Step3 pseudo labels) ----
parser.add_argument("--clip_refine_steps", type=int, default=3,
                    help="If >0, run CLIP-guided refinement on teacher logits in Step3 (continuous correction). Recommended 2-5.")
parser.add_argument("--clip_refine_lr", type=float, default=0.5, help="Step size for refining teacher logits with CLIP guidance")
parser.add_argument("--clip_refine_every", type=int, default=4, help="Run CLIP-guided refinement every N iters (to save compute)")
parser.add_argument("--clip_refine_alpha", type=float, default=0.5, help="Overlay strength when building CLIP evidence image (0..1)")
parser.add_argument("--clip_refine_trust_w", type=float, default=20.0, help="Penalty to keep refined prob close to original prob")
parser.add_argument("--clip_refine_fg_weight", type=float, default=1.0, help="Weight for foreground classes in CLIP relative score")
parser.add_argument("--clip_refine_bg_weight", type=float, default=0.05, help="Weight for background class in CLIP relative score")
parser.add_argument("--clip_refine_prompt_pool", type=str, default="bank",
                    choices=["bank","vrr","init"],
                    help="Which text pool to score against: bank=all prompts in JSON; vrr=current text_feat; init=first prompt per class")
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
parser.add_argument("--distill_feat_w", type=float, default=0.1, help="feature distillation weight")
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
# -----------------------------
# CLIP-guided continuous correction (teacher logits refinement)
# This runs ONLY in Step3 (teacher inference -> pseudo label -> distill student).
# It does NOT backprop into teacher weights; it only adjusts teacher logits a few steps.
# -----------------------------

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def _minmax01(x: torch.Tensor) -> torch.Tensor:
    # x: (B,1,H,W) or (B,3,H,W)
    b = x.shape[0]
    x2 = x.view(b, x.shape[1], -1)
    mn = x2.min(dim=-1, keepdim=True)[0].view(b, x.shape[1], 1, 1)
    mx = x2.max(dim=-1, keepdim=True)[0].view(b, x.shape[1], 1, 1)
    return (x - mn) / (mx - mn + 1e-6)

def build_clip_evidence_image(gray_img: torch.Tensor,
                              prob: torch.Tensor,
                              alpha: float = 0.5,
                              out_size: int = 224) -> torch.Tensor:
    """
    Differentiable overlay evidence for CLIP scoring.
    gray_img: (B,1,H,W) input image tensor (any range)
    prob: (B,C,H,W) soft probabilities (depends on logits)  <-- gradient flows through this
    returns: (B,3,out_size,out_size) normalized for CLIP
    """
    # Normalize gray to [0,1] per-sample
    x = _minmax01(gray_img.float())
    x_rgb = x.repeat(1, 3, 1, 1)

    B, C, H, W = prob.shape
    # Fixed class colors (background=0). For ACDC (4 classes): RV red, MYO green, LV blue.
    # If num_classes differs, colors will repeat cyclically.
    base_colors = torch.tensor([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]], device=prob.device, dtype=prob.dtype)
    if C > base_colors.shape[0]:
        reps = (C + base_colors.shape[0] - 1) // base_colors.shape[0]
        base_colors = base_colors.repeat(reps, 1)
    colors = base_colors[:C]  # (C,3)

    color_map = (prob[:, :, None, :, :] * colors[None, :, :, None, None]).sum(dim=1)  # (B,3,H,W)
    evidence = (1.0 - alpha) * x_rgb.to(color_map.dtype) + alpha * color_map

    evidence = F.interpolate(evidence, size=(out_size, out_size), mode="bilinear", align_corners=False)

    mean = torch.tensor(_CLIP_MEAN, device=evidence.device, dtype=evidence.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(_CLIP_STD, device=evidence.device, dtype=evidence.dtype).view(1, 3, 1, 1)
    evidence = (evidence - mean) / std
    return evidence

def clip_relative_scores(img_feat: torch.Tensor,
                         bank_emb: list,
                         use_pool: str,
                         text_feat_cur: torch.Tensor = None,
                         text_feat_init: torch.Tensor = None) -> torch.Tensor:
    """
    img_feat: (B,512) normalized
    bank_emb: list length K, each (P,512) normalized
    use_pool: 'bank' | 'vrr' | 'init'
    returns: rel_scores (B,K) where rel_scores[:,c] = s_c - logsumexp_{k!=c}(s_k)
    """
    B = img_feat.shape[0]

    # get class similarity s_c
    sims = []
    if use_pool == "bank" and bank_emb is not None:
        for cls_bank in bank_emb:  # (P,512)
            sim = img_feat @ cls_bank.t()  # (B,P)
            s_c = sim.max(dim=1)[0]  # (B,)
            sims.append(s_c)
        s = torch.stack(sims, dim=1)  # (B,K)
    elif use_pool == "vrr" and (text_feat_cur is not None):
        # text_feat_cur: (K,512) normalized
        s = img_feat @ text_feat_cur.t()  # (B,K)
    else:
        # init pool
        if text_feat_init is None:
            raise ValueError("text_feat_init required for init pool scoring")
        s = img_feat @ text_feat_init.t()

    # relative score
    rel = []
    for c in range(s.shape[1]):
        others = torch.cat([s[:, :c], s[:, c+1:]], dim=1)
        rel_c = s[:, c] - torch.logsumexp(others, dim=1)
        rel.append(rel_c)
    rel = torch.stack(rel, dim=1)  # (B,K)
    return rel

def clip_guided_refine_teacher_logits(
    tea_logits: torch.Tensor,
    gray_img: torch.Tensor,
    clip_model,
    bank_emb: list,
    text_feat_cur: torch.Tensor,
    text_feat_init: torch.Tensor,
    args,
) -> torch.Tensor:
    """
    用 CLIP 作为“判别器”对 teacher logits 做连续微调（仅推理阶段使用）。
    - 不会更新 teacher/CLIP 的模型参数；
    - 只对当前 batch 的 logits 做若干步梯度更新；
    - 目标是提高图像与目标文本提示的相对匹配分数，同时约束不要偏离原始预测太远。
    """
    # 无 CLIP 或 bank 模式下缺少 prompt bank 时，直接返回原始 teacher logits。
    if (clip_model is None) or (bank_emb is None and args.clip_refine_prompt_pool == "bank"):
        return tea_logits
    # 配置步数 <= 0 视为关闭 refine。
    steps = int(getattr(args, "clip_refine_steps", 0))
    if steps <= 0:
        return tea_logits

    # 保证 CLIP 处于 eval 且冻结参数（这里只需要对 z 求导）。
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # 将 teacher logits 复制为可优化变量 z；p0 作为“信任锚点”（原始概率分布）。
    z = tea_logits.detach().clone().requires_grad_(True)
    with torch.no_grad():
        p0 = torch.softmax(z.detach(), dim=1)

    # refine 超参数：
    # lr: 每步更新幅度；alpha: 证据图叠加强度；
    # trust_w: 与原始预测保持一致的约束权重；
    # fg_w/bg_w: 前景与背景类别在 CLIP loss 中的权重。
    lr = float(getattr(args, "clip_refine_lr", 0.5))
    alpha = float(getattr(args, "clip_refine_alpha", 0.5))
    trust_w = float(getattr(args, "clip_refine_trust_w", 20.0))
    fg_w = float(getattr(args, "clip_refine_fg_weight", 1.0))
    bg_w = float(getattr(args, "clip_refine_bg_weight", 0.05))

    K = z.shape[1]
    # 类别权重向量：默认前景权重为 fg_w，背景（类0）单独设置为 bg_w。
    w = torch.ones((K,), device=z.device, dtype=z.dtype) * fg_w
    if K > 0:
        w[0] = bg_w

    # 迭代优化 z：z -> p -> evidence -> CLIP score -> loss -> grad(z) -> 更新 z
    for _ in range(steps):
        p = torch.softmax(z, dim=1)
        # 构造可微分的证据图（灰度图 + soft mask 颜色叠加），梯度可回传到 p/z。
        evidence = build_clip_evidence_image(gray_img, p, alpha=alpha, out_size=224)
        # 对齐 CLIP 模型 dtype（例如 fp16/fp32）。
        model_dtype = next(clip_model.parameters()).dtype
        evidence_in = evidence.to(model_dtype)
        # CLIP 图像编码并归一化到单位球，用于和文本特征计算相似度。
        img_feat = clip_model.encode_image(evidence_in).float()
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-6)
        # 计算每个类别的 relative score（相对其它类别的区分性分数）。
        relative_scores = clip_relative_scores(
            img_feat,
            bank_emb=bank_emb,
            use_pool=str(getattr(args, "clip_refine_prompt_pool", "bank")),
            text_feat_cur=text_feat_cur,
            text_feat_init=text_feat_init,
        )  # (B,K)

        # 总损失：
        # 1) clip_loss: 提升目标类别的相对文本匹配分数；
        # 2) trust_loss: 约束当前概率 p 不要偏离初始 p0 太多。
        clip_loss = -(relative_scores * w.view(1, -1)).sum(dim=1).mean()
        trust_loss = F.mse_loss(p, p0)
        loss = clip_loss + trust_w * trust_loss

        # 只对 z 做梯度下降，不构建二阶图；每步后重新开启 z 的 requires_grad。
        g = torch.autograd.grad(loss, z, create_graph=False, retain_graph=False)[0]
        with torch.no_grad():
            z -= lr * g
        z.requires_grad_(True)

    # 返回 refine 后的 logits（与计算图断开，避免影响外部反传）。
    return z.detach()



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


def clip_and_vrr(args, device="cuda"):
    """
    Returns:
    text_feat_init: (K,512) or None 每个类别的初始 CLIP 文本特征
    vrr_callback: callable or None 一个可选的 VRR 回调函数 （在指定扩散步动态更新文本特征）
    vrr_steps: list[int] or None 需要执行 VRR 的扩散步列表
    clip_model: OpenAI CLIP model (frozen) or None (for encode_image/encode_text)
    bank_emb: list[Tensor] or None; length K, each (P,512) normalized text embeddings from JSON prompt bank
    prompt_bank: dict or None; loaded JSON
    class_order: list[str] or None; class name order aligned with indices
    """


    use_clip = True #True
    if not use_clip: # False
        logging.info("[Teacher] CLIP disabled or not available; teacher runs without text.")
        return None, None, None, None, None, None, None

    try:
        clip_model, _ = clip.load(args.clip_model, device=device, jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
    except Exception as e:
        logging.warning(f"[Teacher] CLIP load failed; teacher runs without text. Reason: {e}")
        return None, None, None, None, None, None, None

    # Load prompt bank
    prompt_bank = None
    if args.prompt_bank_path and os.path.isfile(args.prompt_bank_path):
        try:
            with open(args.prompt_bank_path, "r", encoding="utf-8") as f:
                prompt_bank = json.load(f)
            logging.info(f"[Teacher] Loaded prompt bank: {args.prompt_bank_path}")
        except Exception as e:
            logging.warning(f"[Teacher] Failed to read prompt bank; fallback to defaults. Reason: {e}")
            prompt_bank = None

    if prompt_bank is None: # False
        prompt_bank = {
            "background": ["cardiac mri background, non-cardiac tissues"],
            "Right Ventricle": ["Right Ventricle blood pool, crescent cavity"],
            "Myocardium": ["Myocardium, thick ring"],
            "Left Ventricle": ["Left Ventricle blood pool, round cavity"],
        }

    # Class order must match your dataset class indices
    if args.num_classes == 4:
        class_order = ["background", "Right Ventricle", "Myocardium", "Left Ventricle"]
    else:
        class_order = list(prompt_bank.keys())[: args.num_classes]

    bank_emb = []
    for cname in class_order:
        prompts = prompt_bank.get(cname, []) or [cname]
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens).float()
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)
        bank_emb.append(emb)  # (P,512)

    # init text_feat: use first prompt per class
    text_feat_init = torch.stack([e[0] for e in bank_emb], dim=0)  # (K,512)

    # parse vrr steps
    vrr_steps = None
    try:
        vrr_steps = [int(x.strip()) for x in str(args.vrr_steps).split(",") if x.strip() != ""]# [5,10,15]
    except Exception:
        vrr_steps = None

    if  (vrr_steps is None) or len(vrr_steps) == 0:
        logging.info("[Teacher] VRR disabled.")
        return text_feat_init, None, None, clip_model, bank_emb, prompt_bank, class_order

    def make_vrr_callback(bank_emb_list, k_top: int, temp: float):
        # Note: this is a lightweight heuristic VRR. It updates per-class text embedding using teacher x_start statistics.
        def _cb(x_start: torch.Tensor, t_step: int, prev_text: torch.Tensor):
            # x_start: (B,C,H,W)
            v = x_start.mean(dim=(2, 3))  # (B,C)
            v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
            new_feats = []
            for cls_bank in bank_emb_list:  # (P,512)
                # similarity (B,P)
                sim = v @ cls_bank.t()
                topk = min(k_top, cls_bank.shape[0])
                vals, idx = torch.topk(sim, k=topk, dim=-1)
                w = torch.softmax(vals / max(temp, 1e-6), dim=-1)  # (B,topk)
                picked = cls_bank[idx]  # (B,topk,512)
                fused = (w.unsqueeze(-1) * picked).sum(dim=1)  # (B,512)
                fused = fused.mean(dim=0)  # (512,)
                fused = fused / (fused.norm() + 1e-6)
                new_feats.append(fused)
            return torch.stack(new_feats, dim=0)  # (K,512)
        return _cb

    vrr_callback = make_vrr_callback(bank_emb, k_top=args.vrr_k, temp=args.vrr_temp)
    logging.info("[Teacher] CLIP enabled; VRR enabled with steps: {}".format(vrr_steps))
    return text_feat_init, vrr_callback, vrr_steps, clip_model, bank_emb, prompt_bank, class_order


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
    db_val = BaseDataSets(base_dir=args.root_path, split="test")
    db_test = db_val

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
        use_sam_aux=bool(args.use_sam_aux),
        sam_ckpt=args.sam_ckpt,
        sam_model_type=args.sam_model_type,
        sam_img_size=args.sam_img_size,
        sam_max_gain=args.sam_max_gain,
        sam_gate_init=args.sam_gate_init,
        sam_warmup_iters=args.sam_warmup_iters,
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

    # --------- Teacher CLIP + VRR (TRAIN ONLY) ----------
    text_feat_init, vrr_callback, vrr_steps, clip_model, bank_emb, prompt_bank, class_order = clip_and_vrr(args, device="cuda")
    text_feat = text_feat_init

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

    # --------- Color map (for teacher input) ----------
    if args.num_classes == 4:
        color_map = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
    elif args.num_classes == 3:
        color_map = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}
    else:
        color_map = {0: (0, 0, 0), 1: (255, 255, 255)}

    # --------- Distill bottleneck hook (student) ----------
    bn_capture = {"feat": None}

    def _bn_hook(_m, _inp, _out):
        if isinstance(_out, (tuple, list)):
            _out = _out[0]
        if torch.is_tensor(_out) and _out.dim() == 4:
            bn_capture["feat"] = _out

    def _pick_bottleneck_module(net: torch.nn.Module, x: torch.Tensor, num_classes_: int):
        candidates = []
        handles = []

        def _tmp_hook(mod, _inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not (torch.is_tensor(out) and out.dim() == 4):
                return
            b, c, h, w = out.shape
            if c == num_classes_:
                return
            candidates.append((h * w, c, mod))

        for mod in net.modules():
            if isinstance(mod, torch.nn.Conv2d):
                handles.append(mod.register_forward_hook(_tmp_hook))
        try:
            with torch.no_grad():
                _ = net(x)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        if not candidates:
            return None
        candidates.sort(key=lambda t: (t[0], -t[1]))  # smallest spatial, largest channel
        return candidates[0][2]

    bn_handle = None
    try:
        bn_mod = None
        if hasattr(model, "get_bottleneck_hook_module"):
            try:
                bn_mod = model.get_bottleneck_hook_module()
            except Exception:
                bn_mod = None
        if bn_mod is None:
            dummy = torch.zeros((1, args.img_channels, args.patch_size[0], args.patch_size[1]), device="cuda")
            bn_mod = _pick_bottleneck_module(model, dummy, num_classes_=args.num_classes)
        if bn_mod is not None:
            bn_handle = bn_mod.register_forward_hook(_bn_hook)
            logging.info(f"[Distill] selected bottleneck module: {bn_mod}")
        else:
            logging.warning("[Distill] failed to auto-select bottleneck module; feature distill disabled.")
    except Exception as e:
        logging.warning(f"[Distill] hook init failed; feature distill disabled. Reason: {e}")

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
            comp_loss, as_weight = get_comp_loss(out_w_soft, out_s_soft, bs=args.batch_size)

            sup_loss = (
                ce_loss(out_w[: args.labeled_bs], label_batch[: args.labeled_bs].long())
                + dice_loss(out_w_soft[: args.labeled_bs], label_batch[: args.labeled_bs].unsqueeze(1))
            )

            unsup_loss = (
                ce_loss(out_s[args.labeled_bs :], pseudo_outputs[args.labeled_bs :])
                + dice_loss(out_s_soft[args.labeled_bs :], pseudo_outputs[args.labeled_bs :].unsqueeze(1))
                + as_weight * comp_loss
            )

            loss = sup_loss + consistency_weight * unsup_loss
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
            #   Labeled: Strong->Weak (æ®µ1) then midrw->GT (æ®µ2)
            #   Unlabeled: only Strong->Weak (æ®µ1)
            #   ref_comp_loss is removed (æ–¹æ¡ˆ1)
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
                    text_feat=text_feat,
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
                    text_feat=text_feat,
                )
                lat_loss_sup, ref_logits_sup, _aux2 = parse_refine_out(out)
                ref_soft_sup = torch.softmax(ref_logits_sup, dim=1)
                sup_loss_cedice = (
                    ce_loss(ref_logits_sup, label_batch[:lb].long())
                    + dice_loss(ref_soft_sup, label_batch[:lb].unsqueeze(1))
                )
                sup_loss_ref = sup_loss_cedice + lat_loss_sup

            # Segment 1 (Strong -> Weak) for unlabeled only (no ref_comp_loss)
            lat_loss_unsup = torch.tensor(0.0, device="cuda")
            unsup_loss_ref = torch.tensor(0.0, device="cuda")
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
                    text_feat=text_feat,
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

            refine_loss = sup_loss_ref + loss_s2w_lab + ref_consistency_weight * unsup_loss_ref
            refine_optimizer.zero_grad()
            refine_loss.backward()
            refine_optimizer.step()

            # -----------------------------
            # Step3: distill teacher rectification into student (UNLABELED ONLY)
            #   -  硬监督（CE + Dice）：student 学 teacher 的 hard label
            #   -  student 学 teacher 的 soft prob（只在高置信像素上）
            #   - student bottleneck 特征向 teacher 的 rect_feat 对齐
            #   Teacher only at train; test stays student-only.
            # -----------------------------
            if iter_num > args.refine_start:
                with torch.no_grad():
                    # teacher inference uses strong->weak path on unlabeled
                    refine_model.eval()
                    _text = text_feat.detach() if text_feat is not None else None
                    out = refine_forward(
                        pseudo_rgb_s[lb:], # 给 teacher 的伪标签/语义彩色编码
                        t2_unlb_vec,
                        strong_batch[lb:], # 只取 batch 里 unlabeled 部分
                        training=False,
                        text_pos=_text,
                        cfg_pos_scale=args.cfg_pos_scale,
                        cfg_neg_scale=args.cfg_neg_scale,
                        vrr_callback=vrr_callback, # 每一步扩散后做 VRR，更新文本约束
                        vrr_steps=vrr_steps,
                        phys_guidance=bool(args.phys_guidance) and (iter_num >= args.phys_start),
                        phys_lambda0=args.phys_lambda0,
                        phys_gamma=args.phys_gamma,
                        phys_kappa=args.phys_kappa,
                        phys_cmin=args.phys_cmin,
                        phys_iters=args.phys_iters,
                        phys_blend=args.phys_blend,
                        phys_steps=args.phys_steps,
                        return_aux=True,
                    )
                    _lat0, tea_logits, tea_aux = parse_refine_out(out) # teacher 的分类 logits（每像素 C 类）

                    # --- CLIP-guided continuous correction on teacher logits (optional) ---
                    # This runs only occasionally to save compute.
                    if (args.clip_refine_steps > 0) and (args.clip_refine_every > 0) and (iter_num % args.clip_refine_every == 0):
                        # Use the same image input (strong_batch[lb:]) as evidence; refine logits without touching teacher weights.
                        with torch.enable_grad():
                            tea_logits = clip_guided_refine_teacher_logits(
                                tea_logits=tea_logits,
                                gray_img=strong_batch[lb:],  # (B,1,H,W)
                                clip_model=clip_model,
                                bank_emb=bank_emb,
                                text_feat_cur=_text,
                                text_feat_init=text_feat_init,
                                args=args,
                            )
                    tea_prob = torch.softmax(tea_logits, dim=1) # softmax 后的概率图

                    # 置信度筛选：只相信 teacher 高置信的像素
                    tea_mask = (normalize(tea_prob) > args.conf_thresh).float()
                    tea_prob_masked = tea_prob * tea_mask
                    tea_hard = torch.argmax(tea_prob_masked, dim=1)

                    # teacher 的思路特征也拿出来（用于特征蒸馏）
                    tea_feat = None
                    if isinstance(tea_aux, dict):
                        tea_feat = tea_aux.get("rect_feat", None)

                refine_model.train()

                # student forward strong (hook captures bottleneck)
                bn_capture["feat"] = None # 挂 forward hook 抓到的 student bottleneck 特征（稍后用于 feat_loss）
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

                # 3.特征蒸馏：让 student bottleneck 像 teacher rect_feat
                feat_loss = torch.tensor(0.0, device="cuda")
                if args.distill_feat_w > 0 and (tea_feat is not None) and (bn_capture["feat"] is not None):
                    stu_bn = bn_capture["feat"][lb:] # 只取 lb: 作为无标签部分来算硬监督损失
                    # 先把 student和 teacher 的特征图对齐到同样空间大小
                    if stu_bn.shape[-2:] != tea_feat.shape[-2:]:
                        stu_bn = F.interpolate(stu_bn, size=tea_feat.shape[-2:], mode="bilinear", align_corners=False)

                    stu_vec = F.adaptive_avg_pool2d(stu_bn, 1).flatten(1)
                    tea_vec = F.adaptive_avg_pool2d(tea_feat, 1).flatten(1)

                    # 如果维度不一样，做一个固定随机投影（只用于训练，不改网络结构）：
                    if stu_vec.shape[1] != tea_vec.shape[1]:
                        if not hasattr(args, "_distill_proj"):
                            proj = torch.randn((tea_vec.shape[1], stu_vec.shape[1]), device="cuda") / math.sqrt(max(stu_vec.shape[1], 1))
                            setattr(args, "_distill_proj", proj)
                        proj = getattr(args, "_distill_proj")
                        stu_vec = stu_vec @ proj.t()

                    # 用余弦相似度做损失：
                    stu_vec = stu_vec / (stu_vec.norm(dim=1, keepdim=True) + 1e-6)
                    tea_vec = tea_vec / (tea_vec.norm(dim=1, keepdim=True) + 1e-6)
                    feat_loss = 1.0 - (stu_vec * tea_vec).sum(dim=1).mean()

                total_distill = (args.distill_rect_w * rect_loss + args.distill_kl_w * kl_loss + args.distill_feat_w * feat_loss)

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
                    "iter %d | seg_loss %.4f | ref_loss %.4f | distill(rect/kl/feat) %.4f / %.4f / %.4f"
                    % (iter_num, loss.item(), refine_loss.item(),
                       float(rect_loss.item() if 'rect_loss' in locals() else 0.0),
                       float(kl_loss.item() if 'kl_loss' in locals() else 0.0),
                       float(feat_loss.item() if 'feat_loss' in locals() else 0.0))
                )

            # Validation (STUDENT ONLY)
            if iter_num % 50 == 0:
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

    # cleanup hook
    if bn_handle is not None:
        try:
            bn_handle.remove()
        except Exception:
            pass


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

        # 2.2 备份 student 网络文件（根据 net_factory 的 model 名称）
        # 这里给一个你可控的映射：你用哪个网络，就往下加一条
        net_files = []

        # 例：如果你的 student 用的是 UNet 相关文件（按你工程实际路径改）
        # net_files.append("networks/unet.py")

        # 例：如果你要备份 teacher / sequential teacher
        net_files.append("networks/unet_de_512_sequential_TRI_CFG_phys.py")
        # 或者你实际用的是 unet_de_512.py
        # net_files.append("networks/unet_de_512.py")

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
