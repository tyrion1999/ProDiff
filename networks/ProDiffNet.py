from __future__ import division, print_function

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

from .guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from .guided_diffusion.respace import SpacedDiffusion, space_timesteps
from .guided_diffusion.resample import UniformSampler



class TextCrossAttention2D(nn.Module):
    """Cross-attention from 2D image feature maps to a set of text tokens.

    Args:
        dim_img: channel dimension of image feature map (B, C, H, W)
        dim_txt: feature dimension of text tokens (B, K, D) or (K, D)
        num_heads: attention heads
    """
    def __init__(self, dim_img: int, dim_txt: int = 512, num_heads: int = 8):
        super().__init__()
        assert dim_img % num_heads == 0, "dim_img must be divisible by num_heads"
        self.dim_img = dim_img
        self.dim_txt = dim_txt
        self.num_heads = num_heads
        self.head_dim = dim_img // num_heads

        self.q_proj = nn.Conv2d(dim_img, dim_img, kernel_size=1, bias=False)
        self.k_proj = nn.Linear(dim_txt, dim_img, bias=False)
        self.v_proj = nn.Linear(dim_txt, dim_img, bias=False)
        self.out_proj = nn.Conv2d(dim_img, dim_img, kernel_size=1, bias=False)

        self.norm = nn.GroupNorm(num_groups=max(1, dim_img // 32), num_channels=dim_img)

    def forward(self, feat_map: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """feat_map: (B,C,H,W); text_feat: (K,D) or (B,K,D)"""
        if text_feat is None:
            return feat_map
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(0).expand(feat_map.shape[0], -1, -1)  # (B,K,D)
        B, C, H, W = feat_map.shape
        _, K, D = text_feat.shape

        x = self.norm(feat_map)
        q = self.q_proj(x).view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # (B,h,HW,hd)

        k = self.k_proj(text_feat).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # (B,h,K,hd)
        v = self.v_proj(text_feat).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)  # (B,h,K,hd)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,h,HW,K)
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v)  # (B,h,HW,hd)
        ctx = ctx.transpose(2, 3).contiguous().view(B, C, H, W)

        out = self.out_proj(ctx)
        return feat_map + out

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(), )
        self.conv1 = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.temb_proj = torch.nn.Linear(512,
                                         out_channels)

    def forward(self, x, temb):
        x = self.conv0(x)
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        x = self.conv1(x)
        return x


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x, temb):
        x = self.maxpool(x)
        x = self.conv(x, temb)
        return x


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2, temb):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        # print(x1.shape, x2.shape)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, temb)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 6)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])  # 16
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])  # 32
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])  # 64
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])  # 128
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])  # 256
        self.down5 = DownBlock(self.ft_chns[4], self.ft_chns[5], self.dropout[5])  # 512
    def forward(self, x, temb, embeddings=None):
        x0 = self.in_conv(x, temb)
        if embeddings is not None:
            x0 = x0 + embeddings[0]
        x1 = self.down1(x0, temb)
        if embeddings is not None:
            x1 = x1 + embeddings[1]
        x2 = self.down2(x1, temb)
        if embeddings is not None:
            x2 = x2 + embeddings[2]
        x3 = self.down3(x2, temb)
        if embeddings is not None:
            x3 = x3 + embeddings[3]
        x4 = self.down4(x3, temb)
        if embeddings is not None:
            x4 = x4 + embeddings[4]

        x5 = self.down5(x4, temb)
        if embeddings is not None:
            x5 = x5 + embeddings[5]

        return [x0, x1, x2, x3, x4,x5]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        # self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.out_chns = self.params['out_chns']
        assert (len(self.ft_chns) == 6)

        self.up1 = UpBlock(self.ft_chns[5], self.ft_chns[4], self.ft_chns[4], dropout_p=0.0)  # 512->256
        self.up2 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)  # 256->128
        self.up3 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)  # 128->64
        self.up4 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)  # 64->32
        self.up5 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)  # 32->16

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.out_chns, kernel_size=3, padding=1)  # 16 -> out_chns(=4)


    def forward(self, feature, temb, out_multi=False):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x5 = feature[5]

        outs = []

        x = self.up1(x5, x4, temb)  # -> (B,256,16,16)
        x = self.up2(x, x3, temb)  # -> (B,128,32,32)
        outs.append(x)

        x = self.up3(x, x2, temb)  # -> (B,64,64,64)
        outs.append(x)

        x = self.up4(x, x1, temb)  # -> (B,32,128,128)
        outs.append(x)

        x = self.up5(x, x0, temb)  # -> (B,16,256,256)
        output = self.out_conv(x)  # -> (B,out_chns,256,256)  out_chns=4
        outs.append(output)

        if out_multi:
            return outs
        return output


class DeUNet(nn.Module):
    def __init__(self, base_c: int):
        super(DeUNet, self).__init__()
        self.ft_chns = [base_c, int(base_c * 1.5), base_c * 2]

        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], 0.0)
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], 0.0)
        self.up1 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128, 512),
            torch.nn.Linear(512, 512),
        ])

    def forward(self, x, temb, embeddings=None):
        temb = get_timestep_embedding(temb, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        B, C, H, W = x.shape

        x0 = x
        if embeddings is not None and len(embeddings) == 1:
            x0 = x0 + embeddings[0]

        x1 = self.down1(x, temb)
        x2 = self.down2(x1, temb)

        x = self.up1(x2, x1, temb)
        x = self.up2(x, x0, temb)

        assert x.shape == (B, C, H, W)
        return x


def _build_fg_neg_from_pos(text_pos: torch.Tensor) -> torch.Tensor:
    """Build foreground-only negative tokens from positive class tokens.

    Expected positive order: [BG, RV, MYO, LV] => returns [RVneg, MYOneg, LVneg]
    where each neg token is the mean of the other two foreground positives.

    Args:
      text_pos: (K,512) CLIP text embeddings, K>=4

    Returns:
      (3,512) normalized tensor on the same device/dtype.
    """
    if text_pos is None:
        return None
    if (not torch.is_tensor(text_pos)) or text_pos.dim() != 2 or text_pos.shape[0] < 4:
        raise ValueError(f"text_pos must be a tensor of shape (K,512) with K>=4; got {type(text_pos)} {getattr(text_pos, 'shape', None)}")
    rv, myo, lv = text_pos[1], text_pos[2], text_pos[3]
    rvneg = 0.5 * (myo + lv)
    myoneg = 0.5 * (rv + lv)
    lvneg = 0.5 * (rv + myo)
    neg = torch.stack([rvneg, myoneg, lvneg], dim=0)
    neg = neg / (neg.norm(dim=-1, keepdim=True) + 1e-6)
    return neg


def _normalize_per_image_01(x: torch.Tensor) -> torch.Tensor:
    """Normalize each image to [0,1] independently to stabilize edge guidance."""
    if x is None:
        return None
    vmin = x.amin(dim=(2, 3), keepdim=True)
    vmax = x.amax(dim=(2, 3), keepdim=True)
    return (x - vmin) / (vmax - vmin + 1e-6)


def _resize_phys_guide(img: torch.Tensor, size_hw) -> torch.Tensor:
    """Prepare the grayscale image guidance map at the latent spatial size."""
    if img is None:
        return None
    if img.dim() == 3:
        img = img.unsqueeze(1)
    if img.shape[-2:] != size_hw:
        img = F.interpolate(img.float(), size=size_hw, mode="bilinear", align_corners=False)
    else:
        img = img.float()
    return _normalize_per_image_01(img)

def _parse_step_spec(step_spec, total_steps: int, default=None):
    """Parse comma-separated DDIM step indices or ratios in [0,1] into an absolute index set."""
    if total_steps is None or total_steps <= 0:
        total_steps = 1
    out = set()
    src = default if step_spec is None else step_spec
    if isinstance(src, str):
        items = [s.strip() for s in src.split(",") if s.strip() != ""]
    elif isinstance(src, (list, tuple, set)):
        items = list(src)
    else:
        items = [src]
    for s in items:
        try:
            sv = float(s)
        except Exception:
            continue
        if 0.0 <= sv <= 1.0:
            out.add(int(round(sv * (total_steps - 1))))
        else:
            out.add(int(round(sv)))
    return {min(max(int(v), 0), total_steps - 1) for v in out}


def _anisotropic_heat_step(
    x: torch.Tensor,
    guide_img: torch.Tensor,
    lam: float,
    kappa: float = 0.12,
    cmin: float = 0.05,
) -> torch.Tensor:
    """One explicit Perona–Malik-style diffusion step guided by image edges.

    x: latent feature map to be corrected, shape (B,C,H,W)
    guide_img: edge guide derived from original grayscale image, shape (B,1,H,W)
    """
    if (x is None) or (guide_img is None):
        return x
    if x.dim() != 4 or guide_img.dim() != 4:
        return x

    lam = float(max(0.0, min(lam, 0.24)))  # explicit diffusion stability
    kappa = float(max(kappa, 1e-6))
    cmin = float(max(0.0, min(cmin, 1.0)))

    x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    g_pad = F.pad(guide_img, (1, 1, 1, 1), mode="replicate")

    xc = x_pad[:, :, 1:-1, 1:-1]
    xn = x_pad[:, :, :-2, 1:-1]
    xs = x_pad[:, :, 2:, 1:-1]
    xw = x_pad[:, :, 1:-1, :-2]
    xe = x_pad[:, :, 1:-1, 2:]

    gc = g_pad[:, :, 1:-1, 1:-1]
    gn = g_pad[:, :, :-2, 1:-1]
    gs = g_pad[:, :, 2:, 1:-1]
    gw = g_pad[:, :, 1:-1, :-2]
    ge = g_pad[:, :, 1:-1, 2:]

    # Conduction is high inside homogeneous regions and low across edges.
    c_n = torch.exp(-((gn - gc) / kappa) ** 2).clamp(min=cmin, max=1.0)
    c_s = torch.exp(-((gs - gc) / kappa) ** 2).clamp(min=cmin, max=1.0)
    c_w = torch.exp(-((gw - gc) / kappa) ** 2).clamp(min=cmin, max=1.0)
    c_e = torch.exp(-((ge - gc) / kappa) ** 2).clamp(min=cmin, max=1.0)

    update = c_n * (xn - xc) + c_s * (xs - xc) + c_w * (xw - xc) + c_e * (xe - xc)
    return xc + lam * update


def _apply_anisotropic_heat_guidance(
    x: torch.Tensor,
    guide_img: torch.Tensor,
    lam: float,
    kappa: float = 0.12,
    cmin: float = 0.05,
    n_iter: int = 1,
) -> torch.Tensor:
    """Apply a few very light edge-aware diffusion steps to latent features."""
    if (x is None) or (guide_img is None):
        return x
    n_iter = max(int(n_iter), 1)
    lam_each = float(max(lam, 0.0)) / float(n_iter)
    out = x
    for _ in range(n_iter):
        out = _anisotropic_heat_step(out, guide_img, lam=lam_each, kappa=kappa, cmin=cmin)
    return out


class _TriCFGWrapper(nn.Module):
    """Minimal tri-branch CFG wrapper for DeUNet.

    Supports:
      - legacy: embeddings is list/tuple [cond_feat]
      - tri CFG: embeddings is dict with keys:
          'pos','neg','uncond','cfg_pos_scale','cfg_neg_scale'
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, t, embeddings=None):
        if isinstance(embeddings, dict):
            pos_feat = embeddings.get("pos", None)
            neg_feat = embeddings.get("neg", None)
            uncond_feat = embeddings.get("uncond", None)

            if uncond_feat is None:
                raise ValueError("CFG embeddings dict must include 'uncond'.")

            if pos_feat is None:
                pos_feat = uncond_feat
            if neg_feat is None:
                neg_feat = uncond_feat

            s_pos = float(embeddings.get("cfg_pos_scale", embeddings.get("s_pos", 1.0)))
            s_neg = float(embeddings.get("cfg_neg_scale", embeddings.get("s_neg", 0.0)))

            out_u = self.base_model(x, t, embeddings=[uncond_feat])
            out_p = self.base_model(x, t, embeddings=[pos_feat])
            out_n = self.base_model(x, t, embeddings=[neg_feat])

            return out_u + s_pos * (out_p - out_u) - s_neg * (out_n - out_u)

        return self.base_model(x, t, embeddings=embeddings)

class DiffUNet(nn.Module):
    def __init__(self, ts=1000, ts_sample=10, ldm_sch='linear',base_c=256) -> None:
        super().__init__()

        self.model = DeUNet(base_c=base_c)
        self.cfg_model = _TriCFGWrapper(self.model)

        betas = get_named_beta_schedule(ldm_sch, ts)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(ts, [ts]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(ts, [ts_sample]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(ts)

    def forward(
        self,
        x=None,
        pred_type=None,
        step=None,
        embeddings=None,
        *,
        vrr_callback=None,
        vrr_steps=None,
        text_fuser=None,
        base_feat=None,
        phys_img=None,
        phys_lambda0: float = 0.0,
        phys_gamma: float = 1.5,
        phys_kappa: float = 0.12,
        phys_cmin: float = 0.05,
        phys_iters: int = 1,
        phys_blend: float = 0.10,
        phys_steps=None,
    ):
        """Dispatcher for diffusion operations.

        - q_sample: add noise to x, return (x_t, t, noise)
        - denoise: denoise x at timestep 'step' with conditioning 'embeddings'
        - ddim_sample: run DDIM sampling loop to obtain pred_xstart, optionally updating text conditioning via vrr_callback
        """
        if pred_type == "q_sample":
            noise = torch.randn_like(x)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            x_t = self.diffusion.q_sample(x, t, noise=noise)
            return x_t, t, noise

        elif pred_type == "denoise":
            denoise_image = self.model(x, temb=step, embeddings=embeddings)
            return denoise_image

        elif pred_type == "ddim_sample":
            # Decide conditioning shape and kwargs
            if embeddings is None:
                raise ValueError("ddim_sample requires 'embeddings' as conditioning.")

            # Tri-CFG mode: embeddings is a dict {"pos","neg","uncond",...}
            if isinstance(embeddings, dict):
                ref_feat = embeddings.get("pos", None)
                if ref_feat is None:
                    ref_feat = embeddings.get("uncond", None)
                if ref_feat is None:
                    raise ValueError("CFG embeddings dict must include at least 'pos' or 'uncond' feature.")
                shape = ref_feat.shape
                model_kwargs = {"embeddings": embeddings}
                model_to_use = self.cfg_model
            else:
                if len(embeddings) == 0:
                    raise ValueError("ddim_sample requires non-empty embeddings list.")
                if len(embeddings) == 1:
                    shape = embeddings[0].shape
                    model_kwargs = {"embeddings": embeddings}
                else:
                    shape = embeddings[-3].shape
                    model_kwargs = {"embeddings": embeddings[-2:]}
                model_to_use = self.model

            denoised_fn = None
            use_vrr = (vrr_callback is not None) and (text_fuser is not None) and (base_feat is not None)
            use_phys = (phys_img is not None) and (float(phys_lambda0) > 0.0) and (int(phys_iters) > 0)

            if use_vrr or use_phys:
                # Determine total DDIM steps
                total_steps = None
                if hasattr(self.sample_diffusion, "use_timesteps"):
                    try:
                        total_steps = len(self.sample_diffusion.use_timesteps)
                    except Exception:
                        total_steps = None
                if total_steps is None and hasattr(self.sample_diffusion, "num_timesteps"):
                    try:
                        total_steps = int(self.sample_diffusion.num_timesteps)
                    except Exception:
                        total_steps = None
                if total_steps is None:
                    total_steps = 50

                refresh_set = set()
                if use_vrr:
                    refresh_set = _parse_step_spec(vrr_steps, total_steps, default=[0.0, 0.5, 0.9])

                phys_set = set()
                if use_phys:
                    phys_set = _parse_step_spec(phys_steps, total_steps, default=[0])

                guide_img = None
                if use_phys:
                    try:
                        guide_img = _resize_phys_guide(phys_img, shape[-2:])
                    except Exception:
                        guide_img = None

                state = {"i": -1}

                def denoised_fn(x_start, *args, **kwargs):
                    # Called after each denoise step; keep VRR close to baseline and make physics a tiny residual.
                    state["i"] += 1
                    i = state["i"]

                    if use_vrr and (i in refresh_set):
                        try:
                            new_text = vrr_callback(x_start, step_i=i, total_steps=total_steps)
                        except TypeError:
                            try:
                                new_text = vrr_callback(x_start, i, None)
                            except TypeError:
                                new_text = vrr_callback(x_start)
                                if new_text is None:
                                    new_text = None
                        if new_text is not None:
                            with torch.no_grad():
                                if isinstance(embeddings, dict):
                                    embeddings["text_pos"] = new_text.detach() if torch.is_tensor(new_text) else None
                                    embeddings["pos"] = text_fuser(base_feat, new_text).detach()
                                    if embeddings.get("neg_mode", "from_pos") == "from_pos":
                                        try:
                                            neg_tok = _build_fg_neg_from_pos(new_text)
                                            embeddings["text_neg"] = neg_tok.detach()
                                            embeddings["neg"] = text_fuser(base_feat, neg_tok).detach()
                                        except Exception:
                                            pass
                                else:
                                    emb0 = text_fuser(base_feat, new_text)
                                    if len(embeddings) >= 1:
                                        embeddings[0] = emb0.detach()

                    out = x_start
                    if use_phys and (guide_img is not None) and (i in phys_set):
                        norm_t = float(i) / float(max(total_steps - 1, 1))
                        lam_t = float(phys_lambda0) * math.exp(-float(phys_gamma) * norm_t)
                        phys_out = _apply_anisotropic_heat_guidance(
                            x_start,
                            guide_img,
                            lam=lam_t,
                            kappa=float(phys_kappa),
                            cmin=float(phys_cmin),
                            n_iter=int(phys_iters),
                        )
                        blend = float(max(0.0, min(phys_blend, 1.0)))
                        out = x_start + blend * (phys_out - x_start)

                    return out

            sample_out = self.sample_diffusion.ddim_sample_loop(
                model_to_use,
                shape,
                model_kwargs=model_kwargs,
                denoised_fn=denoised_fn,
            )
            sample_out = sample_out["pred_xstart"]
            return sample_out

        else:
            raise ValueError(f"Unknown pred_type: {pred_type}")

class UNet_LDMV2(nn.Module):
    def __init__(self, in_chns, class_num, out_chns, ldm_method='adaptor', ldm_beta_sch='linear', ts=1000,
                 ts_sample=10):
        super(UNet_LDMV2, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256,512],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5,0.5],
                  'class_num': class_num,
                  'out_chns': out_chns,
                  'bilinear': False,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns - 3,  # in chns - Maskige
                   'feature_chns': [16, 32, 64, 128, 256,512],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5,0.5],
                   'class_num': class_num,
                   'out_chns': out_chns,
                   'bilinear': False,
                   'acti_func': 'relu'}

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128, 512),
            torch.nn.Linear(512, 512),
        ])
        self.encoder = Encoder(params)
        self.embedder = Encoder(params2)
        self.decoder = Decoder(params)

        self.deunet = DiffUNet(ts=ts, ts_sample=ts_sample, ldm_sch=ldm_beta_sch,
                               base_c=params['feature_chns'][-1])  # = 512
        self.de_loss = nn.MSELoss()
        # Text conditioning (teacher-side only)
        self.text_fuser = TextCrossAttention2D(dim_img=params['feature_chns'][-1], dim_txt=512, num_heads=8)


        self.ldm_method = ldm_method

        if ldm_method == 'adaptor':
            self.adaptor = ConvBlock(params['feature_chns'][-1] * 2, params['feature_chns'][-1], 0.0)  # 1024->512

    def get_lat_loss(self, pred, gt):
        return self.de_loss(pred, gt)

    def forward(
        self,
        pseudo_rgb,
        t,
        img,
        training: bool = True,
        good=None,
        save_feature_iter: bool = False,
        iter_num: int = -1,
        *,
        text_feat=None,
        # tri-CFG (teacher DDIM sampling only). If text_pos is None, it falls back to text_feat.
        text_pos=None,
        text_neg=None,
        text_uncond=None,
        cfg_pos_scale: float = 1.0,
        cfg_neg_scale: float = 0.0,
        vrr_callback=None,
        vrr_steps=None,
        phys_guidance: bool = False,
        phys_lambda0: float = 0.003,
        phys_gamma: float = 6.0,
        phys_kappa: float = 0.12,
        phys_cmin: float = 0.0,
        phys_iters: int = 1,
        phys_blend: float = 0.10,
        phys_steps=None,
        return_aux: bool = False,
    ):
        """Forward for rectifier U-Net.

        Notes:
        - 'pseudo_rgb' is a 3-channel pseudo-mask embedding (from student predictions).
        - 'img' is the original image (1 channel in ACDC).
        - When training=False, we use DDIM sampling in latent/bottleneck space.
        - Text/VRR are **teacher-side only**; the student learns via distillation and does not need them at test.
        """
        assert pseudo_rgb is not None and img is not None

        # CLIP/text conditioning is optional; we fuse at bottleneck.
        if text_feat is not None and isinstance(text_feat, torch.Tensor) and text_feat.dim() == 2:
            # (K,512) -> (1,K,512) broadcast inside fuser
            pass

        # timestep embedding for the rectifier's encoder/decoder
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        # image encoder used by rectifier
        if img is not None:
            img_embeddings = self.embedder(img, temb)
            x_in = torch.cat([img, pseudo_rgb], dim=1)
        else:
            img_embeddings = None
            x_in = pseudo_rgb

        # Encode pseudo-mask RGB (+ image channel if provided)
        feature = self.encoder(x_in, temb, img_embeddings)
        base_feat = feature[-1]
        pos_tok = text_pos if (text_pos is not None) else text_feat

        _s_pos = 1.0 if cfg_pos_scale is None else float(cfg_pos_scale)
        _s_neg = 0.0 if cfg_neg_scale is None else float(cfg_neg_scale)
        use_tri_cfg = (not training) and (pos_tok is not None) and ((_s_pos != 1.0) or (_s_neg != 0.0))

        if use_tri_cfg:
            # tri-CFG uses unconditional = base_feat (no extra uncond prompts needed)
            cond_pos = self.text_fuser(base_feat, pos_tok)
            cond_uncond = self.text_fuser(base_feat, text_uncond) if (text_uncond is not None) else base_feat

            if text_neg is None:
                neg_tok = _build_fg_neg_from_pos(pos_tok)  # (3,512), excludes BG
                neg_mode = "from_pos"
            else:
                neg_tok = text_neg
                neg_mode = "explicit"

            cond_neg = self.text_fuser(base_feat, neg_tok) if (neg_tok is not None) else cond_uncond

            embeddings = {
                "pos": cond_pos,
                "neg": cond_neg,
                "uncond": cond_uncond,
                "cfg_pos_scale": _s_pos,
                "cfg_neg_scale": _s_neg,
                # Keep tokens for VRR refresh; safe even if unused
                "text_pos": pos_tok.detach() if torch.is_tensor(pos_tok) else None,
                "text_neg": neg_tok.detach() if torch.is_tensor(neg_tok) else None,
                "neg_mode": neg_mode,
            }
            cond_feat = cond_pos
        else:
            cond_feat = self.text_fuser(base_feat, pos_tok) if (pos_tok is not None) else base_feat
            embeddings = [cond_feat]

        if training:
            assert good is not None
            # Encode 'good' pseudo/label to build x_start target in latent space
            good_in = torch.cat([img, good], dim=1)
            feature_good = self.encoder(good_in, temb, img_embeddings)
            x_start = feature_good[-1].detach()

            # add noise to x_start
            x_t, t_latent, noise = self.deunet(x=x_start, pred_type="q_sample")

            # denoise conditioned on fused bottleneck
            pred_xstart = self.deunet(x=x_t, step=t_latent, pred_type="denoise", embeddings=embeddings)

            # compute latent loss
            lat_loss = self.get_lat_loss(pred_xstart, x_start)

            feat_ref = pred_xstart
        else:
            assert good is None
            # DDIM sampling (teacher inference during training)
            sample_xstart = self.deunet(
                pred_type="ddim_sample",
                embeddings=embeddings,
                vrr_callback=vrr_callback,
                vrr_steps=vrr_steps,
                text_fuser=self.text_fuser,
                base_feat=base_feat,
                phys_img=img if phys_guidance else None,
                phys_lambda0=phys_lambda0,
                phys_gamma=phys_gamma,
                phys_kappa=phys_kappa,
                phys_cmin=phys_cmin,
                phys_iters=phys_iters,
                phys_blend=phys_blend,
                phys_steps=phys_steps,
            )

            feat_ref = sample_xstart

        # integrate refined bottleneck into decoder path
        if self.ldm_method == "adaptor":
            feature[-1] = self.adaptor(feat_ref, temb)
        else:
            feature[-1] = feat_ref

        output = self.decoder(feature, temb, out_multi=True)
        if isinstance(output, (list, tuple)):
            output = output[-1]  # 取最高分辨率那层 logits: [B,C,H,W]

        aux = None
        if return_aux:
            aux = {
                "base_feat": base_feat.detach(),
                "cond_feat": cond_feat.detach(),
                "rect_feat": feat_ref.detach(),
                "text_pos": pos_tok.detach() if torch.is_tensor(pos_tok) else None,
                "cfg_pos_scale": _s_pos,
                "cfg_neg_scale": _s_neg,
                "phys_guidance": bool(phys_guidance),
                "phys_lambda0": float(phys_lambda0),
                "phys_blend": float(phys_blend),
                "phys_steps": phys_steps,
            }

        if training:
            return (lat_loss, output, aux) if return_aux else (lat_loss, output)
        else:
            return (output, aux) if return_aux else output

if __name__ == '__main__':
    model = UNet_LDMV2(4, 2, 2)
    x = torch.rand(1, 3, 256, 256)
    image = torch.rand(1, 1, 256, 256)
    t = torch.rand(1)
    output = model(x, t, image, True)
    print(output[0], output[1].shape)
    output = model(x, t, image, False)
    print(output.shape)
