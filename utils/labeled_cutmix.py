"""
Student-only labeled virtual CutMix.

This utility builds an extra labeled batch for supervised student training.
It never modifies the clean weak/strong tensors in-place, so the pseudo-label
source and the diffusion teacher path can keep using clean weak/strong inputs.
"""

import random

import torch


__all__ = ["build_student_labeled_cutmix"]


def _compute_centroid(mask: torch.Tensor):
    if mask.dim() == 3:
        mask = mask[0]
    fg = mask > 0
    if fg.sum().item() == 0:
        return None
    ys, xs = torch.nonzero(fg, as_tuple=True)
    cy = int(round(ys.float().mean().item()))
    cx = int(round(xs.float().mean().item()))
    return cy, cx


def _shift_2d(tensor: torch.Tensor, dy: int, dx: int, fill_value=0):
    if tensor.dim() == 2:
        H, W = tensor.shape
        out = torch.full_like(tensor, fill_value=fill_value)
        src_y0 = max(0, -dy)
        src_y1 = min(H, H - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(W, W - dx)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        if src_y1 > src_y0 and src_x1 > src_x0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = tensor[src_y0:src_y1, src_x0:src_x1]
        return out

    if tensor.dim() == 3:
        C, H, W = tensor.shape
        out = torch.full_like(tensor, fill_value=fill_value)
        src_y0 = max(0, -dy)
        src_y1 = min(H, H - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(W, W - dx)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        if src_y1 > src_y0 and src_x1 > src_x0:
            out[:, dst_y0:dst_y1, dst_x0:dst_x1] = tensor[:, src_y0:src_y1, src_x0:src_x1]
        return out

    raise ValueError(f"_shift_2d expects 2D or 3D tensor, got dim={tensor.dim()}")


def build_student_labeled_cutmix(
    weak_batch: torch.Tensor,
    strong_batch: torch.Tensor,
    label_batch: torch.Tensor,
    labeled_bs: int,
    prob: float = 0.25,
    min_area: int = 200,
    max_area: int = 20000,
):
    """
    Build extra virtual labeled samples from the labeled part of a batch.

    Parameters
    ----------
    weak_batch   : (B, C, H, W), clean weak branch tensor.
    strong_batch : (B, C, H, W), clean strong branch tensor.
    label_batch  : (B, H, W), clean label tensor.
    labeled_bs   : number of labeled samples at the front of the batch.
    prob         : per-receiver probability of applying foreground CutMix.
    min_area     : lower foreground area gate for the donor.
    max_area     : upper foreground area gate for the donor.

    Returns
    -------
    cutmix_weak, cutmix_strong, cutmix_label, applied_mask
        The first three tensors contain only the labeled slice [0:labeled_bs].
        applied_mask marks which receiver samples were actually mixed.
    """
    lb = min(int(labeled_bs), int(weak_batch.shape[0]))
    cutmix_weak = weak_batch[:lb].clone()
    cutmix_strong = strong_batch[:lb].clone()
    cutmix_label = label_batch[:lb].clone()
    applied_mask = torch.zeros(lb, dtype=torch.bool, device=weak_batch.device)

    if lb < 2 or prob <= 0:
        return cutmix_weak, cutmix_strong, cutmix_label, applied_mask

    weak_src = weak_batch[:lb].detach()
    strong_src = strong_batch[:lb].detach()
    label_src = label_batch[:lb].detach()
    donor_indices = torch.roll(torch.arange(lb, device=weak_batch.device), shifts=1)

    for receiver in range(lb):
        if random.random() > prob:
            continue

        donor = int(donor_indices[receiver].item())
        donor_label = label_src[donor]
        receiver_label = label_src[receiver]

        donor_centroid = _compute_centroid(donor_label)
        receiver_centroid = _compute_centroid(receiver_label)
        if donor_centroid is None or receiver_centroid is None:
            continue

        donor_fg = donor_label > 0
        donor_area = int(donor_fg.sum().item())
        if donor_area < min_area or donor_area > max_area:
            continue

        dy = receiver_centroid[0] - donor_centroid[0]
        dx = receiver_centroid[1] - donor_centroid[1]

        donor_fg_shifted = _shift_2d(donor_fg.to(torch.float32), dy, dx, fill_value=0.0)
        donor_fg_shifted = donor_fg_shifted > 0.5
        donor_label_shifted = _shift_2d(donor_label.to(torch.long), dy, dx, fill_value=0)
        donor_weak_shifted = _shift_2d(weak_src[donor], dy, dx, fill_value=0.0)
        donor_strong_shifted = _shift_2d(strong_src[donor], dy, dx, fill_value=0.0)

        img_mask = donor_fg_shifted.to(cutmix_weak.dtype).unsqueeze(0)
        cutmix_weak[receiver] = weak_batch[receiver] * (1.0 - img_mask) + donor_weak_shifted * img_mask
        cutmix_strong[receiver] = strong_batch[receiver] * (1.0 - img_mask) + donor_strong_shifted * img_mask
        cutmix_label[receiver] = torch.where(
            donor_fg_shifted,
            donor_label_shifted,
            label_batch[receiver].to(donor_label_shifted.dtype),
        )
        applied_mask[receiver] = True

    return cutmix_weak, cutmix_strong, cutmix_label, applied_mask
