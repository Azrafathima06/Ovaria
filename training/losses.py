from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(1)
    union = probs.sum(1) + targets.sum(1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce_weight * bce + (1 - bce_weight) * dice


def feature_distill_loss(student_feats: torch.Tensor, teacher_feats: torch.Tensor, mode: str = "mse") -> torch.Tensor:
    teacher_feats = teacher_feats.detach()
    if mode == "mse":
        return F.mse_loss(student_feats, teacher_feats)
    if mode == "cosine":
        return (1 - F.cosine_similarity(student_feats, teacher_feats, dim=1)).mean()
    raise ValueError(f"Unknown feature distillation mode: {mode}")


def kd_logit_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 4.0) -> torch.Tensor:
    teacher_logits = teacher_logits.detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
