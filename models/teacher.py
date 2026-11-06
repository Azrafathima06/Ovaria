from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import build_backbone


class TeacherNet(nn.Module):
    """Sees the privileged pair (CEUS + synced B-mode) and fuses both branches.
    The fused embedding is projected to the same dimensionality as the student's
    single-branch feature vector, so it can be used directly as a distillation
    target (feature MSE/cosine) in addition to producing soft logits for KD."""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        num_classes: int = 2,
        fusion: str = "concat",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.ceus_backbone, feat_dim = build_backbone(backbone_name, pretrained)
        self.bmode_backbone, _ = build_backbone(backbone_name, pretrained)
        self.fusion = fusion
        self.feat_dim = feat_dim
        fused_dim = feat_dim * 2 if fusion == "concat" else feat_dim
        self.fuse_proj = nn.Sequential(nn.Linear(fused_dim, feat_dim), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, ceus: torch.Tensor, bmode: torch.Tensor, return_features: bool = False):
        f_ceus = self.ceus_backbone(ceus)
        f_bmode = self.bmode_backbone(bmode)
        fused = torch.cat([f_ceus, f_bmode], dim=1) if self.fusion == "concat" else f_ceus + f_bmode
        fused = self.fuse_proj(fused)
        logits = self.head(self.dropout(fused))
        if return_features:
            return logits, fused
        return logits
