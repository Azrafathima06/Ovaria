from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class CEUSHallucinator(nn.Module):
    """Decodes the B-mode backbone feature map (512x7x7) into a 56x56 'synthetic CEUS'
    image. Supervised only on the one privileged pair per bag; acts as a
    privileged-information regulariser and a visual explanation at inference."""

    def __init__(self, in_ch: int = 512):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.b1, self.b2, self.b3 = blk(in_ch, 128), blk(128, 64), blk(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, fm):
        x = self.b1(fm)
        x = self.b2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.b3(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.out(x))


class PlusNet(nn.Module):
    """ResNet-18 B-mode classifier (identical to the baseline) + optional privileged
    heads: clinical-descriptor head and CEUS hallucination decoder. At inference only
    `logits` is needed, so deployment cost equals the baseline."""

    def __init__(self, backbone="resnet18", pretrained=True, n_desc=7, desc=False, recon=False, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        d = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d, 2)
        self.desc_head = nn.Linear(d, n_desc) if desc else None
        self.decoder = CEUSHallucinator(d) if recon else None

    def forward(self, x):
        fm = self.backbone.forward_features(x)
        feats = fm.mean((2, 3))
        out = {"logits": self.head(self.dropout(feats)), "feats": feats}
        if self.desc_head is not None:
            out["desc"] = self.desc_head(feats)
        if self.decoder is not None:
            out["ceus_hat"] = self.decoder(fm)
        return out
