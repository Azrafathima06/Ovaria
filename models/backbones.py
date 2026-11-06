from __future__ import annotations

import timm
import torch.nn as nn


def build_backbone(name: str, pretrained: bool) -> tuple:
    """Returns (feature_extractor, feature_dim). Pooled penultimate features only
    (num_classes=0) so the same backbone serves classification heads and
    distillation targets identically."""
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    return model, model.num_features


class ImageClassifier(nn.Module):
    """Single-input B-mode classifier. Used both as the baseline and as the
    distillation student -- identical architecture, so any accuracy delta
    between them is attributable to the distillation loss, not model capacity."""

    def __init__(self, backbone_name: str, pretrained: bool, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.backbone, self.feat_dim = build_backbone(backbone_name, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, return_features: bool = False):
        feats = self.backbone(x)
        logits = self.head(self.dropout(feats))
        if return_features:
            return logits, feats
        return logits

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
