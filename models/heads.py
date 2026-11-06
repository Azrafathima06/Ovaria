"""Stretch-goal interfaces, scaffolded for Review 1 and filled in for Review 2.

MCDropoutUncertainty is fully functional (MC-Dropout is cheap: it reuses the
classifier's existing Dropout layers, no retraining needed). SyntheticCEUSDecoder
is an intentional stub -- the architecture and loss are Review-2 work; only the
interface run_review1.py and the student will call is fixed now.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MCDropoutUncertainty:
    """Wraps a trained classifier that has nn.Dropout layers. Runs `n_passes`
    stochastic forward passes with dropout kept active, and flags a sample for
    "recommend real CEUS" when predictive variance exceeds `threshold`."""

    def __init__(self, model: nn.Module, n_passes: int = 20, threshold: float = 0.15):
        self.model = model
        self.n_passes = n_passes
        self.threshold = threshold

    def _enable_dropout_only(self) -> None:
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        self._enable_dropout_only()
        probs = torch.stack(
            [torch.softmax(self.model(x), dim=1) for _ in range(self.n_passes)], dim=0
        )
        mean_probs = probs.mean(dim=0)
        predictive_variance = probs.var(dim=0).sum(dim=1)
        return {
            "mean_probs": mean_probs,
            "predictive_variance": predictive_variance,
            "recommend_ceus": predictive_variance > self.threshold,
        }


class SyntheticCEUSDecoder(nn.Module):
    """Review-2 stub: decode a B-mode backbone feature map into a synthetic
    CEUS-like perfusion map, supervised against real ZY_left frames on paired
    bags. Fixing this interface now (in_channels/out_channels, forward
    signature) lets training/train_distill.py call it as an optional auxiliary
    branch once implemented -- see synthetic_ceus_loss below.
    """

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO(Review 2): conv-transpose upsampling stack, in_channels -> out_channels

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Synthetic-CEUS decoder is a Review-2 deliverable.")


def synthetic_ceus_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Synthetic-CEUS reconstruction loss is a Review-2 deliverable.")
