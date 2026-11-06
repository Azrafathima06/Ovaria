from __future__ import annotations

import numpy as np
from skimage import measure

import matplotlib.pyplot as plt

from .style import save_fig


def _draw_contours(ax, mask: np.ndarray, color: str, label: str) -> None:
    if mask is None or mask.sum() == 0:
        return
    for contour in measure.find_contours(mask.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.8, label=label)


def plot_segmentation_grid(samples: list, title: str, path, dpi: int = 200) -> None:
    """samples: list of dicts with keys image (H,W), pred (H,W) 0/1, gt (H,W) 0/1 or None, caption (str)"""
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, samples):
        ax.imshow(s["image"], cmap="gray")
        _draw_contours(ax, s.get("gt"), "#2ECC71", "Ground truth")
        _draw_contours(ax, s.get("pred"), "#E84C3D", "Prediction")
        ax.set_title(s.get("caption", ""), fontsize=10)
        ax.axis("off")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            leg = ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=8,
                             framealpha=0.85, facecolor="white")
            leg.get_frame().set_edgecolor("#CCCCCC")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    save_fig(fig, path, dpi)
