"""Extra method figures for the paper: end-to-end pipeline and teacher/student training schematic."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("paper_assets/figures"); OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, t, c, fc="white", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec=c, alpha=.94, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color=fc, fontsize=fs, fontweight="bold", zorder=3)


def arr(ax, a, b, c="#444", ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=c, lw=1.6, ls=ls), zorder=1)


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 3.6)
    steps = [("Raw scan\n(dual-screen:\nCEUS | B-mode)", "#555"), ("QC & parsing\nmask-polarity fix\nDoppler detection", "#6C5B7B"), ("Lesion ROI crop\n(mask bbox + 30%)\nCLAHE", "#2E86AB"),
             ("ResNet-18 student\n+ privileged heads\n(train only)", "#C1443C"), ("Bag-level fusion\n(mean of frames)", "#E8A93A"), ("Malignancy risk\n+ uncertainty\n(defer to CEUS)", "#2ECC71")]
    for i, (t, c) in enumerate(steps):
        x = 0.1 + i * 1.98; box(ax, x, 1.2, 1.75, 1.3, t, c, "#222" if c in ("#E8A93A", "#2ECC71") else "white")
        if i: arr(ax, (x - 0.23, 1.85), (x, 1.85))
    ax.text(0.1, 3.2, "End-to-end pipeline", fontsize=12, fontweight="bold")
    ax.text(0.1, 0.55, "Training uses CEUS + clinical findings as privileged information; deployment needs only the B-mode frame.", fontsize=9.5, color="#C1443C")
    ax.text(0.1, 0.15, "ROI crop uses expert masks in all reported experiments; the U-Net (Dice 0.81 val / 0.71 held-out) is a drop-in replacement (not yet evaluated end-to-end).", fontsize=9.5, color="#2E86AB")
    fig.savefig(OUT / "fig_pipeline.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def schematic():
    fig, ax = plt.subplots(figsize=(11, 5)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(-0.3, 5)
    ax.text(0.1, 4.7, "Stage 1: privileged teacher (per fold)", fontsize=10.5, fontweight="bold", color="#C1443C")
    box(ax, 0.1, 3.5, 1.5, .8, "CEUS crop", "#C1443C"); box(ax, 0.1, 2.5, 1.5, .8, "Synced B-mode", "#2E86AB")
    box(ax, 2.1, 2.7, 1.8, 1.4, "Dual ResNet-18\n+ concat fusion\n→ 512-d embedding", "#8C3B34"); arr(ax, (1.6, 3.9), (2.1, 3.6)); arr(ax, (1.6, 2.9), (2.1, 3.1))
    box(ax, 4.3, 3.1, 1.6, .6, "Embedding z_T\n(frozen)", "#8C3B34"); arr(ax, (3.9, 3.4), (4.3, 3.4))
    ax.text(0.1, 1.9, "Stage 2: B-mode student with privileged supervision", fontsize=10.5, fontweight="bold", color="#2E86AB")
    box(ax, 0.1, 0.5, 1.5, .8, "B-mode ROI\n(paired+unpaired)", "#2E86AB"); box(ax, 2.1, 0.4, 1.7, 1.0, "ResNet-18\nencoder", "#2E86AB"); arr(ax, (1.6, .9), (2.1, .9))
    box(ax, 4.3, 1.15, 1.6, .5, "512-d feature", "#6C9BB5"); box(ax, 4.3, 0.55, 1.6, .5, "Descriptor head", "#6C5B7B"); box(ax, 4.3, -0.05, 1.6, .5, "CEUS decoder", "#E8A93A", "#222")
    for yy in (1.4, .8, .2): arr(ax, (3.8, .9), (4.3, yy))
    box(ax, 6.6, 3.1, 1.6, .6, "MSE(z_S, z_T)", "#777"); arr(ax, (5.9, 3.4), (6.6, 3.4)); arr(ax, (5.9, 1.4), (6.9, 3.1), "#444", "--")
    box(ax, 6.6, 1.15, 1.6, .5, "CE (label)", "#777"); box(ax, 6.6, 0.55, 1.6, .5, "BCE (7 findings)", "#777"); box(ax, 6.6, -0.05, 1.6, .5, "L1 vs real CEUS", "#777")
    for yy in (1.4, .8, .2): arr(ax, (5.9, yy), (6.6, yy))
    box(ax, 8.9, 0.5, 2.0, 2.2, "L = CE\n+ 1.0·MSE\n+ 0.25·BCE\n+ 0.1·L1", "#444", fs=10)
    for yy in (3.4, 1.4, .8, .2): arr(ax, (8.2, yy), (8.9, 1.6 if yy < 3 else 2.5))
    ax.text(8.9, 3.3, "Inference: encoder +\nmalignancy head only", fontsize=9, color="#1e9e5a", fontweight="bold")
    fig.savefig(OUT / "fig_training_schematic.png", dpi=180, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    pipeline(); schematic()
