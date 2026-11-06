"""One-off script: draws the teacher-student LUPI architecture diagram for the
Review-1 slide deck. Pure matplotlib shapes, no model/GPU involved."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch
from matplotlib.lines import Line2D

from viz.style import apply_style, save_fig

C_CEUS = "#C1443C"
C_BMODE = "#2E86AB"
C_FUSE = "#6C5B7B"
C_LOSS = "#E8A93A"
C_OUT = "#2ECC71"
C_TEXT = "#222222"


def box(ax, xy, w, h, text, color, fontsize=10.5, fontcolor="white"):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.92)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=fontcolor, fontweight="bold", wrap=True)
    return patch


def arrow(ax, xy_from, xy_to, color=C_TEXT, style="-", lw=1.8, connectionstyle="arc3,rad=0.0"):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=style,
                                 connectionstyle=connectionstyle, shrinkA=2, shrinkB=2))


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # --- Training time (top): teacher sees privileged pair ---
    ax.text(0.2, 7.6, "TRAINING TIME", fontsize=12, fontweight="bold", color=C_TEXT)
    box(ax, (0.3, 6.2), 2.1, 1.0, "Real CEUS\n(ZY_left)", C_CEUS)
    box(ax, (0.3, 4.7), 2.1, 1.0, "Synced B-mode\n(ZY_right)", C_BMODE)

    box(ax, (3.0, 6.2), 2.0, 1.0, "CEUS\nbackbone", C_CEUS)
    box(ax, (3.0, 4.7), 2.0, 1.0, "B-mode\nbackbone", C_BMODE)
    arrow(ax, (2.4, 6.7), (3.0, 6.7), C_CEUS)
    arrow(ax, (2.4, 5.2), (3.0, 5.2), C_BMODE)

    box(ax, (5.6, 5.45), 1.9, 1.0, "Fusion +\nprojection", C_FUSE)
    arrow(ax, (5.0, 6.7), (5.6, 6.1), C_CEUS)
    arrow(ax, (5.0, 5.2), (5.6, 5.8), C_BMODE)

    box(ax, (8.1, 6.2), 1.9, 1.0, "Teacher\nembedding", C_FUSE, fontsize=10)
    box(ax, (8.1, 4.7), 1.9, 1.0, "Teacher\nsoft logits", C_FUSE, fontsize=10)
    arrow(ax, (7.5, 6.1), (8.1, 6.7), C_FUSE)
    arrow(ax, (7.5, 5.9), (8.1, 5.2), C_FUSE)

    box(ax, (10.5, 5.45), 2.2, 1.0, "Hard label\n(histopathology)", "#8A8D91", fontsize=9.5)

    # --- Distillation loss arrows (teacher output -> student output), labels
    # placed in the clear band between the two halves, not on top of any box.
    arrow(ax, (8.9, 6.2), (7.35, 3.0), C_LOSS, lw=2.0, connectionstyle="arc3,rad=-0.25")
    arrow(ax, (8.9, 4.75), (9.65, 3.0), C_LOSS, lw=2.0, connectionstyle="arc3,rad=0.25")
    label_bg = dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.5)
    ax.text(6.9, 4.15, "Feature loss\n(MSE / cosine)", fontsize=9.3, color=C_LOSS,
             fontweight="bold", ha="center", bbox=label_bg)
    ax.text(10.15, 4.15, "KD-logit loss\n(KL divergence, T)", fontsize=9.3, color=C_LOSS,
             fontweight="bold", ha="center", bbox=label_bg)

    # --- Bottom: student, B-mode only, used at both train and inference time ---
    ax.text(0.2, 3.15, "INFERENCE TIME (and student training)", fontsize=12, fontweight="bold", color=C_TEXT)
    box(ax, (0.3, 1.7), 2.6, 1.0, "B-mode only\n(paired + unpaired pool)", C_BMODE)
    box(ax, (3.6, 1.7), 2.0, 1.0, "Student\nbackbone", C_BMODE)
    arrow(ax, (2.9, 2.2), (3.6, 2.2), C_BMODE)

    box(ax, (6.3, 1.95), 1.9, 1.0, "Student\nembedding", C_BMODE, fontsize=10)
    arrow(ax, (5.6, 2.2), (6.3, 2.45), C_BMODE)

    box(ax, (8.8, 1.95), 1.9, 1.0, "Student\nlogits", C_BMODE, fontsize=10)
    arrow(ax, (8.2, 2.45), (8.8, 2.45), C_BMODE)

    box(ax, (11.2, 1.95), 1.5, 1.0, "Benign /\nMalignant", C_OUT, fontcolor="#0A2E1A")
    arrow(ax, (10.7, 2.45), (11.2, 2.45), C_OUT)

    arrow(ax, (11.4, 5.45), (11.4, 2.95), "#8A8D91", lw=2.0, connectionstyle="arc3,rad=0.25")

    # Divider + note
    ax.plot([0, 13], [3.9, 3.9], linestyle="--", color="#BBBBBB", linewidth=1.2)
    ax.text(6.5, 0.5,
             "At inference, only the B-mode branch runs — the teacher, real CEUS, and all distillation losses are discarded.",
             fontsize=10.5, ha="center", style="italic", color=C_TEXT)

    legend_elems = [
        Line2D([0], [0], color=C_LOSS, lw=2.5, label="Distillation loss (train-time only)"),
        Line2D([0], [0], color="#8A8D91", lw=2.5, label="Hard-label supervision"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=9.5, frameon=False, bbox_to_anchor=(1.0, 1.05))

    fig.suptitle("B2CEUS-Net: LUPI Teacher-Student Distillation", fontsize=15, fontweight="bold", y=0.98)
    save_fig(fig, "figures/architecture_diagram.png", dpi=220)


if __name__ == "__main__":
    main()
