from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

PALETTE = {
    "benign": "#2E86AB",
    "malignant": "#C1443C",
    "baseline": "#8A8D91",
    "distilled": "#2E86AB",
    "teacher": "#C1443C",
    "grid": "#DDDDDD",
    "text": "#222222",
}


def apply_style() -> None:
    matplotlib.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#888888",
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def save_fig(fig, path, dpi: int = 200) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
