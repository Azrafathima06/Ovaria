from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

from .style import PALETTE, save_fig


def plot_training_curves(history: dict, title: str, path, dpi: int = 200) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train", color=PALETTE["baseline"])
    axes[0].plot(epochs, history["val_loss"], label="Val", color=PALETTE["malignant"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title}: loss")
    axes[0].legend()

    metric_key = next((k for k in ("val_dice", "val_auc", "val_acc") if k in history), None)
    if metric_key is not None:
        axes[1].plot(epochs, history[metric_key], color=PALETTE["distilled"])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(metric_key.replace("val_", "val "))
        axes[1].set_title(f"{title}: {metric_key.replace('val_', 'val ')}")
    save_fig(fig, path, dpi)


def plot_confusion_matrix(cm, class_names, title, path, dpi: int = 200) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=12)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    save_fig(fig, path, dpi)


def plot_roc_curves(curves: dict, title: str, path, dpi: int = 200) -> None:
    """curves: {name: (y_true, y_prob, color)}"""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA", label="Chance")
    for name, (y_true, y_prob, color) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        from sklearn.metrics import auc as sk_auc
        auc_val = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})", color=color)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    save_fig(fig, path, dpi)


def plot_metric_comparison(metrics_by_model: dict, title: str, path, dpi: int = 200) -> None:
    """metrics_by_model: {model_name: {'accuracy':.., 'f1':.., 'auc':..}}"""
    metric_names = ["accuracy", "f1", "auc"]
    model_names = list(metrics_by_model.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [PALETTE["baseline"], PALETTE["distilled"], PALETTE["teacher"]]
    for i, name in enumerate(model_names):
        vals = [metrics_by_model[name][m] for m in metric_names]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=name, color=colors[i % len(colors)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [m.upper() for m in metric_names])
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    save_fig(fig, path, dpi)
