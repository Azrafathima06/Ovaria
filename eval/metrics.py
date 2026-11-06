from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def aggregate_bag_predictions(bag_ids, probs, labels):
    """Multiple pool images share one bag -> average probability per bag before
    scoring, so metrics reflect one decision per lesion (clinically meaningful
    and comparable to the reference benchmark), not per-frame."""
    by_bag_prob = defaultdict(list)
    by_bag_label = {}
    for bag_id, p, y in zip(bag_ids, probs, labels):
        by_bag_prob[bag_id].append(p)
        by_bag_label[bag_id] = y
    bag_ids_sorted = sorted(by_bag_prob.keys())
    agg_probs = np.array([np.mean(by_bag_prob[b]) for b in bag_ids_sorted])
    agg_labels = np.array([by_bag_label[b] for b in bag_ids_sorted])
    return bag_ids_sorted, agg_probs, agg_labels


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "n": int(len(y_true)),
    }


def dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2 * intersection + eps) / (pred.sum() + gt.sum() + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)
