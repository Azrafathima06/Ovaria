from __future__ import annotations

import torch
import torch.nn.functional as F

from .metrics import aggregate_bag_predictions, classification_metrics


@torch.no_grad()
def evaluate_classifier(model, loader, device) -> dict:
    """Runs the model over `loader`, averages predicted probability per bag
    (multiple pool images share a bag-level label), and scores bag-level
    accuracy/F1/AUC/confusion matrix. Also returns the raw per-image predictions
    for diagnostics."""
    model.eval()
    all_bag_ids, all_probs, all_labels = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_bag_ids.extend(batch["bag_id"])
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())

    bag_ids, agg_probs, agg_labels = aggregate_bag_predictions(all_bag_ids, all_probs, all_labels)
    metrics = classification_metrics(agg_labels, agg_probs)
    metrics["bag_ids"] = bag_ids
    metrics["image_level_n"] = len(all_labels)
    return metrics


@torch.no_grad()
def evaluate_teacher(model, loader, device) -> dict:
    """Same scoring as evaluate_classifier but for the dual-input (CEUS, B-mode)
    teacher -- one sample per bag already, but we still aggregate through the
    same bag-id path for a single code path and consistent output shape."""
    model.eval()
    all_bag_ids, all_probs, all_labels = [], [], []
    for batch in loader:
        ceus = batch["ceus"].to(device)
        bmode = batch["bmode"].to(device)
        logits = model(ceus, bmode)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_bag_ids.extend(batch["bag_id"])
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())

    bag_ids, agg_probs, agg_labels = aggregate_bag_predictions(all_bag_ids, all_probs, all_labels)
    metrics = classification_metrics(agg_labels, agg_probs)
    metrics["bag_ids"] = bag_ids
    return metrics
