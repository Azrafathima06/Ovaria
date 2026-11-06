"""B-mode-only baseline classifier -- the number the LUPI/distillation student
must beat. Trained and evaluated on the exact same ROI-cropped B-mode pool the
student uses later, so any later comparison isolates the effect of the
distillation loss rather than a data or architecture difference."""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.datasets import ClassificationDataset
from data.index import get_or_build_index
from data.splits import get_or_make_split
from data.transforms import build_classification_transforms
from eval.evaluate import evaluate_classifier
from models.backbones import ImageClassifier
from training.engine import EarlyStopper, load_training_state, save_checkpoint, save_training_state
from utils.config import load_config
from utils.seed import resolve_device, set_seed
from viz.plots import plot_confusion_matrix, plot_training_curves
from viz.style import apply_style


def train_baseline(cfg):
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    apply_style()

    bags = get_or_build_index(cfg)
    bag_split = get_or_make_split(
        bags, cfg.data.resolve_path("split_cache"), cfg.data.val_fraction, cfg.data.split_seed
    )

    bcfg = cfg.baseline
    ckpt_path = bcfg.resolve_path("out_dir") / "baseline_best.pt"
    results_path = bcfg.resolve_path("out_dir") / "baseline_results.json"
    if ckpt_path.exists() and results_path.exists():
        print(f"[baseline] found completed run ({ckpt_path.name} + {results_path.name}) — reusing, skipping retraining")
        model = ImageClassifier(bcfg.backbone, pretrained=cfg.pretrain.backbone_init == "imagenet").to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        with open(results_path) as f:
            saved = json.load(f)
        return model, saved["test"], saved["history"]

    image_size = cfg.data.image_size
    train_tf = build_classification_transforms(image_size, train=True)
    eval_tf = build_classification_transforms(image_size, train=False)

    train_ds = ClassificationDataset(bags, bag_split, "train", train_tf)
    val_ds = ClassificationDataset(bags, bag_split, "val", eval_tf)
    test_ds = ClassificationDataset(bags, bag_split, "test", eval_tf)
    print(f"[baseline] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=bcfg.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=bcfg.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    test_loader = DataLoader(test_ds, batch_size=bcfg.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    pretrained = cfg.pretrain.backbone_init == "imagenet"
    model = ImageClassifier(bcfg.backbone, pretrained=pretrained).to(device)

    last_path = bcfg.resolve_path("out_dir") / "baseline_last.pt"
    resume_epoch = torch.load(last_path, map_location=device)["epoch"] if last_path.exists() else None
    already_unfrozen = (
        resume_epoch is not None and bcfg.freeze_backbone_epochs > 0 and resume_epoch >= bcfg.freeze_backbone_epochs
    )
    if bcfg.freeze_backbone_epochs > 0 and not already_unfrozen:
        model.freeze_backbone(True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=bcfg.lr, weight_decay=bcfg.weight_decay
    )
    stopper = EarlyStopper(patience=bcfg.early_stop_patience, mode="max")

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    start_epoch = 1
    if last_path.exists():
        start_epoch, saved_history = load_training_state(last_path, model, optimizer, stopper, map_location=device)
        history = saved_history or history
        start_epoch += 1
        print(f"[baseline] resuming from epoch {start_epoch} (found {last_path.name})")

    for epoch in range(start_epoch, bcfg.epochs + 1):
        if epoch == bcfg.freeze_backbone_epochs + 1 and bcfg.freeze_backbone_epochs > 0:
            model.freeze_backbone(False)
            optimizer = torch.optim.Adam(model.parameters(), lr=bcfg.lr, weight_decay=bcfg.weight_decay)

        model.train()
        train_loss, n = 0.0, 0
        for batch in train_loader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            n += images.size(0)
        train_loss /= n

        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                vn += images.size(0)
        val_loss /= vn
        val_metrics = evaluate_classifier(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_metrics["auc"])

        improved = stopper.step(val_metrics["auc"])
        if improved:
            save_checkpoint(model, ckpt_path)
        save_training_state(last_path, model, optimizer, epoch, stopper, history)
        print(f"[baseline] epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['accuracy']:.4f}{' *' if improved else ''}")
        if stopper.should_stop:
            print(f"[baseline] early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate_classifier(model, test_loader, device)
    print(f"[baseline] TEST: acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f}")

    figures_dir = cfg.viz.resolve_path("figures_dir")
    plot_training_curves(history, "Baseline B-mode classifier", figures_dir / "baseline_training_curves.png", cfg.viz.dpi)
    plot_confusion_matrix(test_metrics["confusion_matrix"], ["Benign", "Malignant"],
                           "Baseline: confusion matrix (test, bag-level)",
                           figures_dir / "baseline_confusion_matrix.png", cfg.viz.dpi)

    results_path = bcfg.resolve_path("out_dir") / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump({"val": {"auc": stopper.best}, "test": test_metrics, "history": history}, f, indent=2)

    return model, test_metrics, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/review1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_baseline(cfg)
