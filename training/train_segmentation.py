"""Trains the U-Net on the numbered-frame B-mode/Doppler pool (clean, non-empty
masks for every sample -- see data/datasets.py::SegmentationDataset). Then
applies the frozen model to Data/TestData bags (held-out patients, never seen
during training) to report generalization Dice/IoU, plus qualitative overlays
on the CEUS-paired B-mode frame (ZY_right), which mostly lacks its own GT mask."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import SegmentationDataset, load_grayscale
from data.index import get_or_build_index
from data.masks import load_mask
from data.splits import get_or_make_split
from data.transforms import build_segmentation_transforms
from eval.metrics import dice_iou
from models.unet import UNet
from training.engine import EarlyStopper, load_training_state, save_checkpoint, save_training_state
from training.losses import dice_bce_loss
from utils.config import load_config
from utils.seed import resolve_device, set_seed
from viz.plots import plot_training_curves
from viz.segmentation_overlays import plot_segmentation_grid
from viz.style import apply_style


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_dice, total_iou, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = dice_bce_loss(logits, masks)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
        gts = masks.cpu().numpy()
        for p, g in zip(preds, gts):
            d, i = dice_iou(p[0], g[0])
            total_dice += d
            total_iou += i
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n, total_dice / n, total_iou / n


def train_segmentation(cfg):
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    apply_style()

    bags = get_or_build_index(cfg)
    bag_split = get_or_make_split(
        bags, cfg.data.resolve_path("split_cache"), cfg.data.val_fraction, cfg.data.split_seed
    )

    scfg = cfg.segmentation
    ckpt_path = scfg.resolve_path("out_dir") / "unet_best.pt"
    results_path = scfg.resolve_path("out_dir") / "segmentation_results.json"
    if ckpt_path.exists() and results_path.exists():
        print(f"[seg] found completed run ({ckpt_path.name} + {results_path.name}) — reusing, skipping retraining")
        model = UNet(in_channels=scfg.in_channels, out_channels=1, base_channels=scfg.base_channels).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        with open(results_path) as f:
            results = json.load(f)
        return results, model

    image_size = cfg.data.seg_image_size
    train_tf = build_segmentation_transforms(image_size, train=True)
    val_tf = build_segmentation_transforms(image_size, train=False)

    train_ds = SegmentationDataset(bags, bag_split, "train", train_tf)
    val_ds = SegmentationDataset(bags, bag_split, "val", val_tf)
    test_ds = SegmentationDataset(bags, bag_split, "test", val_tf)
    print(f"[seg] train={len(train_ds)} val={len(val_ds)} test(generalization)={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=scfg.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=scfg.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    test_loader = DataLoader(test_ds, batch_size=scfg.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model = UNet(in_channels=scfg.in_channels, out_channels=1, base_channels=scfg.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=scfg.lr, weight_decay=scfg.weight_decay)
    stopper = EarlyStopper(patience=scfg.early_stop_patience, mode="max")
    last_path = scfg.resolve_path("out_dir") / "unet_last.pt"

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    start_epoch = 1
    if last_path.exists():
        start_epoch, saved_history = load_training_state(last_path, model, optimizer, stopper, map_location=device)
        history = saved_history or history
        start_epoch += 1
        print(f"[seg] resuming from epoch {start_epoch} (found {last_path.name})")

    for epoch in range(start_epoch, scfg.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_dice, val_iou = run_epoch(model, val_loader, device, optimizer=None)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        improved = stopper.step(val_dice)
        if improved:
            save_checkpoint(model, ckpt_path)
        save_training_state(last_path, model, optimizer, epoch, stopper, history)
        print(f"[seg] epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_dice={val_dice:.4f} val_iou={val_iou:.4f}{' *' if improved else ''}")
        if stopper.should_stop:
            print(f"[seg] early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, best_val_dice, best_val_iou = run_epoch(model, val_loader, device, optimizer=None)
    test_loss, test_dice, test_iou = run_epoch(model, test_loader, device, optimizer=None)
    print(f"[seg] GENERALIZATION (held-out TestData patients): dice={test_dice:.4f} iou={test_iou:.4f}")

    figures_dir = cfg.viz.resolve_path("figures_dir")
    plot_training_curves(history, "Segmentation (U-Net)", figures_dir / "segmentation_training_curves.png", cfg.viz.dpi)

    _make_overlay_figures(model, bags, bag_split, device, image_size, figures_dir, val_tf)

    results = {
        "val_dice_best": best_val_dice,
        "val_iou_best": best_val_iou,
        "test_generalization_dice": test_dice,
        "test_generalization_iou": test_iou,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }
    results_path = scfg.resolve_path("out_dir") / "segmentation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    return results, model


@torch.no_grad()
def _make_overlay_figures(model, bags, bag_split, device, image_size, figures_dir, tf, n_examples: int = 3):
    model.eval()

    # 1) Quantitative-style overlays: numbered B-mode frames from held-out TestData bags, GT vs prediction
    test_ds = SegmentationDataset(bags, bag_split, "test", tf)
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(test_ds), size=min(n_examples, len(test_ds)), replace=False)
    samples = []
    for idx in idxs:
        image_path, mask_path, bag_id = test_ds.items[idx]
        raw = load_grayscale(image_path)
        gt = load_mask(mask_path, target_hw=raw.shape)
        item = test_ds[idx]
        logits = model(item["image"].unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]
        disp = item["image"][0].cpu().numpy()
        samples.append({"image": disp, "gt": _resize_like(gt, pred.shape) if gt is not None else None,
                         "pred": pred, "caption": f"{bag_id} (held-out test)"})
    plot_segmentation_grid(samples, "Segmentation generalization to unseen TestData patients",
                            figures_dir / "segmentation_generalization_overlays.png")

    # 2) Qualitative-only: apply to the CEUS-paired B-mode frame (ZY_right), a different
    # acquisition style with mostly no GT mask of its own -- shows the model isn't
    # overfit to one frame style.
    paired_bags = [b for b in bags if b.bmode_paired is not None][: n_examples * 3]
    rng.shuffle(paired_bags)
    samples2 = []
    for b in paired_bags[:n_examples]:
        raw = load_grayscale(b.bmode_paired.path)
        aug = tf(image=raw, mask=np.zeros_like(raw, dtype=np.uint8))
        logits = model(aug["image"].unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]
        disp = aug["image"][0].cpu().numpy()
        samples2.append({"image": disp, "gt": None, "pred": pred, "caption": f"{b.bag_id} (ZY_right, qualitative)"})
    plot_segmentation_grid(samples2, "Qualitative transfer to CEUS-paired B-mode frames",
                            figures_dir / "segmentation_zy_transfer_overlays.png")


def _resize_like(mask, shape):
    if mask is None:
        return None
    from PIL import Image
    im = Image.fromarray((mask * 255).astype(np.uint8)).resize((shape[1], shape[0]), Image.NEAREST)
    return (np.array(im) > 127).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/review1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_segmentation(cfg)
