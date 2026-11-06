"""Cross-modal distillation prototype.

1. Train a teacher on the privileged (CEUS, synced B-mode) pair per bag.
2. Freeze the teacher; forward every train/val bag's pair through it once to
   get a per-bag (soft logits, fused embedding) target.
3. Train the student -- identical architecture and identical B-mode-only pool
   as the baseline -- with hard-label CE + KD-logit loss + feature-distillation
   loss against that bag's precomputed teacher target.

At inference, only the student is used; the teacher and all CEUS data are
discardable. We still score the teacher on the test split too (it does see
CEUS there, unlike a real deployment) purely as a research upper bound on how
much signal privileged information carries for this task.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.datasets import ClassificationDataset, PairedDistillDataset
from data.index import get_or_build_index
from data.splits import get_or_make_split
from data.transforms import build_classification_transforms
from eval.evaluate import evaluate_classifier, evaluate_teacher
from models.backbones import ImageClassifier
from models.teacher import TeacherNet
from training.engine import EarlyStopper, load_training_state, save_checkpoint, save_training_state
from training.losses import feature_distill_loss, kd_logit_loss
from utils.config import load_config
from utils.seed import resolve_device, set_seed
from viz.plots import plot_confusion_matrix, plot_training_curves
from viz.style import apply_style


def train_teacher(cfg, bags, bag_split, device):
    dcfg = cfg.distillation
    pretrained = cfg.pretrain.backbone_init == "imagenet"
    ckpt_path = dcfg.resolve_path("out_dir") / "teacher_best.pt"
    last_path = dcfg.resolve_path("out_dir") / "teacher_last.pt"

    if ckpt_path.exists() and not last_path.exists():
        print(f"[teacher] found completed teacher checkpoint ({ckpt_path.name}) — reusing, skipping retraining")
        teacher = TeacherNet(dcfg.teacher.backbone, pretrained=pretrained, fusion=dcfg.teacher.fusion).to(device)
        teacher.load_state_dict(torch.load(ckpt_path, map_location=device))
        return teacher

    image_size = cfg.data.image_size
    train_tf = build_classification_transforms(image_size, train=True)
    eval_tf = build_classification_transforms(image_size, train=False)

    train_ds = PairedDistillDataset(bags, bag_split, "train", train_tf)
    val_ds = PairedDistillDataset(bags, bag_split, "val", eval_tf)
    print(f"[teacher] train={len(train_ds)} val={len(val_ds)} paired bags")

    train_loader = DataLoader(train_ds, batch_size=cfg.baseline.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.baseline.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    teacher = TeacherNet(dcfg.teacher.backbone, pretrained=pretrained, fusion=dcfg.teacher.fusion).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=dcfg.teacher.lr, weight_decay=dcfg.weight_decay)
    stopper = EarlyStopper(patience=cfg.baseline.early_stop_patience, mode="max")

    start_epoch = 1
    if last_path.exists():
        start_epoch, _ = load_training_state(last_path, teacher, optimizer, stopper, map_location=device)
        start_epoch += 1
        print(f"[teacher] resuming from epoch {start_epoch} (found {last_path.name})")

    for epoch in range(start_epoch, dcfg.teacher.epochs + 1):
        teacher.train()
        train_loss, n = 0.0, 0
        for batch in train_loader:
            ceus, bmode, labels = batch["ceus"].to(device), batch["bmode"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            logits = teacher(ceus, bmode)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * ceus.size(0)
            n += ceus.size(0)
        train_loss /= n

        val_metrics = evaluate_teacher(teacher, val_loader, device)
        improved = stopper.step(val_metrics["auc"])
        if improved:
            save_checkpoint(teacher, ckpt_path)
        save_training_state(last_path, teacher, optimizer, epoch, stopper)
        print(f"[teacher] epoch {epoch:03d} train_loss={train_loss:.4f} "
              f"val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['accuracy']:.4f}{' *' if improved else ''}")
        if stopper.should_stop:
            print(f"[teacher] early stopping at epoch {epoch}")
            break

    teacher.load_state_dict(torch.load(ckpt_path, map_location=device))
    return teacher


@torch.no_grad()
def build_teacher_targets(teacher, bags, bag_split, eval_tf, device, splits=("train", "val")):
    teacher.eval()
    targets = {}
    for split_name in splits:
        ds = PairedDistillDataset(bags, bag_split, split_name, eval_tf)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        for batch in loader:
            ceus, bmode = batch["ceus"].to(device), batch["bmode"].to(device)
            logits, feats = teacher(ceus, bmode, return_features=True)
            for bag_id, l, f in zip(batch["bag_id"], logits.cpu(), feats.cpu()):
                targets[bag_id] = (l, f)
    return targets


def train_distill(cfg, teacher=None):
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    apply_style()

    dcfg = cfg.distillation
    pretrained = cfg.pretrain.backbone_init == "imagenet"
    results_path = dcfg.resolve_path("out_dir") / "distill_results.json"
    student_ckpt_path = dcfg.resolve_path("out_dir") / "student_best.pt"
    teacher_ckpt_path = dcfg.resolve_path("out_dir") / "teacher_best.pt"
    if results_path.exists() and student_ckpt_path.exists() and teacher_ckpt_path.exists():
        print(f"[distill] found completed run ({results_path.name}) — reusing, skipping retraining")
        student = ImageClassifier(dcfg.student.backbone, pretrained=pretrained).to(device)
        student.load_state_dict(torch.load(student_ckpt_path, map_location=device))
        teacher_loaded = TeacherNet(dcfg.teacher.backbone, pretrained=pretrained, fusion=dcfg.teacher.fusion).to(device)
        teacher_loaded.load_state_dict(torch.load(teacher_ckpt_path, map_location=device))
        with open(results_path) as f:
            saved = json.load(f)
        return student, teacher_loaded, saved["test"], saved["teacher_oracle_test"], saved["history"]

    bags = get_or_build_index(cfg)
    bag_split = get_or_make_split(
        bags, cfg.data.resolve_path("split_cache"), cfg.data.val_fraction, cfg.data.split_seed
    )

    if teacher is None:
        teacher = train_teacher(cfg, bags, bag_split, device)

    image_size = cfg.data.image_size
    train_tf = build_classification_transforms(image_size, train=True)
    eval_tf = build_classification_transforms(image_size, train=False)

    teacher_targets = build_teacher_targets(teacher, bags, bag_split, eval_tf, device)

    train_ds = ClassificationDataset(bags, bag_split, "train", train_tf)
    val_ds = ClassificationDataset(bags, bag_split, "val", eval_tf)
    test_ds = ClassificationDataset(bags, bag_split, "test", eval_tf)
    print(f"[student] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=dcfg.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=dcfg.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    test_loader = DataLoader(test_ds, batch_size=dcfg.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    student = ImageClassifier(dcfg.student.backbone, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=dcfg.lr, weight_decay=dcfg.weight_decay)
    stopper = EarlyStopper(patience=dcfg.early_stop_patience, mode="max")
    ckpt_path = student_ckpt_path
    last_path = dcfg.resolve_path("out_dir") / "student_last.pt"

    history = {"train_loss": [], "val_loss": [], "val_auc": [],
               "train_hard_loss": [], "train_kd_loss": [], "train_feat_loss": []}
    start_epoch = 1
    if last_path.exists():
        start_epoch, saved_history = load_training_state(last_path, student, optimizer, stopper, map_location=device)
        history = saved_history or history
        start_epoch += 1
        print(f"[student] resuming from epoch {start_epoch} (found {last_path.name})")

    for epoch in range(start_epoch, dcfg.epochs + 1):
        student.train()
        train_loss = hard_loss_sum = kd_loss_sum = feat_loss_sum = 0.0
        n = 0
        for batch in train_loader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            bag_ids = batch["bag_id"]
            teacher_logits = torch.stack([teacher_targets[b][0] for b in bag_ids]).to(device)
            teacher_feats = torch.stack([teacher_targets[b][1] for b in bag_ids]).to(device)

            optimizer.zero_grad()
            student_logits, student_feats = student(images, return_features=True)

            hard_loss = criterion(student_logits, labels)
            kd_loss = kd_logit_loss(student_logits, teacher_logits, dcfg.kd_temperature)
            feat_loss = feature_distill_loss(student_feats, teacher_feats, dcfg.feature_loss)
            loss = (
                dcfg.hard_label_weight * hard_loss
                + dcfg.kd_logit_weight * kd_loss
                + dcfg.feature_loss_weight * feat_loss
            )
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            train_loss += loss.item() * bs
            hard_loss_sum += hard_loss.item() * bs
            kd_loss_sum += kd_loss.item() * bs
            feat_loss_sum += feat_loss.item() * bs
            n += bs

        train_loss /= n
        history["train_loss"].append(train_loss)
        history["train_hard_loss"].append(hard_loss_sum / n)
        history["train_kd_loss"].append(kd_loss_sum / n)
        history["train_feat_loss"].append(feat_loss_sum / n)

        student.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                logits = student(images)
                val_loss += criterion(logits, labels).item() * images.size(0)
                vn += images.size(0)
        val_loss /= vn
        val_metrics = evaluate_classifier(student, val_loader, device)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_metrics["auc"])

        improved = stopper.step(val_metrics["auc"])
        if improved:
            save_checkpoint(student, ckpt_path)
        save_training_state(last_path, student, optimizer, epoch, stopper, history)
        print(f"[student] epoch {epoch:03d} loss={train_loss:.4f} "
              f"(hard={history['train_hard_loss'][-1]:.4f} kd={history['train_kd_loss'][-1]:.4f} "
              f"feat={history['train_feat_loss'][-1]:.4f}) val_auc={val_metrics['auc']:.4f}"
              f"{' *' if improved else ''}")
        if stopper.should_stop:
            print(f"[student] early stopping at epoch {epoch}")
            break

    student.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate_classifier(student, test_loader, device)
    print(f"[student] TEST: acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f}")

    teacher_test_ds = PairedDistillDataset(bags, bag_split, "test", eval_tf)
    teacher_test_loader = DataLoader(teacher_test_ds, batch_size=dcfg.batch_size, shuffle=False)
    teacher_test_metrics = evaluate_teacher(teacher, teacher_test_loader, device)
    print(f"[teacher-oracle] TEST (uses real CEUS, upper bound only): "
          f"acc={teacher_test_metrics['accuracy']:.4f} f1={teacher_test_metrics['f1']:.4f} auc={teacher_test_metrics['auc']:.4f}")

    figures_dir = cfg.viz.resolve_path("figures_dir")
    plot_training_curves(history, "Distillation student", figures_dir / "distill_training_curves.png", cfg.viz.dpi)
    plot_confusion_matrix(test_metrics["confusion_matrix"], ["Benign", "Malignant"],
                           "Distilled student: confusion matrix (test, bag-level)",
                           figures_dir / "distill_confusion_matrix.png", cfg.viz.dpi)

    results_path = dcfg.resolve_path("out_dir") / "distill_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "val_auc_best": stopper.best,
            "test": test_metrics,
            "teacher_oracle_test": teacher_test_metrics,
            "history": history,
        }, f, indent=2)

    return student, teacher, test_metrics, teacher_test_metrics, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/review1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_distill(cfg)
