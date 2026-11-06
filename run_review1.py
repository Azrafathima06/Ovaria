"""Single entry point that reproduces the full Review-1 deliverable:
EDA -> segmentation -> baseline classifier -> distillation (teacher+student)
-> uncertainty-flagging demo -> Grad-CAM wow figure -> final comparison table.

    python run_review1.py --config configs/review1.yaml
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from data.eda import run_eda
from data.index import get_or_build_index
from data.splits import get_or_make_split
from eval.metrics import classification_metrics
from models.heads import MCDropoutUncertainty
from training.train_baseline import train_baseline
from training.train_distill import train_distill
from training.train_segmentation import train_segmentation
from utils.config import load_config
from utils.seed import resolve_device, set_seed
from viz.gradcam import make_wow_figure
from viz.plots import plot_metric_comparison, plot_roc_curves
from viz.style import PALETTE, apply_style

REFERENCE_BENCHMARK = {"accuracy": 0.73, "f1": 0.65}


def run_uncertainty_demo(cfg, student, bags, bag_split, device) -> dict:
    from data.datasets import ClassificationDataset
    from data.transforms import build_classification_transforms
    from torch.utils.data import DataLoader

    ucfg = cfg.uncertainty
    eval_tf = build_classification_transforms(cfg.data.image_size, train=False)
    test_ds = ClassificationDataset(bags, bag_split, "test", eval_tf)
    loader = DataLoader(test_ds, batch_size=cfg.baseline.batch_size, shuffle=False)

    mc = MCDropoutUncertainty(student, n_passes=ucfg.mc_passes, threshold=ucfg.recommend_ceus_threshold)
    n_flagged, n_total = 0, 0
    for batch in loader:
        images = batch["image"].to(device)
        out = mc.predict(images)
        n_flagged += out["recommend_ceus"].sum().item()
        n_total += images.size(0)
    student.eval()  # MCDropoutUncertainty leaves dropout in train() mode; restore
    frac = n_flagged / max(n_total, 1)
    print(f"[uncertainty] {n_flagged}/{n_total} test images ({frac:.1%}) flagged 'recommend real CEUS'")
    return {"n_flagged": n_flagged, "n_total": n_total, "fraction_flagged": frac}


def main(cfg):
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    apply_style()
    print(f"[run_review1] device={device}")

    print("\n=== 1/5 EDA ===")
    eda_summary = run_eda(cfg)

    print("\n=== 2/5 Segmentation ===")
    seg_results, _ = train_segmentation(cfg)

    print("\n=== 3/5 Baseline classifier ===")
    baseline_model, baseline_test_metrics, baseline_history = train_baseline(cfg)

    print("\n=== 4/5 Distillation (teacher + student) ===")
    student_model, teacher_model, student_test_metrics, teacher_test_metrics, distill_history = train_distill(cfg)

    print("\n=== 5/5 Uncertainty demo + Grad-CAM wow figure ===")
    bags = get_or_build_index(cfg)
    bag_split = get_or_make_split(
        bags, cfg.data.resolve_path("split_cache"), cfg.data.val_fraction, cfg.data.split_seed
    )
    uncertainty_summary = run_uncertainty_demo(cfg, student_model, bags, bag_split, device)
    figures_dir = cfg.viz.resolve_path("figures_dir")
    make_wow_figure(student_model, bags, roi_margin=0.3, image_size=cfg.data.image_size, device=device,
                     path=figures_dir / "wow_gradcam_ceus.png")

    # Comparison figures
    metrics_by_model = {
        "Baseline (B-mode)": baseline_test_metrics,
        "Distilled student (B-mode)": student_test_metrics,
        "Teacher oracle (+CEUS)": teacher_test_metrics,
    }
    plot_metric_comparison(
        {k: v for k, v in metrics_by_model.items()},
        "Test-set comparison (bag-level, held-out patients)",
        figures_dir / "final_metric_comparison.png", cfg.viz.dpi,
    )
    plot_roc_curves(
        {
            "Baseline": (baseline_test_metrics["y_true"], baseline_test_metrics["y_prob"], PALETTE["baseline"]),
            "Distilled student": (student_test_metrics["y_true"], student_test_metrics["y_prob"], PALETTE["distilled"]),
            "Teacher oracle": (teacher_test_metrics["y_true"], teacher_test_metrics["y_prob"], PALETTE["teacher"]),
        },
        "ROC: baseline vs. distilled student vs. teacher oracle",
        figures_dir / "final_roc_comparison.png", cfg.viz.dpi,
    )

    # Results table
    rows = []
    for name, m in metrics_by_model.items():
        rows.append({"model": name, "accuracy": m["accuracy"], "f1": m["f1"], "auc": m["auc"], "n_bags": m["n"]})
    rows.append({"model": "Reference benchmark (paper)", "accuracy": REFERENCE_BENCHMARK["accuracy"],
                 "f1": REFERENCE_BENCHMARK["f1"], "auc": float("nan"), "n_bags": float("nan")})
    results_df = pd.DataFrame(rows)
    results_path = cfg.results.resolve_path("table_path")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 72)
    print("REVIEW 1 FINAL SUMMARY")
    print("=" * 72)
    print(results_df.to_string(index=False))
    print("-" * 72)
    delta_acc = student_test_metrics["accuracy"] - baseline_test_metrics["accuracy"]
    delta_f1 = student_test_metrics["f1"] - baseline_test_metrics["f1"]
    delta_auc = student_test_metrics["auc"] - baseline_test_metrics["auc"]
    print(f"Distillation delta over baseline: accuracy {delta_acc:+.3f}, f1 {delta_f1:+.3f}, auc {delta_auc:+.3f}")
    print(f"Segmentation: val dice={seg_results['val_dice_best']:.3f}, "
          f"generalization (held-out TestData) dice={seg_results['test_generalization_dice']:.3f}")
    print(f"Uncertainty head: {uncertainty_summary['fraction_flagged']:.1%} of test images flagged for real CEUS")
    print("=" * 72)

    summary = {
        "eda": eda_summary,
        "segmentation": seg_results,
        "baseline_test": {k: v for k, v in baseline_test_metrics.items() if k not in ("y_true", "y_prob")},
        "student_test": {k: v for k, v in student_test_metrics.items() if k not in ("y_true", "y_prob")},
        "teacher_oracle_test": {k: v for k, v in teacher_test_metrics.items() if k not in ("y_true", "y_prob")},
        "uncertainty_demo": uncertainty_summary,
        "reference_benchmark": REFERENCE_BENCHMARK,
    }
    with open(results_path.parent / "review1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary written to {results_path.parent / 'review1_summary.json'}")
    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/review1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
