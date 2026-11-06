"""Aggregate the final ablation: tables, bootstrap CIs, paired tests, and report figures."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from experiments.cache import build_cache
from experiments.train_variant import CachedDS, VARIANTS
from viz.style import PALETTE, apply_style

RES, FIG = Path("results/final"), Path("figures/final")
LABEL = {"baseline": "Baseline (B-mode CE)", "kd": "+ KD logits", "feat": "+ Feature distill",
         "kd_feat": "KD + Feat (Review 1)", "kd_feat_tcw": "KD + Feat + TCW", "clin": "Clinical-descriptor aux only",
         "recon": "CEUS hallucination only", "kd_feat_tcw_clin": "KD+Feat+TCW+Clinical",
         "full": "B2CEUS-Net++ (all losses, w=1)", "feat_clin": "Feat + Clinical (post-hoc)", "feat_clin_recon": "Feat + Clin + Halluc. low-weight (post-hoc)"}
COL = {"baseline": "#8A8D91", "kd_feat": "#2E86AB", "feat_clin_recon": "#C1443C", "feat_clin": "#E8A93A"}
rng = np.random.default_rng(0)


def load_runs():
    runs = {}
    for f in sorted(glob.glob(str(RES / "*_s*.json"))):
        d = json.load(open(f))
        runs.setdefault(d["variant"], {})[d["seed"]] = d
    return runs


def bag_arrays(run):
    t = run["test"]
    return np.array(t["bag_ids"]), np.array(t["y_true"]), np.array(t["y_prob"])


def ens(runs, variant):
    ids, y, _ = bag_arrays(next(iter(runs[variant].values())))
    P = np.stack([bag_arrays(r)[2] for r in runs[variant].values()])
    return ids, y, P


def ece(y, p, bins=8):
    edges = np.linspace(0, 1, bins + 1); e = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def boot_auc(y, p, n=2000):
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], p[i]))
    return np.percentile(out, [2.5, 97.5])


def paired_boot(y, pa, pb, n=4000):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    return d.mean(), np.percentile(d, [2.5, 97.5]), 2 * min((d <= 0).mean(), (d >= 0).mean())


def tables(runs):
    rows, ens_rows = [], []
    base_ids, y, Pb = ens(runs, "baseline")
    pb = Pb.mean(0)
    for v in VARIANTS:
        if v not in runs:
            continue
        ids, y, P = ens(runs, v)
        per = [dict(auc=roc_auc_score(y, p), acc=((p >= .5) == y).mean(), f1=_f1(y, p), auprc=average_precision_score(y, p),
                    bal=_bal(y, p), brier=brier_score_loss(y, p), ece=ece(y, p)) for p in P]
        df = pd.DataFrame(per)
        r = dict(variant=LABEL[v], n_seeds=len(P))
        for c in df:
            r[c] = f"{df[c].mean():.3f} ± {df[c].std(ddof=0):.3f}"
        rows.append(r)
        pe = P.mean(0)
        lo, hi = boot_auc(y, pe)
        d, ci, pv = paired_boot(y, pe, pb) if v != "baseline" else (0, (0, 0), np.nan)
        ens_rows.append(dict(variant=LABEL[v], ens_auc=roc_auc_score(y, pe), auc_ci=f"[{lo:.3f}, {hi:.3f}]",
                             ens_acc=((pe >= .5) == y).mean(), ens_auprc=average_precision_score(y, pe), ens_ece=ece(y, pe),
                             d_auc_vs_baseline=d, d_ci=f"[{ci[0]:.3f}, {ci[1]:.3f}]", p_vs_baseline=pv))
    return pd.DataFrame(rows), pd.DataFrame(ens_rows)


def _f1(y, p):
    pr = p >= .5; tp = (pr & (y == 1)).sum(); d = pr.sum() + (y == 1).sum()
    return 2 * tp / d if d else 0.0


def _bal(y, p):
    pr = p >= .5
    return 0.5 * ((pr & (y == 1)).sum() / max((y == 1).sum(), 1) + ((~pr) & (y == 0)).sum() / max((y == 0).sum(), 1))


def fig_ablation(runs):
    vs = [v for v in VARIANTS if v in runs]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, v in enumerate(vs):
        a = [roc_auc_score(*bag_arrays(r)[1:]) for r in runs[v].values()]
        ax.bar(i, np.mean(a), yerr=np.std(a), color=COL.get(v, "#9DB4C0"), alpha=.9, capsize=4)
        ax.scatter([i] * len(a), a, color="k", s=14, zorder=3)
    ax.set_xticks(range(len(vs))); ax.set_xticklabels([LABEL[v] for v in vs], rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0.55, 0.95); ax.set_ylabel("Test AUC (67 held-out bags)")
    ax.set_title("Ablation: contribution of each privileged-information signal")
    ax.axhline(np.mean([roc_auc_score(*bag_arrays(r)[1:]) for r in runs["baseline"].values()]), ls="--", c="gray", lw=1)
    fig.savefig(FIG / "ablation_auc.png"); plt.close(fig)


def fig_roc(runs):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for v in ("baseline", "kd_feat", "feat_clin_recon"):
        if v not in runs: continue
        _, y, P = ens(runs, v); p = P.mean(0)
        f, t, _ = roc_curve(y, p); ax.plot(f, t, c=COL[v], lw=2, label=f"{LABEL[v]} (AUC {roc_auc_score(y, p):.3f})")
    d = json.load(open("checkpoints/distill/distill_results.json"))["teacher_oracle_test"]
    f, t, _ = roc_curve(d["y_true"], d["y_prob"]); ax.plot(f, t, c="k", ls=":", lw=1.5, label=f"Teacher oracle, uses CEUS ({d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], c="lightgray", lw=1)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity"); ax.set_title("ROC, 3-seed ensembles (test)"); ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIG / "roc_ensembles.png"); plt.close(fig)


def fig_calibration_selective(runs):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
    for v in ("baseline", "feat_clin_recon"):
        if v not in runs: continue
        _, y, P = ens(runs, v); p = P.mean(0)
        edges = np.linspace(0, 1, 6); xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p <= hi if hi == 1 else p < hi)
            if m.sum() >= 2: xs.append(p[m].mean()); ys.append(y[m].mean())
        axs[0].plot(xs, ys, "o-", c=COL[v], label=f"{LABEL[v]} (ECE {ece(y, p):.3f})")
        # selective prediction: reject the most uncertain bags (entropy of ensemble prob + seed disagreement)
        unc = -(p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9)) + P.std(0)
        order = np.argsort(unc); covs, accs = [], []
        for k in range(int(0.4 * len(y)), len(y) + 1):
            keep = order[:k]; covs.append(k / len(y)); accs.append(((p[keep] >= .5) == y[keep]).mean())
        axs[1].plot(covs, accs, c=COL[v], lw=2, label=LABEL[v])
    axs[0].plot([0, 1], [0, 1], c="lightgray"); axs[0].set_xlabel("Predicted malignancy probability"); axs[0].set_ylabel("Observed malignant fraction")
    axs[0].set_title("Reliability (3-seed ens.)"); axs[0].legend(fontsize=8)
    axs[1].set_xlabel("Coverage (fraction of bags auto-diagnosed)"); axs[1].set_ylabel("Accuracy on retained bags")
    axs[1].set_title("Selective prediction"); axs[1].legend(fontsize=8)
    fig.savefig(FIG / "calibration_selective.png"); plt.close(fig)


class Logits(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)["logits"]


def fig_hallucination(cache, device):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from experiments.model import PlusNet
    from skimage.metrics import structural_similarity as ssim
    models = {}
    for k, kw in (("baseline", {}), ("feat_clin_recon", dict(desc=True, recon=True))):
        m = PlusNet(pretrained=False, **kw).to(device)
        m.load_state_dict(torch.load(RES / f"{k}_s42.pt", map_location=device)); m.eval(); models[k] = m
    ds = CachedDS(cache["test"], False)
    paired = [i for i, r in enumerate(cache["test"]) if r["ceus"] is not None]
    mal = [i for i in paired if cache["test"][i]["label"] == 1][:4]
    ben = [i for i in paired if cache["test"][i]["label"] == 0][:3]
    sel = mal + ben
    fig, axs = plt.subplots(len(sel), 5, figsize=(11, 2.2 * len(sel)))
    stats = []
    for r, i in enumerate(paired and sel):
        s = ds[i]; x = s["image"][None].to(device)
        img = cache["test"][i]["img"]; rgb = np.stack([img] * 3, -1).astype(np.float32) / 255
        cams = []
        for k in ("baseline", "feat_clin_recon"):
            cam = GradCAM(model=Logits(models[k]), target_layers=[models[k].backbone.layer4[-1]])
            c = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
            cams.append(show_cam_on_image(rgb, c, use_rgb=True))
        with torch.no_grad():
            hat = models["feat_clin_recon"](x)["ceus_hat"][0, 0].cpu().numpy()
        real = s["ceus"][0].numpy()
        stats.append(dict(ssim=ssim(hat, real, data_range=1.0), pearson=np.corrcoef(hat.ravel(), real.ravel())[0, 1]))
        for c, (im, t) in enumerate([(img, "B-mode ROI"), (cams[0], "Baseline Grad-CAM"), (cams[1], "B2CEUS-Net++ Grad-CAM"),
                                      (hat, "Hallucinated CEUS"), (real, "Real CEUS (privileged)")]):
            axs[r, c].imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0 if im.ndim == 2 else None, vmax=(255 if im is img else 1) if im.ndim == 2 else None)
            axs[r, c].axis("off"); axs[r, c].grid(False)
            if r == 0: axs[r, c].set_title(t, fontsize=9)
        axs[r, 0].text(-0.05, 0.5, "Malignant" if cache["test"][i]["label"] else "Benign", transform=axs[r, 0].transAxes, rotation=90,
                       va="center", ha="right", fontsize=9, color=PALETTE["malignant" if cache["test"][i]["label"] else "benign"])
    fig.savefig(FIG / "hallucination_gradcam.png", dpi=170); plt.close(fig)
    return pd.DataFrame(stats).describe().loc[["mean", "std"]]


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.8)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 5)
    def box(x, y, w, h, t, c, fc="white"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec=c, alpha=.93, zorder=2)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color=fc, fontsize=8.5, fontweight="bold", zorder=3)
    def arr(a, b): ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#444", lw=1.5), zorder=1)
    ax.text(0.1, 4.75, "Privileged signals (training only)", fontsize=9.5, color="#C1443C", fontweight="bold")
    ax.text(3.9, 4.75, "Loss", fontsize=9.5, fontweight="bold"); ax.text(6.3, 4.75, "Student heads", fontsize=9.5, color="#2E86AB", fontweight="bold")
    rows = [("Frozen teacher embedding\n(CEUS + synced B-mode)", "#C1443C", "Feature MSE", "Feature head (512-d)"),
            ("Real CEUS frame\n(paired bags only)", "#C1443C", "L1 reconstruction", "CEUS hallucination decoder"),
            ("Clinical descriptors\n(7 binary findings)", "#6C5B7B", "BCE", "Descriptor head"),
            ("Histopathology label", "#555555", "Cross-entropy", "Malignancy head")]
    for i, (src, c, loss, head) in enumerate(rows):
        y = 3.55 - i * 1.05
        box(0.1, y, 2.9, .8, src, c); box(3.9, y, 1.7, .8, loss, "#777"); box(6.3, y, 2.3, .8, head, "#E8A93A" if i == 1 else ("#2ECC71" if i == 3 else "#6C9BB5"), "#222" if i in (1, 3) else "white")
        arr((3.0, y + .4), (3.9, y + .4)); arr((6.3, y + .4), (5.6, y + .4))
        arr((9.2, 2.45), (8.6, y + .4))
    box(9.2, 1.7, 1.6, 1.5, "ResNet-18\nB-mode\nencoder", "#2E86AB"); arr((10.0, 0.6), (10.0, 1.7))
    ax.text(10.0, 0.3, "B-mode ROI frame", ha="center", fontsize=8.5)
    ax.text(0.1, 0.1, "Inference: B-mode frame -> encoder -> malignancy head only (no CEUS needed)", fontsize=9, color="#1e9e5a", fontweight="bold")
    fig.savefig(FIG / "architecture_plusplus.png", dpi=170); plt.close(fig)


def main():
    apply_style(); FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    t1, t2 = tables(runs)
    t1.to_csv(RES / "table_seed_mean_std.csv", index=False); t2.to_csv(RES / "table_ensemble_bootstrap.csv", index=False)
    print(t1.to_string(index=False)); print(); print(t2.round(4).to_string(index=False))
    fig_ablation(runs); fig_roc(runs); fig_calibration_selective(runs); fig_architecture()
    import sys
    from utils.seed import resolve_device
    cache = build_cache(None, None, Path("cache/final_cache.pkl"))
    if (RES / "feat_clin_recon_s42.pt").exists():
        st = fig_hallucination(cache, resolve_device("auto")); print("\nHallucination vs real CEUS (test paired bags):\n", st)
        st.to_csv(RES / "hallucination_quality.csv")


if __name__ == "__main__":
    main()
