"""Out-of-fold analysis of the 5-fold CV + 5-model test ensembles. Writes tables/figures to paper_assets/."""
from __future__ import annotations

import glob, json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve, confusion_matrix

from experiments.analyze import ece
from viz.style import apply_style

ASSET = Path("paper_assets"); FIG, TAB = ASSET / "figures", ASSET / "tables"
LAB = {"baseline": "Baseline (B-mode CE)", "feat": "+ Feature distillation", "clin": "+ Clinical findings",
       "feat_clin": "+ Feature + Clinical", "feat_clin_recon": "+ Feature + Clinical + Hallucination"}
COL = {"baseline": "#8A8D91", "feat": "#2E86AB", "clin": "#6C5B7B", "feat_clin": "#E8A93A", "feat_clin_recon": "#C1443C"}
rng = np.random.default_rng(1)


def load():
    R = {}
    for f in glob.glob("results/cv/fold*_*.json"):
        d = json.load(open(f)); R.setdefault(d["variant"], {})[d["fold"]] = d
    return R


def oof(R, v):
    ids, y, p, fold = [], [], [], []
    for k in sorted(R[v]):
        h = R[v][k]["heldout"]; ids += h["bag_ids"]; y += h["y"]; p += h["p"]; fold += [k] * len(p[-0:]) if False else [k] * len(h["y"])
    return np.array(ids), np.array(y), np.array(p), np.array(fold)


def patient_of(bag_id):  # same-patient bags share a numeric prefix before '_' ('54_x', '54_x1')
    return bag_id.split("_")[0] + "_" + "".join(c for c in bag_id.split("_", 1)[1] if c.isalpha())


def cluster_boot(ids, y, pa, pb=None, n=3000):
    pats = np.array([patient_of(i) for i in ids]); up = np.unique(pats); idx = {u: np.where(pats == u)[0] for u in up}
    out = []
    for _ in range(n):
        s = np.concatenate([idx[u] for u in rng.choice(up, len(up))])
        if y[s].min() == y[s].max(): continue
        out.append(roc_auc_score(y[s], pa[s]) - (roc_auc_score(y[s], pb[s]) if pb is not None else 0))
    return np.array(out)


def main():
    apply_style(); FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    R = load(); vs = [v for v in LAB if v in R and len(R[v]) == 5]
    ids0, y0, pb, _ = oof(R, "baseline")
    rows, fold_auc = [], {}
    for v in vs:
        ids, y, p, fold = oof(R, v); assert (ids == ids0).all()
        fa = [roc_auc_score(y[fold == k], p[fold == k]) for k in range(5)]; fold_auc[v] = fa
        bs = cluster_boot(ids, y, p); d = cluster_boot(ids, y, p, pb) if v != "baseline" else np.array([0.0])
        pr = p >= .5
        rows.append(dict(variant=LAB[v], oof_auc=roc_auc_score(y, p), auc_ci=f"[{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]",
                         fold_auc=f"{np.mean(fa):.3f} ± {np.std(fa):.3f}", auprc=average_precision_score(y, p),
                         sens=(pr & (y == 1)).sum() / (y == 1).sum(), spec=((~pr) & (y == 0)).sum() / (y == 0).sum(),
                         acc=(pr == y).mean(), brier=brier_score_loss(y, p), ece=ece(y, p),
                         d_auc=d.mean(), d_ci=f"[{np.percentile(d,2.5):.3f}, {np.percentile(d,97.5):.3f}]" if v != "baseline" else "-",
                         p_boot=(2 * min((d <= 0).mean(), (d >= 0).mean())) if v != "baseline" else np.nan,
                         folds_better=sum(a > b for a, b in zip(fa, fold_auc["baseline"])) if v != "baseline" else np.nan))
    cv = pd.DataFrame(rows); cv.to_csv(TAB / "cv_oof_table.csv", index=False)
    # test ensemble (5 fold models)
    trows, tens = [], {}
    for v in vs:
        P = np.stack([R[v][k]["test"]["p"] for k in range(5)]); ids_t = R[v][0]["test"]["bag_ids"]; yt = np.array(R[v][0]["test"]["y"]); pe = P.mean(0); tens[v] = (yt, pe)
        aucs = [roc_auc_score(yt, q) for q in P]
        trows.append(dict(variant=LAB[v], ens5_auc=roc_auc_score(yt, pe), single_model_auc=f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
                          auprc=average_precision_score(yt, pe), acc=((pe >= .5) == yt).mean(), brier=brier_score_loss(yt, pe), ece=ece(yt, pe)))
    tt = pd.DataFrame(trows); tt.to_csv(TAB / "test_5model_ensemble_table.csv", index=False)
    print(cv.round(4).to_string(index=False)); print(); print(tt.round(4).to_string(index=False))
    (TAB / "cv_oof_table.md").write_text(cv.round(3).to_string(index=False)); (TAB / "test_5model_ensemble_table.md").write_text(tt.round(3).to_string(index=False))

    # Fig: OOF AUC with CI
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, v in enumerate(vs):
        ids, y, p, _ = oof(R, v); bs = cluster_boot(ids, y, p); a = roc_auc_score(y, p)
        ax.bar(i, a, color=COL[v], alpha=.9); ax.errorbar(i, a, yerr=[[a - np.percentile(bs, 2.5)], [np.percentile(bs, 97.5) - a]], c="k", capsize=4)
        ax.scatter([i] * 5, fold_auc[v], c="k", s=12, zorder=3)
        ax.text(i, 0.52, f"{a:.3f}", ha="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(vs))); ax.set_xticklabels([{"baseline":"Baseline","feat":"+Feature\ndistill","clin":"+Clinical\nfindings","feat_clin":"+Feature\n+Clinical","feat_clin_recon":"+Feature+Clin.\n+Halluc."}[v] for v in vs], fontsize=9)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Out-of-fold AUC (257 bags, 83 malignant)"); ax.set_title("5-fold patient-level CV (bars: pooled OOF AUC, 95% cluster-bootstrap CI; dots: folds)")
    ax.title.set_fontsize(10); fig.savefig(FIG / "fig_cv_auc.png"); plt.close(fig)
    # Fig: forest of delta AUC
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for i, v in enumerate([v for v in vs if v != "baseline"]):
        ids, y, p, _ = oof(R, v); d = cluster_boot(ids, y, p, pb)
        ax.errorbar(d.mean(), i, xerr=[[d.mean() - np.percentile(d, 2.5)], [np.percentile(d, 97.5) - d.mean()]], fmt="o", c=COL[v], capsize=4)
    ax.axvline(0, c="gray", ls="--"); ax.set_yticks(range(len(vs) - 1)); ax.set_yticklabels([LAB[v] for v in vs if v != "baseline"], fontsize=8.5)
    ax.set_xlabel("Δ AUC vs baseline (95% CI)"); ax.set_title("Paired improvement over baseline (out-of-fold)"); ax.title.set_fontsize(10)
    fig.savefig(FIG / "fig_cv_delta_forest.png"); plt.close(fig)
    # Fig: OOF ROC
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for v in vs:
        ids, y, p, _ = oof(R, v); f, t, _ = roc_curve(y, p); ax.plot(f, t, c=COL[v], lw=2, label=f"{LAB[v]} ({roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], c="lightgray"); ax.legend(fontsize=7.5, loc="lower right"); ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity"); ax.set_title("Out-of-fold ROC (5-fold CV)")
    fig.savefig(FIG / "fig_cv_roc.png"); plt.close(fig)
    # Fig: per-fold paired lines
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for v in vs: ax.plot(range(1, 6), fold_auc[v], "o-", c=COL[v], label=LAB[v], lw=2 if v in ("baseline", "feat_clin_recon") else 1, alpha=1 if v in ("baseline", "feat_clin_recon") else .6)
    ax.set_xlabel("Fold"); ax.set_ylabel("Held-out AUC"); ax.set_xticks(range(1, 6)); ax.legend(fontsize=7); ax.set_title("Per-fold AUC (same patients across variants)")
    fig.savefig(FIG / "fig_cv_per_fold.png"); plt.close(fig)
    # Fig: calibration OOF + confusion matrices
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for v in ("baseline", "feat_clin_recon"):
        if v not in vs: continue
        ids, y, p, _ = oof(R, v); e = np.linspace(0, 1, 7); xs, ys = [], []
        for lo, hi in zip(e[:-1], e[1:]):
            m = (p >= lo) & (p <= hi if hi == 1 else p < hi)
            if m.sum() >= 3: xs.append(p[m].mean()); ys.append(y[m].mean())
        axs[0].plot(xs, ys, "o-", c=COL[v], label=f"{LAB[v]} (ECE {ece(y, p):.3f})")
    axs[0].plot([0, 1], [0, 1], c="lightgray"); axs[0].legend(fontsize=7); axs[0].set_xlabel("Predicted probability"); axs[0].set_ylabel("Observed malignant fraction"); axs[0].set_title("OOF reliability")
    for ax, v in zip(axs[1:], ("baseline", "feat_clin_recon")):
        ids, y, p, _ = oof(R, v); cm = confusion_matrix(y, p >= .5); ax.imshow(cm, cmap="Blues"); ax.grid(False)
        for (i, j), z in np.ndenumerate(cm): ax.text(j, i, z, ha="center", va="center", fontsize=14, color="white" if z > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Malignant"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "Malignant"]); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(LAB[v], fontsize=9)
    fig.savefig(FIG / "fig_cv_calibration_confusion.png"); plt.close(fig)
    # Fig: test ensemble ROC
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for v in vs:
        yt, pe = tens[v]; f, t, _ = roc_curve(yt, pe); ax.plot(f, t, c=COL[v], lw=2, label=f"{LAB[v]} ({roc_auc_score(yt, pe):.3f})")
    ax.plot([0, 1], [0, 1], c="lightgray"); ax.legend(fontsize=7.5, loc="lower right"); ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity"); ax.set_title("Held-out test (67 bags): 5-fold-model ensembles"); ax.title.set_fontsize(10)
    fig.savefig(FIG / "fig_test_roc_5model.png"); plt.close(fig)
    # stat across folds
    for v in vs:
        if v != "baseline":
            diffs = np.array(fold_auc[v]) - np.array(fold_auc["baseline"])
            print(f"{v}: mean fold ΔAUC {diffs.mean():+.3f}, folds better {int((diffs>0).sum())}/5, Wilcoxon p={wilcoxon(diffs).pvalue:.3f}")


if __name__ == "__main__":
    main()
