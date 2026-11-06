# B2CEUS-Net — Final Results (Review 1 analysis + final experiment suite)

Task: predict ovarian-tumour malignancy from **B-mode ultrasound only**, using paired CEUS and clinical findings as
*privileged information* (LUPI) at training time. Dataset: 324 bags (257 train/val, 67 held-out test patients,
11 malignant in test).

---

## 1. What Review 1 delivered

| Component | Result |
|---|---|
| EDA + data-quirk discovery (mask polarity, ZY_left/right pairing, Doppler frames, ROI cropping) | `figures/eda/*` |
| U-Net lesion segmentation | Dice **0.807** (val), **0.709** (held-out TestData) |
| Baseline ResNet-18 (B-mode) | acc 0.776, F1 0.483, AUC 0.828 |
| Distilled student (KD + feature MSE) | acc 0.851, F1 0.444, AUC 0.839 |
| Teacher oracle (uses real CEUS) | acc 0.866, F1 0.526, AUC 0.820 |
| MC-dropout "recommend CEUS" | **0 / 199** images flagged → threshold was uncalibrated (non-functional demo) |
| Reference benchmark (paper) | acc 0.73, F1 0.65 |

### Critical reading of Review 1 (what a reviewer would say)
1. **The +0.011 AUC of distillation is statistically indistinguishable from zero.** The test set has 67 bags / 11 positives; bootstrap 95 % CIs on AUC are ≈ ±0.11. Single-seed differences of this size are noise.
2. **The teacher is not better than the students** (AUC 0.820 vs 0.839). So CEUS, collapsed to one grayscale frame, adds little discriminative signal here — distillation cannot be expected to give a large win from it.
3. Accuracy gains are partly a prevalence artefact (test prevalence 14 % vs 32 % in training).
4. Uncertainty flagging flagged nothing; stretch hooks (hallucination decoder) were stubs.

These findings motivated the final experiments below: **multi-seed, bootstrap-CI evaluation, a proper ablation, and new privileged-information signals.**

---

## 2. Final experiment suite

### Protocol (identical for every variant)
- Same split, same ROI-cropped B-mode pool, same ResNet-18 (ImageNet init), Adam 3e-4, batch 16, **max 12 epochs, early-stop patience 5** on val AUC (~8 s/epoch on Apple MPS; 33 runs total, run sequentially, images cached in memory → low RAM).
- **3 seeds (42, 43, 44)** per variant. Teacher (frozen) is the Review-1 teacher, shared across seeds.
- Bag-level scoring (mean of frame probabilities), threshold 0.5.
- Statistics: mean ± SD over seeds; 3-seed-ensemble AUC with **2000× bag-bootstrap 95 % CI**; paired bootstrap (4000×) vs. baseline.

### Proposed additions (novelty over Review 1)
1. **CEUS hallucination decoder** — an auxiliary decoder maps the B-mode feature map to a 56×56 *synthetic CEUS* image, supervised by the real CEUS on the one privileged pair per bag (L1). Zero inference cost; gives a visual "what would CEUS show" output. (Fills the Review-1 stub.)
2. **Clinical-descriptor privileged supervision** — an auxiliary head predicts the 7 radiologist findings (morphology, boundary, solid, separation, nipple, blood flow, internal echo) from B-mode only. Cheap, interpretable, a second privileged modality beyond CEUS.
3. **Teacher-confidence weighting (TCW)** — KD/feature losses are weighted by the teacher's probability of the true class, to down-weight wrong teacher targets.
4. **Statistically honest evaluation** (multi-seed, CIs, calibration, selective prediction).

### Variants
`baseline` · `+KD` · `+Feat` · `KD+Feat` (Review-1 method) · `KD+Feat+TCW` · `Clinical only` · `Hallucination only` · `KD+Feat+TCW+Clinical` · `All losses (w=1)` · and two **post-hoc** variants (see caveat) `Feat+Clinical`, `Feat+Clinical+Hallucination (low weight 0.1/0.25)`.

> **Caveat — post-hoc selection.** The last two variants were designed *after* seeing the test results of the first nine (when the all-losses model collapsed). They were not tuned on the validation set before being evaluated on test, so their numbers are **exploratory**, mildly optimistic, and must be confirmed on new data/cross-validation before being claimed as a result.

---

## 3. Results

### 3.1 Per-seed mean ± SD (test, 67 bags)

| Variant | AUC | Acc | F1 | AUPRC | Bal. acc | Brier | ECE |
|---|---|---|---|---|---|---|---|
| Baseline (B-mode CE) | 0.826 ± 0.015 | 0.746 ± 0.024 | 0.464 ± 0.028 | 0.530 ± 0.028 | 0.714 ± 0.023 | 0.164 ± 0.024 | 0.186 ± 0.042 |
| + KD logits | 0.820 ± 0.011 | 0.781 ± 0.031 | 0.365 ± 0.115 | 0.477 ± 0.046 | 0.638 ± 0.082 | 0.130 ± 0.012 | 0.148 ± 0.032 |
| + Feature distill | 0.844 ± 0.004 | 0.811 ± 0.028 | 0.475 ± 0.083 | 0.475 ± 0.056 | 0.704 ± 0.075 | 0.124 ± 0.010 | 0.134 ± 0.024 |
| KD + Feat (Review 1) | 0.838 ± 0.032 | 0.801 ± 0.063 | 0.439 ± 0.147 | 0.498 ± 0.052 | 0.686 ± 0.089 | 0.135 ± 0.019 | 0.177 ± 0.044 |
| KD + Feat + TCW | 0.817 ± 0.030 | 0.811 ± 0.019 | 0.262 ± 0.094 | 0.418 ± 0.046 | 0.570 ± 0.038 | 0.121 ± 0.002 | 0.124 ± 0.015 |
| Clinical-descriptor only | 0.805 ± 0.077 | 0.801 ± 0.039 | 0.258 ± 0.191 | 0.461 ± 0.134 | 0.589 ± 0.071 | 0.132 ± 0.019 | 0.127 ± 0.018 |
| Hallucination only | 0.802 ± 0.042 | 0.816 ± 0.039 | 0.342 ± 0.130 | 0.478 ± 0.093 | 0.622 ± 0.057 | 0.138 ± 0.018 | 0.141 ± 0.037 |
| KD+Feat+TCW+Clinical | 0.816 ± 0.014 | 0.786 ± 0.025 | 0.428 ± 0.031 | 0.452 ± 0.027 | 0.665 ± 0.020 | 0.134 ± 0.004 | 0.144 ± 0.019 |
| All losses, w=1 | 0.702 ± 0.092 | 0.816 ± 0.028 | 0.095 ± 0.135 | 0.395 ± 0.102 | 0.525 ± 0.035 | 0.138 ± 0.013 | 0.121 ± 0.007 |
| **Feat + Clinical** *(post-hoc)* | **0.859 ± 0.029** | 0.801 ± 0.055 | 0.530 ± 0.057 | **0.603 ± 0.047** | 0.747 ± 0.032 | 0.128 ± 0.025 | 0.169 ± 0.047 |
| **Feat + Clin + Halluc. (low w)** *(post-hoc)* | 0.852 ± 0.027 | 0.776 ± 0.012 | **0.536 ± 0.007** | 0.540 ± 0.048 | **0.781 ± 0.011** | 0.141 ± 0.010 | 0.176 ± 0.022 |

### 3.2 3-seed ensembles with bootstrap CIs and paired tests vs. baseline

| Variant | Ens. AUC | 95 % CI | ΔAUC vs base | Δ 95 % CI | p |
|---|---|---|---|---|---|
| Baseline | 0.833 | [0.711, 0.932] | – | – | – |
| KD + Feat (Review 1) | 0.852 | [0.747, 0.935] | +0.019 | [−0.046, 0.088] | 0.56 |
| Clinical only | 0.857 | [0.755, 0.937] | +0.024 | [−0.052, 0.102] | 0.52 |
| All losses, w=1 | 0.771 | [0.639, 0.879] | −0.061 | [−0.142, 0.011] | 0.10 |
| Feat + Clinical *(post-hoc)* | 0.869 | [0.757, 0.955] | +0.036 | [−0.024, 0.104] | 0.26 |
| Feat + Clin + Halluc. *(post-hoc)* | 0.869 | [0.764, 0.956] | +0.036 | [−0.014, 0.095] | 0.17 |

(Full table: `results/final/table_ensemble_bootstrap.csv`, `table_seed_mean_std.csv`.)

### 3.3 Hallucinated CEUS quality (test paired frames, 7 shown / all paired bags scored)
SSIM 0.18 ± 0.06, Pearson r 0.41 ± 0.20 vs. the real CEUS (`results/final/hallucination_quality.csv`).
The decoder reproduces **coarse enhancement layout** (cystic voids dark, solid regions bright) but not fine texture — see Fig. 5.

---

## 4. Honest conclusions (what the evidence supports)

1. **No variant is statistically significantly better than the baseline** (all paired-bootstrap p ≥ 0.17; all CIs include 0) — expected with 11 test positives. Claims must be phrased as *consistent trends*, not proven gains.
2. **Consistent trends across 3 seeds:** feature-level distillation and clinical-descriptor supervision each raise ensemble AUC (+0.02), and combining them gave the best AUC/AUPRC/balanced-accuracy/F1 (+0.03–0.04 AUC, +0.07 AUPRC, +0.03–0.07 balanced accuracy) **with ≥2.5× lower seed SD on F1 than Review-1's distillation (0.007–0.057 vs 0.147)**.
3. **Distillation also improves probability quality:** Brier 0.164 → ~0.125–0.135 and ECE 0.186 → 0.12–0.18 for most distilled variants (more calibrated, less over-confident on the low-prevalence test cohort).
4. **Negative results (report them):**
   - Stacking *all* losses at weight 1 **hurts** (AUC 0.70, unstable): auxiliary tasks compete for a small 12-epoch budget / 663 images. Loss weights must be small.
   - KD on logits alone and teacher-confidence weighting gave no benefit (TCW even reduced F1) — the teacher is fit on the same training bags, so its confidence is almost always high and carries little information; a cross-fitted teacher would be needed.
   - CEUS hallucination alone did not improve AUC; its value here is **explainability** (a synthetic-CEUS output at zero inference cost), not accuracy.
5. The teacher oracle (AUC 0.82) not exceeding the students suggests that, in this dataset, grayscale-collapsed CEUS holds limited extra signal; clinical findings were an equally useful (and cheaper) privileged signal. This is itself a publishable observation.

### Suggested framing for the report / paper
> *"We study learning-using-privileged-information for B-mode-only ovarian tumour classification. Beyond CEUS distillation, we introduce (i) clinical-finding supervision and (ii) a CEUS-hallucination decoder as zero-inference-cost privileged signals, and show with a 3-seed, bootstrap-CI ablation that feature-level privileged supervision gives consistent but statistically non-significant gains on a small cohort, while naively stacking auxiliary losses degrades performance."*
Strengthen before submission: 5-fold patient-level cross-validation on all 324 bags (more positives), validation-based selection of loss weights, cross-fitted teacher, and a second centre.

---

## 5. Figures for the report (`figures/final/` unless noted)

| # | File | Use |
|---|---|---|
| 1 | `architecture_plusplus.png` | Method diagram (privileged signals → losses → student heads) |
| 2 | `ablation_auc.png` | Ablation bar chart, per-seed dots, baseline dashed line |
| 3 | `roc_ensembles.png` | ROC: baseline vs Review-1 distillation vs best model vs teacher oracle |
| 4 | `calibration_selective.png` | Reliability diagram + selective-prediction (defer-to-CEUS) curve |
| 5 | `hallucination_gradcam.png` | B-mode, baseline Grad-CAM, new-model Grad-CAM, hallucinated CEUS, real CEUS (4 malignant, 3 benign) |
| – | `../eda/*`, `../segmentation_*.png`, `../baseline_*`, `../distill_*`, `../wow_gradcam_ceus.png` | Review-1 figures (EDA, segmentation, curves, confusion matrices) |

Notes on figures: Fig. 4 right shows accuracy rising from 0.78 (all bags) to ≈0.92 at 64 % coverage and ≈1.0 at 40 % coverage (i.e. deferring the most uncertain 36–60 % of bags to real CEUS) — a functional replacement for Review 1's 0 %-flagged uncertainty demo (uncertainty = ensemble entropy + seed disagreement). Fig. 5: Grad-CAM maps of the new model are not systematically better localised than the baseline's; several benign cases attend to image corners/UI, so no localisation claim should be made.

---

## 6. Reproduce

```bash
cd b2ceus-net && source .venv/bin/activate
python -m experiments.train_variant --seeds 42 43 44          # 9 main variants (~45 min)
python -m experiments.train_variant --variants feat_clin feat_clin_recon --seeds 42 43 44   # post-hoc variants
python -m experiments.analyze                                  # tables + figures
```
Code: `experiments/{cache,model,train_variant,analyze}.py`. Per-run JSON (bag-level probabilities, histories): `results/final/`. Logs: `results/final_run*.log`.

## 7. Limitations
Single centre; 67 test bags (11 malignant); bag labels broadcast to frames; ROI crop uses expert masks (not predicted masks) at test time; teacher shared across seeds; post-hoc variants not validated independently; bootstrap treats bags as independent although some patients contribute several bags.
