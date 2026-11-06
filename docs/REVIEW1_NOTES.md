# B2CEUS-Net — Review 1

Predicts ovarian tumor malignancy from B-mode ultrasound alone, distilling
knowledge from paired CEUS (contrast-enhanced ultrasound) that is only
available at training time (Learning Using Privileged Information / teacher
student). At inference, only a B-mode frame is needed.

## Setup

```bash
cd b2ceus-net
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12 / PyTorch 2.11 on Apple Silicon (MPS). `utils/seed.py`
auto-selects CUDA > MPS > CPU; override with `device:` in the config.

## Data layout this expects

```
Data/dataset/{Malignant,Benign}/<bag_id>/Original/{ZY_left,ZY_right,1,2,...}.*
Data/dataset/{Malignant,Benign}/<bag_id>/Mask/{ZY_left,ZY_right,1,2,...}.*
Data/TestData/<bag_id>/Original|Mask/...                 (same layout, held-out patients)
labels/clinical_data.csv, labels/test_clinical_data.csv  (English_Name, ..., Malignant)
```

Paths live in `configs/review1.yaml` under `data:` (relative to `b2ceus-net/`
by default) — point them at your copy if the layout differs. Nothing else
needs to change; the loader in `data/index.py` scans whatever it finds there.

## Run everything

```bash
python run_review1.py --config configs/review1.yaml
```

Reproduces, in order: EDA → segmentation → baseline B-mode classifier →
distillation (teacher + student) → uncertainty-flagging demo → Grad-CAM "wow"
figure → final comparison table. Everything lands in `figures/` and `results/`.

Individual stages can also be run standalone:

```bash
python -m data.eda --config configs/review1.yaml
python -m training.train_segmentation --config configs/review1.yaml
python -m training.train_baseline --config configs/review1.yaml
python -m training.train_distill --config configs/review1.yaml
```

## Data quirks this pipeline was built around

These were discovered by actually opening the images/masks, not assumed from
the brief — see the docstrings in `data/index.py` and `data/masks.py` for the
verification details.

1. **Every one of the 324 bags has a genuine privileged pair.** `Original/ZY_left.*`
   is real CEUS; `Original/ZY_right.*` is the B-mode frame captured
   *simultaneously* on the scanner's dual-screen display (pixel-aligned,
   confirmed visually). Additional numbered frames (0–3 per bag: `1.jpg`,
   `2.jpg`, ...) are B-mode/Doppler-only, no CEUS counterpart. So "paired vs.
   unpaired" is real in this dataset, but at the *frame* level, not the
   patient level — every bag contributes exactly one paired sample plus a
   variable unpaired B-mode pool.
2. **Some numbered frames are Color Doppler**, not plain grayscale 2D —
   detected with a colorfulness heuristic (not filename), grayscale-converted,
   and included in the B-mode pool (toggle: `data.include_doppler_frames`).
3. **Mask files use two opposite value-polarity encodings.** Numbered-frame
   masks (`.nii.gz`) use background=0. `ZY_*` masks (`.jpg`/`.bmp`) encode
   background as whichever value is the *majority* pixel — 0, 1, or 2
   depending on the file. Thresholding `>0` silently inverts the `.bmp` masks.
   `data/masks.py::binarize_mask` handles this generically: foreground = "not
   the modal value," with a minimum-area floor to reject sub-pixel
   compression-artifact "masks."
4. **`ZY_right`'s own mask file is essentially never a real annotation** —
   323/324 are blank or tiny compression artifacts, not lesion outlines. ROI
   cropping for that frame always falls back to the bag's CEUS mask instead of
   trusting its own mask file.
5. **`Data/TestData`** (67 bags, disjoint patients) is the untouched, official
   held-out test set. **`Data/dataset/{Malignant,Benign}`** (257 bags) is
   split patient-level, stratified by malignancy, into train/val (85/15).
6. **The test set's malignancy prevalence (~14%) is much lower than
   train/val (~32%)** — it's a separate clinical cohort, not a stratified
   split of the same pool. Read accuracy/F1 with that in mind; AUC is
   prevalence-invariant.

## Architecture / design decisions

- **ROI cropping.** Every B-mode image is cropped to the lesion neighborhood
  (mask bounding box + 30% margin) before resizing. Raw frames carry heavy
  on-screen scanner UI/text; without cropping, a classifier can learn
  shortcuts from that instead of tissue appearance.
- **Bag-level labels, per-frame images.** Every frame in a bag inherits that
  bag's histopathology label (standard weak/MIL-style labeling for
  multi-image-per-case ultrasound sets). At eval time, per-frame probabilities
  are averaged per bag before scoring (`eval/metrics.py::aggregate_bag_predictions`),
  so reported metrics are one decision per lesion, comparable to the reference
  benchmark.
- **Baseline and student train on the identical B-mode pool.** The baseline
  classifier and the distillation student see the exact same ROI-cropped
  images (paired `ZY_right` frame + all unpaired frames), same split, same
  backbone. This isolates the distillation loss as the *only* difference, so
  any accuracy delta between them is attributable to it, not to more data or
  a bigger model.
- **Teacher** (`models/teacher.py`): dual-branch network (separate CEUS and
  B-mode backbones) trained on the one true privileged pair per bag, fused
  into an embedding with the same dimensionality as the student's feature
  vector — needed so it can serve directly as a feature-distillation target.
- **Distillation** (`training/train_distill.py`): the teacher is trained once
  and frozen; its per-bag (soft logits, fused embedding) are precomputed once
  and broadcast as the distillation target to *every* B-mode-only frame from
  that bag (hard-label CE + KD-logit KL-divergence + feature MSE/cosine).
- **Segmentation** (`models/unet.py`): U-Net trained only on the numbered-frame
  pool — the only mask source verified 100% clean and non-empty. Evaluated for
  generalization on `Data/TestData`'s numbered frames (genuinely unseen
  patients), plus qualitative-only overlays on the CEUS-paired B-mode frame
  (which mostly lacks its own usable mask, see quirk #4).
- **CEUS color.** The false-color contrast overlay is collapsed to grayscale
  intensity for Review 1 — it's a fixed display LUT, not an independent
  signal. Preserving hue/enhancement-timing as extra channels is a natural
  Review-2 extension.

## Stretch hooks (scaffolded now, Review 2 fills them in — `models/heads.py`)

- `MCDropoutUncertainty` is **fully functional**: MC-Dropout over the trained
  student's existing `nn.Dropout` layers, flags a sample "recommend real CEUS"
  when predictive variance crosses `uncertainty.recommend_ceus_threshold`.
  Demoed at the end of `run_review1.py`.
- `SyntheticCEUSDecoder` / `synthetic_ceus_loss` are **intentional stubs** —
  interface (channel dims, forward/loss signatures) is fixed so Review 2 can
  wire in the decoder and loss without touching the training loop; both raise
  `NotImplementedError` for now.

## Outputs

- `figures/eda/*.png` — class balance, pairing composition, bag-per-patient
  histogram, image stats sanity checks.
- `figures/segmentation_*.png` — training curves, generalization overlays on
  held-out `TestData`, qualitative ZY-transfer overlays.
- `figures/baseline_*.png`, `figures/distill_*.png` — training curves,
  confusion matrices.
- `figures/wow_gradcam_ceus.png` — Grad-CAM (B-mode-only student) vs. the real
  CEUS lesion outline, for the same bag.
- `figures/final_metric_comparison.png`, `figures/final_roc_comparison.png` —
  baseline vs. distilled student vs. teacher oracle (uses real CEUS — an
  upper-bound reference, not a deployable model).
- `results/metrics_review1.csv`, `results/review1_summary.json` — final
  numbers; also printed to console at the end of `run_review1.py`.

## Known limitations / Review 2 TODOs

- Small, single-center dataset (324 bags, ~293–312 unique patients) — numbers
  are directional, not a clinically validated result.
- MMOTU pretraining is wired as a config toggle (`pretrain.mmotu`) but off by
  default (no local copy available at Review-1 time) — point
  `pretrain.mmotu.root` at a local copy and flip `enabled: true` to use it.
- Synthetic-CEUS reconstruction branch and evidential-uncertainty calibration
  are stubs only (see above).
- CEUS color/enhancement-timing is currently discarded (collapsed to
  grayscale); worth revisiting once the reconstruction branch exists.
