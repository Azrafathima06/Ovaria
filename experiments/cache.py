"""One-off in-memory cache of ROI-cropped, 224x224 grayscale B-mode images (plus the
aligned CEUS crop for the one privileged pair per bag). Removes per-epoch disk I/O,
nii.gz parsing and cropping, which keeps each run light on an 8 GB machine."""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from data.datasets import ClassificationDataset, load_grayscale
from data.masks import load_mask
from data.roi import crop_to_roi

SIZE = 224
DESC_KEYS = ["Morphology", "Boundary", "Solid", "Separation", "Nipple", "Blood_Flow"]


def descriptor_vector(clinical: dict) -> np.ndarray:
    v = [float(clinical.get(k, 0) > 0) for k in DESC_KEYS]
    v.append(float(clinical.get("Internal_Echo", 0) >= 2))
    return np.array(v, dtype=np.float32)


def _resize(a: np.ndarray) -> np.ndarray:
    return cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def build_cache(bags, bag_split, path: Path, roi_margin: float = 0.3) -> dict:
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    by_id = {b.bag_id: b for b in bags}
    out = {}
    for split in ("train", "val", "test"):
        ds = ClassificationDataset(bags, bag_split, split, None, roi_margin)
        recs = []
        for it in ds.items:
            img = load_grayscale(it.image_path)
            mask = ds._resolve_mask(it, img.shape)
            crop, _ = crop_to_roi(img, mask, roi_margin)
            ceus_crop = None
            if it.kind == "bmode_paired":
                b = by_id[it.bag_id]
                ceus = load_grayscale(b.ceus.path)
                cmask = load_mask(b.ceus.mask_path, target_hw=ceus.shape) if b.ceus.mask_path else None
                cc, _ = crop_to_roi(ceus, cmask, roi_margin)
                ceus_crop = _resize(cc)
            recs.append(dict(img=_resize(crop), ceus=ceus_crop, bag_id=it.bag_id, label=it.label,
                             kind=it.kind, desc=descriptor_vector(by_id[it.bag_id].clinical)))
        out[split] = recs
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out
