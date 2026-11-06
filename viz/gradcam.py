"""The 'wow' figure: does the B-mode-only student's Grad-CAM attention land on
the same region a radiologist marked as the lesion on the real CEUS frame?
The student never sees CEUS at inference -- this is a purely visual argument
that distillation transferred CEUS-relevant spatial attention, not a metric."""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage import measure

import matplotlib.pyplot as plt

from data.datasets import load_grayscale
from data.masks import load_mask
from data.roi import crop_to_roi
from data.transforms import build_classification_transforms
from .style import save_fig


def _compute_gradcam(model, image_tensor: torch.Tensor, device, target_class: int = 1) -> np.ndarray:
    model.eval()
    target_layers = [model.backbone.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0).to(device), targets=targets)
    return grayscale_cam[0]


def make_wow_figure(student, bags, roi_margin: float, image_size: int, device, path, n_examples: int = 3, dpi: int = 200):
    tf = build_classification_transforms(image_size, train=False)
    candidates = [b for b in bags if b.ceus and b.ceus.mask_path and b.bmode_paired and b.label == 1]
    random.Random(0).shuffle(candidates)
    candidates = candidates[:n_examples]
    if not candidates:
        return

    fig, axes = plt.subplots(len(candidates), 3, figsize=(12, 4 * len(candidates)))
    if len(candidates) == 1:
        axes = axes[None, :]

    for row, b in enumerate(candidates):
        bmode_raw = load_grayscale(b.bmode_paired.path)
        # ZY_right's own mask file is unreliable (see data/index.py docstring) --
        # crop using the bag's CEUS mask, the co-registered reliable annotation.
        roi_mask = load_mask(b.ceus.mask_path, target_hw=bmode_raw.shape)
        crop, _ = crop_to_roi(bmode_raw, roi_mask, margin=roi_margin)
        crop_rgb = np.stack([crop] * 3, axis=-1)
        tensor = tf(image=crop_rgb)["image"]

        cam = _compute_gradcam(student, tensor, device, target_class=1)
        rgb_float = crop_rgb.astype(np.float32) / 255.0
        rgb_resized = cv2.resize(rgb_float, (cam.shape[1], cam.shape[0]))
        overlay = show_cam_on_image(rgb_resized, cam, use_rgb=True)

        ceus_raw = load_grayscale(b.ceus.path)
        ceus_mask = load_mask(b.ceus.mask_path, target_hw=ceus_raw.shape)

        axes[row, 0].imshow(crop, cmap="gray")
        axes[row, 0].set_title(f"{b.bag_id}\nB-mode (student input, ROI crop)", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title("Student Grad-CAM (B-mode only)", fontsize=10)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(ceus_raw, cmap="gray")
        if ceus_mask is not None:
            for contour in measure.find_contours(ceus_mask.astype(float), 0.5):
                axes[row, 2].plot(contour[:, 1], contour[:, 0], color="#2ECC71", linewidth=1.6)
        axes[row, 2].set_title("Real CEUS + radiologist-outlined lesion", fontsize=10)
        axes[row, 2].axis("off")

    fig.suptitle("Does the B-mode-only student attend to the CEUS-relevant region?",
                 fontsize=13, fontweight="bold")
    save_fig(fig, path, dpi)
