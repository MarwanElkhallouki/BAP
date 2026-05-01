"""Grad-CAM via captum.attr.LayerGradCam (thesis §3.5).

Public API:
    compute_gradcam(model, images, target_class, device) -> np.ndarray (N, H, W)
    mean_heatmap(heatmaps)                               -> np.ndarray (H, W)
    change_map(pre_mean, post_mean)                      -> np.ndarray (H, W)
    compute_ada(heatmap, defect_mask)                    -> float  [IoU]
"""

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam

from config import GRADCAM_TOP_PERCENTILE, IMAGE_SIZE


def _target_layer(model: torch.nn.Module):
    """Final residual block of ResNet-50 — compatible with Grad-CAM out of the box."""
    return model.layer4[-1]


def compute_gradcam(
    model: torch.nn.Module,
    images: torch.Tensor,   # (N, 3, H, W) already on device
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """Return per-image Grad-CAM heatmaps, shape (N, IMAGE_SIZE, IMAGE_SIZE), values in [0,1]."""
    model.eval()
    gc = LayerGradCam(model, _target_layer(model))
    all_attrs = []
    for i in range(len(images)):
        img = images[i : i + 1].to(device)
        attr = gc.attribute(img, target=target_class)          # (1, C, h, w)
        attr = F.interpolate(
            attr, size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear", align_corners=False,
        ).squeeze(1)                                           # (1, H, W)
        attr = torch.clamp(attr, min=0).detach().cpu().numpy()
        all_attrs.append(attr)
        torch.cuda.empty_cache()
    attrs = np.concatenate(all_attrs, axis=0)                  # (N, H, W)

    # Normalise each heatmap independently to [0, 1]
    for i in range(len(attrs)):
        mx = attrs[i].max()
        if mx > 0:
            attrs[i] /= mx

    return attrs


def mean_heatmap(heatmaps: np.ndarray) -> np.ndarray:
    """Average a stack of heatmaps → (H, W)."""
    return heatmaps.mean(axis=0)


def change_map(pre_mean: np.ndarray, post_mean: np.ndarray) -> np.ndarray:
    """Signed attribution shift: positive = more activated post-drift."""
    return post_mean - pre_mean


def compute_ada(heatmap: np.ndarray, defect_mask: np.ndarray) -> float:
    """Attribution-to-Defect Alignment (ADA) = IoU(binarised CAM, ground-truth mask).

    CAM is binarised by thresholding at the GRADCAM_TOP_PERCENTILE-th percentile
    (default 80th → top 20 % of activated pixels).
    """
    if defect_mask.shape != heatmap.shape:
        mask_tensor = torch.tensor(defect_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(mask_tensor, size=heatmap.shape, mode="nearest")
        defect_mask = resized.squeeze(0).squeeze(0).cpu().numpy()

    threshold = np.percentile(heatmap, GRADCAM_TOP_PERCENTILE)
    binary_cam  = (heatmap >= threshold).astype(np.uint8)
    binary_mask = (defect_mask > 0).astype(np.uint8)

    intersection = int((binary_cam & binary_mask).sum())
    union        = int((binary_cam | binary_mask).sum())
    return intersection / union if union > 0 else 0.0
