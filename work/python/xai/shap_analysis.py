"""Superpixel-level KernelSHAP for drift attribution analysis (thesis §3.5).

Approach:
  1. Segment each image into superpixels with quickshift.
  2. Build a predict function that toggles superpixels on/off (off = mean colour).
  3. Run SHAP KernelExplainer; each feature = one superpixel.
  4. Compare pre/post-drift distributions with a KS test.

Public API:
    compute_shap_values(model, images_np, transform, device, n) -> list[np.ndarray]
    ks_test(pre_shap, post_shap)                               -> dict
"""

from __future__ import annotations

import numpy as np
import shap
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import ks_2samp
from skimage.segmentation import quickshift

from config import SHAP_N_PERTURBATIONS, SHAP_SIGNIFICANCE_LEVEL, XAI_CHUNK_SIZE, XAI_SAMPLE_SIZE


def _build_predict_fn(model: torch.nn.Module, transform, device: torch.device):
    """Return a function that maps (n_masks, n_segs) → (n_masks, n_classes) probabilities."""
    def predict(masks: np.ndarray, img_np: np.ndarray, segments: np.ndarray) -> np.ndarray:
        bg = img_np.mean(axis=(0, 1), keepdims=True)  # mean-colour background
        tensors = []
        for mask in masks:
            masked = img_np.copy().astype(float)
            for seg_id, keep in enumerate(mask):
                if not keep:
                    masked[segments == seg_id] = bg
            pil = Image.fromarray(masked.clip(0, 255).astype(np.uint8))
            tensors.append(transform(pil))
        all_probs = []
        for start in range(0, len(tensors), XAI_CHUNK_SIZE):
            chunk = torch.stack(tensors[start : start + XAI_CHUNK_SIZE]).to(device)
            with torch.no_grad():
                probs = F.softmax(model(chunk), dim=1).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)
    return predict


def compute_shap_values(
    model: torch.nn.Module,
    images_np: list[np.ndarray],   # list of uint8 H×W×3 arrays
    transform,
    device: torch.device,
    n_samples: int = XAI_SAMPLE_SIZE,
) -> list[np.ndarray]:
    """Return list of (n_segments,) mean-absolute SHAP arrays, one per image."""
    model.eval()
    predict_fn = _build_predict_fn(model, transform, device)
    results = []

    for img_np in images_np[:n_samples]:
        segments = quickshift(img_np, kernel_size=4, max_dist=200, ratio=0.2)
        n_segs = int(segments.max()) + 1

        def _predict(masks: np.ndarray) -> np.ndarray:
            return predict_fn(masks, img_np, segments)

        explainer = shap.KernelExplainer(_predict, np.zeros((1, n_segs)))
        shap_vals = explainer.shap_values(
            np.ones((1, n_segs)),
            nsamples=SHAP_N_PERTURBATIONS,
            silent=True,
        )
        # shap_vals: list[n_classes] of (1, n_segs) → mean-abs across classes
        mean_abs = np.abs(np.array(shap_vals)).mean(axis=0).squeeze()
        results.append(mean_abs)

    return results


def ks_test(
    pre_shap: list[np.ndarray],
    post_shap: list[np.ndarray],
) -> dict:
    """Two-sample KS test on pooled superpixel SHAP distributions.

    Returns {"statistic": float, "pvalue": float, "significant": bool}.
    """
    pre  = np.concatenate([v.flatten() for v in pre_shap])
    post = np.concatenate([v.flatten() for v in post_shap])
    stat, pval = ks_2samp(pre, post)
    return {
        "statistic":   float(stat),
        "pvalue":      float(pval),
        "significant": bool(pval < SHAP_SIGNIFICANCE_LEVEL),
    }
