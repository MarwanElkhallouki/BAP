"""LIME superpixel explanations for drift attribution analysis (thesis §3.5).

Per-window analysis:
  - Identify top-K most important superpixels per image (K = LIME_TOP_K_SUPERPIXELS).
  - Compute overlap coefficient between pre- and post-drift top-K sets.
  - Overlap < LIME_OVERLAP_SHIFT_THRESHOLD indicates substantial attribution shift.

Public API:
    compute_lime_top_k(model, images_np, transform, device, n) -> list[set[int]]
    overlap_coefficient(pre, post)                              -> float
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_image import LimeImageExplainer
from PIL import Image

from config import (
    LIME_N_PERTURBATIONS,
    LIME_OVERLAP_SHIFT_THRESHOLD,
    LIME_TOP_K_SUPERPIXELS,
    XAI_CHUNK_SIZE,
    XAI_SAMPLE_SIZE,
)


def _build_predict_fn(model: torch.nn.Module, transform, device: torch.device):
    def predict(images: np.ndarray) -> np.ndarray:
        """images: (N, H, W, 3) uint8 numpy array from LIME."""
        model.eval()
        tensors = [transform(Image.fromarray(img.astype(np.uint8))) for img in images]
        all_probs = []
        for start in range(0, len(tensors), XAI_CHUNK_SIZE):
            chunk = torch.stack(tensors[start : start + XAI_CHUNK_SIZE]).to(device)
            with torch.no_grad():
                probs = F.softmax(model(chunk), dim=1).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)
    return predict


def compute_lime_top_k(
    model: torch.nn.Module,
    images_np: list[np.ndarray],
    transform,
    device: torch.device,
    n_samples: int = XAI_SAMPLE_SIZE,
) -> list[set[int]]:
    """Return one set of top-K superpixel IDs per image."""
    model.eval()
    predict_fn = _build_predict_fn(model, transform, device)
    explainer  = LimeImageExplainer()
    results: list[set[int]] = []

    for img_np in images_np[:n_samples]:
        explanation = explainer.explain_instance(
            img_np.astype(np.uint8),
            predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=LIME_N_PERTURBATIONS,
        )
        # Importance for the defective class (label 1)
        _, mask = explanation.get_image_and_mask(
            label=1,
            positive_only=True,
            num_features=LIME_TOP_K_SUPERPIXELS,
            hide_rest=True,
        )
        top_ids = set(int(x) for x in np.unique(mask) if x > 0)
        results.append(top_ids)

    return results


def overlap_coefficient(
    pre_top_k: list[set[int]],
    post_top_k: list[set[int]],
) -> float:
    """Mean overlap coefficient across image pairs.

    overlap(A, B) = |A ∩ B| / min(|A|, |B|)
    A value below LIME_OVERLAP_SHIFT_THRESHOLD signals substantial attribution shift.
    """
    coeffs = []
    for pre, post in zip(pre_top_k, post_top_k):
        denom = min(len(pre), len(post))
        if denom > 0:
            coeffs.append(len(pre & post) / denom)
    return float(np.mean(coeffs)) if coeffs else 0.0
