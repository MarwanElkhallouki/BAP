"""Geometric transformations for drift detection: rotations and translations.

Phase 7: Rigid way to separate geometric corruptions from pixel-level corruptions.

Public API:
    apply_rotation(image_np, degrees, scale) -> np.ndarray
    apply_translation(image_np, pixels_or_frac, scale) -> np.ndarray
    get_geometric_magnitude(scale) -> float
"""

from typing import Tuple
import numpy as np
from PIL import Image as PILImage
import random

from config import (
    ROTATION_SMALL_DEG,
    ROTATION_LARGE_DEG,
    TRANSLATION_SMALL_FRAC,
    TRANSLATION_LARGE_FRAC,
)


def apply_rotation(image_np: np.ndarray, scale: str = "small") -> np.ndarray:
    """Apply random rotation ±θ degrees to image (uint8 RGB).
    
    Args:
        image_np: uint8 H×W×3 numpy array
        scale: "small" (±15°) or "large" (±30°)
    
    Returns:
        Rotated uint8 H×W×3 image; may have white borders from rotation.
    """
    max_deg = ROTATION_SMALL_DEG if scale == "small" else ROTATION_LARGE_DEG
    angle = random.uniform(-max_deg, max_deg)
    
    img_pil = PILImage.fromarray(image_np)
    rotated = img_pil.rotate(angle, fillcolor=(255, 255, 255), expand=False)
    return np.array(rotated, dtype=np.uint8)


def apply_translation(image_np: np.ndarray, scale: str = "small") -> np.ndarray:
    """Apply random translation (shift) to image; pad shifted regions with mean color.
    
    Args:
        image_np: uint8 H×W×3 numpy array
        scale: "small" (±5%) or "large" (±10%)
    
    Returns:
        Translated uint8 H×W×3 image.
    """
    max_frac = TRANSLATION_SMALL_FRAC if scale == "small" else TRANSLATION_LARGE_FRAC
    h, w = image_np.shape[:2]
    
    # Random shift in fraction of image dimensions
    shift_x_pix = int(random.uniform(-max_frac * w, max_frac * w))
    shift_y_pix = int(random.uniform(-max_frac * h, max_frac * h))
    
    # Roll (circular shift)
    translated = np.roll(image_np, shift_y_pix, axis=0)
    translated = np.roll(translated, shift_x_pix, axis=1)
    
    # Fill shifted borders with mean color to avoid circular artifacts
    mean_color = np.array(image_np).mean(axis=(0, 1)).astype(np.uint8)
    
    if shift_y_pix > 0:
        translated[:shift_y_pix, :] = mean_color
    elif shift_y_pix < 0:
        translated[shift_y_pix:, :] = mean_color
    
    if shift_x_pix > 0:
        translated[:, :shift_x_pix] = mean_color
    elif shift_x_pix < 0:
        translated[:, shift_x_pix:] = mean_color
    
    return translated.astype(np.uint8)


def get_geometric_magnitude(scale: str) -> float:
    """Return magnitude descriptor for scale (used in logging/reporting).
    
    Args:
        scale: "small" or "large"
    
    Returns:
        Magnitude as float (e.g., 0.05 for small, 0.10 for large)
    """
    return TRANSLATION_SMALL_FRAC if scale == "small" else TRANSLATION_LARGE_FRAC
