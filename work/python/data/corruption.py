"""Helpers for applying ImageNet-C corruptions via the local robustness/ImageNet-C package.

Usage:
    fn = make_corruption_fn("gaussian_noise", severity=3)
    corrupted_np = fn(image_np)   # image_np: uint8 H×W×3

Stream ordering (per thesis §3.2):
    severity 0 (clean) → severity 1 → 2 → 3 → 4 → 5
    Each severity transition is one injected drift event.
"""

from typing import Callable, Generator, Iterable, Tuple
import sys
from pathlib import Path

# Use local robustness/ImageNet-C package from the repository.
ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGENET_C_SRC = ROOT_DIR.parent / "robustness" / "ImageNet-C" / "imagenet_c"
if str(IMAGENET_C_SRC) not in sys.path:
    sys.path.insert(0, str(IMAGENET_C_SRC))

import numpy as np
from imagenet_c import corrupt

from config import CORRUPTION_SEVERITIES


def make_corruption_fn(corruption_name: str, severity: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function that applies one corruption type at a fixed severity."""
    def _apply(image: np.ndarray) -> np.ndarray:
        return corrupt(image, corruption_name=corruption_name, severity=severity)
    _apply.__name__ = f"{corruption_name}_sev{severity}"
    return _apply


def build_severity_stream(
    clean_samples: Iterable[Tuple],
    corrupted_by_severity: dict[int, Iterable[Tuple]],
) -> Generator[Tuple, None, None]:
    """Yield (image, label, path, severity) tuples ordered severity 0 → 5.

    clean_samples:         iterable of (image, label, path)
    corrupted_by_severity: {severity: iterable of (image, label, path)}
    """
    for item in clean_samples:
        yield (*item, 0)
    for sev in CORRUPTION_SEVERITIES:
        for item in corrupted_by_severity.get(sev, []):
            yield (*item, sev)
