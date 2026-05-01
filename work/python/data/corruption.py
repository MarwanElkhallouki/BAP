"""Helpers for corruption application and precomputed manifest loading.

Usage:
    fn = make_corruption_fn("gaussian_noise", severity=3)
    corrupted_np = fn(image_np)   # image_np: uint8 H×W×3

Stream ordering (per thesis §3.2):
    severity 0 (clean) → severity 1 → 2 → 3 → 4 → 5
    Each severity transition is one injected drift event.
"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Tuple

import numpy as np

from config import CORRUPTION_SEVERITIES

# Use local robustness/ImageNet-C package from the repository.
ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGENET_C_SRC = ROOT_DIR.parent / "robustness" / "ImageNet-C" / "imagenet_c"
if str(IMAGENET_C_SRC) not in sys.path:
    sys.path.insert(0, str(IMAGENET_C_SRC))


def make_corruption_fn(corruption_name: str, severity: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function that applies one corruption type at a fixed severity."""
    try:
        from imagenet_c import corrupt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "imagenet_c is required for runtime corruption generation. "
            "Install ImageNet-C dependencies or use precomputed manifest datasets."
        ) from exc

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


_MANIFEST_REQUIRED_FIELDS = (
    "image_path",
    "label",
    "original_path",
    "corruption_type",
    "scale",
    "category",
    "severity",
)


def _resolve_existing_path(raw_path: str, manifest_path: Path, field_name: str, idx: int) -> Path:
    """Resolve and validate a manifest path with explicit error messages."""
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [
        manifest_path.parent / path,
        Path(__file__).resolve().parents[1] / path,  # work/python/<raw_path>
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    candidate_str = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Manifest validation error in {manifest_path} at entry {idx}: "
        f"field '{field_name}' path '{raw_path}' does not exist. "
        f"Tried: {candidate_str}"
    )


def load_manifest_entries(
    manifest_path: Path,
    *,
    expected_category: str | None = None,
    expected_corruption_type: str | None = None,
) -> list[dict[str, Any]]:
    """Load and validate a corruption dataset manifest.

    Required fields per entry:
    image_path, label, original_path, corruption_type, scale, category, severity.
    Also validates image_path/original_path existence with explicit errors.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Manifest must contain a list of entries: {manifest_path}")

    validated: list[dict[str, Any]] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(
                f"Manifest validation error in {manifest_path} at entry {idx}: expected object, got {type(row).__name__}"
            )

        missing = [k for k in _MANIFEST_REQUIRED_FIELDS if k not in row]
        if missing:
            raise ValueError(
                f"Manifest validation error in {manifest_path} at entry {idx}: missing fields {missing}"
            )

        if expected_category is not None and row["category"] != expected_category:
            raise ValueError(
                f"Manifest validation error in {manifest_path} at entry {idx}: "
                f"category '{row['category']}' != expected '{expected_category}'"
            )

        if expected_corruption_type is not None and row["corruption_type"] != expected_corruption_type:
            raise ValueError(
                f"Manifest validation error in {manifest_path} at entry {idx}: "
                f"corruption_type '{row['corruption_type']}' != expected '{expected_corruption_type}'"
            )

        try:
            row["label"] = int(row["label"])
            row["severity"] = int(row["severity"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Manifest validation error in {manifest_path} at entry {idx}: "
                f"label/severity must be integers (got label={row['label']}, severity={row['severity']})"
            ) from exc
        if row["label"] not in (0, 1):
            raise ValueError(
                f"Manifest validation error in {manifest_path} at entry {idx}: "
                f"label {row['label']} not in supported labels [0, 1]"
            )
        allowed_severities = {0} if row["corruption_type"] == "held_out_defects" else set(CORRUPTION_SEVERITIES)
        if row["severity"] not in allowed_severities:
            raise ValueError(
                f"Manifest validation error in {manifest_path} at entry {idx}: "
                f"severity {row['severity']} not in supported severities {sorted(allowed_severities)}"
            )

        row["resolved_image_path"] = str(
            _resolve_existing_path(row["image_path"], manifest_path, "image_path", idx)
        )
        row["resolved_original_path"] = str(
            _resolve_existing_path(row["original_path"], manifest_path, "original_path", idx)
        )

        validated.append(row)

    return validated
