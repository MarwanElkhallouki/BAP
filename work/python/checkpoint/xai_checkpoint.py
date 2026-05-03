"""Phase 4: Checkpoint persistence for decoupled XAI analysis.

Save detector state + sample buffer on drift alarm, enabling XAI to rerun
independently without affecting results or requiring re-inference.

Public API:
    save_xai_checkpoint(checkpoint_data, filepath) -> None
    load_xai_checkpoint(filepath) -> XAICheckpoint
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


def minimal_scale_interpretation_dict(
    interpretation: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Subset of scale_interpretation safe for JSON checkpoint metadata."""
    if not interpretation:
        return None
    keys = (
        "combined_large_score",
        "gradcam_change_magnitude",
        "lime_overlap_coefficient",
        "combined_large_threshold",
        "gradcam_component_score",
        "lime_component_score",
        "status",
        "rationale",
        "used_components",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        if key not in interpretation:
            continue
        val = interpretation[key]
        if hasattr(val, "item"):
            val = val.item()
        out[key] = val
    return out or None


class XAICheckpoint:
    """Container for drift detection state to enable decoupled XAI replay."""
    
    def __init__(
        self,
        detector_name: str,
        sample_index: int,
        drift_type: str,  # "corruption" | "defect" | "geometric"
        scale: Optional[str],  # "small" | "large" | None
        category: str,
        corruption_type: str,
        severity: Optional[int],
        model_state_dict: Dict,
        pre_drift_samples: List[Dict],
        post_drift_samples: List[Dict],
        *,
        inferred_scale: Optional[str] = None,
        scale_interpretation: Optional[Dict[str, Any]] = None,
    ):
        """Initialize checkpoint with drift metadata and sample buffers.
        
        Args:
            detector_name: e.g., "ADWIN", "DDM"
            sample_index: global sample index at alarm
            drift_type: categorization of drift source
            scale: magnitude scale if applicable
            category: MVTec category
            corruption_type: corruption applied (e.g., "gaussian_noise")
            severity: corruption severity level
            model_state_dict: ResNet-50 state dict (for XAI)
            pre_drift_samples: list of dicts with 'image_np', 'prediction', 'confidence', 'label', 'path'
            post_drift_samples: list of dicts with same keys
        """
        self.detector_name = detector_name
        self.sample_index = sample_index
        self.drift_type = drift_type
        self.scale = scale
        self.category = category
        self.corruption_type = corruption_type
        self.severity = severity
        
        self.model_state_dict = model_state_dict
        self.pre_drift_samples = pre_drift_samples
        self.post_drift_samples = post_drift_samples
        self.inferred_scale = inferred_scale
        self.scale_interpretation = scale_interpretation

        # Metadata
        self.n_pre = len(pre_drift_samples)
        self.n_post = len(post_drift_samples)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for JSON-compatible parts)."""
        return {
            "metadata": {
                "detector_name": self.detector_name,
                "sample_index": int(self.sample_index),
                "drift_type": self.drift_type,
                "scale": self.scale,
                "category": self.category,
                "corruption_type": self.corruption_type,
                "severity": int(self.severity) if self.severity is not None else None,
                "n_pre": self.n_pre,
                "n_post": self.n_post,
                "inferred_scale": self.inferred_scale,
                "scale_interpretation": self.scale_interpretation,
            },
            "samples": {
                "pre_drift": [
                    {
                        "prediction": int(s["prediction"]),
                        "confidence": float(s["confidence"]),
                        "label": int(s["label"]),
                        "path": str(s["path"]),
                    }
                    for s in self.pre_drift_samples
                ],
                "post_drift": [
                    {
                        "prediction": int(s["prediction"]),
                        "confidence": float(s["confidence"]),
                        "label": int(s["label"]),
                        "path": str(s["path"]),
                    }
                    for s in self.post_drift_samples
                ],
            },
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any], model_state_dict: Dict, images: Dict[str, List[np.ndarray]]) -> "XAICheckpoint":
        """Deserialize from dictionary."""
        meta = data["metadata"]
        samp = data["samples"]
        
        pre_drift = [
            {**s, "image_np": images["pre"][i]}
            for i, s in enumerate(samp["pre_drift"])
        ]
        post_drift = [
            {**s, "image_np": images["post"][i]}
            for i, s in enumerate(samp["post_drift"])
        ]
        
        return XAICheckpoint(
            detector_name=meta["detector_name"],
            sample_index=meta["sample_index"],
            drift_type=meta["drift_type"],
            scale=meta.get("scale"),
            category=meta["category"],
            corruption_type=meta["corruption_type"],
            severity=meta.get("severity"),
            model_state_dict=model_state_dict,
            pre_drift_samples=pre_drift,
            post_drift_samples=post_drift,
            inferred_scale=meta.get("inferred_scale"),
            scale_interpretation=meta.get("scale_interpretation"),
        )


def _raise_schema_error(path: str, message: str) -> None:
    raise ValueError(f"Invalid checkpoint schema at '{path}': {message}")


def _raise_payload_error(path: str, message: str) -> None:
    raise ValueError(f"Invalid checkpoint payload at '{path}': {message}")


def _validate_type(value: Any, expected_type: Any, path: str) -> None:
    if not isinstance(value, expected_type):
        _raise_schema_error(path, f"expected {expected_type}, got {type(value).__name__}")


def _validate_int(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise_schema_error(path, f"expected int, got {type(value).__name__}")


def _require_key(obj: Dict[str, Any], key: str, path: str) -> Any:
    if key not in obj:
        _raise_schema_error(path, f"missing required key '{key}'")
    return obj[key]


def _validate_json_schema(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        _raise_schema_error("root", f"expected dict, got {type(data).__name__}")

    metadata = _require_key(data, "metadata", "root")
    samples = _require_key(data, "samples", "root")
    _validate_type(metadata, dict, "metadata")
    _validate_type(samples, dict, "samples")

    required_metadata_str_fields = [
        "detector_name",
        "drift_type",
        "category",
        "corruption_type",
    ]
    for key in required_metadata_str_fields:
        value = _require_key(metadata, key, "metadata")
        _validate_type(value, str, f"metadata.{key}")

    required_metadata_int_fields = ["sample_index", "n_pre", "n_post"]
    for key in required_metadata_int_fields:
        value = _require_key(metadata, key, "metadata")
        _validate_int(value, f"metadata.{key}")

    scale = metadata.get("scale")
    if scale is not None and not isinstance(scale, str):
        _raise_schema_error("metadata.scale", f"expected str or None, got {type(scale).__name__}")

    severity = metadata.get("severity")
    if severity is not None:
        _validate_int(severity, "metadata.severity")

    inferred_scale = metadata.get("inferred_scale")
    if inferred_scale is not None and not isinstance(inferred_scale, str):
        _raise_schema_error("metadata.inferred_scale", f"expected str or None, got {type(inferred_scale).__name__}")

    scale_interpretation = metadata.get("scale_interpretation")
    if scale_interpretation is not None:
        _validate_type(scale_interpretation, dict, "metadata.scale_interpretation")

    pre_samples = _require_key(samples, "pre_drift", "samples")
    post_samples = _require_key(samples, "post_drift", "samples")
    _validate_type(pre_samples, list, "samples.pre_drift")
    _validate_type(post_samples, list, "samples.post_drift")

    required_sample_str_fields = ["path"]
    required_sample_int_fields = ["prediction", "label"]
    for group_name, sample_list in (("pre_drift", pre_samples), ("post_drift", post_samples)):
        for i, sample in enumerate(sample_list):
            sample_path = f"samples.{group_name}[{i}]"
            _validate_type(sample, dict, sample_path)
            for key in required_sample_int_fields:
                value = _require_key(sample, key, sample_path)
                _validate_int(value, f"{sample_path}.{key}")
            for key in required_sample_str_fields:
                value = _require_key(sample, key, sample_path)
                _validate_type(value, str, f"{sample_path}.{key}")
            confidence = _require_key(sample, "confidence", sample_path)
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                _raise_schema_error(f"{sample_path}.confidence", f"expected float-compatible number, got {type(confidence).__name__}")

    if metadata["n_pre"] != len(pre_samples):
        _raise_schema_error(
            "metadata.n_pre",
            f"value {metadata['n_pre']} does not match samples.pre_drift length {len(pre_samples)}",
        )
    if metadata["n_post"] != len(post_samples):
        _raise_schema_error(
            "metadata.n_post",
            f"value {metadata['n_post']} does not match samples.post_drift length {len(post_samples)}",
        )


def _validate_npz_payload(npz_obj: np.lib.npyio.NpzFile, data: Dict[str, Any]) -> Dict[str, Any]:
    required_arrays = {"pre_drift_images", "post_drift_images", "model_state_pickle"}
    missing_arrays = sorted(required_arrays - set(npz_obj.files))
    if missing_arrays:
        _raise_payload_error("npz", f"missing required arrays: {', '.join(missing_arrays)}")

    pre_images = npz_obj["pre_drift_images"]
    post_images = npz_obj["post_drift_images"]
    model_state_arr = npz_obj["model_state_pickle"]

    try:
        model_state_payload = model_state_arr.item()
    except Exception as exc:
        _raise_payload_error("npz.model_state_pickle", f"could not extract scalar payload: {exc}")

    if not isinstance(model_state_payload, (bytes, bytearray)):
        _raise_payload_error(
            "npz.model_state_pickle",
            f"expected bytes payload, got {type(model_state_payload).__name__}",
        )

    try:
        model_state = pickle.loads(model_state_payload)
    except Exception as exc:
        _raise_payload_error("npz.model_state_pickle", f"failed to decode pickled model state: {exc}")

    if not isinstance(model_state, dict):
        _raise_payload_error("npz.model_state_pickle", f"decoded model state must be dict, got {type(model_state).__name__}")

    expected_pre = data["metadata"]["n_pre"]
    expected_post = data["metadata"]["n_post"]
    if len(pre_images) != expected_pre:
        _raise_payload_error(
            "npz.pre_drift_images",
            f"length {len(pre_images)} does not match metadata.n_pre {expected_pre}",
        )
    if len(post_images) != expected_post:
        _raise_payload_error(
            "npz.post_drift_images",
            f"length {len(post_images)} does not match metadata.n_post {expected_post}",
        )

    if len(pre_images) != len(data["samples"]["pre_drift"]):
        _raise_payload_error(
            "npz.pre_drift_images",
            f"length {len(pre_images)} does not match samples.pre_drift length {len(data['samples']['pre_drift'])}",
        )
    if len(post_images) != len(data["samples"]["post_drift"]):
        _raise_payload_error(
            "npz.post_drift_images",
            f"length {len(post_images)} does not match samples.post_drift length {len(data['samples']['post_drift'])}",
        )

    return {
        "pre_images": pre_images,
        "post_images": post_images,
        "model_state": model_state,
    }


def save_xai_checkpoint(checkpoint: XAICheckpoint, filepath: Path) -> None:
    """Save checkpoint to disk.
    
    Uses two files:
      - {filepath}.json: metadata + predictions (text-based)
      - {filepath}.npz: model state dict + images (binary)
    
    Args:
        checkpoint: XAICheckpoint instance
        filepath: base path (without extension)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save metadata + sample predictions as JSON
    json_path = filepath.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(checkpoint.to_dict(), f, indent=2)
    
    # Save images + model state as NPZ
    npz_path = filepath.with_suffix(".npz")
    
    # Prepare image arrays
    pre_images = np.array([s["image_np"] for s in checkpoint.pre_drift_samples], dtype=object)
    post_images = np.array([s["image_np"] for s in checkpoint.post_drift_samples], dtype=object)
    
    # Save: images, model state dict (as pickle within npz)
    np.savez(
        npz_path,
        pre_drift_images=pre_images,
        post_drift_images=post_images,
        model_state_pickle=pickle.dumps(checkpoint.model_state_dict),
    )
    
    print(f"Saved checkpoint to {json_path} and {npz_path}")
    print(f"  Pre-drift: {checkpoint.n_pre} samples")
    print(f"  Post-drift: {checkpoint.n_post} samples")


def load_xai_checkpoint(filepath: Path) -> XAICheckpoint:
    """Load checkpoint from disk.
    
    Args:
        filepath: base path (without extension)
    
    Returns:
        Reconstructed XAICheckpoint instance.
    """
    filepath = Path(filepath)
    
    # Load metadata
    json_path = filepath.with_suffix(".json")
    with open(json_path, "r") as f:
        data = json.load(f)
    _validate_json_schema(data)
    
    # Load images + model state
    npz_path = filepath.with_suffix(".npz")
    with np.load(npz_path, allow_pickle=True) as arr:
        payload = _validate_npz_payload(arr, data)
    pre_images = payload["pre_images"]
    post_images = payload["post_images"]
    model_state = payload["model_state"]
    
    # Reconstruct checkpoint
    checkpoint = XAICheckpoint.from_dict(
        data,
        model_state_dict=model_state,
        images={
            "pre": [img for img in pre_images],
            "post": [img for img in post_images],
        },
    )
    
    print(f"Loaded checkpoint from {json_path} and {npz_path}")
    return checkpoint


def checkpoint_filename(
    category: str,
    corruption_type: str,
    detector_name: str,
    alarm_index: int,
) -> str:
    """Generate checkpoint filename.
    
    Format: {category}_{corruption}_{detector}_{alarm_index}
    """
    return f"{category}_{corruption_type}_{detector_name}_{alarm_index:03d}"
