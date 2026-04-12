"""Phase 4: Checkpoint persistence for decoupled XAI analysis.

Save detector state + sample buffer on drift alarm, enabling XAI to rerun
independently without affecting results or requiring re-inference.

Public API:
    save_xai_checkpoint(checkpoint_data, filepath) -> None
    load_xai_checkpoint(filepath) -> dict
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch

from config import XAI_CHECKPOINT_DIR


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
        )


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
    
    # Load images + model state
    npz_path = filepath.with_suffix(".npz")
    arr = np.load(npz_path, allow_pickle=True)
    
    pre_images = arr["pre_drift_images"]
    post_images = arr["post_drift_images"]
    model_state = pickle.loads(arr["model_state_pickle"].item())
    
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
