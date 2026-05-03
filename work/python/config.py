"""Central configuration for the XAI drift-detection pipeline.

All paths, hyperparameters, and experimental constants live here so that
individual modules never need hard-coded magic numbers.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT              = Path("data/raw")
MVTEC_ROOT             = DATA_ROOT / "mvtec"
IMAGENET_C_ROOT        = DATA_ROOT / "imagenet-c"
CORRUPTION_DATASETS    = Path("data/corruption_datasets")  # pre-computed
CHECKPOINT_DIR         = Path("checkpoints")
OUTPUT_DIR             = Path("output")
XAI_CHECKPOINT_DIR     = OUTPUT_DIR / "xai_checkpoints"    # drift detection checkpoints
XAI_SUPERPIXEL_EXPORT_DIR = OUTPUT_DIR / "xai_superpixels"  # Quickshift exports (notebook)
# High-resolution figure export for the XAI visualization notebook (optional)
SAVE_NOTEBOOK_FIGURES = False
NOTEBOOK_FIGURE_EXPORT_DIR = OUTPUT_DIR / "notebook_figures"
MODEL_CHECKPOINT_DIR   = Path("models/checkpoints")        # DINO-V3 fine-tuned weights

# ---------------------------------------------------------------------------
# MVTec AD
# ---------------------------------------------------------------------------
MVTEC_CATEGORIES = ["carpet", "bottle", "metal_nut", "transistor", "leather"]

# Explicit, reproducible defect split policy per category.
# - train:   "known" defect types used during supervised binary classifier training
# - holdout: "novel" defect types used only for evaluation (no train/eval overlap)
MVTEC_DEFECT_SPLIT_POLICY: dict[str, dict[str, list[str]]] = {
    "carpet": {
        "train": ["color", "cut"],
        "holdout": ["hole", "metal_contamination", "thread"],
    },
    "bottle": {
        "train": ["broken_large", "broken_small"],
        "holdout": ["contamination"],
    },
    "metal_nut": {
        "train": ["bent", "color"],
        "holdout": ["flip", "scratch"],
    },
    "transistor": {
        "train": ["bent_lead", "cut_lead"],
        "holdout": ["damaged_case", "misplaced"],
    },
    "leather": {
        "train": ["color", "cut"],
        "holdout": ["fold", "glue", "poke"],
    },
}

# Backward-compatible views used by existing callers.
MVTEC_TRAIN_DEFECT_TYPES: dict[str, list[str]] = {
    category: split["train"][:] for category, split in MVTEC_DEFECT_SPLIT_POLICY.items()
}
MVTEC_HELDOUT_DEFECT_TYPES: dict[str, list[str]] = {
    category: split["holdout"][:] for category, split in MVTEC_DEFECT_SPLIT_POLICY.items()
}


def get_mvtec_defect_split(category: str) -> dict[str, list[str]]:
    """Return configured train/holdout defect split for a category.

    Raises:
        ValueError: if category is not configured.
    """
    if category not in MVTEC_DEFECT_SPLIT_POLICY:
        raise ValueError(
            f"Unknown MVTec category '{category}'. "
            f"Expected one of: {sorted(MVTEC_DEFECT_SPLIT_POLICY)}"
        )
    split = MVTEC_DEFECT_SPLIT_POLICY[category]
    return {
        "train": split["train"][:],
        "holdout": split["holdout"][:],
    }


def _validate_mvtec_split_policy() -> None:
    """Validate split policy at import time for deterministic behavior."""
    missing_categories = set(MVTEC_CATEGORIES) - set(MVTEC_DEFECT_SPLIT_POLICY)
    extra_categories = set(MVTEC_DEFECT_SPLIT_POLICY) - set(MVTEC_CATEGORIES)
    if missing_categories or extra_categories:
        raise ValueError(
            "MVTEC_DEFECT_SPLIT_POLICY keys must match MVTEC_CATEGORIES. "
            f"missing={sorted(missing_categories)}, extra={sorted(extra_categories)}"
        )

    for category, split in MVTEC_DEFECT_SPLIT_POLICY.items():
        train = split.get("train", [])
        holdout = split.get("holdout", [])
        if len(train) != len(set(train)):
            raise ValueError(f"Category '{category}' has duplicate train defects: {train}")
        if len(holdout) != len(set(holdout)):
            raise ValueError(f"Category '{category}' has duplicate holdout defects: {holdout}")
        overlap = set(train) & set(holdout)
        if overlap:
            raise ValueError(
                f"Category '{category}' has overlapping train/holdout defects: {sorted(overlap)}"
            )
        if not train:
            raise ValueError(f"Category '{category}' has empty train defect split.")
        if not holdout:
            raise ValueError(f"Category '{category}' has empty holdout defect split.")


_validate_mvtec_split_policy()

# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Corruptions (imagecorruptions / ImageNet-C)
# ---------------------------------------------------------------------------
# Pixel-level corruptions (ImageNet-C)
CORRUPTION_PIXEL_LEVEL = [
    "gaussian_noise",
    "defocus_blur",
    "brightness",
    "jpeg_compression",
]

# Geometric transformations (Translations & Big Anomalies)
CORRUPTION_GEOMETRIC = ["rotation", "translation"]

# Defect-type drift (held-out MVTec defects per category)
CORRUPTION_DEFECT_TYPE = ["held_out_defects"]

# All corruption types (for iteration)
CORRUPTION_TYPES = CORRUPTION_PIXEL_LEVEL + CORRUPTION_GEOMETRIC + CORRUPTION_DEFECT_TYPE

CORRUPTION_SEVERITIES = [1, 2, 3, 4, 5]  # severity 0 = clean baseline

# Scale threshold — small vs large corruptions
SCALE_SMALL_SEVERITY_MAX = 2        # severity 1-2 = small
SCALE_LARGE_SEVERITY_MIN = 3        # severity 3-5 = large
GEOMETRIC_SMALL_MAGNITUDE = 0.05    # translation ±5% = small
GEOMETRIC_LARGE_MAGNITUDE = 0.10    # translation ±10% = large

# Geometric transform magnitudes
ROTATION_SMALL_DEG      = 15.0      # ±15° for small scale
ROTATION_LARGE_DEG      = 30.0      # ±30° for large scale
TRANSLATION_SMALL_FRAC  = 0.05      # ±5% of image dimensions
TRANSLATION_LARGE_FRAC  = 0.10      # ±10% of image dimensions

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
LEARNING_RATE_HEAD      = 1e-4   # final FC layer
LEARNING_RATE_BACKBONE  = 1e-5   # convolutional backbone
EARLY_STOPPING_PATIENCE = 5      # epochs without val-F1 improvement
BATCH_SIZE              = 32
NUM_WORKERS             = 4
MAX_EPOCHS              = 50
VAL_SPLIT               = 0.20   # fraction of train set used for validation

# ---------------------------------------------------------------------------
# Runtime profile (laptop-first defaults)
# ---------------------------------------------------------------------------
LAPTOP_MODE = False

# Guardrail for stream buffering during drift/XAI processing to avoid OOM.
MAX_STREAM_IMAGES_IN_RAM = 512

# ---------------------------------------------------------------------------
# Drift detectors — ADWIN + DDM only
# ---------------------------------------------------------------------------
DRIFT_DETECTORS = ["DDM", "ADWIN"]
DDM_WARNING_THRESHOLD = 2.0    # ~95 % confidence
DDM_DRIFT_THRESHOLD   = 3.0    # ~99 % confidence
ADWIN_DELTA           = 0.002

# Detection tolerance window for TPR calculation
DETECTION_TOLERANCE_WINDOW = 500  # samples

# TODO: Naive baseline

# ---------------------------------------------------------------------------
# XAI — Grad-CAM + LIME only
# ---------------------------------------------------------------------------
XAI_METHODS = ["gradcam", "lime"]
XAI_PRE_DRIFT_WINDOW  = 200    # images before alarm
XAI_POST_DRIFT_WINDOW = 200    # images after alarm
XAI_SAMPLE_SIZE       = 10     # images sampled for XAI analysis in laptop mode

GRADCAM_TOP_PERCENTILE       = 80   # binarise at 80th percentile → top-20 % activated
LIME_TOP_K_SUPERPIXELS       = 5
LIME_OVERLAP_SHIFT_THRESHOLD = 0.50  # overlap < 0.5 → substantial attribution shift
LIME_N_PERTURBATIONS         = 128   # laptop-first default for faster LIME execution

# Alarm-level scale interpretation from XAI outputs (deterministic rules)
# Grad-CAM component: larger mean absolute attribution shift => more likely "large"
XAI_SCALE_GRADCAM_LARGE_THRESHOLD = 0.10
# LIME component: lower overlap => more likely "large"
XAI_SCALE_LIME_LARGE_THRESHOLD = 0.50
# Combined score (weighted mean of available components) threshold for "large"
XAI_SCALE_GRADCAM_WEIGHT = 0.5
XAI_SCALE_LIME_WEIGHT = 0.5
XAI_SCALE_COMBINED_LARGE_THRESHOLD = 0.50

# GPU inference chunk size for LIME perturbation batches
# 8 = 4 GB, 32 = 12 GB
XAI_CHUNK_SIZE = 4

# Checkpoint persistence
SAVE_XAI_CHECKPOINTS        = True   # save drift detection state before XAI analysis
ENABLE_XAI_ONLY_MODE        = True   # support run_xai_only.py for decoupled XAI

# If True, each detector is disabled after its first alarm and XAI runs exactly once per
# detector. The stream ends early once all detectors have fired.
# If False, detectors run continuously and XAI is throttled by XAI_POST_DRIFT_WINDOW.
DISABLE_DETECTOR_AFTER_ALARM = True

# ---------------------------------------------------------------------------
# Hyperparameter tuning placeholders (disabled)
# ---------------------------------------------------------------------------
ENABLE_HPARAM_TUNING = False
HPARAM_SEARCH_SPACE = {
    "DDM_WARNING_THRESHOLD": [1.8, 2.0, 2.2],
    "DDM_DRIFT_THRESHOLD": [2.8, 3.0, 3.2],
    "ADWIN_DELTA": [0.001, 0.002, 0.005],
    "XAI_SAMPLE_SIZE": [8, 10, 16],
    "LIME_N_PERTURBATIONS": [96, 128, 192],
}

# ---------------------------------------------------------------------------
# Integrated Grad-CAM placeholders (deferred / disabled)
# ---------------------------------------------------------------------------
ENABLE_INTEGRATED_GRADCAM = False
IGC_STEPS = 32
IGC_BASELINE = "zeros"

# ---------------------------------------------------------------------------
# DINO-V3 Integration
# ---------------------------------------------------------------------------
USE_DINO_FEATURE_DRIFT = True              # enable DINO feature extraction alongside error signal
DINO_MODEL_NAME        = "dinov2_vits14"   # small ViT model (14 patch size); fast & low-mem
DINO_FEATURE_REDUCTION = "mean"            # pool features to (384,) vector
DINO_FEATURE_DISTANCE  = "cosine"          # distance metric for distribution shift
DINO_CHECKPOINT_PATH   = MODEL_CHECKPOINT_DIR / "dino_finetuned"
