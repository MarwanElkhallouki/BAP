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
CORRUPTION_DATASETS    = Path("data/corruption_datasets")  # Phase 1: pre-computed
CHECKPOINT_DIR         = Path("checkpoints")
OUTPUT_DIR             = Path("output")
XAI_CHECKPOINT_DIR     = OUTPUT_DIR / "xai_checkpoints"    # Phase 4: drift detection checkpoints
MODEL_CHECKPOINT_DIR   = Path("models/checkpoints")        # Phase 6: DINO-V3 fine-tuned weights

# ---------------------------------------------------------------------------
# MVTec AD
# ---------------------------------------------------------------------------
MVTEC_CATEGORIES = ["carpet", "bottle", "metal_nut", "transistor", "leather"]

# Two defect types added to the training split per category (see methodology §3.2)
MVTEC_TRAIN_DEFECT_TYPES: dict[str, list[str]] = {
    "carpet":      ["color", "cut"],
    "bottle":      ["broken_large", "broken_small"],
    "metal_nut":   ["bent", "color"],
    "transistor":  ["bent_lead", "cut_lead"],
    "leather":     ["color", "cut"],
}

# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Corruptions (imagecorruptions / ImageNet-C) — Phase 1 & Phase 7
# ---------------------------------------------------------------------------
# Pixel-level corruptions (ImageNet-C)
CORRUPTION_PIXEL_LEVEL = [
    "gaussian_noise",
    "defocus_blur",
    "brightness",
    "jpeg_compression",
]

# Geometric transformations (Phase 7: Translations & Big Anomalies)
CORRUPTION_GEOMETRIC = ["rotation", "translation"]

# Defect-type drift (held-out MVTec defects per category)
CORRUPTION_DEFECT_TYPE = ["held_out_defects"]

# All corruption types (for iteration)
CORRUPTION_TYPES = CORRUPTION_PIXEL_LEVEL + CORRUPTION_GEOMETRIC + CORRUPTION_DEFECT_TYPE

CORRUPTION_SEVERITIES = [1, 2, 3, 4, 5]  # severity 0 = clean baseline

# Phase 7: Scale threshold — small vs large corruptions
SCALE_SMALL_SEVERITY_MAX = 2        # severity 1-2 = small
SCALE_LARGE_SEVERITY_MIN = 3        # severity 3-5 = large
GEOMETRIC_SMALL_MAGNITUDE = 0.05    # translation ±5%, rotation ±15° = small
GEOMETRIC_LARGE_MAGNITUDE = 0.10    # translation ±10%, rotation ±30° = large

# Geometric transform magnitudes (Phase 7)
ROTATION_SMALL_DEG      = 15.0      # ±15° for small scale
ROTATION_LARGE_DEG      = 30.0      # ±30° for large scale
TRANSLATION_SMALL_FRAC  = 0.05      # ±5% of image dimensions
TRANSLATION_LARGE_FRAC  = 0.10      # ±10% of image dimensions

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
LEARNING_RATE_HEAD      = 1e-4   # final FC layer
LEARNING_RATE_BACKBONE  = 1e-5   # convolutional backbone
EARLY_STOPPING_PATIENCE = 10    # epochs without val-F1 improvement
BATCH_SIZE              = 32
NUM_WORKERS             = 4
MAX_EPOCHS              = 50
VAL_SPLIT               = 0.20             # fraction of train set used for validation

# ---------------------------------------------------------------------------
# Drift detectors — Phase 5: ADWIN + DDM only (removed EDDM, MDDM, Naive)
# ---------------------------------------------------------------------------
DDM_WARNING_THRESHOLD = 2.0    # ~95 % confidence
DDM_DRIFT_THRESHOLD   = 3.0    # ~99 % confidence
ADWIN_DELTA           = 0.002

# Detection tolerance window for TPR calculation
DETECTION_TOLERANCE_WINDOW = 500  # samples

# ---------------------------------------------------------------------------
# XAI — Phase 5: Grad-CAM + LIME only (removed SHAP)
# ---------------------------------------------------------------------------
XAI_PRE_DRIFT_WINDOW  = 200    # images before alarm
XAI_POST_DRIFT_WINDOW = 200    # images after alarm
XAI_SAMPLE_SIZE       = 20     # images used for LIME; Grad-CAM on all

GRADCAM_TOP_PERCENTILE       = 80   # binarise at 80th percentile → top-20 % activated
LIME_TOP_K_SUPERPIXELS       = 5
LIME_OVERLAP_SHIFT_THRESHOLD = 0.50  # overlap < 0.5 → substantial attribution shift
LIME_N_PERTURBATIONS         = 256   # perturbation samples (can reduce to 128-150 for speed)

# GPU inference chunk size for LIME perturbation batches
# 8 for 4 GB cards, 32 for 12 GB cards (RTX 3060+)
XAI_CHUNK_SIZE = 32

# Phase 4: Checkpoint persistence
SAVE_XAI_CHECKPOINTS        = True   # save drift detection state before XAI analysis
ENABLE_XAI_ONLY_MODE        = True   # support run_xai_only.py for decoupled XAI

# If True, each detector is disabled after its first alarm and XAI runs exactly once per
# detector. The stream ends early once all detectors have fired.
# If False, detectors run continuously and XAI is throttled by XAI_POST_DRIFT_WINDOW.
DISABLE_DETECTOR_AFTER_ALARM = True

# ---------------------------------------------------------------------------
# Phase 6: DINO-V3 Integration (Approach A: Feature Distribution Shift)
# ---------------------------------------------------------------------------
USE_DINO_FEATURE_DRIFT = True              # enable DINO feature extraction alongside error signal
DINO_MODEL_NAME        = "dinov2_vits14"   # small ViT model (14 patch size); fast & low-mem
DINO_FEATURE_REDUCTION = "mean"            # pool features to (384,) vector
DINO_FEATURE_DISTANCE  = "cosine"          # distance metric for distribution shift
DINO_CHECKPOINT_PATH   = MODEL_CHECKPOINT_DIR / "dino_finetuned"
