"""Central configuration for the XAI drift-detection pipeline.

All paths, hyperparameters, and experimental constants live here so that
individual modules never need hard-coded magic numbers.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT       = Path("data/raw")
MVTEC_ROOT      = DATA_ROOT / "mvtec"
IMAGENET_C_ROOT = DATA_ROOT / "imagenet-c"
CHECKPOINT_DIR  = Path("checkpoints")
OUTPUT_DIR      = Path("output")

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
# Corruptions (imagecorruptions / ImageNet-C)
# ---------------------------------------------------------------------------
CORRUPTION_TYPES = [
    "gaussian_noise",
    "defocus_blur",
    "brightness",
    "jpeg_compression",
]
CORRUPTION_SEVERITIES = [1, 2, 3, 4, 5]  # severity 0 = clean baseline

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
# Drift detectors (River defaults from original publications)
# ---------------------------------------------------------------------------
DDM_WARNING_THRESHOLD = 2.0    # ~95 % confidence
DDM_DRIFT_THRESHOLD   = 3.0    # ~99 % confidence
EDDM_ALPHA            = 0.95   # warning threshold
EDDM_BETA             = 0.90   # drift threshold
ADWIN_DELTA           = 0.002
MDDM_WINDOW_SIZE      = 100

# Naive rolling-accuracy baseline
BASELINE_WINDOW_SIZE       = 500
BASELINE_ACCURACY_FRACTION = 0.80  # alarm when accuracy < 80 % of train accuracy

# Detection tolerance window for TPR calculation
DETECTION_TOLERANCE_WINDOW = 500  # samples

# ---------------------------------------------------------------------------
# XAI
# ---------------------------------------------------------------------------
XAI_PRE_DRIFT_WINDOW  = 200    # images before alarm
XAI_POST_DRIFT_WINDOW = 200    # images after alarm
XAI_SAMPLE_SIZE       = 20     # images used for SHAP / LIME

GRADCAM_TOP_PERCENTILE       = 80   # binarise at 80th percentile → top-20 % activated
SHAP_SIGNIFICANCE_LEVEL      = 0.05
LIME_TOP_K_SUPERPIXELS       = 5
LIME_OVERLAP_SHIFT_THRESHOLD = 0.50  # overlap < 0.5 → substantial attribution shift

# Number of SHAP/LIME perturbation samples
SHAP_N_PERTURBATIONS = 50
LIME_N_PERTURBATIONS = 256

# GPU inference chunk size for SHAP/LIME perturbation batches
# 8 for 4 GB cards, 32 for 12 GB cards (RTX 3060+)
XAI_CHUNK_SIZE = 32

# If True, each detector is disabled after its first alarm and XAI runs exactly once per
# detector. The stream ends early once all detectors have fired.
# If False, detectors run continuously and XAI is throttled by XAI_POST_DRIFT_WINDOW.
DISABLE_DETECTOR_AFTER_ALARM = True
