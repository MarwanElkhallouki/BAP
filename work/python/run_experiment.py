"""Run the full drift-detection + XAI experiment for one MVTec AD category.

Steps:
  1. Load fine-tuned ResNet-50 from checkpoint.
  2. Build the corruption stream (severity 0 → 5, one corruption type at a time).
  3. Run the Pipeline, collecting alarms and detector metrics.
  4. Compute AUROC degradation curve and ADA curve.
  5. Print summary tables.

Usage:
    python run_experiment.py --category carpet --corruption gaussian_noise
    python run_experiment.py --category bottle --corruption defocus_blur --all_corruptions
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR,
    CORRUPTION_SEVERITIES,
    CORRUPTION_TYPES,
    MVTEC_CATEGORIES,
    MVTEC_ROOT,
    OUTPUT_DIR,
)
from data.corruption import make_corruption_fn
from data.mvtec import MVTecDataset, get_transforms
from evaluation.metrics import print_detector_summary
from models.resnet import load_checkpoint
from pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _dataset_to_list(ds: MVTecDataset) -> list[tuple[np.ndarray, int, str]]:
    """Load an entire dataset into memory as (np_image, label, path) triples."""
    items = []
    for img_tensor, label, path in ds:
        # Undo normalisation to get back a raw uint8 image for XAI methods
        img_np = np.array(Image.open(path).convert("RGB"))
        items.append((img_np, int(label), path))
    return items


def build_stream(
    category: str,
    corruption_type: str,
    data_root: Path,
) -> tuple[list[tuple], list[int]]:
    """Build the ordered sample stream and note drift onset indices.

    Phase 7: Stream now includes (image, label, severity, path, drift_type, scale).
    Stream order: all severity-0 images, then severity-1, …, severity-5.
    Each severity transition is a drift event.

    Returns (stream_list, drift_onsets) where drift_onsets[i] is the sample
    index at which severity i+1 begins.
    """
    from config import SCALE_SMALL_SEVERITY_MAX, SCALE_LARGE_SEVERITY_MIN
    
    tf = get_transforms(augment=False)
    base_ds = MVTecDataset(data_root, category, split="test", transform=tf)
    samples_clean = _dataset_to_list(base_ds)

    stream: list[tuple] = [
        (img, lbl, 0, path, "corruption", None) for img, lbl, path in samples_clean
    ]
    drift_onsets: list[int] = []

    for sev in CORRUPTION_SEVERITIES:
        drift_onsets.append(len(stream))
        corrupt_fn = make_corruption_fn(corruption_type, sev)
        
        # Phase 7: Determine scale
        scale = "small" if sev <= SCALE_SMALL_SEVERITY_MAX else "large"
        
        ds = MVTecDataset(data_root, category, split="test",
                          transform=tf, corruption_fn=corrupt_fn)
        for img_tensor, label, path in ds:
            img_np = corrupt_fn(np.array(Image.open(path).convert("RGB")))
            stream.append((img_np, int(label), sev, path, "corruption", scale))

    return stream, drift_onsets


def run_one(category: str, corruption_type: str, global_pbar=None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = CHECKPOINT_DIR / f"resnet50_{category}.pth"

    if not ckpt.exists():
        log.error("Checkpoint not found: %s  (run train.py first)", ckpt)
        return

    model, train_acc = load_checkpoint(ckpt, device)
    transform = get_transforms(augment=False)
    log.info("Loaded %s  (train_acc=%.4f)", ckpt, train_acc)

    log.info("Building stream: category=%s  corruption=%s", category, corruption_type)
    stream, drift_onsets = build_stream(category, corruption_type, MVTEC_ROOT)
    log.info("Stream length: %d samples,  drift onsets: %s", len(stream), drift_onsets)
    if global_pbar is not None:
        global_pbar.total = (global_pbar.total or 0) + len(stream)
        global_pbar.set_postfix({"run": f"{category}/{corruption_type}"}, refresh=True)

    pipe = Pipeline(
        model=model,
        device=device,
        transform=transform,
        train_accuracy=train_acc,
        category=category,
        corruption_type=corruption_type,  # Phase 1/4: for checkpoint metadata
    )

    alarms, det_results = pipe.run(
        iter(stream),
        drift_onsets=drift_onsets,
        stream_len=len(stream),
        global_pbar=global_pbar,
    )

    # --- AUROC curve ---
    auroc_curve = pipe.compute_auroc_curve()
    print(f"\nAUROC degradation curve  [{category} / {corruption_type}]")
    for sev, auc in sorted(auroc_curve.items()):
        print(f"  severity {sev}: AUROC = {auc:.4f}")

    # --- ADA curve (requires MVTec masks) ---
    try:
        ada_curve = pipe.compute_ada_curve()
        print(f"\nADA curve  [{category} / {corruption_type}]")
        for sev, ada in sorted(ada_curve.items()):
            print(f"  severity {sev}: ADA = {ada:.4f}")
    except Exception as exc:
        log.warning("ADA curve skipped: %s", exc)

    # --- Detector summary ---
    print(f"\nDetector summary  [{category} / {corruption_type}]")
    print_detector_summary(list(det_results.values()))

    # --- Alarm details ---
    print(f"\nTotal alarms: {len(alarms)}")
    for a in alarms:
        cam_change = a.xai.get("gradcam_change_map", "—")
        lime_ovlp = a.xai.get("lime_overlap_coefficient", "—")
        print(f"  [{a.detector_name}] sample={a.sample_index}  sev={a.severity}  "
              f"drift_type={a.drift_type}  scale={a.scale}  "
              f"cam_change={'OK' if cam_change != '—' else '—'}  lime_overlap={lime_ovlp}")

    # --- Save results ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{category}_{corruption_type}_results.npz"
    np.savez(
        out,
        auroc_severities=list(auroc_curve.keys()),
        auroc_values=list(auroc_curve.values()),
    )
    log.info("Results saved to %s", out)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the drift-detection + XAI experiment for one MVTec AD category.\n\n"
            "Steps: load ResNet-50 checkpoint → build corruption stream (sev 0→5) → "
            "run Pipeline → print AUROC / ADA / detector summary → save .npz results."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_experiment.py --category carpet --corruption gaussian_noise\n"
            "  python run_experiment.py --category bottle --all_corruptions\n"
            "  python run_experiment.py --all_categories --corruption gaussian_noise\n"
            "  python run_experiment.py --all_categories --all_corruptions\n"
            f"  Available categories : {', '.join(MVTEC_CATEGORIES)}\n"
            f"  Available corruptions: {', '.join(CORRUPTION_TYPES)}"
        ),
    )
    parser.add_argument(
        "--category",
        choices=MVTEC_CATEGORIES,
        default="carpet",
        help=f"MVTec AD product category to evaluate (default: carpet). Choices: {', '.join(MVTEC_CATEGORIES)}",
    )
    parser.add_argument(
        "--corruption",
        choices=CORRUPTION_TYPES,
        default="gaussian_noise",
        help=f"ImageNet-C corruption type to apply (default: gaussian_noise). Choices: {', '.join(CORRUPTION_TYPES)}",
    )
    parser.add_argument(
        "--all_corruptions",
        action="store_true",
        help="Run all corruption types for the selected category instead of just --corruption",
    )
    parser.add_argument(
        "--all_categories",
        action="store_true",
        help="Run all categories for the selected corruption instead of just --category",
    )
    args = parser.parse_args()

    categories = MVTEC_CATEGORIES if args.all_categories else [args.category]
    corruptions = CORRUPTION_TYPES if args.all_corruptions else [args.corruption]
    pairs = [(cat, corr) for corr in corruptions for cat in categories]
    with tqdm(total=0, desc="Experiment", unit="sample", position=0, leave=True) as global_pbar:
        for cat, corr in pairs:
            run_one(cat, corr, global_pbar=global_pbar)


if __name__ == "__main__":
    main()
