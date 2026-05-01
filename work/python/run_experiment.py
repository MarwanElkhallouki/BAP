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
    python run_experiment.py --runtime-profile full --all_categories --corruption gaussian_noise
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from config import (
    CHECKPOINT_DIR,
    CORRUPTION_DATASETS,
    CORRUPTION_GEOMETRIC,
    HPARAM_SEARCH_SPACE,
    LAPTOP_MODE,
    LIME_N_PERTURBATIONS,
    CORRUPTION_PIXEL_LEVEL,
    CORRUPTION_SEVERITIES,
    CORRUPTION_TYPES,
    MVTEC_CATEGORIES,
    MVTEC_ROOT,
    OUTPUT_DIR,
    XAI_CHUNK_SIZE,
    XAI_SAMPLE_SIZE,
)
from data.corruption import load_manifest_entries
from data.mvtec import MVTecDataset, get_transforms
from evaluation.metrics import print_detector_summary
from models.resnet import load_checkpoint
from pipeline import Pipeline
import config as config_module
import pipeline as pipeline_module
import xai.lime_analysis as lime_analysis_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STREAM_SCENARIO_SINGLE = "single_progressive"
STREAM_SCENARIO_MIXED = "mixed_progressive"
STREAM_SCENARIOS = (STREAM_SCENARIO_SINGLE, STREAM_SCENARIO_MIXED)
CAMERA_CORRUPTION_TYPES = tuple(CORRUPTION_PIXEL_LEVEL + CORRUPTION_GEOMETRIC)


def _load_rgb_np(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _dataset_to_list(ds: MVTecDataset) -> list[tuple[np.ndarray, int, str]]:
    """Load an entire dataset into memory as (np_image, label, path) triples."""
    items = []
    for _img_tensor, label, path in ds:
        img_np = _load_rgb_np(path)
        items.append((img_np, int(label), path))
    return items


def _expected_manifest_paths(category: str, corruption_type: str) -> list[Path]:
    if corruption_type in CORRUPTION_PIXEL_LEVEL or corruption_type in CORRUPTION_GEOMETRIC:
        scales = ["small", "large"]
    else:
        scales = ["all"]
    return [
        CORRUPTION_DATASETS / f"{category}_{corruption_type}_{scale}" / "manifest.json"
        for scale in scales
    ]


def _full_profile_sample_size() -> int:
    values = [XAI_SAMPLE_SIZE]
    values.extend(HPARAM_SEARCH_SPACE.get("XAI_SAMPLE_SIZE", []))
    return int(max(values))


def _full_profile_lime_perturbations() -> int:
    values = [LIME_N_PERTURBATIONS]
    values.extend(HPARAM_SEARCH_SPACE.get("LIME_N_PERTURBATIONS", []))
    return int(max(values))


def _validate_manifests_present(manifest_paths: list[Path], *, category: str, corruption_type: str) -> None:
    missing_manifests = [p for p in manifest_paths if not p.exists()]
    if missing_manifests:
        missing_str = "\n  - ".join(str(p) for p in missing_manifests)
        raise FileNotFoundError(
            f"Missing required precomputed manifest file(s) for category={category}, corruption={corruption_type}:\n"
            f"  - {missing_str}"
        )


def _expected_severities_for_corruption(
    corruption_type: str,
    entries_by_severity: dict[int, list[dict[str, Any]]],
) -> list[int]:
    if corruption_type in CORRUPTION_PIXEL_LEVEL:
        return list(CORRUPTION_SEVERITIES)
    if corruption_type in CORRUPTION_GEOMETRIC:
        return [1]
    return sorted(entries_by_severity.keys())


def _load_validated_entries_by_severity(
    category: str,
    corruption_type: str,
) -> dict[int, list[dict[str, Any]]]:
    manifest_paths = _expected_manifest_paths(category, corruption_type)
    _validate_manifests_present(manifest_paths, category=category, corruption_type=corruption_type)

    manifest_entries: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest_entries.extend(
            load_manifest_entries(
                manifest_path,
                expected_category=category,
                expected_corruption_type=corruption_type,
            )
        )
    if not manifest_entries:
        expected_paths = "\n  - ".join(str(p) for p in manifest_paths)
        raise FileNotFoundError(
            f"No manifest entries found for category={category}, corruption={corruption_type}, "
            f"expected manifest path(s):\n  - {expected_paths}"
        )

    entries_by_severity: dict[int, list[dict[str, Any]]] = {}
    for entry in manifest_entries:
        entries_by_severity.setdefault(int(entry["severity"]), []).append(entry)

    expected_severities = _expected_severities_for_corruption(corruption_type, entries_by_severity)
    missing_severities = [sev for sev in expected_severities if sev not in entries_by_severity]
    if missing_severities:
        raise ValueError(
            f"Manifest data missing expected severities {missing_severities} for "
            f"category={category}, corruption={corruption_type}"
        )

    for sev, severity_entries in entries_by_severity.items():
        entries_by_severity[sev] = sorted(
            severity_entries,
            key=lambda e: (
                int(e.get("image_idx", 0)),
                str(e.get("resolved_original_path", "")),
                str(e.get("resolved_image_path", "")),
            ),
        )

    return entries_by_severity


def _append_stage(
    stream: list[tuple],
    drift_onsets: list[int],
    stage_plan: list[dict[str, Any]],
    *,
    stage_name: str,
    severity: int,
    entries: list[dict[str, Any]],
    drift_type: str,
) -> None:
    if not entries:
        return

    stage_plan.append(
        {
            "stage_name": stage_name,
            "severity": int(severity),
            "count": len(entries),
            "drift_type": drift_type,
            "onset_index": len(stream),
        }
    )
    drift_onsets.append(len(stream))
    for entry in entries:
        img_np = _load_rgb_np(entry["resolved_image_path"])
        stream.append(
            (
                img_np,
                int(entry["label"]),
                int(severity),
                str(entry["resolved_original_path"]),
                drift_type,
                str(entry["scale"]),
            )
        )


def _iter_progressive_stages(
    corruption_type: str,
    entries_by_severity: dict[int, list[dict[str, Any]]],
):
    ordered_severities = _expected_severities_for_corruption(corruption_type, entries_by_severity)
    for sev in ordered_severities:
        severity_entries = entries_by_severity.get(sev, [])
        if not severity_entries:
            continue
        if corruption_type in CORRUPTION_GEOMETRIC:
            by_scale: dict[str, list[dict[str, Any]]] = {}
            for entry in severity_entries:
                by_scale.setdefault(str(entry.get("scale", "all")), []).append(entry)
            for scale in ("small", "large", "all"):
                scale_entries = by_scale.get(scale, [])
                if not scale_entries:
                    continue
                yield {
                    "stage_name": f"{corruption_type}_sev{sev}_{scale}",
                    "severity": int(sev),
                    "entries": scale_entries,
                }
            continue
        yield {
            "stage_name": f"{corruption_type}_sev{sev}",
            "severity": int(sev),
            "entries": severity_entries,
        }


def _build_stream_single_progressive(
    category: str,
    corruption_type: str,
    data_root: Path,
) -> tuple[list[tuple], list[int], list[dict[str, Any]]]:
    tf = get_transforms(augment=False)
    base_ds = MVTecDataset(data_root, category, split="test", transform=tf)
    samples_clean = _dataset_to_list(base_ds)

    stream: list[tuple] = [
        (img, lbl, 0, path, "clean", None) for img, lbl, path in samples_clean
    ]
    drift_onsets: list[int] = []
    stage_plan: list[dict[str, Any]] = []

    entries_by_severity = _load_validated_entries_by_severity(category, corruption_type)
    for stage in _iter_progressive_stages(corruption_type, entries_by_severity):
        _append_stage(
            stream,
            drift_onsets,
            stage_plan,
            stage_name=stage["stage_name"],
            severity=stage["severity"],
            entries=stage["entries"],
            drift_type=str(corruption_type),
        )

    return stream, drift_onsets, stage_plan


def _build_stream_mixed_progressive(
    category: str,
    data_root: Path,
    *,
    camera_corruptions: list[str],
    include_held_out_defects: bool,
) -> tuple[list[tuple], list[int], list[dict[str, Any]]]:
    if not camera_corruptions:
        raise ValueError("Mixed progressive scenario requires at least one camera corruption.")
    invalid = [corr for corr in camera_corruptions if corr not in CAMERA_CORRUPTION_TYPES]
    if invalid:
        raise ValueError(
            "Mixed progressive scenario only supports camera corruptions "
            f"{list(CAMERA_CORRUPTION_TYPES)}; got invalid entries: {invalid}"
        )

    tf = get_transforms(augment=False)
    base_ds = MVTecDataset(data_root, category, split="test", transform=tf)
    samples_clean = _dataset_to_list(base_ds)

    stream: list[tuple] = [
        (img, lbl, 0, path, "clean", None) for img, lbl, path in samples_clean
    ]
    drift_onsets: list[int] = []
    stage_plan: list[dict[str, Any]] = []

    for corruption_type in camera_corruptions:
        entries_by_severity = _load_validated_entries_by_severity(category, corruption_type)
        for stage in _iter_progressive_stages(corruption_type, entries_by_severity):
            _append_stage(
                stream,
                drift_onsets,
                stage_plan,
                stage_name=stage["stage_name"],
                severity=stage["severity"],
                entries=stage["entries"],
                drift_type=str(corruption_type),
            )

    if include_held_out_defects:
        defect_entries_by_severity = _load_validated_entries_by_severity(category, "held_out_defects")
        defect_entries: list[dict[str, Any]] = []
        for sev in sorted(defect_entries_by_severity.keys()):
            defect_entries.extend(defect_entries_by_severity[sev])
        _append_stage(
            stream,
            drift_onsets,
            stage_plan,
            stage_name="held_out_defects",
            severity=max(CORRUPTION_SEVERITIES) + 1,
            entries=defect_entries,
            drift_type="held_out_defects",
        )

    return stream, drift_onsets, stage_plan


def build_stream_for_scenario(
    *,
    category: str,
    data_root: Path,
    stream_scenario: str,
    corruption_type: str,
    mixed_camera_corruptions: list[str] | None = None,
    include_held_out_defects: bool = False,
) -> tuple[list[tuple], list[int], list[dict[str, Any]]]:
    if stream_scenario == STREAM_SCENARIO_SINGLE:
        return _build_stream_single_progressive(category, corruption_type, data_root)
    if stream_scenario == STREAM_SCENARIO_MIXED:
        return _build_stream_mixed_progressive(
            category,
            data_root,
            camera_corruptions=list(mixed_camera_corruptions or []),
            include_held_out_defects=include_held_out_defects,
        )
    raise ValueError(f"Unsupported stream scenario: {stream_scenario}")


RUNTIME_PROFILE_SETTINGS: dict[str, dict[str, int | bool]] = {
    "laptop": {
        "LAPTOP_MODE": bool(LAPTOP_MODE),
        "XAI_SAMPLE_SIZE": int(XAI_SAMPLE_SIZE),
        "LIME_N_PERTURBATIONS": int(LIME_N_PERTURBATIONS),
        "XAI_CHUNK_SIZE": int(XAI_CHUNK_SIZE),
    },
    "full": {
        "LAPTOP_MODE": False,
        "XAI_SAMPLE_SIZE": _full_profile_sample_size(),
        "LIME_N_PERTURBATIONS": _full_profile_lime_perturbations(),
        "XAI_CHUNK_SIZE": max(int(XAI_CHUNK_SIZE), 32),
    },
}


def _apply_runtime_profile(profile: str) -> None:
    settings = RUNTIME_PROFILE_SETTINGS[profile]

    # config module values
    config_module.LAPTOP_MODE = bool(settings["LAPTOP_MODE"])
    config_module.XAI_SAMPLE_SIZE = int(settings["XAI_SAMPLE_SIZE"])
    config_module.LIME_N_PERTURBATIONS = int(settings["LIME_N_PERTURBATIONS"])
    config_module.XAI_CHUNK_SIZE = int(settings["XAI_CHUNK_SIZE"])

    # imported module-level runtime knobs
    pipeline_module.XAI_SAMPLE_SIZE = int(settings["XAI_SAMPLE_SIZE"])
    lime_analysis_module.LIME_N_PERTURBATIONS = int(settings["LIME_N_PERTURBATIONS"])
    lime_analysis_module.XAI_CHUNK_SIZE = int(settings["XAI_CHUNK_SIZE"])

    log.info(
        "Runtime profile '%s' active (LAPTOP_MODE=%s, XAI_SAMPLE_SIZE=%d, LIME_N_PERTURBATIONS=%d, XAI_CHUNK_SIZE=%d)",
        profile,
        settings["LAPTOP_MODE"],
        settings["XAI_SAMPLE_SIZE"],
        settings["LIME_N_PERTURBATIONS"],
        settings["XAI_CHUNK_SIZE"],
    )


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
    stream, drift_onsets, _stage_plan = build_stream_for_scenario(
        category=category,
        data_root=data_root,
        stream_scenario=STREAM_SCENARIO_SINGLE,
        corruption_type=corruption_type,
    )
    return stream, drift_onsets


def _resolve_mixed_camera_corruptions(args: argparse.Namespace) -> list[str]:
    if args.mixed_camera_corruptions:
        return list(dict.fromkeys(args.mixed_camera_corruptions))
    if (args.corruption in CAMERA_CORRUPTION_TYPES) and (not args.all_corruptions):
        return [args.corruption]
    return list(CAMERA_CORRUPTION_TYPES)


def _log_stage_plan(stage_plan: list[dict[str, Any]]) -> None:
    if not stage_plan:
        log.info("Stage composition: no drift stages found")
        return
    log.info("Stage composition (%d stages):", len(stage_plan))
    for stage in stage_plan:
        log.info(
            "  - onset=%d  stage=%s  severity=%d  count=%d  drift_type=%s",
            stage["onset_index"],
            stage["stage_name"],
            stage["severity"],
            stage["count"],
            stage["drift_type"],
        )


def _result_suffix(
    *,
    stream_scenario: str,
    mixed_camera_corruptions: list[str] | None,
    include_held_out_defects: bool,
) -> str:
    if stream_scenario != STREAM_SCENARIO_MIXED:
        return ""
    camera_part = "-".join(mixed_camera_corruptions or ["none"])
    heldout_part = "heldout" if include_held_out_defects else "noheldout"
    return f"{stream_scenario}_{camera_part}_{heldout_part}"


def run_one(
    category: str,
    corruption_type: str,
    *,
    stream_scenario: str = STREAM_SCENARIO_SINGLE,
    mixed_camera_corruptions: list[str] | None = None,
    include_held_out_defects: bool = False,
    dry_run_stream: bool = False,
    global_pbar=None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = CHECKPOINT_DIR / f"resnet50_{category}.pth"

    log.info(
        "Building stream: category=%s  scenario=%s  corruption=%s",
        category,
        stream_scenario,
        corruption_type,
    )
    stream, drift_onsets, stage_plan = build_stream_for_scenario(
        category=category,
        data_root=MVTEC_ROOT,
        stream_scenario=stream_scenario,
        corruption_type=corruption_type,
        mixed_camera_corruptions=mixed_camera_corruptions,
        include_held_out_defects=include_held_out_defects,
    )
    log.info("Stream length: %d samples,  drift onsets: %s", len(stream), drift_onsets)
    _log_stage_plan(stage_plan)

    if dry_run_stream:
        log.info("Dry-run stream mode active; skipping model loading and pipeline execution.")
        return

    if not ckpt.exists():
        log.error("Checkpoint not found: %s  (run train.py first)", ckpt)
        return

    model, train_acc = load_checkpoint(ckpt, device)
    transform = get_transforms(augment=False)
    log.info("Loaded %s  (train_acc=%.4f)", ckpt, train_acc)

    if global_pbar is not None:
        global_pbar.total = (global_pbar.total or 0) + len(stream)
        global_pbar.set_postfix({"run": f"{category}/{corruption_type}/{stream_scenario}"}, refresh=True)

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
        cam_change = a.xai.get("gradcam_change_map")
        lime_ovlp = a.xai.get("lime_overlap_coefficient", "—")
        true_scale = a.scale if a.scale is not None else "—"
        inferred_scale = a.inferred_scale if a.inferred_scale is not None else "—"
        scale_score = a.scale_interpretation.get("combined_large_score")
        scale_score_str = f"{scale_score:.3f}" if isinstance(scale_score, float) else "—"
        lime_ovlp_str = f"{lime_ovlp:.3f}" if isinstance(lime_ovlp, float) else "—"
        cam_change_status = "OK" if cam_change is not None else "—"
        print(f"  [{a.detector_name}] sample={a.sample_index}  sev={a.severity}  "
               f"drift_type={a.drift_type}  true_scale={true_scale}  "
               f"inferred_scale={inferred_scale}  scale_score={scale_score_str}  "
               f"cam_change={cam_change_status}  lime_overlap={lime_ovlp_str}")

    # --- Save results ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = _result_suffix(
        stream_scenario=stream_scenario,
        mixed_camera_corruptions=mixed_camera_corruptions,
        include_held_out_defects=include_held_out_defects,
    )
    if suffix:
        out = OUTPUT_DIR / f"{category}_{corruption_type}_{suffix}_results.npz"
    else:
        out = OUTPUT_DIR / f"{category}_{corruption_type}_results.npz"
    np.savez(
        out,
        category=category,
        corruption=corruption_type,
        stream_scenario=stream_scenario,
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
            "  python run_experiment.py --runtime-profile full --all_categories --corruption gaussian_noise\n"
            f"  Available categories : {', '.join(MVTEC_CATEGORIES)}\n"
            f"  Available corruptions: {', '.join(CORRUPTION_TYPES)}"
        ),
    )
    parser.add_argument(
        "--stream-scenario",
        choices=STREAM_SCENARIOS,
        default=STREAM_SCENARIO_SINGLE,
        help=(
            "Stream construction scenario. "
            "'single_progressive' keeps existing behavior (clean then one corruption over severities). "
            "'mixed_progressive' composes a deterministic multi-stage schedule of camera degradations "
            "with optional held-out defect stage."
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
    parser.add_argument(
        "--runtime-profile",
        choices=("laptop", "full"),
        default="laptop",
        help=(
            "Runtime tuning profile. "
            "'laptop' keeps lightweight defaults from config; "
            "'full' increases XAI sample size/perturbations for higher-fidelity analysis."
        ),
    )
    parser.add_argument(
        "--mixed-camera-corruptions",
        nargs="+",
        choices=CAMERA_CORRUPTION_TYPES,
        help=(
            "Camera degradations used by --stream-scenario mixed_progressive. "
            "If omitted: defaults to --corruption when that is a camera corruption and --all_corruptions is not set; "
            f"otherwise uses all camera corruptions ({', '.join(CAMERA_CORRUPTION_TYPES)})."
        ),
    )
    parser.add_argument(
        "--mixed-include-heldout-defects",
        action="store_true",
        help="Append held-out defect stage to mixed_progressive stream.",
    )
    parser.add_argument(
        "--dry-run-stream",
        action="store_true",
        help="Build stream(s) and print drift onsets/stage composition without model inference.",
    )
    args = parser.parse_args()
    _apply_runtime_profile(args.runtime_profile)

    categories = MVTEC_CATEGORIES if args.all_categories else [args.category]
    if args.stream_scenario == STREAM_SCENARIO_SINGLE:
        corruptions = CORRUPTION_TYPES if args.all_corruptions else [args.corruption]
        runs = [
            {
                "category": cat,
                "corruption_type": corr,
                "mixed_camera_corruptions": None,
                "include_held_out_defects": False,
            }
            for corr in corruptions
            for cat in categories
        ]
    else:
        mixed_camera_corruptions = _resolve_mixed_camera_corruptions(args)
        runs = [
            {
                "category": cat,
                "corruption_type": "mixed_progressive",
                "mixed_camera_corruptions": mixed_camera_corruptions,
                "include_held_out_defects": bool(args.mixed_include_heldout_defects),
            }
            for cat in categories
        ]

    with tqdm(total=0, desc="Experiment", unit="sample", position=0, leave=True) as global_pbar:
        for run in runs:
            run_one(
                run["category"],
                run["corruption_type"],
                stream_scenario=args.stream_scenario,
                mixed_camera_corruptions=run["mixed_camera_corruptions"],
                include_held_out_defects=run["include_held_out_defects"],
                dry_run_stream=bool(args.dry_run_stream),
                global_pbar=global_pbar,
            )


if __name__ == "__main__":
    main()
