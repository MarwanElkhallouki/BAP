"""Phase 4: XAI-Only Replay Script

Rerun XAI analysis on a previously saved checkpoint without re-running drift detection.
This enables:
  - Separate scheduling: drift detection on cloud; XAI locally
  - Reproducibility: same XAI outputs regardless of when XAI is run
  - Iterative analysis: tweak XAI parameters and rerun without re-inference

Usage:
    python run_xai_only.py --checkpoint output/xai_checkpoints/carpet_gaussian_noise_ADWIN_000 \\
                           --override-lime-perturbations 128
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch

from config import (
    XAI_CHECKPOINT_DIR,
    XAI_SAMPLE_SIZE,
    LIME_N_PERTURBATIONS,
)
from checkpoint.xai_checkpoint import XAICheckpoint, load_xai_checkpoint
from models.resnet import build_model
from xai.gradcam import compute_gradcam, mean_heatmap, compute_ada
from xai.lime_analysis import compute_lime_top_k, overlap_coefficient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return state dict with any leading DataParallel 'module.' prefixes removed."""
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def _load_replay_model(checkpoint_state: Mapping[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    """Reconstruct model and load checkpoint weights for XAI replay."""
    model = build_model(pretrained=False)
    normalized_state = _normalize_state_dict_keys(checkpoint_state)
    model.load_state_dict(normalized_state)
    model.to(device)
    model.eval()
    return model


def run_xai_analysis(
    checkpoint_data: XAICheckpoint,
    device: torch.device,
    output_dir: Path,
    lime_perturbations: Optional[int] = None,
):
    """Perform XAI analysis on checkpoint data.
    
    Args:
        checkpoint_data: loaded XAICheckpoint instance
        device: torch device
        output_dir: where to save XAI results
    """
    
    # Reconstruct model
    logger.info(f"Loading model for {checkpoint_data.category}...")
    model = _load_replay_model(checkpoint_data.model_state_dict, device)
    
    # Get samples
    pre_samples = checkpoint_data.pre_drift_samples
    post_samples = checkpoint_data.post_drift_samples

    if not pre_samples or not post_samples:
        raise ValueError(
            f"Checkpoint missing replay samples: pre={len(pre_samples)}, post={len(post_samples)}"
        )
    
    logger.info(f"Pre-drift: {len(pre_samples)} samples")
    logger.info(f"Post-drift: {len(post_samples)} samples")
    
    # Prepare images
    pre_images_np = [np.asarray(s["image_np"], dtype=np.uint8) for s in pre_samples]
    post_images_np = [np.asarray(s["image_np"], dtype=np.uint8) for s in post_samples]
    
    from data.mvtec import get_transforms
    transform = get_transforms(augment=False)
    
    from PIL import Image

    pre_tensors = torch.stack([transform(Image.fromarray(img_np)) for img_np in pre_images_np]).to(device)
    post_tensors = torch.stack([transform(Image.fromarray(img_np)) for img_np in post_images_np]).to(device)
    
    # ====================================================================
    # Grad-CAM Analysis
    # ====================================================================
    logger.info("Computing Grad-CAM heatmaps...")
    
    pre_gradcam = compute_gradcam(model, pre_tensors, target_class=1, device=device)
    post_gradcam = compute_gradcam(model, post_tensors, target_class=1, device=device)
    
    pre_mean = mean_heatmap(pre_gradcam)
    post_mean = mean_heatmap(post_gradcam)
    
    logger.info(f"  Pre-drift mean heatmap range: [{pre_mean.min():.3f}, {pre_mean.max():.3f}]")
    logger.info(f"  Post-drift mean heatmap range: [{post_mean.min():.3f}, {post_mean.max():.3f}]")
    
    # Save heatmaps
    results = {
        "detector": checkpoint_data.detector_name,
        "category": checkpoint_data.category,
        "corruption": checkpoint_data.corruption_type,
        "drift_type": checkpoint_data.drift_type,
        "severity": checkpoint_data.severity,
        "gradcam": {
            "pre_mean": pre_mean.tolist(),
            "post_mean": post_mean.tolist(),
        },
    }
    
    # ====================================================================
    # LIME Analysis (on sample subset)
    # ====================================================================
    logger.info("Computing LIME analysis...")

    import xai.lime_analysis as lime_analysis_module
    effective_lime_perturbations = lime_perturbations or LIME_N_PERTURBATIONS
    lime_analysis_module.LIME_N_PERTURBATIONS = effective_lime_perturbations
    logger.info(f"  LIME perturbations: {effective_lime_perturbations}")
    
    sample_size = min(XAI_SAMPLE_SIZE, len(pre_samples), len(post_samples))
    rng = np.random.default_rng(int(checkpoint_data.sample_index))
    pre_sample_indices = rng.choice(len(pre_samples), sample_size, replace=False)
    post_sample_indices = rng.choice(len(post_samples), sample_size, replace=False)
    
    pre_sample_imgs = [pre_images_np[i] for i in pre_sample_indices]
    post_sample_imgs = [post_images_np[i] for i in post_sample_indices]

    old_lime_perturbations = lime_analysis_module.LIME_N_PERTURBATIONS
    lime_analysis_module.LIME_N_PERTURBATIONS = effective_lime_perturbations

    try:
        pre_lime_topk = compute_lime_top_k(
            model=model,
            images_np=pre_sample_imgs,
            transform=transform,
            device=device,
            n_samples=sample_size,
        )
        post_lime_topk = compute_lime_top_k(
            model=model,
            images_np=post_sample_imgs,
            transform=transform,
            device=device,
            n_samples=sample_size,
        )

        mean_overlap = overlap_coefficient(pre_lime_topk, post_lime_topk)
        logger.info(f"  LIME overlap coefficient: {mean_overlap:.3f} (threshold: 0.50)")
        results["lime"] = {
            "mean_overlap": float(mean_overlap),
            "n_samples": int(sample_size),
            "n_perturbations": int(effective_lime_perturbations),
        }
    except Exception:
        logger.exception("LIME replay failed")
    finally:
        lime_analysis_module.LIME_N_PERTURBATIONS = old_lime_perturbations
    
    # ====================================================================
    # Save Results
    # ====================================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"{checkpoint_data.category}_{checkpoint_data.detector_name}_xai_results.json"
    
    import json
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"XAI results saved to {result_file}")


def main():
    parser = argparse.ArgumentParser(description="Rerun XAI analysis from checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (without extension)")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--override-lime-perturbations", type=int, default=None, help="Override LIME perturbations")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = load_xai_checkpoint(checkpoint_path)
    
    # Run XAI
    output_dir = Path(args.output_dir) if args.output_dir else XAI_CHECKPOINT_DIR / "xai_results"
    run_xai_analysis(
        checkpoint_data=checkpoint,
        device=device,
        output_dir=output_dir,
        lime_perturbations=args.override_lime_perturbations,
    )


if __name__ == "__main__":
    main()
