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
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from config import (
    OUTPUT_DIR,
    XAI_CHECKPOINT_DIR,
    XAI_SAMPLE_SIZE,
    LIME_N_PERTURBATIONS,
)
from checkpoint.xai_checkpoint import load_xai_checkpoint
from models.resnet import load_checkpoint
from xai.gradcam import compute_gradcam, mean_heatmap, compute_ada
from xai.lime_analysis import compute_lime_top_k, overlap_coefficient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_xai_analysis(checkpoint_data: Dict, device: torch.device, output_dir: Path):
    """Perform XAI analysis on checkpoint data.
    
    Args:
        checkpoint_data: loaded XAICheckpoint instance
        device: torch device
        output_dir: where to save XAI results
    """
    
    # Reconstruct model
    logger.info(f"Loading model for {checkpoint_data.category}...")
    model = torch.nn.DataParallel(torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True))
    model.module.fc = torch.nn.Linear(2048, 2)
    model.load_state_dict(checkpoint_data.model_state_dict)
    model.to(device)
    model.eval()
    
    # Get samples
    pre_samples = checkpoint_data.pre_drift_samples
    post_samples = checkpoint_data.post_drift_samples
    
    logger.info(f"Pre-drift: {len(pre_samples)} samples")
    logger.info(f"Post-drift: {len(post_samples)} samples")
    
    # Prepare images
    pre_images = [Image.fromarray(s["image_np"]) for s in pre_samples]
    post_images = [Image.fromarray(s["image_np"]) for s in post_samples]
    
    from data.mvtec import get_transforms
    transform = get_transforms(augment=False)
    
    pre_tensors = torch.stack([transform(img) for img in pre_images]).to(device)
    post_tensors = torch.stack([transform(img) for img in post_images]).to(device)
    
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
    
    sample_size = min(XAI_SAMPLE_SIZE, len(pre_samples), len(post_samples))
    pre_sample_indices = np.random.choice(len(pre_samples), sample_size, replace=False)
    post_sample_indices = np.random.choice(len(post_samples), sample_size, replace=False)
    
    pre_sample_imgs = [pre_images[i] for i in pre_sample_indices]
    post_sample_imgs = [post_images[i] for i in post_sample_indices]
    
    pre_lime_topk = []
    post_lime_topk = []
    
    for img in pre_sample_imgs:
        try:
            topk = compute_lime_top_k(model, np.array(img), k=5, device=device)
            pre_lime_topk.append(topk)
        except Exception as e:
            logger.warning(f"LIME failed on pre-drift sample: {e}")
    
    for img in post_sample_imgs:
        try:
            topk = compute_lime_top_k(model, np.array(img), k=5, device=device)
            post_lime_topk.append(topk)
        except Exception as e:
            logger.warning(f"LIME failed on post-drift sample: {e}")
    
    if pre_lime_topk and post_lime_topk:
        # Compute overlap
        overlap_scores = []
        for pre_set in pre_lime_topk:
            for post_set in post_lime_topk:
                ov = overlap_coefficient(pre_set, post_set)
                overlap_scores.append(ov)
        
        if overlap_scores:
            mean_overlap = np.mean(overlap_scores)
            logger.info(f"  LIME overlap coefficient: {mean_overlap:.3f} (threshold: 0.50)")
            results["lime"] = {"mean_overlap": float(mean_overlap)}
    
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
    run_xai_analysis(checkpoint, device, output_dir)


if __name__ == "__main__":
    main()
