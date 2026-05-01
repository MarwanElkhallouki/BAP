"""Phase 1: Pre-compute corruption datasets to eliminate runtime generation.

Instead of applying corruptions on-the-fly during experiments, pre-generate and save
all corrupted images to disk. This provides:
  - Reproducibility (same images each run)
  - Speed (disk I/O vs compute)
  - Scale comparison (small vs large corruptions in separate datasets)

Usage:
    python generate_datasets.py --category all --scale all
    python generate_datasets.py --category carpet --corruption gaussian_noise --scale small
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure the script is run with the repository virtual environment.
# if ".venv" not in str(Path(sys.executable).resolve()).lower():
#     print("ERROR: Please run this script with the repository virtual environment:")
#     print("  f:/git/BAP/.venv/Scripts/python.exe generate_datasets.py")
#     print("If you already installed dependencies, use the venv python interpreter.")
#     sys.exit(1)

# Use the local robustness/ImageNet-C package instead of a PyPI-installed one.
ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGENET_C_SRC = ROOT_DIR.parent / "robustness" / "ImageNet-C" / "imagenet_c"
if str(IMAGENET_C_SRC) not in sys.path:
    sys.path.insert(0, str(IMAGENET_C_SRC))

from config import (
    MVTEC_ROOT,
    MVTEC_CATEGORIES,
    CORRUPTION_PIXEL_LEVEL,
    CORRUPTION_GEOMETRIC,
    CORRUPTION_SEVERITIES,
    SCALE_SMALL_SEVERITY_MAX,
    SCALE_LARGE_SEVERITY_MIN,
    CORRUPTION_DATASETS,
)
from data.corruption import make_corruption_fn
from data.mvtec import MVTecDataset, get_transforms
from data.geometric import apply_rotation, apply_translation
from data.defects import get_holdout_defect_types, load_holdout_defects


def _load_clean_test_images(category: str) -> List[tuple]:
    """Load clean (severity 0) test images for a category.
    
    Returns:
        List of (image_np, label, path_str) tuples.
    """
    tf = get_transforms(augment=False)
    ds = MVTecDataset(MVTEC_ROOT, category, split="test", transform=None)
    
    images = []
    for img_tensor, label, path in ds:
        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)
        images.append((img_np, int(label), str(path)))
    
    return images


def _save_dataset_manifest(
    output_dir: Path,
    images: List[tuple],
    corruption_type: str,
    scale: str,
    category: str,
):
    """Save corrupted images and metadata manifest.
    
    Args:
        output_dir: where to save images + manifest
        images: list of (image_np, label, path, original_path) tuples
        corruption_type: e.g., "gaussian_noise", "rotation", "held_out_defects"
        scale: "small" or "large"
        category: MVTec category
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    for idx, (img_np, label, path, original_path, severity) in enumerate(images):
        # Save image
        img_filename = f"{idx:06d}.jpg"
        img_path = images_dir / img_filename
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        img_pil.save(img_path, quality=95)
        
        # Add to manifest
        manifest.append({
            "image_path": str(img_path),
            "image_idx": idx,
            "label": int(label),
            "original_path": str(original_path),
            "corruption_type": corruption_type,
            "scale": scale,
            "category": category,
            "severity": int(severity),
        })
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved {len(manifest)} images + manifest to {output_dir}")


def generate_pixel_corruptions(category: str, scales: List[str]):
    """Generate pixel-level corruptions (Gaussian noise, blur, brightness, JPEG).
    
    Phase 1 Step 1.2: Pre-compute ImageNet-C corruptions at two scales.
    """
    clean_images = _load_clean_test_images(category)
    
    for corruption_type in CORRUPTION_PIXEL_LEVEL:
        for scale in scales:
            severity_range = (
                range(1, SCALE_SMALL_SEVERITY_MAX + 1)
                if scale == "small"
                else range(SCALE_LARGE_SEVERITY_MIN, max(CORRUPTION_SEVERITIES) + 1)
            )
            
            corrupted_images = []
            print(f"\nGenerating {category} × {corruption_type} × {scale}...")
            
            for severity in severity_range:
                corrupt_fn = make_corruption_fn(corruption_type, severity)
                
                pbar = tqdm(clean_images, desc=f"  severity {severity}")
                for img_np, label, path in pbar:
                    try:
                        corrupted_np = corrupt_fn(img_np)
                        corrupted_images.append((
                            corrupted_np,
                            label,
                            path,
                            path,
                            severity
                        ))
                    except Exception as e:
                        print(f"Error corrupting {path}: {e}")
            
            # Save dataset
            output_dir = CORRUPTION_DATASETS / f"{category}_{corruption_type}_{scale}"
            _save_dataset_manifest(
                output_dir,
                corrupted_images,
                corruption_type,
                scale,
                category,
            )


def generate_geometric_transformations(category: str, scales: List[str]):
    """Generate geometric transformations (rotation, translation).
    
    Phase 1 Step 1.3: Pre-compute geometric corruptions at two scales.
    """
    clean_images = _load_clean_test_images(category)
    
    for transform_type in CORRUPTION_GEOMETRIC:
        for scale in scales:
            print(f"\nGenerating {category} × {transform_type} × {scale}...")
            
            transformed_images = []
            
            # Apply transform to each clean image (severity = 1 for geometric since it's binary)
            pbar = tqdm(clean_images, desc=f"  {transform_type}")
            for img_np, label, path in pbar:
                try:
                    if transform_type == "rotation":
                        transformed_np = apply_rotation(img_np, scale=scale)
                    elif transform_type == "translation":
                        transformed_np = apply_translation(img_np, scale=scale)
                    else:
                        continue
                    
                    transformed_images.append((
                        transformed_np,
                        label,
                        path,
                        path,
                        1,  # severity (binary: applied or not)
                    ))
                except Exception as e:
                    print(f"Error transforming {path}: {e}")
            
            # Save dataset
            output_dir = CORRUPTION_DATASETS / f"{category}_{transform_type}_{scale}"
            _save_dataset_manifest(
                output_dir,
                transformed_images,
                transform_type,
                scale,
                category,
            )


def generate_holdout_defects(category: str):
    """Generate held-out defect dataset (no corruptions, pure defects).
    
    Phase 1 Step 1.4: Pre-compute held-out defect images.
    """
    holdout_types = get_holdout_defect_types(category)
    
    if not holdout_types:
        print(f"No held-out defects for {category}; skipping.")
        return
    
    print(f"\nGenerating {category} × held_out_defects...")
    
    images_list, labels_list, paths_list = load_holdout_defects(category, transform=None)
    
    defect_images = []
    for img_arr, label, path in zip(images_list, labels_list, paths_list):
        if isinstance(img_arr, np.ndarray) and img_arr.dtype == np.float32:
            # Handle normalized tensors (convert back to uint8)
            img_arr = (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)
        
        defect_images.append((
            img_arr,
            label,
            path,
            path,
            0,  # severity (not applicable for defects)
        ))
    
    # Save dataset
    output_dir = CORRUPTION_DATASETS / f"{category}_held_out_defects_all"
    _save_dataset_manifest(
        output_dir,
        defect_images,
        "held_out_defects",
        "all",
        category,
    )


def main():
    parser = argparse.ArgumentParser(description="Pre-generate corruption datasets.")
    parser.add_argument("--category", default="all", help="Category or 'all'")
    parser.add_argument("--corruption", default="all", help="Corruption type or 'all'")
    parser.add_argument("--scale", default="all", help="Scale ('small', 'large', or 'all')")
    parser.add_argument("--skip-geometric", action="store_true", help="Skip geometric transforms")
    parser.add_argument("--skip-defects", action="store_true", help="Skip held-out defects")
    
    args = parser.parse_args()
    
    categories = MVTEC_CATEGORIES if args.category == "all" else [args.category]
    scales = ["small", "large"] if args.scale == "all" else [args.scale]
    
    for category in categories:
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")
        
        # Pixel-level corruptions
        if args.corruption in ["all"] or args.corruption in CORRUPTION_PIXEL_LEVEL:
            generate_pixel_corruptions(category, scales)
        
        # Geometric transformations
        if not args.skip_geometric and (args.corruption == "all" or args.corruption in CORRUPTION_GEOMETRIC):
            generate_geometric_transformations(category, scales)
        
        # Held-out defects
        if not args.skip_defects and args.corruption in ["all", "held_out_defects"]:
            generate_holdout_defects(category)
    
    print(f"\n{'='*60}")
    print("Dataset generation complete!")
    print(f"Datasets saved to: {CORRUPTION_DATASETS}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
