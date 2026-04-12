"""Phase 1: Held-out defect dataset loader for concept drift simulation.

Loads defect types NOT included in training set per MVTec AD category,
simulating arrival of novel defect types in production.

Public API:
    get_holdout_defect_types(category) -> list[str]
    load_holdout_defects(mvtec_root, category, transform) -> (images, labels, paths)
"""

from pathlib import Path
from typing import Callable, List, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from config import MVTEC_TRAIN_DEFECT_TYPES, MVTEC_ROOT
from data.mvtec import MVTecDataset


def get_holdout_defect_types(category: str) -> List[str]:
    """Return defect types NOT in training for this category.
    
    Training includes 2 defect types per category (see config.MVTEC_TRAIN_DEFECT_TYPES).
    Holdout types are all remaining defects in the test set.
    
    Args:
        category: one of ["carpet", "bottle", "metal_nut", "transistor", "leather"]
    
    Returns:
        List of defect type names (e.g., ["hole", "contamination"] for carpet).
    """
    if category not in MVTEC_TRAIN_DEFECT_TYPES:
        raise ValueError(f"Unknown category: {category}")
    
    # All test split subdirectories for this category
    cat_path = MVTEC_ROOT / category / "test"
    all_defects = set()
    
    if cat_path.exists():
        for subdir in cat_path.iterdir():
            if subdir.is_dir() and subdir.name != "good":
                all_defects.add(subdir.name)
    
    # Remove training defects to get holdouts
    train_defects = set(MVTEC_TRAIN_DEFECT_TYPES.get(category, []))
    holdout_defects = list(sorted(all_defects - train_defects))
    
    return holdout_defects


def load_holdout_defects(
    category: str,
    transform: Callable = None,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Load all held-out defect images from test set.
    
    Args:
        category: MVTec category
        transform: optional transform to apply (e.g., resizing + normalization)
    
    Returns:
        (images, labels, paths) where:
          - images: list of np.ndarray (uint8 RGB or normalized tensors)
          - labels: list of int (1 for defective)
          - paths: list of str (original file paths)
    """
    holdout_types = get_holdout_defect_types(category)
    images = []
    labels = []
    paths = []
    
    cat_root = MVTEC_ROOT / category
    test_root = cat_root / "test"
    
    for defect_type in holdout_types:
        defect_path = test_root / defect_type
        if not defect_path.exists():
            continue
        
        for img_file in sorted(defect_path.glob("*.png")) + sorted(defect_path.glob("*.jpg")):
            try:
                img_pil = Image.open(img_file).convert("RGB")
                
                if transform:
                    # transform returns tensor; convert back to np for consistency
                    transformed = transform(img_pil)
                    if hasattr(transformed, 'numpy'):
                        img_arr = transformed.numpy()
                    else:
                        img_arr = np.array(transformed)
                else:
                    img_arr = np.array(img_pil)
                
                images.append(img_arr)
                labels.append(1)  # defective
                paths.append(str(img_file))
            except Exception as e:
                print(f"Warning: Failed to load {img_file}: {e}")
    
    return images, labels, paths


class HoldoutDefectDataset(Dataset):
    """PyTorch Dataset for held-out defects."""
    
    def __init__(self, category: str, transform: Callable = None):
        self.category = category
        self.transform = transform
        self.images, self.labels, self.paths = load_holdout_defects(category, transform=None)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple:
        img_arr = self.images[idx]
        img_pil = Image.fromarray(img_arr) if isinstance(img_arr, np.ndarray) else img_arr
        
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = img_arr
        
        label = self.labels[idx]
        path = self.paths[idx]
        
        return img_tensor, label, path
