"""Phase 1: Held-out defect dataset loader for concept drift simulation.

Loads held-out defect types configured in ``config.MVTEC_DEFECT_SPLIT_POLICY``,
simulating arrival of novel defect types in production.

Public API:
    get_holdout_defect_types(category) -> list[str]
    load_holdout_defects(category, transform) -> (images, labels, paths)
"""

from pathlib import Path
from typing import Callable, List, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from config import MVTEC_ROOT, get_mvtec_defect_split


def get_holdout_defect_types(category: str) -> List[str]:
    """Return configured held-out defect types for this category.
    
    Args:
        category: MVTec category present in split config.
    
    Returns:
        Sorted list of held-out defect type names.
    """
    split = get_mvtec_defect_split(category)
    return sorted(split["holdout"])


def load_holdout_defects(
    category: str,
    transform: Callable = None,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Load all configured held-out defect images from test set.
    
    Args:
        category: MVTec category
        transform: optional transform to apply (e.g., resizing + normalization)
    
    Returns:
        (images, labels, paths) where:
          - images: list of np.ndarray (uint8 RGB or normalized tensors)
          - labels: list of int (1 for defective)
          - paths: list of str (original file paths)
    """
    split = get_mvtec_defect_split(category)
    holdout_types = sorted(split["holdout"])
    images = []
    labels = []
    paths = []
    
    cat_root = MVTEC_ROOT / category
    test_root = cat_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"Missing test directory for category '{category}': {test_root}")
    
    for defect_type in holdout_types:
        defect_path = test_root / defect_type
        if not defect_path.is_dir():
            raise FileNotFoundError(
                f"Configured held-out defect directory not found for "
                f"category='{category}', defect='{defect_type}': {defect_path}"
            )
        
        for img_file in sorted(defect_path.glob("*.png")) + sorted(defect_path.glob("*.jpg")):
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
