"""MVTec AD dataset loader for binary classification (normal vs defective).

Split policy is category-specific and configured in ``config.MVTEC_DEFECT_SPLIT_POLICY``:
- Training split: ``train/good`` + configured "known" train defect types.
- Test split: ``test/good`` + configured held-out (novel) defect types only.

MVTec directory layout expected:
    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/<name>_mask.png
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, get_mvtec_defect_split
)


class MVTecDataset(Dataset):
    """Binary classification dataset built from MVTec AD.

    Labels:  0 = normal,  1 = defective.
    """

    def __init__(
        self,
        root: Path,
        category: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        corruption_fn: Optional[Callable] = None,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.root = Path(root) / category
        self.category = category
        self.split = split
        self.transform = transform
        self.corruption_fn = corruption_fn
        self.samples: list[tuple[Path, int]] = []
        self.defect_split = get_mvtec_defect_split(category)

        if split == "train":
            self._load_train()
        else:
            self._load_test()

    @staticmethod
    def _iter_image_files(directory: Path) -> list[Path]:
        return sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg"))

    def _load_train(self) -> None:
        good_dir = self.root / "train" / "good"
        if not good_dir.is_dir():
            raise FileNotFoundError(f"Missing MVTec training good directory: {good_dir}")

        for p in self._iter_image_files(good_dir):
            self.samples.append((p, 0))

        test_dir = self.root / "test"
        for defect_type in self.defect_split["train"]:
            defect_dir = test_dir / defect_type
            if not defect_dir.is_dir():
                raise FileNotFoundError(
                    f"Configured training defect directory not found for "
                    f"category='{self.category}', defect='{defect_type}': {defect_dir}"
                )
            for p in self._iter_image_files(defect_dir):
                self.samples.append((p, 1))

    def _load_test(self) -> None:
        test_dir = self.root / "test"
        good_dir = test_dir / "good"
        if not good_dir.is_dir():
            raise FileNotFoundError(f"Missing MVTec test good directory: {good_dir}")

        for p in self._iter_image_files(good_dir):
            self.samples.append((p, 0))

        for defect_type in self.defect_split["holdout"]:
            defect_dir = test_dir / defect_type
            if not defect_dir.is_dir():
                raise FileNotFoundError(
                    f"Configured held-out defect directory not found for "
                    f"category='{self.category}', defect='{defect_type}': {defect_dir}"
                )
            for p in self._iter_image_files(defect_dir):
                self.samples.append((p, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.corruption_fn is not None:
            img_np = np.array(img)
            img_np = self.corruption_fn(img_np)
            img = Image.fromarray(img_np.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        return img, label, str(path)


def get_transforms(augment: bool = False) -> transforms.Compose:
    base = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    if augment:
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(base)


def load_defect_mask(root: Path, category: str, image_path: str) -> Optional[np.ndarray]:
    """Return the binary ground-truth defect mask for an MVTec image, or None for 'good'."""
    p = Path(image_path)
    defect_type = p.parent.name
    if defect_type == "good":
        return None
    mask_path = root / category / "ground_truth" / defect_type / (p.stem + "_mask.png")
    if not mask_path.exists():
        return None
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 0).astype(np.uint8)
