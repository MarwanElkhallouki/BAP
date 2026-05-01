"""Fine-tune ResNet-50 on each MVTec AD category and save checkpoints.

Usage:
    python train.py                          # all 5 categories
    python train.py --category carpet
    python train.py --data_root /path/to/mvtec --category bottle
"""

import argparse
import logging
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_BACKBONE,
    LEARNING_RATE_HEAD,
    MAX_EPOCHS,
    MVTEC_CATEGORIES,
    MVTEC_ROOT,
    NUM_WORKERS,
    VAL_SPLIT,
    get_mvtec_defect_split,
)
from data.mvtec import MVTecDataset, get_transforms
from models.resnet import build_model, fine_tune, get_optimizer, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
SPLIT_SEED = 42


def _normalized_paths(dataset: MVTecDataset, indices: tuple[int, ...]) -> set[str]:
    return {str(dataset.samples[i][0]).replace("\\", "/") for i in indices}


def _assert_split_integrity(
    category: str,
    train_ds_for_paths: MVTecDataset,
    val_ds_for_paths: MVTecDataset,
    train_indices: tuple[int, ...],
    val_indices: tuple[int, ...],
) -> None:
    train_paths = _normalized_paths(train_ds_for_paths, train_indices)
    val_paths = _normalized_paths(val_ds_for_paths, val_indices)

    overlap = train_paths & val_paths
    if overlap:
        raise ValueError(f"Train/val overlap detected ({len(overlap)} files)")

    all_paths = train_paths | val_paths
    split_cfg = get_mvtec_defect_split(category)
    train_defect_markers = tuple(f"/test/{defect}/" for defect in split_cfg["train"])
    heldout_markers = tuple(f"/test/{defect}/" for defect in split_cfg["holdout"])

    unexpected_test_paths = {
        p for p in all_paths if "/test/" in p and not any(marker in p for marker in train_defect_markers)
    }
    if unexpected_test_paths:
        raise ValueError(
            f"Found unexpected test/ paths in train/val split ({len(unexpected_test_paths)} files). "
            f"Allowed test defect types: {split_cfg['train']}"
        )

    heldout_overlap = {p for p in all_paths if any(marker in p for marker in heldout_markers)}
    if heldout_overlap:
        raise ValueError(
            f"Configured held-out defects leaked into train/val split ({len(heldout_overlap)} files): "
            f"{split_cfg['holdout']}"
        )


def _build_train_val_indices(
    dataset: MVTecDataset,
    val_split: float,
    split_seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    total = len(dataset)
    if total < 2:
        raise ValueError(
            f"Need at least 2 samples to split train/val, found {total} for category='{dataset.category}'"
        )

    indices = list(range(total))
    labels = [dataset.samples[idx][1] for idx in indices]
    val_size = max(1, int(val_split * total))
    val_size = min(val_size, total - 1)

    try:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_size,
            random_state=split_seed,
            shuffle=True,
            stratify=labels,
        )
    except ValueError as exc:
        log.warning(
            "Stratified split not possible for category='%s' (%s). Falling back to deterministic split.",
            dataset.category,
            exc,
        )
        g = torch.Generator().manual_seed(split_seed)
        permuted_indices = torch.randperm(total, generator=g).tolist()
        val_indices = permuted_indices[:val_size]
        train_indices = permuted_indices[val_size:]

    return tuple(sorted(train_indices)), tuple(sorted(val_indices))


def train_category(category: str, data_root: Path, device: torch.device) -> Path:
    log.info("=== %s ===", category)

    train_ds_aug = MVTecDataset(
        data_root, category, split="train",
        transform=get_transforms(augment=True),
    )
    train_ds_no_aug = MVTecDataset(
        data_root, category, split="train",
        transform=get_transforms(augment=False),
    )
    if len(train_ds_aug) != len(train_ds_no_aug):
        raise ValueError("Train datasets must be aligned")
    for idx, ((aug_path, aug_label), (plain_path, plain_label)) in enumerate(
        zip(train_ds_aug.samples, train_ds_no_aug.samples)
    ):
        aug_path_str = str(aug_path).replace("\\", "/")
        plain_path_str = str(plain_path).replace("\\", "/")
        if aug_path_str != plain_path_str or aug_label != plain_label:
            raise ValueError(
                f"Train datasets must be sample-aligned (mismatch at idx={idx}: "
                f"{aug_path_str} vs {plain_path_str})"
            )

    train_indices, val_indices = _build_train_val_indices(train_ds_aug, VAL_SPLIT, SPLIT_SEED)
    if len(train_indices) + len(val_indices) != len(train_ds_aug):
        raise ValueError("Train/val split sizes do not add up to dataset size")

    train_labels = [train_ds_aug.samples[i][1] for i in train_indices]
    val_labels = [train_ds_aug.samples[i][1] for i in val_indices]
    log.info(
        "Split stats %s: train=%d (good=%d defect=%d), val=%d (good=%d defect=%d)",
        category,
        len(train_indices),
        sum(label == 0 for label in train_labels),
        sum(label == 1 for label in train_labels),
        len(val_indices),
        sum(label == 0 for label in val_labels),
        sum(label == 1 for label in val_labels),
    )

    train_sub = Subset(train_ds_aug, train_indices)
    val_sub = Subset(train_ds_no_aug, val_indices)
    _assert_split_integrity(category, train_ds_aug, train_ds_no_aug, train_indices, val_indices)

    train_loader = DataLoader(
        train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    model     = build_model(pretrained=True).to(device)
    optimizer = get_optimizer(model, LEARNING_RATE_HEAD, LEARNING_RATE_BACKBONE)

    train_acc = fine_tune(
        model, train_loader, val_loader, optimizer, device,
        max_epochs=MAX_EPOCHS, patience=EARLY_STOPPING_PATIENCE,
    )

    ckpt = CHECKPOINT_DIR / f"resnet50_{category}.pth"
    save_checkpoint(model, train_acc, ckpt)
    log.info("Saved %s (train_acc=%.4f)", ckpt, train_acc)
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category",  choices=MVTEC_CATEGORIES + ["all"], default="all")
    parser.add_argument("--data_root", type=Path, default=MVTEC_ROOT)
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = MVTEC_CATEGORIES if args.category == "all" else [args.category]

    log.info("Device: %s", device)
    for cat in categories:
        train_category(cat, args.data_root, device)


if __name__ == "__main__":
    main()
