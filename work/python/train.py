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
from torch.utils.data import DataLoader, random_split

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
)
from data.mvtec import MVTecDataset, get_transforms
from models.resnet import build_model, fine_tune, get_optimizer, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def train_category(category: str, data_root: Path, device: torch.device) -> Path:
    log.info("=== %s ===", category)

    train_ds = MVTecDataset(
        data_root, category, split="train",
        transform=get_transforms(augment=True),
    )
    val_size   = max(1, int(VAL_SPLIT * len(train_ds)))
    train_size = len(train_ds) - val_size
    train_sub, val_sub = random_split(train_ds, [train_size, val_size])

    # Val subset: no augmentation
    val_sub.dataset = MVTecDataset(
        data_root, category, split="train",
        transform=get_transforms(augment=False),
    )

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
