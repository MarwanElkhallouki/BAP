"""ResNet-50 fine-tuned for binary classification (normal vs defective).

Architecture choice (from thesis §3.3):
  - Pre-trained on ImageNet (IMAGENET1K_V1 weights)
  - Final FC layer replaced with a 2-class head
  - Differential learning rates: lr_head for FC, lr_backbone for conv layers
  - Early stopping on validation F1 (patience = 10 epochs)
  - Grad-CAM compatible without any reshaping (uses model.layer4[-1])
"""

from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def get_optimizer(model: nn.Module, lr_head: float, lr_backbone: float) -> Adam:
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    return Adam([
        {"params": model.fc.parameters(), "lr": lr_head},
        {"params": backbone_params,        "lr": lr_backbone},
    ])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return (accuracy, macro-F1)."""
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels, _ in loader:
        preds = model(imgs.to(device)).argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return float(acc), float(f1)


def fine_tune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 10,
) -> float:
    """Fine-tune with early stopping on val-F1.

    Returns the training accuracy of the best checkpoint
    (used as the baseline for the naive drift detector).
    """
    all_train_labels = []
    for _, lbls, _ in train_loader:
        all_train_labels.extend(lbls.tolist())
    counts = torch.bincount(torch.tensor(all_train_labels))
    weights = (1.0 / counts.float())
    weights = (weights / weights.sum() * len(counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_f1 = 0.0
    patience_counter = 0
    best_state: dict | None = None
    best_train_acc = 0.0

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        train_acc, _ = evaluate(model, train_loader, device)

        print(f"  epoch {epoch+1:03d}  loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_train_acc = train_acc
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_train_acc


def save_checkpoint(model: nn.Module, train_accuracy: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "train_accuracy": train_accuracy}, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, float]:
    ckpt = torch.load(path, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, ckpt["train_accuracy"]
