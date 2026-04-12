"""Phase 6: DINO-V3 Integration — Approach A: Feature Distribution Shift Detection

DINO (Distillation without Labels) pre-trained features are invariant to corruptions
and augmentations, making them useful for detecting distribution shifts that ResNet-50
predictions might miss.

This module extracts DINO-V3 features and uses them as a secondary drift signal:
  - Primary signal: ResNet-50 binary error (corruptions/defects visible to model)
  - Secondary signal: DINO feature distribution shift (dataset-level distribution change)

Approach A: Feature distribution; feed feature distance to ADWIN alongside error signal.

Public API:
    DINOFeatureExtractor.__init__(model_name, device, checkpoint_path)
    DINOFeatureExtractor.extract(image_np) -> np.ndarray  # (384,) for vits14
    DINOFeatureExtractor.distance(feat1, feat2) -> float  # cosine distance
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from config import (
    DINO_MODEL_NAME,
    DINO_FEATURE_REDUCTION,
    DINO_FEATURE_DISTANCE,
    DINO_CHECKPOINT_PATH,
)


class DINOFeatureExtractor:
    """Extract DINO-V3 features for drift detection (Approach A).
    
    DINO pre-trained on unlabeled data; features robust to corruptions.
    Use for: detecting dataset-level distribution shifts (e.g., new defect types).
    """
    
    def __init__(
        self,
        model_name: str = DINO_MODEL_NAME,
        device: torch.device = None,
        checkpoint_path: Optional[Path] = None,
    ):
        """Initialize DINO feature extractor.
        
        Args:
            model_name: e.g., "dinov2_vits14" (small ViT, 14 patch size)
            device: torch device ("cuda" or "cpu")
            checkpoint_path: optional path to fine-tuned DINO weights
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # Load pre-trained DINO via timm or torchvision
        try:
            import timm
            self.model = timm.create_model(model_name, pretrained=True)
        except Exception:
            print(f"Warning: Could not load {model_name} via timm; falling back to basic ViT")
            # Fallback: use basic ViT; user must provide checkpoint
            from torchvision.models import vit_b_16
            self.model = vit_b_16(weights=None)
        
        # Load fine-tuned weights if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                state = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state)
                print(f"Loaded fine-tuned DINO from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load fine-tuned DINO: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Feature dimension (depends on model; dinov2_vits14 = 384)
        self.feature_dim = self._infer_feature_dim()
        
        # Reduction method
        self.reduction = DINO_FEATURE_REDUCTION
        self.distance_metric = DINO_FEATURE_DISTANCE
    
    def _infer_feature_dim(self) -> int:
        """Infer feature dimension from model."""
        try:
            with torch.no_grad():
                # Dummy forward to get feature shape
                dummy = torch.randn(1, 3, 224, 224).to(self.device)
                feat = self._forward(dummy)
                return feat.shape[-1]
        except Exception:
            # Fallback
            return 384
    
    @torch.no_grad()
    def _forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass; extract features before classification head.
        
        Args:
            img_tensor: (B, 3, H, W) on device
        
        Returns:
            Feature tensor (B, D) where D is feature dimension.
        """
        # Remove classification head to get intermediate features
        if hasattr(self.model, "forward_features"):
            # timm models
            feat = self.model.forward_features(img_tensor)
        elif hasattr(self.model, "fc"):
            # Standard PyTorch models: extract from before FC layer
            # This is a simplified version; depends on model architecture
            feat = self.model.forward(img_tensor)
        else:
            feat = self.model(img_tensor)
        
        # Flatten if needed
        if len(feat.shape) > 2:
            feat = feat.reshape(feat.shape[0], -1)
        
        return feat
    
    @torch.no_grad()
    def extract(self, image_np: np.ndarray) -> np.ndarray:
        """Extract DINO feature from RGB image.
        
        Args:
            image_np: uint8 H×W×3 numpy array
        
        Returns:
            Feature vector (D,) as np.ndarray.
        """
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Normalize to ImageNet statistics
        img_pil = Image.fromarray(image_np)
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        img_tensor = tf(img_pil).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        
        # Forward
        feat = self._forward(img_tensor).squeeze(0)  # (D,)
        
        # Reduction (pooling)
        if self.reduction == "mean":
            feat = feat.mean(dim=0, keepdim=True).squeeze()
        elif self.reduction == "max":
            feat = feat.max(dim=0, keepdim=True)[0].squeeze()
        
        return feat.cpu().numpy()
    
    def distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute distance between two feature vectors.
        
        Args:
            feat1, feat2: (D,) feature vectors
        
        Returns:
            Distance as float.
        """
        if self.distance_metric == "cosine":
            # Cosine distance
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return 1.0 - np.dot(feat1, feat2) / (norm1 * norm2)
        elif self.distance_metric == "l2":
            return float(np.linalg.norm(feat1 - feat2))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def fine_tune(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        learning_rate: float = 1e-5,
    ) -> Tuple[float, float]:
        """Fine-tune DINO on MVTec data (if needed for Approach A variant).
        
        Note: Approach A doesn't require fine-tuning; pre-trained features usually work.
        This is optional for improved performance.
        
        Args:
            train_loader: DataLoader of (image, label) tuples
            val_loader: DataLoader for validation
            num_epochs: number of epochs
            learning_rate: learning rate
        
        Returns:
            (best_train_acc, best_val_acc)
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_train_acc = 0.0
        
        # Add classification head if not present
        feature_dim = self.feature_dim
        if not hasattr(self.model, "fc_for_mvtec"):
            self.model.fc_for_mvtec = nn.Linear(feature_dim, 2).to(self.device)
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_acc = 0.0
            for img_batch, label_batch in train_loader:
                img_batch = img_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                
                # Forward
                feat = self._forward(img_batch)
                logits = self.model.fc_for_mvtec(feat)
                loss = criterion(logits, label_batch)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Accuracy
                preds = logits.argmax(dim=1)
                train_acc += (preds == label_batch).float().mean().item()
            
            train_acc /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_acc = 0.0
            with torch.no_grad():
                for img_batch, label_batch in val_loader:
                    img_batch = img_batch.to(self.device)
                    label_batch = label_batch.to(self.device)
                    
                    feat = self._forward(img_batch)
                    logits = self.model.fc_for_mvtec(feat)
                    preds = logits.argmax(dim=1)
                    val_acc += (preds == label_batch).float().mean().item()
            
            val_acc /= len(val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc
                # Save checkpoint
                if self.checkpoint_path:
                    Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), self.checkpoint_path)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")
        
        print(f"Best validation accuracy: {best_val_acc:.3f}")
        return best_train_acc, best_val_acc
