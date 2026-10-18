"""ResNet-18 CNN architecture, checkpointing, and inference interface."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Automatically use local workspace torch cache if present
_local_cache = Path(__file__).resolve().parents[3] / ".cache" / "torch"
if _local_cache.is_dir() and "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = str(_local_cache)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CheckpointMetadataMismatchError(Exception):
    """Raised when loaded checkpoint metadata fails compatibility checks."""
    pass


class ResNet18BinaryClassifier(nn.Module):
    """ResNet-18 adapted for binary classification (Real vs AI-Generated)."""

    def __init__(
        self,
        pretrained: bool = False,
        weights: Optional[str] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        torch_weights = None
        if pretrained or weights == "DEFAULT":
            torch_weights = models.ResNet18_Weights.DEFAULT

        self.backbone = models.resnet18(weights=torch_weights)
        in_features = self.backbone.fc.in_features  # 512

        # Replace final classification head with binary logit output
        if dropout > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Logits of shape [B, 1]
        """
        return self.backbone(x)

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 1D logit tensor [B]."""
        return self.forward(x).view(-1)

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 1D probabilities [B] via sigmoid."""
        logits = self.predict_logits(x)
        return torch.sigmoid(logits)


def get_inference_transform(input_size: int = 224) -> transforms.Compose:
    """Standard inference transform for ResNet-18."""
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_training_transform(input_size: int = 224) -> transforms.Compose:
    """Standard label-preserving training augmentations."""
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@dataclass
class CNNModelBundle:
    """Bundle containing model, device, and training metadata."""

    model: ResNet18BinaryClassifier
    metadata: Dict[str, Any] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def to(self, device: Union[str, torch.device]) -> "CNNModelBundle":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def predict_score(self, image_tensor: torch.Tensor) -> float:
        """Predicts AI-generated score for a single normalized tensor [1, 3, H, W] or [3, H, W]."""
        self.model.eval()
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            prob = self.model.predict_probs(image_tensor)
            return float(prob.cpu().item())

    def predict_image(
        self,
        image: Union[np.ndarray, Image.Image],
        input_size: int = 224,
    ) -> float:
        """Predicts score directly from an RGB numpy array or PIL Image."""
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image, mode="RGB")
        else:
            pil_img = image

        transform = get_inference_transform(input_size=input_size)
        tensor = transform(pil_img)
        return self.predict_score(tensor)

    def save(self, path: Union[str, Path]) -> None:
        """Saves checkpoint and metadata to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "metadata": self.metadata,
        }
        torch.save(checkpoint, save_path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        expected_architecture: str = "resnet18",
    ) -> "CNNModelBundle":
        """Loads checkpoint and verifies metadata."""
        load_path = Path(path)
        if not load_path.is_file():
            raise FileNotFoundError(f"CNN checkpoint not found at: {load_path}")

        if device is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        checkpoint = torch.load(load_path, map_location=dev, weights_only=False)
        metadata = checkpoint.get("metadata", {})

        arch = metadata.get("architecture", "resnet18")
        if arch != expected_architecture:
            raise CheckpointMetadataMismatchError(
                f"Checkpoint architecture mismatch: expected {expected_architecture}, got {arch}"
            )

        model = ResNet18BinaryClassifier(pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(dev)
        model.eval()

        return cls(model=model, metadata=metadata, device=dev)

