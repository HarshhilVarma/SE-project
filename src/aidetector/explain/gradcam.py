"""Grad-CAM (Gradient-weighted Class Activation Mapping) explainability module."""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from aidetector.models.cnn import ResNet18BinaryClassifier, get_inference_transform


class GradCAM:
    """Computes Grad-CAM heatmaps for ResNet18BinaryClassifier.

    Target layer: layer4[-1] (the final residual block before global pooling).

    Important Notice:
        Grad-CAM visualizes features that contributed positively to the model's
        activation score. It is an explanatory aid, NOT legal or forensic proof
        that an image is AI-generated or authentic.
    """

    def __init__(
        self,
        model: ResNet18BinaryClassifier,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.target_layer = target_layer or model.backbone.layer4[-1]

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """Generates a 2D float32 heatmap [H, W] normalized to [0, 1].

        Args:
            input_tensor: Normalized image tensor [1, 3, H, W] or [3, H, W].
            target_class: 1 for AI-generated logit.

        Returns:
            2D numpy array [H, W] with values in [0, 1].
        """
        self.model.eval()
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor).view(-1)
        score = logits[0] if target_class == 1 else -logits[0]

        # Backward pass
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured during Grad-CAM.")

        # Global average pooling of gradients: weights alpha_k
        alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        # Upsample to input tensor resolution
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        max_val, min_val = float(np.max(cam_np)), float(np.min(cam_np))
        if max_val > min_val:
            cam_norm = (cam_np - min_val) / (max_val - min_val)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm.astype(np.float32)

    def overlay_heatmap(
        self,
        original_image: Union[np.ndarray, Image.Image],
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Overlays heatmap on original RGB image.

        Args:
            original_image: RGB image [H, W, 3] uint8 or PIL Image.
            heatmap: 2D float32 array [H', W'] in [0, 1].
            alpha: Heatmap blend weight (0 = only original, 1 = only heatmap).
            colormap: OpenCV colormap enum.

        Returns:
            Tuple of (overlaid_image: np.ndarray [H, W, 3] uint8, colored_heatmap: np.ndarray)
        """
        if isinstance(original_image, Image.Image):
            rgb_arr = np.array(original_image.convert("RGB"), dtype=np.uint8)
        else:
            rgb_arr = original_image.copy()

        h, w = rgb_arr.shape[:2]

        # Resize heatmap to original image dimensions
        resized_cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        uint8_cam = np.clip(resized_cam * 255.0, 0, 255).astype(np.uint8)

        # Apply colormap
        colored_cam = cv2.applyColorMap(uint8_cam, colormap)
        colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = (alpha * colored_cam + (1.0 - alpha) * rgb_arr).clip(0, 255).astype(np.uint8)

        return overlay, colored_cam
