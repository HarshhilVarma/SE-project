"""Tests for aidetector.explain.gradcam module."""

import numpy as np
import torch
from aidetector.explain.gradcam import GradCAM
from aidetector.models.cnn import ResNet18BinaryClassifier


def test_gradcam_heatmap_generation():
    model = ResNet18BinaryClassifier(pretrained=False)
    gradcam = GradCAM(model=model, device=torch.device("cpu"))

    dummy_tensor = torch.randn(1, 3, 224, 224)
    heatmap = gradcam.generate_heatmap(dummy_tensor, target_class=1)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert np.all(heatmap >= 0.0)
    assert np.all(heatmap <= 1.0)
    assert np.all(np.isfinite(heatmap))


def test_gradcam_overlay():
    model = ResNet18BinaryClassifier(pretrained=False)
    gradcam = GradCAM(model=model, device=torch.device("cpu"))

    dummy_tensor = torch.randn(1, 3, 224, 224)
    heatmap = gradcam.generate_heatmap(dummy_tensor, target_class=1)

    orig_img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    overlay, colored_cam = gradcam.overlay_heatmap(orig_img, heatmap, alpha=0.5)

    assert overlay.shape == (300, 400, 3)
    assert overlay.dtype == np.uint8
    assert colored_cam.shape == (300, 400, 3)
    assert colored_cam.dtype == np.uint8

