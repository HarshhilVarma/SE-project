"""Tests for aidetector.models.cnn module."""

from pathlib import Path
import numpy as np
import pytest
import torch
from aidetector.models.cnn import (
    ResNet18BinaryClassifier,
    CNNModelBundle,
    CheckpointMetadataMismatchError,
    get_inference_transform,
    get_training_transform,
)
from aidetector.training.dataset import AIDetectionTorchDataset


def test_resnet18_forward_shape():
    model = ResNet18BinaryClassifier(pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)

    logits = model(dummy_input)
    assert logits.shape == (4, 1)

    probs = model.predict_probs(dummy_input)
    assert probs.shape == (4,)
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)


def test_cnn_transforms():
    train_tf = get_training_transform(input_size=224)
    inf_tf = get_inference_transform(input_size=224)

    from PIL import Image
    img = Image.new("RGB", (256, 256), color=(100, 150, 200))

    t_train = train_tf(img)
    t_inf = inf_tf(img)

    assert t_train.shape == (3, 224, 224)
    assert t_inf.shape == (3, 224, 224)


def test_cnn_bundle_save_load(tmp_path: Path):
    model = ResNet18BinaryClassifier(pretrained=False)
    metadata = {
        "architecture": "resnet18",
        "input_size": 224,
        "seed": 42,
    }
    bundle = CNNModelBundle(model=model, metadata=metadata, device=torch.device("cpu"))

    save_path = tmp_path / "cnn_checkpoint.pt"
    bundle.save(save_path)

    loaded_bundle = CNNModelBundle.load(save_path, device="cpu")
    assert loaded_bundle.metadata["architecture"] == "resnet18"
    assert loaded_bundle.metadata["seed"] == 42

    dummy_tensor = torch.randn(1, 3, 224, 224)
    score = loaded_bundle.predict_score(dummy_tensor)
    assert 0.0 <= score <= 1.0


def test_cnn_checkpoint_metadata_mismatch(tmp_path: Path):
    model = ResNet18BinaryClassifier(pretrained=False)
    metadata = {"architecture": "vit_base"}  # wrong architecture
    bundle = CNNModelBundle(model=model, metadata=metadata, device=torch.device("cpu"))

    save_path = tmp_path / "mismatch.pt"
    bundle.save(save_path)

    with pytest.raises(CheckpointMetadataMismatchError):
        CNNModelBundle.load(save_path, expected_architecture="resnet18")

