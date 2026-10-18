"""Model definitions and bundles."""

from aidetector.models.classical_model import (
    ClassicalModelBundle,
    ClassicalModelNotFittedError,
    SCHEMA_VERSION as CLASSICAL_SCHEMA_VERSION,
)
from aidetector.models.cnn import (
    ResNet18BinaryClassifier,
    CNNModelBundle,
    CheckpointMetadataMismatchError,
    get_inference_transform,
    get_training_transform,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

__all__ = [
    "ClassicalModelBundle",
    "ClassicalModelNotFittedError",
    "CLASSICAL_SCHEMA_VERSION",
    "ResNet18BinaryClassifier",
    "CNNModelBundle",
    "CheckpointMetadataMismatchError",
    "get_inference_transform",
    "get_training_transform",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
