"""Training package for aidetector."""

from aidetector.training.dataset import (
    validate_manifest,
    extract_defactify_subset,
    ManifestValidationError,
    ManifestReport,
    AIDetectionTorchDataset,
)
from aidetector.training.train_classical import (
    train_classical_baseline,
    extract_features_from_manifest,
)
from aidetector.training.train_cnn import (
    train_cnn_baseline,
    evaluate_cnn,
)

__all__ = [
    "validate_manifest",
    "extract_defactify_subset",
    "ManifestValidationError",
    "ManifestReport",
    "AIDetectionTorchDataset",
    "train_classical_baseline",
    "extract_features_from_manifest",
    "train_cnn_baseline",
    "evaluate_cnn",
]


