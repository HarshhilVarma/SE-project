"""Training package for aidetector."""

from aidetector.training.dataset import (
    validate_manifest,
    extract_defactify_subset,
    ManifestValidationError,
    ManifestReport,
)
from aidetector.training.train_classical import (
    train_classical_baseline,
    extract_features_from_manifest,
)

__all__ = [
    "validate_manifest",
    "extract_defactify_subset",
    "ManifestValidationError",
    "ManifestReport",
    "train_classical_baseline",
    "extract_features_from_manifest",
]

