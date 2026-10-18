"""Classical forensics package."""

from aidetector.classical.features import (
    extract_classical_features,
    validate_feature_schema,
    FeatureSchemaMismatchError,
)
from aidetector.classical.ela import compute_ela
from aidetector.classical.color_features import compute_color_restoration_features
from aidetector.classical.residuals import compute_residual_features
from aidetector.classical.frequency import compute_frequency_features

__all__ = [
    "extract_classical_features",
    "validate_feature_schema",
    "FeatureSchemaMismatchError",
    "compute_ela",
    "compute_color_restoration_features",
    "compute_residual_features",
    "compute_frequency_features",
]

