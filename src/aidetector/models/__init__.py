"""Model definitions and bundles."""

from aidetector.models.classical_model import (
    ClassicalModelBundle,
    ClassicalModelNotFittedError,
    SCHEMA_VERSION as CLASSICAL_SCHEMA_VERSION,
)

__all__ = [
    "ClassicalModelBundle",
    "ClassicalModelNotFittedError",
    "CLASSICAL_SCHEMA_VERSION",
]

