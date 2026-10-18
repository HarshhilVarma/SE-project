"""Unified classical feature extractor and schema enforcement."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from aidetector.config import ClassicalConfig
from aidetector.classical.ela import compute_ela
from aidetector.classical.color_features import compute_color_restoration_features
from aidetector.classical.residuals import compute_residual_features
from aidetector.classical.frequency import compute_frequency_features


class FeatureSchemaMismatchError(Exception):
    """Raised when extracted feature names or ordering does not match expected schema."""
    pass


def extract_classical_features(
    image: np.ndarray,
    config: Optional[ClassicalConfig] = None,
    is_jpeg: bool = True,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """Extracts all classical forensic features into a unified 1D float32 array.

    Combines:
        1. ELA (Error Level Analysis)
        2. Color restoration / distribution descriptor
        3. High-pass residual, Laplacian, and gradient textures
        4. 2D FFT spectral and directional energies

    Args:
        image: RGB numpy array [H, W, 3], uint8.
        config: ClassicalConfig configuration object.
        is_jpeg: Boolean indicating if source image was native JPEG.

    Returns:
        Tuple of:
            - features: 1D finite float32 array
            - names: list of strings of identical length with deterministic order
            - visualizations: dict mapping feature map names to 2D uint8 image arrays

    Raises:
        ValueError: Invalid image dimensions or channels.
        FloatingPointError: NaN or Inf encountered in extracted features.
    """
    if config is None:
        config = ClassicalConfig()

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image of shape (H, W, 3), got {image.shape}")

    # 1. ELA features
    ela_feats, ela_names, ela_map = compute_ela(image, config=config.ela, is_jpeg=is_jpeg)

    # 2. Color restoration features
    color_feats, color_names, color_map = compute_color_restoration_features(
        image, config=config.color_restoration
    )

    # 3. Residual features
    res_feats, res_names, res_map = compute_residual_features(image, config=config.residuals)

    # 4. Frequency features
    freq_feats, freq_names, freq_map = compute_frequency_features(image, config=config.frequency)

    # Concatenate features and names in strict deterministic order
    all_features = np.concatenate([ela_feats, color_feats, res_feats, freq_feats]).astype(np.float32)
    all_names = ela_names + color_names + res_names + freq_names

    # Check for NaN or Inf
    if not np.all(np.isfinite(all_features)):
        nan_indices = np.where(~np.isfinite(all_features))[0]
        bad_names = [all_names[i] for i in nan_indices]
        raise FloatingPointError(
            f"Non-finite values (NaN/Inf) detected in classical features: {bad_names}"
        )

    visualizations = {
        "ela_map": ela_map,
        "color_difference_map": color_map,
        "residual_map": res_map,
        "frequency_spectrum": freq_map,
    }

    return all_features, all_names, visualizations


def validate_feature_schema(actual_names: List[str], expected_names: List[str]) -> None:
    """Validates that actual feature names and ordering strictly match expected schema.

    Args:
        actual_names: Feature names extracted from an image.
        expected_names: Feature schema recorded during model training.

    Raises:
        FeatureSchemaMismatchError: If counts, names, or ordering differ.
    """
    if len(actual_names) != len(expected_names):
        raise FeatureSchemaMismatchError(
            f"Feature count mismatch: expected {len(expected_names)}, got {len(actual_names)}"
        )

    mismatches = []
    for idx, (act, exp) in enumerate(zip(actual_names, expected_names)):
        if act != exp:
            mismatches.append(f"Index {idx}: expected '{exp}', got '{act}'")

    if mismatches:
        raise FeatureSchemaMismatchError(
            f"Feature schema mismatch detected ({len(mismatches)} errors): " + "; ".join(mismatches[:5])
        )

