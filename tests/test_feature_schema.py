"""Tests for unified classical feature extraction and schema enforcement."""

import numpy as np
import pytest
from aidetector.classical.features import (
    extract_classical_features,
    validate_feature_schema,
    FeatureSchemaMismatchError,
)


def test_feature_extraction_contract():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    feats, names, visualizations = extract_classical_features(img, is_jpeg=True)

    assert isinstance(feats, np.ndarray)
    assert feats.ndim == 1
    assert feats.dtype == np.float32
    assert len(feats) == len(names)
    assert np.all(np.isfinite(feats))

    # Required visualizations
    assert "ela_map" in visualizations
    assert "color_difference_map" in visualizations
    assert "residual_map" in visualizations
    assert "frequency_spectrum" in visualizations


def test_feature_names_stable_order():
    img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)

    _, names1, _ = extract_classical_features(img1)
    _, names2, _ = extract_classical_features(img2)

    assert names1 == names2


def test_validate_feature_schema_success():
    names = ["feat_a", "feat_b", "feat_c"]
    # Should not raise
    validate_feature_schema(names, names)


def test_validate_feature_schema_length_mismatch():
    names1 = ["feat_a", "feat_b"]
    names2 = ["feat_a", "feat_b", "feat_c"]
    with pytest.raises(FeatureSchemaMismatchError, match="Feature count mismatch"):
        validate_feature_schema(names1, names2)


def test_validate_feature_schema_order_mismatch():
    names1 = ["feat_a", "feat_b"]
    names2 = ["feat_b", "feat_a"]
    with pytest.raises(FeatureSchemaMismatchError, match="Feature schema mismatch detected"):
        validate_feature_schema(names1, names2)

