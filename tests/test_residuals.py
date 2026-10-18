"""Tests for aidetector.classical.residuals module."""

import numpy as np
from aidetector.config import ResidualConfig
from aidetector.classical.residuals import compute_residual_features


def test_residual_features_random_image():
    img = np.random.RandomState(42).randint(0, 256, (80, 80, 3), dtype=np.uint8)
    config = ResidualConfig(laplacian_ksize=3, edge_threshold=20.0, grid_tiles=4)

    feats, names, visual_map = compute_residual_features(img, config=config)

    assert isinstance(feats, np.ndarray)
    assert len(feats) == len(names)
    assert np.all(np.isfinite(feats))
    assert visual_map.shape == (80, 80)
    assert visual_map.dtype == np.uint8


def test_residual_features_constant_image():
    # Constant image has 0 residual everywhere
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    feats, names, visual_map = compute_residual_features(img)

    assert len(feats) == len(names)
    assert np.all(np.isfinite(feats))

    # Mean residual should be 0.0
    mean_idx = names.index("residual_mean")
    assert feats[mean_idx] == 0.0

    # Edge density should be 0.0
    edge_idx = names.index("residual_edge_density")
    assert feats[edge_idx] == 0.0

