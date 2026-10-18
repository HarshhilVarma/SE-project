"""Tests for aidetector.classical.ela module."""

import numpy as np
import pytest
from aidetector.config import ELAConfig
from aidetector.classical.ela import compute_ela


def test_compute_ela_jpeg():
    # 64x64 synthetic RGB image
    arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    config = ELAConfig(quality=90, scale=10.0, grid_tiles=4)

    feats, names, visual_map = compute_ela(arr, config=config, is_jpeg=True)

    assert isinstance(feats, np.ndarray)
    assert feats.dtype == np.float32
    assert len(feats) == len(names)
    assert np.all(np.isfinite(feats))

    # Check is_jpeg feature flag
    assert names[0] == "ela_is_native_jpeg"
    assert feats[0] == 1.0

    # Visual map shape and type
    assert visual_map.shape == (64, 64)
    assert visual_map.dtype == np.uint8


def test_compute_ela_non_jpeg():
    arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    feats, names, _ = compute_ela(arr, is_jpeg=False)

    assert names[0] == "ela_is_native_jpeg"
    assert feats[0] == 0.0


def test_compute_ela_invalid_shape():
    gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_ela(gray)

