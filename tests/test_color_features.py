"""Tests for aidetector.classical.color_features module."""

import numpy as np
from aidetector.config import ColorFeaturesConfig
from aidetector.classical.color_features import (
    compute_color_restoration_features,
    quantize_color,
)


def test_quantize_color():
    arr = np.array([0.0, 7.0, 8.0, 15.0, 255.0], dtype=np.float32)
    # 5 bits -> step = 8, bin centers at 4, 12, 20...
    q = quantize_color(arr, bits=5)
    assert q[0] == 4.0
    assert q[1] == 4.0
    assert q[2] == 12.0
    assert q[3] == 12.0


def test_color_restoration_deterministic_seed():
    img = np.random.RandomState(123).randint(0, 256, (64, 64, 3), dtype=np.uint8)
    config1 = ColorFeaturesConfig(quantization_bits=5, num_samples=6, noise_scale=4.0, seed=42)
    config2 = ColorFeaturesConfig(quantization_bits=5, num_samples=6, noise_scale=4.0, seed=42)

    feats1, names1, map1 = compute_color_restoration_features(img, config1)
    feats2, names2, map2 = compute_color_restoration_features(img, config2)

    assert names1 == names2
    np.testing.assert_array_equal(feats1, feats2)
    np.testing.assert_array_equal(map1, map2)


def test_color_restoration_features_finite():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    feats, names, map_img = compute_color_restoration_features(img)

    assert len(feats) == len(names)
    assert np.all(np.isfinite(feats))
    assert map_img.shape == (64, 64)

