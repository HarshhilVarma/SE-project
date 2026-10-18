"""Tests for aidetector.classical.frequency module."""

import numpy as np
from aidetector.config import FrequencyConfig
from aidetector.classical.frequency import compute_frequency_features


def test_frequency_features_random_image():
    img = np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8)
    config = FrequencyConfig(num_radial_bins=16)

    feats, names, visual_map = compute_frequency_features(img, config=config)

    assert isinstance(feats, np.ndarray)
    assert len(feats) == len(names)
    assert np.all(np.isfinite(feats))
    assert visual_map.shape == (64, 64)
    assert visual_map.dtype == np.uint8

    # Radial bin count check
    radial_names = [n for n in names if n.startswith("freq_radial_bin_")]
    assert len(radial_names) == 16


def test_frequency_zero_energy_image():
    # All zero image should not cause ZeroDivisionError or NaNs
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    feats, names, visual_map = compute_frequency_features(img)

    assert len(feats) == len(names)
    assert np.all(np.isfinite(feats))
    assert not np.any(np.isnan(feats))


def test_frequency_directional_energy():
    # Horizontal line pattern has vertical frequency energy and vice versa
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[::4, :, :] = 255  # horizontal stripes

    feats, names, _ = compute_frequency_features(img)
    vert_norm_idx = names.index("freq_dir_vert_norm")
    horiz_norm_idx = names.index("freq_dir_horiz_norm")

    # Horizontal spatial stripes produce vertical frequency concentration
    assert feats[vert_norm_idx] > feats[horiz_norm_idx]

