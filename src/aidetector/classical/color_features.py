"""Color distribution and restoration features inspired by Jia et al. (CVPR 2025)."""

from typing import List, Optional, Tuple
import numpy as np
from scipy import stats
from aidetector.config import ColorFeaturesConfig


def quantize_color(arr: np.ndarray, bits: int) -> np.ndarray:
    """Quantizes an image array to a specified bit depth.

    Args:
        arr: Array with values in [0, 255].
        bits: Bit depth per channel (e.g., 5).

    Returns:
        Quantized array with reconstruction levels centered in each bin.
    """
    levels = 2 ** bits
    step = 256.0 / levels
    quantized = np.floor(arr / step) * step + (step / 2.0)
    return np.clip(quantized, 0, 255)


def compute_color_restoration_features(
    image: np.ndarray,
    config: Optional[ColorFeaturesConfig] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Computes color quantization/restoration difference features.

    Inspired by Jia et al., "Secret Lies in Color: Enhancing AI-Generated
    Images Detection with Color Distribution Analysis", CVPR 2025.

    Args:
        image: RGB numpy array [H, W, 3], uint8.
        config: ColorFeaturesConfig instance.

    Returns:
        Tuple of:
            - features: 1D float32 array
            - feature_names: List of corresponding feature names
            - visual_map: 2D uint8 image (scaled difference map)
    """
    if config is None:
        config = ColorFeaturesConfig()

    h, w, c = image.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels (RGB), got shape: {image.shape}")

    img_float = image.astype(np.float32)

    # Deterministic RNG for reproducible controlled Gaussian noise
    rng = np.random.RandomState(config.seed)

    restored_accum = np.zeros_like(img_float)
    for _ in range(config.num_samples):
        noise = rng.normal(loc=0.0, scale=config.noise_scale, size=img_float.shape).astype(np.float32)
        noisy = np.clip(img_float + noise, 0, 255)
        quantized = quantize_color(noisy, config.quantization_bits)
        restored_accum += quantized

    restored = restored_accum / float(config.num_samples)

    # Absolute difference between original image and restored image
    diff = np.abs(img_float - restored)
    gray_diff = 0.299 * diff[:, :, 0] + 0.587 * diff[:, :, 1] + 0.114 * diff[:, :, 2]

    # Visual map: scaled difference map
    visual_map = np.clip(gray_diff * 10.0, 0, 255).astype(np.uint8)

    features: List[float] = []
    names: List[str] = []

    # Channel-wise means
    for ch, ch_name in enumerate(["r", "g", "b"]):
        names.append(f"color_res_mean_{ch_name}")
        features.append(float(np.mean(diff[:, :, ch])))

    names.append("color_res_mean_gray")
    features.append(float(np.mean(gray_diff)))

    # Channel-wise stds
    for ch, ch_name in enumerate(["r", "g", "b"]):
        names.append(f"color_res_std_{ch_name}")
        features.append(float(np.std(diff[:, :, ch])))

    names.append("color_res_std_gray")
    features.append(float(np.std(gray_diff)))

    # Skewness and kurtosis
    flat_gray = gray_diff.ravel()
    skew_val = float(stats.skew(flat_gray))
    names.append("color_res_skewness_gray")
    features.append(skew_val if np.isfinite(skew_val) else 0.0)

    kurt_val = float(stats.kurtosis(flat_gray))
    names.append("color_res_kurtosis_gray")
    features.append(kurt_val if np.isfinite(kurt_val) else 0.0)

    # Quantiles
    quantiles = [10, 25, 50, 75, 90, 95, 99]
    q_vals = np.percentile(gray_diff, quantiles)
    for q, val in zip(quantiles, q_vals):
        names.append(f"color_res_q{q}")
        features.append(float(val))

    # Histogram summaries (10 bins over [0, 30])
    hist, _ = np.histogram(gray_diff, bins=10, range=(0.0, 30.0), density=True)
    for b_idx, val in enumerate(hist):
        names.append(f"color_res_hist_bin_{b_idx}")
        features.append(float(val))

    # Grid tile statistics
    tiles = config.grid_tiles
    tile_h = max(1, h // tiles)
    tile_w = max(1, w // tiles)
    tile_means: List[float] = []

    for i in range(tiles):
        for j in range(tiles):
            y_start = i * tile_h
            y_end = (i + 1) * tile_h if i < tiles - 1 else h
            x_start = j * tile_w
            x_end = (j + 1) * tile_w if j < tiles - 1 else w

            patch = gray_diff[y_start:y_end, x_start:x_end]
            mean_val = float(np.mean(patch)) if patch.size > 0 else 0.0
            std_val = float(np.std(patch)) if patch.size > 0 else 0.0

            tile_means.append(mean_val)
            names.append(f"color_res_tile_{i}_{j}_mean")
            features.append(mean_val)
            names.append(f"color_res_tile_{i}_{j}_std")
            features.append(std_val)

    names.append("color_res_tile_spatial_std")
    features.append(float(np.std(tile_means)) if len(tile_means) > 0 else 0.0)

    feature_array = np.array(features, dtype=np.float32)
    return feature_array, names, visual_map
