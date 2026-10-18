"""Error Level Analysis (ELA) feature extraction."""

import io
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from aidetector.config import ELAConfig


def _calculate_entropy(arr: np.ndarray) -> float:
    """Calculates Shannon entropy of an integer/quantized array."""
    hist, _ = np.histogram(arr, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    return float(-np.sum(hist * np.log2(hist)))


def compute_ela(
    image: np.ndarray,
    config: Optional[ELAConfig] = None,
    is_jpeg: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Computes Error Level Analysis (ELA) on an RGB image.

    For JPEG images, re-encodes at configured quality, decodes, and computes
    absolute error. For non-JPEG inputs, computes derived recompression error
    and explicitly marks it via `ela_is_native_jpeg` feature.

    Args:
        image: RGB numpy array [H, W, 3], uint8.
        config: ELA configuration.
        is_jpeg: Boolean indicating if input was originally a JPEG image.

    Returns:
        Tuple of:
            - features: 1D float32 array
            - feature_names: List of corresponding feature names
            - visual_map: 2D uint8 image (scaled grayscale difference map)
    """
    if config is None:
        config = ELAConfig()

    h, w, c = image.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels (RGB), got shape: {image.shape}")

    # Recompress copy to JPEG in memory
    pil_src = Image.fromarray(image, mode="RGB")
    buffer = io.BytesIO()
    pil_src.save(buffer, format="JPEG", quality=config.quality)
    buffer.seek(0)
    recompressed_pil = Image.open(buffer)
    recompressed = np.array(recompressed_pil, dtype=np.float32)

    # Compute absolute difference
    diff = np.abs(image.astype(np.float32) - recompressed)
    gray_diff = 0.299 * diff[:, :, 0] + 0.587 * diff[:, :, 1] + 0.114 * diff[:, :, 2]

    # Scaled visual map
    visual_map = np.clip(gray_diff * config.scale, 0, 255).astype(np.uint8)

    features: List[float] = []
    names: List[str] = []

    # Flag indicating whether this is native JPEG ELA or derived recompression
    names.append("ela_is_native_jpeg")
    features.append(1.0 if is_jpeg else 0.0)

    # Channel-wise means
    for ch, ch_name in enumerate(["r", "g", "b"]):
        names.append(f"ela_channel_mean_{ch_name}")
        features.append(float(np.mean(diff[:, :, ch])))

    names.append("ela_gray_mean")
    features.append(float(np.mean(gray_diff)))

    # Channel-wise standard deviations
    for ch, ch_name in enumerate(["r", "g", "b"]):
        names.append(f"ela_channel_std_{ch_name}")
        features.append(float(np.std(diff[:, :, ch])))

    names.append("ela_gray_std")
    features.append(float(np.std(gray_diff)))

    # Quantiles
    quantiles = [10, 25, 50, 75, 90, 95, 99]
    q_vals = np.percentile(gray_diff, quantiles)
    for q, val in zip(quantiles, q_vals):
        names.append(f"ela_gray_q{q}")
        features.append(float(val))

    # Entropy
    entropy = _calculate_entropy(visual_map)
    names.append("ela_entropy")
    features.append(entropy)

    # Grid tile statistics (e.g. 4x4 grid)
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
            names.append(f"ela_tile_{i}_{j}_mean")
            features.append(mean_val)
            names.append(f"ela_tile_{i}_{j}_std")
            features.append(std_val)

    # Spatial uniformity / variance across tiles
    names.append("ela_tile_spatial_std")
    features.append(float(np.std(tile_means)) if len(tile_means) > 0 else 0.0)

    feature_array = np.array(features, dtype=np.float32)
    return feature_array, names, visual_map

