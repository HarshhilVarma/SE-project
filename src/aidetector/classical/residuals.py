"""High-pass residual, Laplacian, and texture feature extraction."""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from scipy import stats
from aidetector.config import ResidualConfig


def _calculate_entropy(arr: np.ndarray) -> float:
    """Calculates Shannon entropy of an integer/quantized array."""
    hist, _ = np.histogram(arr, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    return float(-np.sum(hist * np.log2(hist)))


def compute_residual_features(
    image: np.ndarray,
    config: Optional[ResidualConfig] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Computes high-pass residual, Laplacian, and gradient texture features.

    Args:
        image: RGB numpy array [H, W, 3], uint8.
        config: ResidualConfig instance.

    Returns:
        Tuple of:
            - features: 1D float32 array
            - feature_names: List of corresponding feature names
            - visual_map: 2D uint8 image (scaled absolute residual map)
    """
    if config is None:
        config = ResidualConfig()

    h, w, c = image.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels (RGB), got shape: {image.shape}")

    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 1. Laplacian high-pass residual
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=config.laplacian_ksize)
    abs_laplacian = np.abs(laplacian)

    # Visual map: scaled normalized absolute Laplacian
    visual_map = np.clip(abs_laplacian * 4.0, 0, 255).astype(np.uint8)

    # 2. Sobel gradients for edge density
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    features: List[float] = []
    names: List[str] = []

    # Residual basic statistics
    names.append("residual_mean")
    features.append(float(np.mean(abs_laplacian)))

    names.append("residual_std")
    features.append(float(np.std(abs_laplacian)))

    flat_lap = abs_laplacian.ravel()
    skew_val = float(stats.skew(flat_lap))
    names.append("residual_skewness")
    features.append(skew_val if np.isfinite(skew_val) else 0.0)

    kurt_val = float(stats.kurtosis(flat_lap))
    names.append("residual_kurtosis")
    features.append(kurt_val if np.isfinite(kurt_val) else 0.0)

    # Quantiles
    quantiles = [10, 25, 50, 75, 90, 95, 99]
    q_vals = np.percentile(abs_laplacian, quantiles)
    for q, val in zip(quantiles, q_vals):
        names.append(f"residual_q{q}")
        features.append(float(val))

    # Gradient & edge density statistics
    edge_mask = grad_mag > config.edge_threshold
    edge_density = float(np.mean(edge_mask))
    names.append("residual_edge_density")
    features.append(edge_density)

    names.append("residual_grad_mean")
    features.append(float(np.mean(grad_mag)))

    names.append("residual_grad_std")
    features.append(float(np.std(grad_mag)))

    names.append("residual_grad_max")
    features.append(float(np.max(grad_mag)))

    # Entropy of residual map
    residual_entropy = _calculate_entropy(visual_map)
    names.append("residual_entropy")
    features.append(residual_entropy)

    # Local variance summary: variance in small non-overlapping 8x8 patches
    patch_size = 8
    patches_h = max(1, h // patch_size)
    patches_w = max(1, w // patch_size)
    local_vars: List[float] = []
    for pi in range(patches_h):
        for pj in range(patches_w):
            patch = gray[pi * patch_size : (pi + 1) * patch_size, pj * patch_size : (pj + 1) * patch_size]
            if patch.size > 1:
                local_vars.append(float(np.var(patch)))

    if len(local_vars) > 0:
        names.append("residual_local_var_mean")
        features.append(float(np.mean(local_vars)))
        names.append("residual_local_var_std")
        features.append(float(np.std(local_vars)))
    else:
        names.append("residual_local_var_mean")
        features.append(0.0)
        names.append("residual_local_var_std")
        features.append(0.0)

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

            patch = abs_laplacian[y_start:y_end, x_start:x_end]
            mean_val = float(np.mean(patch)) if patch.size > 0 else 0.0
            std_val = float(np.std(patch)) if patch.size > 0 else 0.0

            tile_means.append(mean_val)
            names.append(f"residual_tile_{i}_{j}_mean")
            features.append(mean_val)
            names.append(f"residual_tile_{i}_{j}_std")
            features.append(std_val)

    names.append("residual_tile_spatial_std")
    features.append(float(np.std(tile_means)) if len(tile_means) > 0 else 0.0)

    feature_array = np.array(features, dtype=np.float32)
    return feature_array, names, visual_map
