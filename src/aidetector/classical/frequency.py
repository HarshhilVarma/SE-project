"""Frequency domain (2D FFT) feature extraction."""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from aidetector.config import FrequencyConfig


def compute_frequency_features(
    image: np.ndarray,
    config: Optional[FrequencyConfig] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Computes classical 2D FFT spectral features.

    Extracts radial annular bin energies, high/low frequency ratio,
    spectral entropy, and directional quadrant energies.

    Args:
        image: RGB numpy array [H, W, 3], uint8.
        config: FrequencyConfig instance.

    Returns:
        Tuple of:
            - features: 1D float32 array
            - feature_names: List of corresponding feature names
            - visual_map: 2D uint8 image (normalized log-magnitude spectrum)
    """
    if config is None:
        config = FrequencyConfig()

    h, w, c = image.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels (RGB), got shape: {image.shape}")

    # Convert to grayscale float32
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 2D FFT and zero-frequency shift to center
    fft2 = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft2)
    mag = np.abs(fft_shifted)
    power = mag ** 2
    eps = config.eps

    log_mag = np.log(mag + eps)

    # Normalized log-magnitude visual map
    min_val, max_val = float(np.min(log_mag)), float(np.max(log_mag))
    if max_val > min_val:
        visual_map = np.clip((log_mag - min_val) / (max_val - min_val) * 255.0, 0, 255).astype(np.uint8)
    else:
        visual_map = np.zeros((h, w), dtype=np.uint8)

    features: List[float] = []
    names: List[str] = []

    # Overall spectral energy statistics
    total_power = float(np.sum(power))
    names.append("freq_total_power")
    features.append(total_power)

    names.append("freq_log_mag_mean")
    features.append(float(np.mean(log_mag)))

    names.append("freq_log_mag_std")
    features.append(float(np.std(log_mag)))

    names.append("freq_log_mag_max")
    features.append(float(np.max(log_mag)))

    # Coordinate grid centered at (cy, cx)
    cy, cx = h / 2.0, w / 2.0
    y_coords, x_coords = np.ogrid[:h, :w]
    norm_y = (y_coords - cy) / max(cy, 1.0)
    norm_x = (x_coords - cx) / max(cx, 1.0)
    radius = np.sqrt(norm_y**2 + norm_x**2)

    # Radial frequency bins (0.0 to 1.0)
    num_bins = config.num_radial_bins
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    radial_energies: List[float] = []

    norm_factor = total_power if total_power > eps else 1.0

    for b in range(num_bins):
        mask = (radius >= bin_edges[b]) & (radius < bin_edges[b + 1])
        bin_power = float(np.sum(power[mask]))
        radial_energies.append(bin_power)
        norm_bin_power = bin_power / norm_factor
        names.append(f"freq_radial_bin_{b}")
        features.append(norm_bin_power)

    # High / Low frequency energy ratio
    half_bins = num_bins // 2
    low_freq_energy = sum(radial_energies[:half_bins])
    high_freq_energy = sum(radial_energies[half_bins:])
    ratio = (high_freq_energy + eps) / (low_freq_energy + eps)
    names.append("freq_high_low_ratio")
    features.append(float(ratio))

    # Spectral entropy
    prob_dist = np.array(radial_energies, dtype=np.float64) / (sum(radial_energies) + eps)
    prob_dist = prob_dist[prob_dist > 0]
    if len(prob_dist) > 0:
        spec_entropy = float(-np.sum(prob_dist * np.log2(prob_dist + eps)))
    else:
        spec_entropy = 0.0
    names.append("freq_spectral_entropy")
    features.append(spec_entropy)

    # Directional energy analysis
    # Angle theta in [-pi, pi]
    angles = np.arctan2(y_coords - cy, x_coords - cx)

    # Horizontal: |theta| < pi/8 or |theta| > 7pi/8
    mask_horiz = (np.abs(angles) < np.pi / 8.0) | (np.abs(angles) > 7.0 * np.pi / 8.0)
    # Vertical: 3pi/8 < |theta| < 5pi/8
    mask_vert = (np.abs(angles) > 3.0 * np.pi / 8.0) & (np.abs(angles) < 5.0 * np.pi / 8.0)
    # Diagonals: remaining sectors
    mask_diag = (~mask_horiz) & (~mask_vert)

    horiz_power = float(np.sum(power[mask_horiz]))
    vert_power = float(np.sum(power[mask_vert]))
    diag_power = float(np.sum(power[mask_diag]))

    names.append("freq_dir_horiz_norm")
    features.append(horiz_power / norm_factor)

    names.append("freq_dir_vert_norm")
    features.append(vert_power / norm_factor)

    names.append("freq_dir_diag_norm")
    features.append(diag_power / norm_factor)

    names.append("freq_horiz_vert_ratio")
    features.append(float((horiz_power + eps) / (vert_power + eps)))

    feature_array = np.array(features, dtype=np.float32)
    return feature_array, names, visual_map

