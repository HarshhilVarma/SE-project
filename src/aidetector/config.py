"""Configuration dataclasses and loaders for aidetector."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class ELAConfig:
    quality: int = 90
    scale: float = 10.0
    grid_tiles: int = 4


@dataclass
class ColorFeaturesConfig:
    quantization_bits: int = 5
    num_samples: int = 8
    noise_scale: float = 4.0
    seed: int = 42
    grid_tiles: int = 4


@dataclass
class ResidualConfig:
    laplacian_ksize: int = 3
    edge_threshold: float = 20.0
    grid_tiles: int = 4


@dataclass
class FrequencyConfig:
    num_radial_bins: int = 16
    eps: float = 1e-7


@dataclass
class QualityConfig:
    min_width: int = 64
    min_height: int = 64
    max_image_pixels: int = 89478485


@dataclass
class ClassicalConfig:
    ela: ELAConfig = field(default_factory=ELAConfig)
    color_restoration: ColorFeaturesConfig = field(default_factory=ColorFeaturesConfig)
    residuals: ResidualConfig = field(default_factory=ResidualConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassicalConfig":
        ela_data = data.get("ela", {})
        color_data = data.get("color_restoration", {})
        res_data = data.get("residuals", {})
        freq_data = data.get("frequency", {})
        return cls(
            ela=ELAConfig(**ela_data),
            color_restoration=ColorFeaturesConfig(**color_data),
            residuals=ResidualConfig(**res_data),
            frequency=FrequencyConfig(**freq_data),
        )


def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}

