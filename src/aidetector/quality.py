"""Quality assessment and flags for image evaluation."""

from typing import List, Optional
from aidetector.io import ImageMetadata


KNOWN_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP", "TIFF"}


def assess_image_quality(
    metadata: ImageMetadata,
    min_width: int = 64,
    min_height: int = 64,
    max_aspect_ratio: float = 6.0,
    decode_warnings: Optional[List[str]] = None,
) -> List[str]:
    """Generates quality flags based on image metadata and decoding conditions.

    Args:
        metadata: ImageMetadata instance from load_image.
        min_width: Minimum acceptable width.
        min_height: Minimum acceptable height.
        max_aspect_ratio: Maximum aspect ratio before flagging extreme proportions.
        decode_warnings: Any warnings logged during decoding.

    Returns:
        List of string flags representing potential quality issues or conditions.
    """
    flags: List[str] = []

    # Resolution check
    if metadata.width < min_width or metadata.height < min_height:
        flags.append("low_resolution")

    # Format check
    if not metadata.format or metadata.format.upper() not in KNOWN_FORMATS:
        flags.append("unsupported_or_unknown_format")

    # ELA native check
    if not metadata.is_jpeg:
        flags.append("ela_not_native_jpeg")

    # Extreme aspect ratio check
    shorter = min(metadata.width, metadata.height)
    longer = max(metadata.width, metadata.height)
    if shorter > 0 and (longer / shorter) > max_aspect_ratio:
        flags.append("extreme_aspect_ratio")

    # Decode warnings
    if decode_warnings:
        flags.append("decode_warning")

    return flags

