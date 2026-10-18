"""Image loading, decoding, and validation utilities."""

from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps


class ImageLoadingError(Exception):
    """Base exception for image loading failures."""
    pass


class ImageNotFoundError(ImageLoadingError):
    """Raised when the specified image path does not exist."""
    pass


class ImageDecodeError(ImageLoadingError):
    """Raised when image decoding fails or file is corrupted."""
    pass


class ImageDecompressionBombError(ImageLoadingError):
    """Raised when image exceeds safe decompression limits."""
    pass


@dataclass(frozen=True)
class ImageMetadata:
    format: Optional[str]
    width: int
    height: int
    color_mode: str
    is_jpeg: bool
    has_alpha: bool
    has_exif: bool
    exif_orientation: Optional[int]
    file_size_bytes: int
    quality_flags: List[str] = field(default_factory=list)


def load_image(
    source: Union[str, Path, bytes, io.BytesIO, Image.Image],
    max_pixels: int = 89478485,
) -> Tuple[np.ndarray, ImageMetadata]:
    """Loads, validates, and standardizes an image into an RGB numpy array.

    Args:
        source: Path to image file, raw bytes, BytesIO stream, or PIL Image.
        max_pixels: Maximum allowed pixels to prevent decompression bombs.

    Returns:
        Tuple of (rgb_image_array: np.ndarray [H, W, 3] uint8, metadata: ImageMetadata)

    Raises:
        ImageNotFoundError: File does not exist.
        ImageDecodeError: Decoding failed or file is corrupted.
        ImageDecompressionBombError: Decompression bomb limit exceeded.
    """
    original_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels

    file_size_bytes = 0
    raw_stream: Optional[io.BytesIO] = None

    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise ImageNotFoundError(f"Image file does not exist: {path}")
            file_size_bytes = path.stat().st_size
            try:
                pil_img = Image.open(path)
            except Image.DecompressionBombError as e:
                raise ImageDecompressionBombError(f"Decompression bomb detected: {e}") from e
            except Exception as e:
                raise ImageDecodeError(f"Failed to open image file {path}: {e}") from e

        elif isinstance(source, (bytes, bytearray)):
            file_size_bytes = len(source)
            raw_stream = io.BytesIO(source)
            try:
                pil_img = Image.open(raw_stream)
            except Image.DecompressionBombError as e:
                raise ImageDecompressionBombError(f"Decompression bomb detected: {e}") from e
            except Exception as e:
                raise ImageDecodeError(f"Failed to decode image from bytes: {e}") from e

        elif isinstance(source, io.BytesIO):
            source.seek(0, io.SEEK_END)
            file_size_bytes = source.tell()
            source.seek(0)
            try:
                pil_img = Image.open(source)
            except Image.DecompressionBombError as e:
                raise ImageDecompressionBombError(f"Decompression bomb detected: {e}") from e
            except Exception as e:
                raise ImageDecodeError(f"Failed to decode image from BytesIO: {e}") from e

        elif isinstance(source, Image.Image):
            pil_img = source
            file_size_bytes = getattr(source, "size", (0, 0))[0] * getattr(source, "size", (0, 0))[1] * 3
        else:
            raise TypeError(f"Unsupported image source type: {type(source)}")

        # Verify integrity
        raw_format = pil_img.format
        raw_mode = pil_img.mode
        raw_width, raw_height = pil_img.size

        if raw_width <= 0 or raw_height <= 0:
            raise ImageDecodeError(f"Invalid image dimensions: {raw_width}x{raw_height}")

        if raw_width * raw_height > max_pixels:
            raise ImageDecompressionBombError(
                f"Image size ({raw_width * raw_height} pixels) exceeds limit of {max_pixels} pixels."
            )

        has_alpha = raw_mode in ("RGBA", "LA", "PA")

        # Check EXIF orientation
        has_exif = False
        exif_orientation = None
        try:
            exif = pil_img.getexif()
            if exif:
                has_exif = True
                exif_orientation = exif.get(0x0112)
        except Exception:
            pass

        # Handle EXIF orientation rotation
        try:
            transposed = ImageOps.exif_transpose(pil_img)
            if transposed is not None:
                pil_img = transposed
        except Exception:
            pass

        # Convert to RGB safely
        if pil_img.mode != "RGB":
            # If RGBA, blend onto neutral background or convert directly
            if pil_img.mode == "RGBA":
                background = Image.new("RGB", pil_img.size, (255, 255, 255))
                background.paste(pil_img, mask=pil_img.split()[3])
                rgb_pil = background
            else:
                rgb_pil = pil_img.convert("RGB")
        else:
            rgb_pil = pil_img

        rgb_array = np.array(rgb_pil, dtype=np.uint8)

        is_jpeg = raw_format is not None and raw_format.upper() in ("JPEG", "JPG")

        metadata = ImageMetadata(
            format=raw_format.upper() if raw_format else None,
            width=raw_width,
            height=raw_height,
            color_mode=raw_mode,
            is_jpeg=is_jpeg,
            has_alpha=has_alpha,
            has_exif=has_exif,
            exif_orientation=exif_orientation,
            file_size_bytes=file_size_bytes,
        )

        return rgb_array, metadata

    finally:
        Image.MAX_IMAGE_PIXELS = original_max_pixels
        if raw_stream is not None:
            raw_stream.close()


def save_image_array(image_array: np.ndarray, path: Union[str, Path], format: Optional[str] = None) -> None:
    """Saves a numpy image array to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if image_array.ndim == 2:
        pil_img = Image.fromarray(image_array, mode="L")
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        pil_img = Image.fromarray(image_array, mode="RGB")
    elif image_array.ndim == 3 and image_array.shape[2] == 4:
        pil_img = Image.fromarray(image_array, mode="RGBA")
    else:
        raise ValueError(f"Unsupported array shape for saving image: {image_array.shape}")

    pil_img.save(out_path, format=format)
