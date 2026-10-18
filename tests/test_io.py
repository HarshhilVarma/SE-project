"""Tests for aidetector.io module."""

import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from aidetector.io import (
    load_image,
    ImageNotFoundError,
    ImageDecodeError,
    ImageDecompressionBombError,
)


def test_load_valid_rgb_image(tmp_path: Path):
    img = Image.new("RGB", (100, 80), color=(128, 64, 32))
    img_path = tmp_path / "test.jpg"
    img.save(img_path, format="JPEG")

    arr, meta = load_image(img_path)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (80, 100, 3)
    assert arr.dtype == np.uint8
    assert meta.format == "JPEG"
    assert meta.width == 100
    assert meta.height == 80
    assert meta.is_jpeg is True
    assert meta.has_alpha is False


def test_load_rgba_converts_to_rgb(tmp_path: Path):
    img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
    img_path = tmp_path / "test.png"
    img.save(img_path, format="PNG")

    arr, meta = load_image(img_path)
    assert arr.shape == (50, 50, 3)
    assert meta.format == "PNG"
    assert meta.is_jpeg is False
    assert meta.has_alpha is True


def test_load_from_bytes():
    img = Image.new("RGB", (32, 32), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    arr, meta = load_image(raw_bytes)
    assert arr.shape == (32, 32, 3)
    assert meta.is_jpeg is True


def test_load_nonexistent_file():
    with pytest.raises(ImageNotFoundError):
        load_image("nonexistent_image_file_12345.jpg")


def test_load_corrupt_file(tmp_path: Path):
    bad_file = tmp_path / "corrupt.jpg"
    bad_file.write_bytes(b"NOT_A_VALID_IMAGE_DATA_BYTES")

    with pytest.raises(ImageDecodeError):
        load_image(bad_file)


def test_decompression_bomb_protection(tmp_path: Path):
    # Image with 100x100 = 10000 pixels
    img = Image.new("RGB", (100, 100))
    p = tmp_path / "bomb.png"
    img.save(p)

    # Set max_pixels smaller than 10000
    with pytest.raises(ImageDecompressionBombError):
        load_image(p, max_pixels=5000)

