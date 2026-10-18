"""Tests for aidetector.quality module."""

from aidetector.io import ImageMetadata
from aidetector.quality import assess_image_quality


def test_quality_clean_image():
    meta = ImageMetadata(
        format="JPEG",
        width=512,
        height=512,
        color_mode="RGB",
        is_jpeg=True,
        has_alpha=False,
        has_exif=True,
        exif_orientation=1,
        file_size_bytes=10240,
    )
    flags = assess_image_quality(meta)
    assert flags == []


def test_quality_low_resolution():
    meta = ImageMetadata(
        format="JPEG",
        width=32,
        height=32,
        color_mode="RGB",
        is_jpeg=True,
        has_alpha=False,
        has_exif=False,
        exif_orientation=None,
        file_size_bytes=500,
    )
    flags = assess_image_quality(meta, min_width=64, min_height=64)
    assert "low_resolution" in flags


def test_quality_non_jpeg_ela_flag():
    meta = ImageMetadata(
        format="PNG",
        width=256,
        height=256,
        color_mode="RGB",
        is_jpeg=False,
        has_alpha=False,
        has_exif=False,
        exif_orientation=None,
        file_size_bytes=5000,
    )
    flags = assess_image_quality(meta)
    assert "ela_not_native_jpeg" in flags


def test_quality_extreme_aspect_ratio():
    meta = ImageMetadata(
        format="JPEG",
        width=1000,
        height=50,
        color_mode="RGB",
        is_jpeg=True,
        has_alpha=False,
        has_exif=False,
        exif_orientation=None,
        file_size_bytes=5000,
    )
    flags = assess_image_quality(meta, max_aspect_ratio=5.0)
    assert "extreme_aspect_ratio" in flags

