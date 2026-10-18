"""Tests for dataset manifest validation and leakage detection."""

from pathlib import Path
import pandas as pd
import pytest
from aidetector.training.dataset import validate_manifest, ManifestValidationError


def create_dummy_manifest(tmp_path: Path, rows) -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / "manifest.csv"
    df.to_csv(p, index=False)
    return p


def test_valid_manifest(tmp_path: Path):
    # Create dummy images
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img1.write_bytes(b"dummy")
    img2.write_bytes(b"dummy")

    rows = [
        {
            "image_path": str(img1),
            "label": 0,
            "source_dataset": "coco",
            "generator_family": "real",
            "content_group": "group_a",
            "split": "train",
            "format": "jpeg",
        },
        {
            "image_path": str(img2),
            "label": 1,
            "source_dataset": "defactify",
            "generator_family": "sdxl",
            "content_group": "group_b",
            "split": "validation",
            "format": "jpeg",
        },
    ]
    manifest_p = create_dummy_manifest(tmp_path, rows)
    report = validate_manifest(manifest_p, check_files_exist=True)

    assert report.total_rows == 2
    assert report.split_counts == {"train": 1, "validation": 1}
    assert report.leakage_detected is False


def test_manifest_missing_column(tmp_path: Path):
    rows = [{"image_path": "test.jpg", "label": 0}]
    manifest_p = create_dummy_manifest(tmp_path, rows)

    with pytest.raises(ManifestValidationError, match="missing required columns"):
        validate_manifest(manifest_p, check_files_exist=False)


def test_manifest_invalid_label(tmp_path: Path):
    rows = [
        {
            "image_path": "img1.jpg",
            "label": 99,  # invalid label
            "source_dataset": "coco",
            "generator_family": "real",
            "content_group": "group_a",
            "split": "train",
            "format": "jpeg",
        }
    ]
    manifest_p = create_dummy_manifest(tmp_path, rows)
    with pytest.raises(ManifestValidationError, match="Invalid labels found"):
        validate_manifest(manifest_p, check_files_exist=False)


def test_manifest_image_leakage(tmp_path: Path):
    # Same image in both train and validation
    rows = [
        {
            "image_path": "shared_image.jpg",
            "label": 0,
            "source_dataset": "coco",
            "generator_family": "real",
            "content_group": "group_a",
            "split": "train",
            "format": "jpeg",
        },
        {
            "image_path": "shared_image.jpg",
            "label": 0,
            "source_dataset": "coco",
            "generator_family": "real",
            "content_group": "group_b",
            "split": "validation",
            "format": "jpeg",
        },
    ]
    manifest_p = create_dummy_manifest(tmp_path, rows)
    with pytest.raises(ManifestValidationError, match="Data leakage detected"):
        validate_manifest(manifest_p, check_files_exist=False)


def test_manifest_content_group_leakage(tmp_path: Path):
    # Same content group in both train and validation
    rows = [
        {
            "image_path": "image1.jpg",
            "label": 0,
            "source_dataset": "coco",
            "generator_family": "real",
            "content_group": "identical_scene_101",
            "split": "train",
            "format": "jpeg",
        },
        {
            "image_path": "image2.jpg",
            "label": 1,
            "source_dataset": "defactify",
            "generator_family": "sdxl",
            "content_group": "identical_scene_101",
            "split": "validation",
            "format": "jpeg",
        },
    ]
    manifest_p = create_dummy_manifest(tmp_path, rows)
    with pytest.raises(ManifestValidationError, match="Content-group leakage detected"):
        validate_manifest(manifest_p, check_files_exist=False)

