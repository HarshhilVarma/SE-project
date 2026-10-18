"""Command-line interface for the AI Image Detector."""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional
import numpy as np

from aidetector.config import ClassicalConfig, load_yaml_config
from aidetector.io import load_image, save_image_array
from aidetector.quality import assess_image_quality
from aidetector.classical.features import extract_classical_features
from aidetector.training.dataset import validate_manifest, extract_defactify_subset
from aidetector.training.train_classical import train_classical_baseline


def cmd_features(args: argparse.Namespace) -> int:
    """Extracts classical features and visual maps for a single image."""
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading image from: {image_path}")
    img_arr, metadata = load_image(image_path)
    quality_flags = assess_image_quality(metadata)

    config = None
    if args.config:
        cfg_dict = load_yaml_config(args.config)
        config = ClassicalConfig.from_dict(cfg_dict.get("classical", {}))

    print("Extracting classical features...")
    features, names, visualizations = extract_classical_features(
        img_arr, config=config, is_jpeg=metadata.is_jpeg
    )

    # Save feature json
    features_dict = {
        "metadata": {
            "format": metadata.format,
            "width": metadata.width,
            "height": metadata.height,
            "is_jpeg": metadata.is_jpeg,
            "quality_flags": quality_flags,
        },
        "feature_count": len(features),
        "features": {k: float(v) for k, v in zip(names, features)},
    }

    json_path = output_dir / f"{image_path.stem}_features.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(features_dict, f, indent=2)
    print(f"Saved feature values to: {json_path}")

    # Save visualization maps
    for map_name, map_arr in visualizations.items():
        map_path = output_dir / f"{image_path.stem}_{map_name}.png"
        save_image_array(map_arr, map_path)
        print(f"Saved {map_name} visualization to: {map_path}")

    return 0


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    """Validates a dataset manifest CSV."""
    manifest_path = Path(args.manifest)
    print(f"Validating manifest: {manifest_path}...")
    try:
        report = validate_manifest(
            manifest_path,
            check_files_exist=not args.skip_file_check,
            base_dir=args.base_dir,
        )
        print("Manifest is valid!")
        print(f"  Total records: {report.total_rows}")
        print(f"  Split counts: {report.split_counts}")
        print(f"  Label counts per split: {report.label_counts}")
        print(f"  Generator counts: {report.generator_counts}")
        print(f"  Format counts: {report.format_counts}")
        return 0
    except Exception as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        return 1


def cmd_prepare_data(args: argparse.Namespace) -> int:
    """Extracts a balanced image dataset from Defactify parquet files."""
    parquet_dir = Path(args.parquet_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_out)

    print(f"Extracting Defactify dataset from {parquet_dir} to {output_dir}...")
    df = extract_defactify_subset(
        parquet_dir=parquet_dir,
        output_images_dir=output_dir,
        output_manifest_path=manifest_path,
        max_train_samples=args.train_samples,
        max_val_samples=args.val_samples,
        max_test_samples=args.test_samples,
        seed=args.seed,
    )
    print(f"Extraction complete! Manifest written with {len(df)} entries to {manifest_path}")
    return 0


def cmd_train_classical(args: argparse.Namespace) -> int:
    """Trains and compares classical baseline classifiers."""
    manifest_path = Path(args.manifest)
    config_dict = load_yaml_config(args.config) if args.config else {}

    output_bundle = args.output_bundle or "artifacts/classical_bundle.joblib"
    reports_dir = args.reports_dir or "reports"

    print(f"Starting classical training with manifest {manifest_path}...")
    bundle, metrics = train_classical_baseline(
        manifest_path=manifest_path,
        config_dict=config_dict,
        output_bundle_path=output_bundle,
        reports_dir=reports_dir,
    )
    print("Training finished successfully!")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="aidetector",
        description="AI-Generated Image Detection Framework (Classical Forensics + CNN)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: features
    p_feat = subparsers.add_parser("features", help="Extract classical features and visualization maps")
    p_feat.add_argument("image", help="Path to image file")
    p_feat.add_argument("--output-dir", "-o", default="reports/features", help="Directory to save features and maps")
    p_feat.add_argument("--config", "-c", help="Path to config YAML")

    # Command: validate-manifest
    p_val = subparsers.add_parser("validate-manifest", help="Validate dataset manifest CSV")
    p_val.add_argument("--manifest", "-m", required=True, help="Path to manifest CSV")
    p_val.add_argument("--base-dir", "-b", help="Base directory for relative image paths")
    p_val.add_argument("--skip-file-check", action="store_true", help="Skip checking if files exist on disk")

    # Command: prepare-data
    p_prep = subparsers.add_parser("prepare-data", help="Prepare image files and manifest from Defactify dataset")
    p_prep.add_argument("--parquet-dir", "-p", default="Defactify_Image_Dataset", help="Directory containing parquet files")
    p_prep.add_argument("--output-dir", "-o", default="data/images", help="Directory to save unpacked images")
    p_prep.add_argument("--manifest-out", "-m", default="data/manifests/dataset.csv", help="Output path for manifest CSV")
    p_prep.add_argument("--train-samples", type=int, default=1000, help="Number of training samples")
    p_prep.add_argument("--val-samples", type=int, default=300, help="Number of validation samples")
    p_prep.add_argument("--test-samples", type=int, default=300, help="Number of test samples")
    p_prep.add_argument("--seed", type=int, default=42, help="Random seed")

    # Command: train-classical
    p_train = subparsers.add_parser("train-classical", help="Train classical baseline classifier")
    p_train.add_argument("--manifest", "-m", required=True, help="Path to dataset manifest CSV")
    p_train.add_argument("--config", "-c", default="configs/train.yaml", help="Path to training config YAML")
    p_train.add_argument("--output-bundle", "-o", default="artifacts/classical_bundle.joblib", help="Output bundle path")
    p_train.add_argument("--reports-dir", "-r", default="reports", help="Reports output directory")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "features":
        return cmd_features(args)
    elif args.command == "validate-manifest":
        return cmd_validate_manifest(args)
    elif args.command == "prepare-data":
        return cmd_prepare_data(args)
    elif args.command == "train-classical":
        return cmd_train_classical(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

