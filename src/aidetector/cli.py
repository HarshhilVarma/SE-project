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


def cmd_train_cnn(args: argparse.Namespace) -> int:
    """Trains the ResNet-18 CNN detector."""
    from aidetector.training.train_cnn import train_cnn_baseline

    manifest_path = Path(args.manifest)
    output_checkpoint = args.output_checkpoint or "artifacts/cnn_resnet18_best.pt"
    reports_dir = args.reports_dir or "reports"

    print(f"Starting CNN training with manifest {manifest_path}...")
    bundle, metrics = train_cnn_baseline(
        manifest_path=manifest_path,
        output_checkpoint_path=output_checkpoint,
        reports_dir=reports_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        pretrained=args.pretrained,
        seed=args.seed,
    )
    print("CNN training finished successfully!")
    return 0


def cmd_train_fusion(args: argparse.Namespace) -> int:
    """Trains late fusion model combining Classical and CNN branches."""
    from aidetector.training.train_fusion import train_fusion_model

    manifest_path = Path(args.manifest)
    classical_bundle = args.classical_bundle or "artifacts/classical_bundle.joblib"
    cnn_checkpoint = args.cnn_checkpoint or "artifacts/cnn_resnet18_best.pt"
    output_fusion = args.output_fusion or "artifacts/fusion_model.joblib"
    reports_dir = args.reports_dir or "reports"

    print(f"Starting late fusion training with manifest {manifest_path}...")
    fusion, metrics = train_fusion_model(
        manifest_path=manifest_path,
        classical_bundle_path=classical_bundle,
        cnn_checkpoint_path=cnn_checkpoint,
        output_fusion_path=output_fusion,
        reports_dir=reports_dir,
    )
    print("Late fusion training finished successfully!")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluates all branches on test split and generates graphs and reports."""
    from aidetector.training.evaluate import evaluate_test_split

    manifest_path = Path(args.manifest)
    classical_bundle = args.classical_bundle or "artifacts/classical_bundle.joblib"
    cnn_checkpoint = args.cnn_checkpoint or "artifacts/cnn_resnet18_best.pt"
    fusion_model = args.fusion_model or "artifacts/fusion_model.joblib"
    reports_dir = args.reports_dir or "reports"

    print(f"Evaluating detector on {manifest_path} test split...")
    report = evaluate_test_split(
        manifest_path=manifest_path,
        classical_bundle_path=classical_bundle,
        cnn_checkpoint_path=cnn_checkpoint,
        fusion_model_path=fusion_model,
        reports_dir=reports_dir,
    )
    print("Evaluation finished! Check reports/ for full metrics and figures.")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Predicts veracity of a single image using the full combined detector."""
    from aidetector.inference import FullDetector
    from aidetector.models.classical_model import ClassicalModelBundle
    from aidetector.models.cnn import CNNModelBundle
    from aidetector.models.fusion import LateFusionModel

    image_path = Path(args.image)
    c_bundle_path = Path(args.classical_bundle or "artifacts/classical_bundle.joblib")
    n_ckpt_path = Path(args.cnn_checkpoint or "artifacts/cnn_resnet18_best.pt")
    f_model_path = Path(args.fusion_model or "artifacts/fusion_model.joblib")

    c_bundle = ClassicalModelBundle.load(c_bundle_path) if c_bundle_path.is_file() else None
    n_bundle = CNNModelBundle.load(n_ckpt_path) if n_ckpt_path.is_file() else None
    f_model = LateFusionModel.load(f_model_path) if f_model_path.is_file() else None

    if c_bundle is None and n_bundle is None:
        print("Error: Neither classical nor CNN checkpoint was found. Please train models first.", file=sys.stderr)
        return 1

    detector = FullDetector(
        classical_bundle=c_bundle,
        cnn_bundle=n_bundle,
        fusion_model=f_model,
    )

    result = detector.predict_image(image_path)
    print(json.dumps(result, indent=2))
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

    # Command: train-cnn
    p_cnn = subparsers.add_parser("train-cnn", help="Train ResNet-18 CNN detector")
    p_cnn.add_argument("--manifest", "-m", required=True, help="Path to dataset manifest CSV")
    p_cnn.add_argument("--output-checkpoint", "-o", default="artifacts/cnn_resnet18_best.pt", help="Checkpoint save path")
    p_cnn.add_argument("--reports-dir", "-r", default="reports", help="Reports output directory")
    p_cnn.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p_cnn.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p_cnn.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p_cnn.add_argument("--pretrained", action="store_true", help="Initialize backbone with ImageNet weights")
    p_cnn.add_argument("--seed", type=int, default=42, help="Random seed")

    # Command: train-fusion
    p_fus = subparsers.add_parser("train-fusion", help="Train late fusion model")
    p_fus.add_argument("--manifest", "-m", required=True, help="Path to dataset manifest CSV")
    p_fus.add_argument("--classical-bundle", default="artifacts/classical_bundle.joblib", help="Classical bundle path")
    p_fus.add_argument("--cnn-checkpoint", default="artifacts/cnn_resnet18_best.pt", help="CNN checkpoint path")
    p_fus.add_argument("--output-fusion", "-o", default="artifacts/fusion_model.joblib", help="Fusion save path")
    p_fus.add_argument("--reports-dir", "-r", default="reports", help="Reports output directory")

    # Command: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate models on test split and generate reports & figures")
    p_eval.add_argument("--manifest", "-m", required=True, help="Path to dataset manifest CSV")
    p_eval.add_argument("--classical-bundle", default="artifacts/classical_bundle.joblib", help="Classical bundle path")
    p_eval.add_argument("--cnn-checkpoint", default="artifacts/cnn_resnet18_best.pt", help="CNN checkpoint path")
    p_eval.add_argument("--fusion-model", default="artifacts/fusion_model.joblib", help="Fusion model path")
    p_eval.add_argument("--reports-dir", "-r", default="reports", help="Reports output directory")

    # Command: predict
    p_pred = subparsers.add_parser("predict", help="Predict veracity of a single image")
    p_pred.add_argument("image", help="Path to image file")
    p_pred.add_argument("--classical-bundle", default="artifacts/classical_bundle.joblib", help="Classical bundle path")
    p_pred.add_argument("--cnn-checkpoint", default="artifacts/cnn_resnet18_best.pt", help="CNN checkpoint path")
    p_pred.add_argument("--fusion-model", default="artifacts/fusion_model.joblib", help="Fusion model path")

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
    elif args.command == "train-cnn":
        return cmd_train_cnn(args)
    elif args.command == "train-fusion":
        return cmd_train_fusion(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "predict":
        return cmd_predict(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())


