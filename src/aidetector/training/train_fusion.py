"""Training pipeline for Late Fusion model combining Classical and CNN branches."""

import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aidetector.classical.features import extract_classical_features
from aidetector.config import ClassicalConfig
from aidetector.io import load_image
from aidetector.models.classical_model import ClassicalModelBundle
from aidetector.models.cnn import CNNModelBundle, get_inference_transform
from aidetector.models.fusion import LateFusionModel
from aidetector.training.dataset import validate_manifest


def train_fusion_model(
    manifest_path: Union[str, Path],
    classical_bundle_path: Union[str, Path] = "artifacts/classical_bundle.joblib",
    cnn_checkpoint_path: Union[str, Path] = "artifacts/cnn_resnet18_best.pt",
    output_fusion_path: Union[str, Path] = "artifacts/fusion_model.joblib",
    reports_dir: Union[str, Path] = "reports",
    split: str = "validation",
) -> Tuple[LateFusionModel, Dict[str, Any]]:
    """Trains late fusion model on held-out validation predictions to avoid data leakage."""
    manifest_p = Path(manifest_path)
    validate_manifest(manifest_p, check_files_exist=True)

    df = pd.read_csv(manifest_p)
    val_df = df[df["split"] == split].reset_index(drop=True)

    if len(val_df) == 0:
        raise ValueError(f"No samples found for split '{split}' in manifest.")

    print(f"Loading classical model from: {classical_bundle_path}")
    classical_bundle = ClassicalModelBundle.load(classical_bundle_path)

    print(f"Loading CNN model from: {cnn_checkpoint_path}")
    cnn_bundle = CNNModelBundle.load(cnn_checkpoint_path)

    classical_cfg = ClassicalConfig()
    cnn_transform = get_inference_transform(input_size=224)

    s_classical_list = []
    s_cnn_list = []
    y_list = []

    print(f"Extracting branch predictions on {len(val_df)} validation images for fusion...")
    for idx, row in val_df.iterrows():
        img_p = Path(row["image_path"])
        label = int(row["label"])

        try:
            img_arr, meta = load_image(img_p)

            # Classical score
            feats, names, _ = extract_classical_features(
                img_arr, config=classical_cfg, is_jpeg=meta.is_jpeg
            )
            s_c = classical_bundle.predict_score(feats, feature_names=names)

            # CNN score
            pil_img = Image.fromarray(img_arr, mode="RGB")
            tensor = cnn_transform(pil_img)
            s_n = cnn_bundle.predict_score(tensor)

            s_classical_list.append(s_c)
            s_cnn_list.append(s_n)
            y_list.append(label)
        except Exception as e:
            print(f"Warning: skipped {img_p}: {e}")
            continue

    s_classical = np.array(s_classical_list, dtype=np.float32)
    s_cnn = np.array(s_cnn_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    fusion = LateFusionModel()
    fusion.fit(s_classical, s_cnn, y, calibrate=True, calibration_method="platt")

    fused_scores = fusion.predict_score(s_classical, s_cnn)
    preds = (fused_scores >= 0.5).astype(int)

    acc = float(accuracy_score(y, preds))
    prec = float(precision_score(y, preds, zero_division=0))
    rec = float(recall_score(y, preds, zero_division=0))
    f1 = float(f1_score(y, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y, fused_scores))
    pr_auc = float(average_precision_score(y, fused_scores))

    print(f"Late Fusion Validation ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, F1: {f1:.4f}")

    metadata = {
        "fusion_type": "logistic_regression_platt_calibrated",
        "validation_samples": len(y),
        "val_metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        },
        "branch_coefficients": {
            "classical_weight": float(fusion.classifier.coef_[0][0]),
            "cnn_weight": float(fusion.classifier.coef_[0][1]),
            "disagreement_weight": float(fusion.classifier.coef_[0][2]),
            "intercept": float(fusion.classifier.intercept_[0]),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    fusion.metadata = metadata

    out_p = Path(output_fusion_path)
    fusion.save(out_p)
    print(f"Saved fusion model to: {out_p}")

    rep_p = Path(reports_dir)
    rep_p.mkdir(parents=True, exist_ok=True)
    report_file = rep_p / "fusion_training_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved fusion report to: {report_file}")

    return fusion, metadata

