"""Training pipeline for classical forensic baseline classifiers."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from aidetector.config import ClassicalConfig
from aidetector.io import load_image
from aidetector.classical.features import extract_classical_features
from aidetector.models.classical_model import ClassicalModelBundle
from aidetector.training.dataset import validate_manifest


def extract_features_from_manifest(
    manifest_df: pd.DataFrame,
    config: ClassicalConfig,
    cache_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    """Extracts classical features from a manifest dataframe with optional caching.

    Args:
        manifest_df: DataFrame with image_path and label columns.
        config: ClassicalConfig instance.
        cache_path: Optional path to .npz cache file.

    Returns:
        Tuple of (X: np.ndarray, y: np.ndarray, feature_names: List[str], metadata_list: List[Dict])
    """
    if cache_path is not None:
        c_path = Path(cache_path)
        if c_path.is_file():
            loaded = np.load(c_path, allow_pickle=True)
            X = loaded["X"]
            y = loaded["y"]
            feature_names = list(loaded["feature_names"])
            metadata = list(loaded["metadata"])
            return X, y, feature_names, metadata

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    feature_names: List[str] = []
    metadata_list: List[Dict[str, Any]] = []

    for idx, row in manifest_df.iterrows():
        img_p = Path(row["image_path"])
        label = int(row["label"])
        try:
            img_arr, meta = load_image(img_p)
            feats, names, _ = extract_classical_features(
                img_arr, config=config, is_jpeg=meta.is_jpeg
            )
            if not feature_names:
                feature_names = names
            X_list.append(feats)
            y_list.append(label)
            metadata_list.append({
                "image_path": str(img_p),
                "label": label,
                "format": meta.format,
                "generator_family": row.get("generator_family", "unknown"),
            })
        except Exception as e:
            print(f"Warning: Failed to process image {img_p}: {e}")
            continue

    if not X_list:
        raise RuntimeError("No features could be extracted from manifest images.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)

    if cache_path is not None:
        c_path = Path(cache_path)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            c_path,
            X=X,
            y=y,
            feature_names=np.array(feature_names, dtype=object),
            metadata=np.array(metadata_list, dtype=object),
        )

    return X, y, feature_names, metadata_list


def evaluate_classifier(clf, X_scaled: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Computes standard evaluation metrics."""
    preds = clf.predict(X_scaled)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_scaled)[:, 1]
    elif hasattr(clf, "decision_function"):
        dec = clf.decision_function(X_scaled)
        probs = 1.0 / (1.0 + np.exp(-dec))
    else:
        probs = preds.astype(float)

    acc = float(accuracy_score(y, preds))
    prec = float(precision_score(y, preds, zero_division=0))
    rec = float(recall_score(y, preds, zero_division=0))
    f1 = float(f1_score(y, preds, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y, probs))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y, probs))
    except Exception:
        pr_auc = 0.5

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def train_classical_baseline(
    manifest_path: Union[str, Path],
    config_dict: Dict[str, Any],
    output_bundle_path: Union[str, Path] = "artifacts/classical_bundle.joblib",
    reports_dir: Union[str, Path] = "reports",
    cache_dir: Optional[Union[str, Path]] = "data/cache",
) -> Tuple[ClassicalModelBundle, Dict[str, Any]]:
    """Trains and compares classical baseline classifiers on manifest train split.

    Args:
        manifest_path: Path to dataset manifest CSV.
        config_dict: Full config dictionary.
        output_bundle_path: Where to save winning model bundle.
        reports_dir: Directory to save evaluation reports.
        cache_dir: Directory to cache extracted features.

    Returns:
        Tuple of (winning_model_bundle, metrics_report)
    """
    manifest_p = Path(manifest_path)
    validate_manifest(manifest_p, check_files_exist=True)

    df = pd.read_csv(manifest_p)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    if len(train_df) == 0:
        raise ValueError("Manifest contains no rows with split == 'train'")
    if len(val_df) == 0:
        raise ValueError("Manifest contains no rows with split == 'validation'")

    classical_cfg = ClassicalConfig.from_dict(config_dict.get("classical", {}))
    seed = int(config_dict.get("random_seed", 42))

    train_cache = Path(cache_dir) / "features_train.npz" if cache_dir else None
    val_cache = Path(cache_dir) / "features_val.npz" if cache_dir else None

    print(f"Extracting features for {len(train_df)} training images...")
    X_train, y_train, feat_names, _ = extract_features_from_manifest(
        train_df, classical_cfg, cache_path=train_cache
    )

    print(f"Extracting features for {len(val_df)} validation images...")
    X_val, y_val, _, _ = extract_features_from_manifest(
        val_df, classical_cfg, cache_path=val_cache
    )

    # Fit scaler strictly on training features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    candidate_classifiers = {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=seed, max_iter=150, early_stopping=True
        ),
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=seed
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    best_name = ""
    best_score = -1.0
    best_clf = None

    for name, clf in candidate_classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train_scaled, y_train)
        metrics = evaluate_classifier(clf, X_val_scaled, y_val)
        results[name] = metrics
        print(f"[{name}] Val ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, F1: {metrics['f1']:.4f}")

        # Choose best by validation ROC-AUC
        score = metrics["roc_auc"]
        if score > best_score:
            best_score = score
            best_name = name
            best_clf = clf

    print(f"Best classical model: {best_name} with ROC-AUC {best_score:.4f}")

    training_metadata = {
        "model_type": best_name,
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "train_class_balance": {int(k): int(v) for k, v in pd.Series(y_train).value_counts().items()},
        "val_class_balance": {int(k): int(v) for k, v in pd.Series(y_val).value_counts().items()},
        "metrics_all": results,
        "best_metrics": results[best_name],
        "random_seed": seed,
        "feature_count": len(feat_names),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    winning_bundle = ClassicalModelBundle(
        classifier=best_clf,
        scaler=scaler,
        feature_names=feat_names,
        model_type=best_name,
        training_metadata=training_metadata,
    )

    out_p = Path(output_bundle_path)
    winning_bundle.save(out_p)
    print(f"Saved classical model bundle to: {out_p}")

    rep_p = Path(reports_dir)
    rep_p.mkdir(parents=True, exist_ok=True)
    report_file = rep_p / "classical_baseline_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2)
    print(f"Saved report to: {report_file}")

    return winning_bundle, training_metadata

