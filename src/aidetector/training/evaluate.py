"""Comprehensive evaluation and graph generation over test split."""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Configure writable matplotlib cache
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aidetector.classical.features import extract_classical_features
from aidetector.config import ClassicalConfig
from aidetector.io import load_image
from aidetector.models.classical_model import ClassicalModelBundle
from aidetector.models.cnn import CNNModelBundle, get_inference_transform
from aidetector.models.fusion import LateFusionModel
from aidetector.training.dataset import validate_manifest


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes full set of binary classification and probability metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    brier = float(brier_score_loss(y_true, y_prob))

    # Expected Calibration Error (ECE) with 10 bins
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ece = float(np.mean(np.abs(prob_true - prob_pred))) if len(prob_true) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr_at_threshold": fpr,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def apply_perturbation(image: np.ndarray, pert_type: str) -> np.ndarray:
    """Applies realistic image degradations for robustness testing."""
    if pert_type == "clean":
        return image.copy()
    elif pert_type == "jpeg_70":
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    elif pert_type == "jpeg_50":
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    elif pert_type == "gaussian_blur":
        return cv2.GaussianBlur(image, (7, 7), sigmaX=1.5)
    elif pert_type == "gaussian_noise":
        noise = np.random.RandomState(42).normal(0, 10, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif pert_type == "resize_50":
        h, w = image.shape[:2]
        down = cv2.resize(image, (max(w // 2, 1), max(h // 2, 1)), interpolation=cv2.INTER_AREA)
        up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
        return up
    else:
        return image.copy()


def evaluate_test_split(
    manifest_path: Union[str, Path] = "data/manifests/dataset.csv",
    classical_bundle_path: Union[str, Path] = "artifacts/classical_bundle.joblib",
    cnn_checkpoint_path: Union[str, Path] = "artifacts/cnn_resnet18_best.pt",
    fusion_model_path: Union[str, Path] = "artifacts/fusion_model.joblib",
    reports_dir: Union[str, Path] = "reports",
) -> Dict[str, Any]:
    """Runs full evaluation of Classical, CNN, and Fusion models over the test split."""
    manifest_p = Path(manifest_path)
    validate_manifest(manifest_p, check_files_exist=True)

    df = pd.read_csv(manifest_p)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if len(test_df) == 0:
        raise ValueError("No test samples found in manifest.")

    rep_path = Path(reports_dir)
    figures_dir = rep_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    classical_bundle = ClassicalModelBundle.load(classical_bundle_path)
    cnn_bundle = CNNModelBundle.load(cnn_checkpoint_path)
    fusion_model = LateFusionModel.load(fusion_model_path)

    classical_cfg = ClassicalConfig()
    cnn_transform = get_inference_transform(input_size=224)

    y_true_list = []
    scores_classical_list = []
    scores_cnn_list = []
    scores_fusion_list = []
    generators_list = []
    test_images: List[np.ndarray] = []

    # Measure latency
    classical_latencies = []
    cnn_latencies = []

    print(f"Evaluating {len(test_df)} test images across Classical, CNN, and Late Fusion...")
    for idx, row in test_df.iterrows():
        img_p = Path(row["image_path"])
        label = int(row["label"])
        gen_family = row.get("generator_family", "unknown")

        img_arr, meta = load_image(img_p)
        test_images.append(img_arr)
        y_true_list.append(label)
        generators_list.append(gen_family)

        # Classical
        t0 = time.perf_counter()
        feats, names, _ = extract_classical_features(
            img_arr, config=classical_cfg, is_jpeg=meta.is_jpeg
        )
        s_c = classical_bundle.predict_score(feats, feature_names=names)
        classical_latencies.append((time.perf_counter() - t0) * 1000.0)

        # CNN
        t1 = time.perf_counter()
        pil_img = Image.fromarray(img_arr, mode="RGB")
        tensor = cnn_transform(pil_img)
        s_n = cnn_bundle.predict_score(tensor)
        cnn_latencies.append((time.perf_counter() - t1) * 1000.0)

        # Fusion
        s_f = float(fusion_model.predict_score(s_c, s_n))

        scores_classical_list.append(s_c)
        scores_cnn_list.append(s_n)
        scores_fusion_list.append(s_f)

    y_true = np.array(y_true_list, dtype=int)
    s_c = np.array(scores_classical_list, dtype=float)
    s_n = np.array(scores_cnn_list, dtype=float)
    s_f = np.array(scores_fusion_list, dtype=float)

    # 1. Overall Metrics
    metrics_classical = compute_metrics(y_true, s_c)
    metrics_cnn = compute_metrics(y_true, s_n)
    metrics_fusion = compute_metrics(y_true, s_f)

    # 2. Per-Generator Family Breakdown
    gen_families = sorted(list(set(generators_list)))
    generator_breakdown: Dict[str, Dict[str, Any]] = {}
    for gen in gen_families:
        mask = np.array([g == gen for g in generators_list])
        y_sub = y_true[mask]
        sc_sub = s_c[mask]
        sn_sub = s_n[mask]
        sf_sub = s_f[mask]

        generator_breakdown[gen] = {
            "sample_count": int(np.sum(mask)),
            "classical_acc": float(accuracy_score(y_sub, (sc_sub >= 0.5).astype(int))),
            "cnn_acc": float(accuracy_score(y_sub, (sn_sub >= 0.5).astype(int))),
            "fusion_acc": float(accuracy_score(y_sub, (sf_sub >= 0.5).astype(int))),
            "fusion_mean_score": float(np.mean(sf_sub)),
        }

    # 3. Robustness under perturbations (Ablation)
    perturbation_types = ["clean", "jpeg_70", "jpeg_50", "gaussian_blur", "gaussian_noise", "resize_50"]
    robustness_results: Dict[str, Dict[str, float]] = {}

    print("Running post-processing robustness ablation...")
    for pert in perturbation_types:
        pert_sc = []
        pert_sn = []
        pert_sf = []
        for img_arr in test_images:
            p_img = apply_perturbation(img_arr, pert)
            f, n, _ = extract_classical_features(p_img, config=classical_cfg, is_jpeg=True)
            sc = classical_bundle.predict_score(f, feature_names=n)
            sn = cnn_bundle.predict_image(p_img)
            sf = float(fusion_model.predict_score(sc, sn))
            pert_sc.append(sc)
            pert_sn.append(sn)
            pert_sf.append(sf)

        robustness_results[pert] = {
            "classical_roc_auc": float(roc_auc_score(y_true, pert_sc)),
            "cnn_roc_auc": float(roc_auc_score(y_true, pert_sn)),
            "fusion_roc_auc": float(roc_auc_score(y_true, pert_sf)),
            "fusion_accuracy": float(accuracy_score(y_true, (np.array(pert_sf) >= 0.5).astype(int))),
        }

    # 4. Generate Publication-Quality Plots

    # Plot 1: ROC Curves
    plt.figure(figsize=(8, 6), dpi=150)
    fpr_c, tpr_c, _ = roc_curve(y_true, s_c)
    fpr_n, tpr_n, _ = roc_curve(y_true, s_n)
    fpr_f, tpr_f, _ = roc_curve(y_true, s_f)

    plt.plot(fpr_c, tpr_c, label=f"Branch A: Classical (AUC = {metrics_classical['roc_auc']:.3f})", color="#2b5c8f", lw=2)
    plt.plot(fpr_n, tpr_n, label=f"Branch B: CNN ResNet-18 (AUC = {metrics_cnn['roc_auc']:.3f})", color="#d95f02", lw=2)
    plt.plot(fpr_f, tpr_f, label=f"Late Fusion A+B (AUC = {metrics_fusion['roc_auc']:.3f})", color="#1b9e77", lw=2.5)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) - Test Split", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    roc_plot_path = figures_dir / "roc_curves.png"
    plt.savefig(roc_plot_path)
    plt.close()

    # Plot 2: Precision-Recall Curves
    plt.figure(figsize=(8, 6), dpi=150)
    p_c, r_c, _ = precision_recall_curve(y_true, s_c)
    p_n, r_n, _ = precision_recall_curve(y_true, s_n)
    p_f, r_f, _ = precision_recall_curve(y_true, s_f)

    plt.plot(r_c, p_c, label=f"Classical (PR-AUC = {metrics_classical['pr_auc']:.3f})", color="#2b5c8f", lw=2)
    plt.plot(r_n, p_n, label=f"CNN ResNet-18 (PR-AUC = {metrics_cnn['pr_auc']:.3f})", color="#d95f02", lw=2)
    plt.plot(r_f, p_f, label=f"Late Fusion (PR-AUC = {metrics_fusion['pr_auc']:.3f})", color="#1b9e77", lw=2.5)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves - Test Split", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    pr_plot_path = figures_dir / "pr_curves.png"
    plt.savefig(pr_plot_path)
    plt.close()

    # Plot 3: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    models_cm = [
        ("Classical (Branch A)", metrics_classical["confusion_matrix"]),
        ("CNN ResNet-18 (Branch B)", metrics_cnn["confusion_matrix"]),
        ("Late Fusion (A + B)", metrics_fusion["confusion_matrix"]),
    ]

    for ax, (m_title, cm_data) in zip(axes, models_cm):
        mat = np.array([[cm_data["tn"], cm_data["fp"]], [cm_data["fn"], cm_data["tp"]]])
        im = ax.imshow(mat, cmap="Blues", interpolation="nearest")
        ax.set_title(m_title, fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real (0)", "AI (1)"])
        ax.set_yticklabels(["Real (0)", "AI (1)"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                color = "white" if val > mat.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    plt.tight_layout()
    cm_plot_path = figures_dir / "confusion_matrices.png"
    plt.savefig(cm_plot_path)
    plt.close()

    # Plot 4: Per-Generator Breakdown
    plt.figure(figsize=(10, 5), dpi=150)
    x_indices = np.arange(len(gen_families))
    bar_width = 0.25

    c_accs = [generator_breakdown[g]["classical_acc"] * 100 for g in gen_families]
    n_accs = [generator_breakdown[g]["cnn_acc"] * 100 for g in gen_families]
    f_accs = [generator_breakdown[g]["fusion_acc"] * 100 for g in gen_families]

    plt.bar(x_indices - bar_width, c_accs, width=bar_width, label="Classical", color="#2b5c8f")
    plt.bar(x_indices, n_accs, width=bar_width, label="CNN ResNet-18", color="#d95f02")
    plt.bar(x_indices + bar_width, f_accs, width=bar_width, label="Late Fusion", color="#1b9e77")

    plt.xticks(x_indices, [g.upper() for g in gen_families], fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Detection Accuracy by Generator Family", fontsize=14, fontweight="bold")
    plt.ylim(0, 110)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    gen_plot_path = figures_dir / "per_generator_accuracy.png"
    plt.savefig(gen_plot_path)
    plt.close()

    # Plot 5: Robustness Degradation under Perturbations
    plt.figure(figsize=(10, 5.5), dpi=150)
    pert_labels = [p.replace("_", " ").title() for p in perturbation_types]
    c_pert_auc = [robustness_results[p]["classical_roc_auc"] for p in perturbation_types]
    n_pert_auc = [robustness_results[p]["cnn_roc_auc"] for p in perturbation_types]
    f_pert_auc = [robustness_results[p]["fusion_roc_auc"] for p in perturbation_types]

    plt.plot(pert_labels, c_pert_auc, marker="o", lw=2, label="Classical (Branch A)", color="#2b5c8f")
    plt.plot(pert_labels, n_pert_auc, marker="s", lw=2, label="CNN (Branch B)", color="#d95f02")
    plt.plot(pert_labels, f_pert_auc, marker="^", lw=2.5, label="Late Fusion (A+B)", color="#1b9e77")

    plt.ylabel("ROC-AUC", fontsize=12)
    plt.title("Perturbation Robustness Degradation", fontsize=14, fontweight="bold")
    plt.ylim(0.4, 1.05)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    rob_plot_path = figures_dir / "robustness_degradation.png"
    plt.savefig(rob_plot_path)
    plt.close()

    # Latency report
    latency_summary = {
        "classical_latency_ms": {
            "mean": float(np.mean(classical_latencies)),
            "std": float(np.std(classical_latencies)),
            "min": float(np.min(classical_latencies)),
            "max": float(np.max(classical_latencies)),
        },
        "cnn_latency_ms": {
            "mean": float(np.mean(cnn_latencies)),
            "std": float(np.std(cnn_latencies)),
            "min": float(np.min(cnn_latencies)),
            "max": float(np.max(cnn_latencies)),
        },
        "device": str(cnn_bundle.device),
    }

    full_report = {
        "dataset": "Defactify_Image_Dataset",
        "split": "test",
        "total_test_samples": len(y_true),
        "class_balance": {
            "real_count": int(np.sum(y_true == 0)),
            "ai_count": int(np.sum(y_true == 1)),
        },
        "branch_a_classical": metrics_classical,
        "branch_b_cnn": metrics_cnn,
        "late_fusion": metrics_fusion,
        "generator_family_breakdown": generator_breakdown,
        "robustness_ablation": robustness_results,
        "inference_latency": latency_summary,
        "figures": {
            "roc_curves": str(roc_plot_path.relative_to(rep_path.parent)),
            "pr_curves": str(pr_plot_path.relative_to(rep_path.parent)),
            "confusion_matrices": str(cm_plot_path.relative_to(rep_path.parent)),
            "per_generator_accuracy": str(gen_plot_path.relative_to(rep_path.parent)),
            "robustness_degradation": str(rob_plot_path.relative_to(rep_path.parent)),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report_json_path = rep_path / "test_evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Evaluation report written to: {report_json_path}")
    print(f"Generated figures saved to: {figures_dir}")

    return full_report

