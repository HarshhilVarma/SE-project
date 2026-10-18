"""Training pipeline for ResNet-18 CNN detector."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aidetector.models.cnn import (
    CNNModelBundle,
    ResNet18BinaryClassifier,
    get_inference_transform,
    get_training_transform,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from aidetector.training.dataset import AIDetectionTorchDataset, validate_manifest


def evaluate_cnn(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates CNN on validation or test dataloader."""
    model.eval()
    total_loss = 0.0
    all_targets: List[float] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs).view(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    avg_loss = total_loss / max(len(y_true), 1)
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

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def train_cnn_baseline(
    manifest_path: Union[str, Path],
    output_checkpoint_path: Union[str, Path] = "artifacts/cnn_resnet18_best.pt",
    reports_dir: Union[str, Path] = "reports",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    pretrained: bool = False,
    seed: int = 42,
    device_name: Optional[str] = None,
) -> Tuple[CNNModelBundle, Dict[str, Any]]:
    """Trains ResNet-18 binary classifier on manifest dataset.

    Args:
        manifest_path: Path to dataset manifest CSV.
        output_checkpoint_path: Destination for best model weights.
        reports_dir: Directory for training metrics report.
        epochs: Number of training epochs.
        batch_size: DataLoader batch size.
        learning_rate: AdamW initial learning rate.
        weight_decay: Weight decay penalty.
        pretrained: Whether to load ImageNet pretrained weights.
        seed: Random seed for reproducibility.
        device_name: 'cuda' or 'cpu'. If None, auto-selects available accelerator.

    Returns:
        Tuple of (best_model_bundle, training_summary_dict)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device for CNN training: {device}")

    manifest_p = Path(manifest_path)
    validate_manifest(manifest_p, check_files_exist=True)

    # Prepare datasets and loaders
    train_transform = get_training_transform(input_size=224)
    val_transform = get_inference_transform(input_size=224)

    train_ds = AIDetectionTorchDataset(manifest_p, split="train", transform=train_transform)
    val_ds = AIDetectionTorchDataset(manifest_p, split="validation", transform=val_transform)

    if len(train_ds) == 0:
        raise ValueError("Train split is empty in manifest.")
    if len(val_ds) == 0:
        raise ValueError("Validation split is empty in manifest.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")
    )

    print(f"Loaded {len(train_ds)} train samples and {len(val_ds)} validation samples.")

    # Initialize model
    model = ResNet18BinaryClassifier(pretrained=pretrained, dropout=0.2)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_roc_auc = -1.0
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    best_state_dict = None
    epoch_history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs).view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)

        scheduler.step()
        epoch_train_loss = train_loss / len(train_ds)

        # Validation evaluation
        val_metrics = evaluate_cnn(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) - "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_metrics": val_metrics,
            "time_sec": elapsed,
        }
        epoch_history.append(epoch_record)

        if val_metrics["roc_auc"] > best_val_roc_auc:
            best_val_roc_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            best_metrics = val_metrics
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Best model achieved at Epoch {best_epoch} with Val ROC-AUC: {best_val_roc_auc:.4f}")

    # Restore best state dict
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    metadata = {
        "architecture": "resnet18",
        "input_size": 224,
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "class_mapping": {
            "real": 0,
            "ai_generated": 1,
        },
        "random_seed": seed,
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": epoch_history,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    best_bundle = CNNModelBundle(model=model, metadata=metadata, device=device)
    out_path = Path(output_checkpoint_path)
    best_bundle.save(out_path)
    print(f"Saved best CNN checkpoint to: {out_path}")

    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    report_file = rep_path / "cnn_training_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved CNN training report to: {report_file}")

    return best_bundle, metadata

