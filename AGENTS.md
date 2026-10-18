# Agent Implementation Guide

This project follows the milestone breakdown from `AGENT_BUILD_SPEC_AI_IMAGE_DETECTOR.md`.

## Milestones

- **Milestone 1: Project Scaffold & Classical Forensics (Branch A)**
  - Unified I/O and Quality validation (`io.py`, `quality.py`).
  - Error Level Analysis (`classical/ela.py`).
  - Color Distribution Analysis (`classical/color_features.py`).
  - High-Pass Residuals & Textures (`classical/residuals.py`).
  - 2D FFT Frequency Analysis (`classical/frequency.py`).
  - Unified feature extractor with strict schema validation (`classical/features.py`).
  - Dataset manifest validation and leakage detector (`training/dataset.py`).
  - Classical classifier training pipeline (`training/train_classical.py`).
  - Comprehensive unit test suite (`tests/`).

- **Milestone 2: CNN Baseline (Branch B)**
  - ResNet-18 binary classifier (`models/cnn.py`).
  - Torch Dataset & DataLoader with balanced sampling and augmentations (`training/dataset.py`).
  - Training loop with ROC-AUC validation and checkpointing (`training/train_cnn.py`).
  - Grad-CAM explainability module (`explain/gradcam.py`).

- **Milestone 3: Late Fusion & Uncertainty Calibration**
  - Fusion classifier combining Classical + CNN probabilities (`models/fusion.py`).
  - Isotonic / Platt scaling calibration (`models/calibration.py`).
  - Three-way prediction policy: `likely_real`, `likely_ai_generated`, `uncertain` (`inference.py`).

- **Milestone 4: Explainability & Unified Evaluation**
  - Robustness testing under perturbations (blur, compression, resizing).
  - Cross-generator evaluation reports.
