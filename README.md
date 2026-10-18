# AI-Generated Image Detector (Classical Forensics + CNN)

A modular, reproducible research prototype for detecting AI-generated images using classical image forensic analysis (Error Level Analysis, Color Quantization/Restoration, High-Pass Residuals, and 2D FFT Frequency Analysis) and deep convolutional neural networks (ResNet-18).

## Overview

This repository implements a dual-branch detector:
- **Branch A (Classical Forensics):** Extracts 187 interpretable forensic features across four domains:
  1. **Error Level Analysis (ELA):** Recompression artifacts, channel-wise error statistics, entropy, and spatial grid uniformity.
  2. **Color Distribution Analysis (Jia et al., CVPR 2025 inspired):** Quantization and stochastic noise restoration differences in color distributions.
  3. **High-Pass Residuals & Texture:** Laplacian residuals, Sobel gradient edge densities, local patch variance, and spatial textures.
  4. **2D FFT Frequency Features:** Radial annular frequency-bin energies, spectral entropy, high/low frequency ratios, and directional quadrant energies.
- **Branch B (CNN):** Pretrained ResNet-18 fine-tuned for synthetic vs. authentic discrimination.
- **Late Fusion:** Combines branch scores with calibrated probability outputs and an explicit uncertainty band.

## Repository Layout

```text
├── configs/
│   ├── train.yaml
│   └── inference.yaml
├── data/
│   ├── README.md
│   └── manifests/
├── artifacts/
│   └── classical_bundle.joblib
├── reports/
│   └── classical_baseline_report.json
├── src/
│   └── aidetector/
│       ├── __init__.py
│       ├── config.py
│       ├── io.py
│       ├── quality.py
│       ├── classical/
│       │   ├── __init__.py
│       │   ├── ela.py
│       │   ├── color_features.py
│       │   ├── residuals.py
│       │   ├── frequency.py
│       │   └── features.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── classical_model.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   └── train_classical.py
│       └── cli.py
├── tests/
│   ├── test_io.py
│   ├── test_quality.py
│   ├── test_ela.py
│   ├── test_color_features.py
│   ├── test_residuals.py
│   ├── test_frequency.py
│   ├── test_feature_schema.py
│   ├── test_manifest.py
│   └── test_classical_model.py
├── pyproject.toml
└── requirements.txt
```

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run test suite:
```bash
pytest tests/
```

## CLI Usage

### 1. Extract Features & Visual Maps from an Image
```bash
aidetector features path/to/image.jpg --output-dir reports/features
```
Saves JSON feature statistics alongside visual maps (`ela_map.png`, `color_difference_map.png`, `residual_map.png`, `frequency_spectrum.png`).

### 2. Prepare Balanced Dataset from Defactify
```bash
aidetector prepare-data --parquet-dir Defactify_Image_Dataset --output-dir data/images --manifest-out data/manifests/dataset.csv
```

### 3. Validate Manifest Integrity & Leakage
```bash
aidetector validate-manifest --manifest data/manifests/dataset.csv
```

### 4. Train Classical Baseline
```bash
aidetector train-classical --manifest data/manifests/dataset.csv --config configs/train.yaml
```

## Running the Web Application

To launch both the FastAPI verification backend and the React frontend with a single command, execute:

```bash
./run_app.sh
```

### Options:
- `./run_app.sh --backend-only`: Run only the FastAPI backend server on port 8000.
- `./run_app.sh --frontend-only`: Run only the Vite React dev server on port 5173.
- `./run_app.sh --host 0.0.0.0 --backend-port 8000`: Specify custom host binding and port.

### Services:
- **Web App UI:** `http://localhost:5173`
- **FastAPI Backend:** `http://127.0.0.1:8000`
- **Swagger Interactive API Docs:** `http://127.0.0.1:8000/docs`

You can also start the FastAPI server directly via CLI:
```bash
aidetector serve --host 127.0.0.1 --port 8000
```

## Known Limitations

- **Not Proof of Authenticity:** Model scores indicate statistical similarity to training artifacts, not legal or definitive proof of provenance.
- **Format Dependence:** ELA is native to JPEG images; for non-JPEG formats (PNG, WEBP), it represents derived recompression difference.
- **Unseen Generator Shifts:** Performance may vary on generative models unseen during training.

## References

1. Jia et al., "Secret Lies in Color: Enhancing AI-Generated Images Detection with Color Distribution Analysis", *CVPR*, 2025.
2. Sarnikar et al., "FakeVision++: A Computationally Efficient Dual-Descriptor Image Forensic Framework using HFRD–FCDS Modeling", *IEEE IATMSI*, 2026.
3. Roy et al., "A Comprehensive Dataset for Human vs. AI Generated Image Detection", *arXiv:2601.00553*, 2026 (Defactify).
4. Park, Na and Choi, "Performance Comparison and Visualization of AI-Generated-Image Detection Methods", *IEEE Access*, 2024.

