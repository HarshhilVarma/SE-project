# Agent Build Spec: AI-Generated Image Detection (Classical Forensics + CNN)

> **Purpose:** Give coding agents an implementation contract for building a runnable research prototype that classifies images as `likely_real`, `likely_ai_generated`, or `uncertain` using classical image forensics and CNN features.
>
> **Important:** This is a proposed design. Do not claim accuracy until the project is trained and evaluated on a leakage-resistant test set. A model score is not proof of authenticity.

## 0. Instructions for the coding agent

Work in small, verifiable milestones. Before coding, inspect the repository and report its current structure. Do not overwrite user files without explaining the change. Implement tests with each component. Prefer clear, typed, modular Python over one large script.

**Required behaviour**
- Build Branch A: classical image-processing features.
- Build Branch B: pretrained ResNet-18 CNN.
- Build late fusion as the first integrated classifier.
- Keep all preprocessing deterministic and shared between training and inference.
- Make the system runnable on CPU; use CUDA if available.
- Do not invent a trained model or fabricate evaluation results.
- If a dataset or model checkpoint is missing, provide setup instructions and fail with a clear error rather than pretending inference succeeded.
- Do not make internet downloads mandatory at runtime. Any pretrained-weight download should be explicit and documented.
- Do not implement SPAI or ViT in this version. Keep interfaces extensible for future branches.

## 1. Product objective

Given one image, return:
1. a prediction: `likely_real`, `likely_ai_generated`, or `uncertain`;
2. an AI-generated score (only call it a calibrated probability if calibration has actually been fitted and validated);
3. branch-level scores when available;
4. image-quality flags;
5. paths/references to optional ELA, color-difference, residual and Grad-CAM visualizations;
6. model version and preprocessing version;
7. a brief limitation notice.

The primary task is **fully AI-generated image vs real image**. Manually edited/tampered-image detection is a separate future task and must not be conflated with this binary label.

## 2. Research basis

Use these sources as design motivation, not as proof that this exact combined system will achieve their reported results:

1. Sarnikar et al., “FakeVision++: A Computationally Efficient Dual-Descriptor Image Forensic Framework using HFRD–FCDS Modeling for Reliable Detection of AI-Generated and Tampered Images,” IEEE IATMSI, 2026. DOI: `10.1109/IATMSI68868.2026.11465929`. User-provided PDF.
2. Jia et al., “Secret Lies in Color: Enhancing AI-Generated Images Detection with Color Distribution Analysis,” CVPR 2025, pp. 13445–13454. DOI: `10.1109/CVPR52734.2025.01255`. Paper: <https://openaccess.thecvf.com/content/CVPR2025/html/Jia_Secret_Lies_in_Color_Enhancing_AI-Generated_Images_Detection_with_Color_CVPR_2025_paper.html>
3. Karageorgiou et al., “Any-Resolution AI-Generated Image Detection by Spectral Learning,” CVPR 2025, pp. 18706–18717. DOI: `10.1109/CVPR52734.2025.01743`. Paper: <https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html>. SPAI is future work, not part of this implementation.
4. Park, Na and Choi, “Performance Comparison and Visualization of AI-Generated-Image Detection Methods,” IEEE Access, 2024. DOI: `10.1109/ACCESS.2024.3394250`. User-provided PDF.
5. Lamichhane, “Advanced Detection of AI-Generated Images Through Vision Transformers,” IEEE Access, 2025. DOI: `10.1109/ACCESS.2024.3522759`. User-provided PDF; ViT is not part of this version.
6. van Lierop et al., “Interpol review of detection of AI-generated image and video deepfakes, 2022–2025,” *Forensic Science International: Synergy*, 13 (2026), 100704. DOI: <https://doi.org/10.1016/j.fsisyn.2026.100704>.
7. Guillaro et al., “A Bias-Free Training Paradigm for More General AI-generated Image Detection,” CVPR 2025. <https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html>
8. Park and Owens, “Community Forensics: Using Thousands of Generators to Train Fake Image Detectors,” CVPR 2025. <https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html>

Research implications:
- Color-quantization/restoration differences can be a useful feature, but color/style shortcuts are a risk.
- ELA is a recompression comparison, most directly interpretable for JPEG images; it is not a definitive AI detector.
- Diverse training sources and held-out generator-family evaluation are essential.
- Published scores from different papers/datasets are not directly transferable to this project.

## 3. Required technology

Use:
- Python 3.11 or a compatible version supported by the selected PyTorch build
- PyTorch and torchvision
- Pillow
- OpenCV (`opencv-python-headless` is suitable for server use)
- NumPy, SciPy
- scikit-learn
- joblib
- pytest
- Optional: FastAPI for API or Streamlit for a demo UI

Pin tested dependency versions in `requirements.txt` or a lock file. Avoid adding heavy dependencies without need.

## 4. Repository layout

Create this structure unless an existing repository has a well-established alternative:

```text
ai-image-detector/
├── README.md
├── AGENTS.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── train.yaml
│   └── inference.yaml
├── data/
│   ├── README.md
│   └── manifests/
├── artifacts/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
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
│       │   ├── cnn.py
│       │   ├── classical_model.py
│       │   ├── fusion.py
│       │   └── calibration.py
│       ├── training/
│       │   ├── dataset.py
│       │   ├── train_classical.py
│       │   ├── train_cnn.py
│       │   ├── train_fusion.py
│       │   └── evaluate.py
│       ├── explain/
│       │   ├── gradcam.py
│       │   └── render.py
│       ├── inference.py
│       └── cli.py
└── tests/
    ├── test_io.py
    ├── test_ela.py
    ├── test_color_features.py
    ├── test_residuals.py
    ├── test_frequency.py
    ├── test_feature_schema.py
    └── test_inference_contract.py
```

## 5. Input handling and quality assessment

Implement a single image-loading function:
- Accept a filesystem path initially; optionally accept bytes later.
- Validate that the file exists, can be decoded and has supported dimensions.
- Correctly handle EXIF orientation.
- Convert to RGB for model input; preserve alpha/metadata as quality flags if relevant.
- Do not overwrite or recompress the source image.
- Protect against decompression bombs and unreasonable file sizes.
- Return original format, width, height, color mode and whether the source is JPEG.
- Raise clear exceptions for invalid/corrupt images.

Quality flags should include at least: `low_resolution`, `unsupported_or_unknown_format`, `ela_not_native_jpeg`, `decode_warning`. Do not infer “AI” from metadata absence or file type.

## 6. Branch A: classical feature extractor

Every extractor must:
- Accept a decoded image and configuration.
- Return a fixed-size numeric feature vector plus optional visual map(s).
- Have deterministic output for fixed inputs/configuration.
- Document feature names and ordering.
- Handle small images and edge cases.
- Never use labels to change preprocessing.

### 6.1 ELA
For JPEG input, re-encode a copy at a configured quality (initial baseline: 90), decode it, and compute absolute per-pixel difference from the decoded source. Provide:
- a normalized visual map;
- channel-wise mean and standard deviation;
- selected quantiles;
- map entropy or a documented alternative;
- tile-level statistics (fixed grid).
For non-JPEG input, either mark ELA as not applicable or explicitly label it as a derived recompression difference; never present it as original JPEG evidence. Keep the policy consistent across training/inference.

### 6.2 Color distribution / CoD-inspired restoration
Implement a project-defined, reproducible color-quantization/restoration descriptor inspired by Jia et al.:
1. Quantize RGB values to a configurable bit depth.
2. Generate several controlled Gaussian-noise variants.
3. Quantize each noisy variant.
4. Average quantized variants into a restored image.
5. Compute absolute difference between original and restored image.
6. Extract channel-wise difference statistics, histogram summaries and optional local-tile statistics.
Expose quantization bit depth, noise scale, number of samples and seed in configuration. Use the paper/supplement for parameter reproduction if accessible. If exact parameters are not reproduced, document these as project-defined settings, not a faithful reproduction of the published model.

### 6.3 High-pass residual and texture features
Compute one or more documented high-pass/Laplacian residual maps and extract:
- residual mean, standard deviation and quantiles;
- local variance/entropy summaries;
- edge density;
- optional wavelet-band energies if SciPy/PyWavelets is added and justified.
Avoid silently changing feature count/order across images.

### 6.4 Frequency features
Compute FFT magnitude (or DCT if implemented) and stable log magnitude. Extract a fixed set of radial frequency-bin energies, high/low frequency energy ratio, spectral entropy and directional energy. Handle zero-energy inputs and numerical stability. This is classical frequency analysis, **not SPAI**.

### 6.5 Feature contract
Expose a function like:

```python
features, names, visualizations = extract_classical_features(image, config)
```

- `features`: one-dimensional finite `float32` array
- `names`: same length, stable order
- `visualizations`: optional map references/arrays
- Fail loudly if feature names/order differ from the trained schema.
- Save schema and scaler with the trained classical model.
- Fit scaling/imputation on training data only.

## 7. Branch B: CNN

### 7.1 RGB ResNet-18 baseline
- Use `torchvision.models.resnet18` with pretrained ImageNet weights when explicitly enabled.
- Replace the final fully connected layer with a binary logit.
- Initial input size: 224×224; allow a documented configuration override.
- Use the normalization expected by the selected pretrained weights.
- Use `BCEWithLogitsLoss` for one logit, or two-class cross-entropy consistently.
- Support CPU inference; use CUDA when available.
- Save checkpoint metadata: architecture, image size, normalization, class mapping, seed, epoch and validation metric.
- Inference must load only a checkpoint whose metadata matches the preprocessing configuration.

### 7.2 Augmentation
Training-only, label-preserving transforms may include horizontal flip, small rotation/crop, resizing, mild blur/noise and JPEG recompression. Use a range of transformation strengths.
**Critical:** Do not apply a transform only to one label class. Otherwise the network may learn the data-generation pipeline rather than synthetic-image traces. Keep validation/test transformations separate from training augmentation.

### 7.3 Optional forensic-map CNN
After the RGB baseline works, add a separate small CNN for ELA/residual maps only as an experiment. Keep map normalization fixed. Train and compare:
- RGB-only CNN;
- forensic-map-only CNN;
- RGB + forensic-map fusion.
Keep the extra branch only if held-out tests show a meaningful improvement.

## 8. Dataset contract and leakage prevention

Use a CSV manifest with columns such as:

```text
image_path,label,source_dataset,generator_family,content_group,split,format
```

Labels: `real=0`, `ai_generated=1`.

Requirements:
- No image or near-duplicate group may cross train/validation/test.
- Keep original/derived variants in one group.
- Split by generator family for a dedicated unseen-generator test.
- Balance format, resolution and compression across classes where possible.
- Include varied content categories and multiple real-image sources.
- Keep test data out of feature selection, calibration, threshold selection and model selection.
- Do not fabricate or download copyrighted/private data without checking source terms.
- If no dataset is provided, create manifest validation and sample toy tests only; do not train a claimed useful detector on toy data.

Implement a manifest validator that checks missing paths, invalid labels, duplicates, split overlap, class counts and generator-family overlap.

## 9. Training plan

### Milestone 1: classical baseline
- Extract features into a versioned table/cache.
- Fit scaler and compare logistic regression, RBF-SVM and gradient boosting.
- Save classifier, scaler, feature names and training metadata.

### Milestone 2: CNN baseline
- Fine-tune ResNet-18.
- Select checkpoint using validation PR-AUC or ROC-AUC.
- Use early stopping and save learning curves.
- Do not use test data to choose epochs.

### Milestone 3: late fusion
- Generate out-of-fold predictions on training data, or use a dedicated fusion-training split.
- Train a logistic-regression fusion model using classical and CNN scores.
- Fit calibration only on validation data or an explicitly reserved calibration split.
- Record thresholds and calibration metadata.

### Milestone 4: optional dual-branch experiment
- Add the forensic-map branch.
- Perform ablation with identical splits.
- Do not assume added complexity improves generalization.

All training scripts must set and record random seeds, log class counts, write configs to the output directory, and fail clearly on missing data or checkpoints.

## 10. Inference contract

Implement a function like:

```python
result = predict_image(image_path, model_bundle, config)
```

Return a JSON-serializable dictionary:

```json
{
  "prediction": "uncertain",
  "ai_score": null,
  "score_type": "uncalibrated_model_score",
  "threshold": null,
  "model_version": "aid-ab-v1",
  "quality_flags": [],
  "branch_scores": {
    "classical": null,
    "cnn_rgb": null,
    "cnn_forensic": null
  },
  "artifacts": {
    "ela_map": null,
    "color_difference_map": null,
    "residual_map": null,
    "gradcam": null
  },
  "limitations": [
    "This model output is not proof of image authenticity."
  ]
}
```

- Use `null` when a branch is unavailable; do not fabricate a score.
- Only label `ai_score` as calibrated probability if a calibration step was trained and evaluated.
- `uncertain` should be available for a near-threshold score, unsupported quality conditions or strong branch disagreement. Make rules configurable and document them.
- Include errors as clear exceptions/API errors, not false/real predictions.
- Include model and preprocessing versions in every response.

## 11. Explainability

- Generate ELA, color-difference and residual maps as optional outputs.
- Implement Grad-CAM for the CNN and document the target layer.
- Do not describe Grad-CAM as proof of the model’s reasoning or proof that an image is fake.
- Save visualization parameters and avoid modifying the original image.
- Add a report that displays prediction, branch scores, quality flags and maps.

## 12. Evaluation requirements

Report at minimum:
- accuracy, precision, recall, F1;
- ROC-AUC and PR-AUC;
- false-positive rate at the chosen threshold;
- calibration/reliability metrics if presenting probabilities;
- per-generator-family and per-content-domain metrics;
- performance after JPEG compression, resizing, blur, noise and crop;
- CPU/GPU inference latency and memory use.

Required tracks:
1. in-distribution held-out test;
2. held-out generator-family test;
3. post-processing robustness test;
4. cross-domain test.

Required ablations:
- A only: classical features;
- B1 only: RGB ResNet-18;
- A + B1: late fusion;
- B2 only: forensic-map CNN (optional);
- B1 + B2: dual-CNN fusion (optional);
- A + B1 + B2: full optional model.

Every report must identify dataset, split protocol, generator families, class counts, threshold and model version. Never state a target score as achieved unless an evaluation script actually measured it.

## 13. CLI requirements

Implement these commands (names can vary if documented):
- `aidetector features IMAGE --output-dir DIR`
- `aidetector train-classical --manifest PATH --config PATH`
- `aidetector train-cnn --manifest PATH --config PATH`
- `aidetector train-fusion --manifest PATH --config PATH`
- `aidetector evaluate --manifest PATH --model-dir PATH --output-dir DIR`
- `aidetector predict IMAGE --model-dir PATH --output-dir DIR`
- `aidetector validate-manifest --manifest PATH`

If a trained model bundle is missing, `predict` must explain how to train/download a legitimate checkpoint. It must not silently create random weights and report a prediction.

## 14. Tests and acceptance criteria

Write unit tests for:
- invalid and corrupt image input;
- RGB conversion and EXIF orientation;
- ELA map shape/range and non-JPEG policy;
- deterministic color restoration for a fixed seed;
- residual/frequency feature vector finite values;
- stable feature schema and ordering;
- manifest split leakage detection;
- model output shape and CPU inference;
- missing checkpoint handling;
- JSON serializability of inference result;
- threshold/uncertainty behavior;
- end-to-end smoke test with a fixture image and an explicitly initialized untrained model, clearly labelled as a smoke test only.

Acceptance criteria:
- `pytest` passes;
- lint/type checks pass if configured;
- CLI help works;
- training and inference preprocessing are consistent;
- no fake metrics or default random-model predictions are presented as valid detection;
- README contains setup, dataset schema, training, inference, evaluation, known limitations and citations.

## 15. Suggested implementation order for agents

1. **Agent 1 — Project scaffold and data validation:** repository structure, configs, manifest validator, tests.
2. **Agent 2 — Classical forensics:** ELA, color restoration, residuals, FFT features and unit tests.
3. **Agent 3 — CNN baseline:** dataset, ResNet-18 training, checkpoint metadata and inference tests.
4. **Agent 4 — Fusion and calibration:** out-of-fold/held-out scores, fusion, uncertainty policy.
5. **Agent 5 — Evaluation and explainability:** metrics, perturbation tests, Grad-CAM and visual reports.
6. **Integrator agent:** review all interfaces, run tests, resolve schema mismatches, update README and produce a final reproducibility report.

Agents should work on separate modules/branches when possible. The integrator owns shared interfaces and must run the full test suite after merges. Do not allow multiple agents to independently redefine feature schemas, labels, preprocessing or result JSON.

## 16. Definition of done

The project is complete when:
- a user can install dependencies and validate a dataset manifest;
- classical features can be extracted and tested independently;
- a ResNet-18 can be trained and evaluated;
- late fusion can be trained without data leakage;
- inference returns a documented result and optional visualizations;
- cross-generator and perturbation evaluation reports are generated;
- all metrics are traceable to a specific dataset split and checkpoint;
- limitations and references are present in the README.

## 17. References and links

- Jia et al., CVPR 2025, *Secret Lies in Color: Enhancing AI-Generated Images Detection with Color Distribution Analysis*: <https://openaccess.thecvf.com/content/CVPR2025/html/Jia_Secret_Lies_in_Color_Enhancing_AI-Generated_Images_Detection_with_Color_CVPR_2025_paper.html>
- Karageorgiou et al., CVPR 2025, *Any-Resolution AI-Generated Image Detection by Spectral Learning*: <https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html>
- Guillaro et al., CVPR 2025, *A Bias-Free Training Paradigm for More General AI-generated Image Detection*: <https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html>
- Park and Owens, CVPR 2025, *Community Forensics: Using Thousands of Generators to Train Fake Image Detectors*: <https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html>
- Sarnikar et al., *FakeVision++*, IEEE IATMSI 2026, DOI: `10.1109/IATMSI68868.2026.11465929` (user-provided PDF).
- Park, Na and Choi, *Performance Comparison and Visualization of AI-Generated-Image Detection Methods*, IEEE Access 2024, DOI: `10.1109/ACCESS.2024.3394250` (user-provided PDF).
- Lamichhane, *Advanced Detection of AI-Generated Images Through Vision Transformers*, IEEE Access 2025, DOI: `10.1109/ACCESS.2024.3522759` (user-provided PDF; ViT is outside current scope).
- van Lierop et al., *Interpol review of detection of AI-generated image and video deepfakes, 2022–2025*, Forensic Science International: Synergy 13 (2026), 100704. DOI: <https://doi.org/10.1016/j.fsisyn.2026.100704>
