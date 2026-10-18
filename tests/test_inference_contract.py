"""Tests for aidetector.inference contract and three-way policy."""

import numpy as np
import pytest
from aidetector.inference import FullDetector
from aidetector.models.classical_model import ClassicalModelBundle
from aidetector.models.cnn import CNNModelBundle, ResNet18BinaryClassifier
from aidetector.models.fusion import LateFusionModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def test_inference_contract_json_structure():
    # Mock untrained / simple bundles for interface smoke test
    c_clf = LogisticRegression()
    c_scaler = StandardScaler()
    # Dummy fit with 187 features
    feats = np.random.randn(10, 187).astype(np.float32)
    labels = np.random.randint(0, 2, 10)
    c_scaler.fit(feats)
    c_clf.fit(c_scaler.transform(feats), labels)

    from aidetector.classical.features import extract_classical_features
    _, dummy_names, _ = extract_classical_features(np.zeros((64, 64, 3), dtype=np.uint8))
    c_bundle = ClassicalModelBundle(
        classifier=c_clf, scaler=c_scaler, feature_names=dummy_names
    )

    n_model = ResNet18BinaryClassifier(pretrained=False)
    n_bundle = CNNModelBundle(model=n_model)

    fusion = LateFusionModel()
    fusion.fit(np.array([0.1, 0.9]), np.array([0.2, 0.8]), np.array([0, 1]))

    detector = FullDetector(
        classical_bundle=c_bundle,
        cnn_bundle=n_bundle,
        fusion_model=fusion,
        ai_threshold=0.60,
        real_threshold=0.40,
    )

    test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    res = detector.predict_image(test_img)

    assert "prediction" in res
    assert res["prediction"] in ("likely_real", "likely_ai_generated", "uncertain")
    assert "ai_score" in res
    assert "score_type" in res
    assert "branch_scores" in res
    assert "artifacts" in res
    assert "limitations" in res
    assert res["model_version"] == "aid-ab-v1"

