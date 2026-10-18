"""Tests for ClassicalModelBundle and inference interface."""

from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aidetector.classical.features import FeatureSchemaMismatchError
from aidetector.models.classical_model import ClassicalModelBundle


def test_classical_bundle_fit_predict_save_load(tmp_path: Path):
    # Create synthetic dataset with 10 features
    num_samples = 40
    num_features = 10
    rng = np.random.RandomState(42)
    X = rng.randn(num_samples, num_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    feature_names = [f"feat_{i}" for i in range(num_features)]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_scaled, y)

    bundle = ClassicalModelBundle(
        classifier=clf,
        scaler=scaler,
        feature_names=feature_names,
        training_metadata={"accuracy": 0.95},
    )

    # Test predictions
    single_sample = X[0]
    prob = bundle.predict_proba(single_sample, feature_names=feature_names)
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0, 1] <= 1.0

    score = bundle.predict_score(single_sample, feature_names=feature_names)
    assert score == pytest.approx(prob[0, 1])

    # Save and reload
    save_path = tmp_path / "classical_bundle.joblib"
    bundle.save(save_path)

    loaded_bundle = ClassicalModelBundle.load(save_path)
    loaded_prob = loaded_bundle.predict_proba(single_sample, feature_names=feature_names)
    np.testing.assert_array_almost_equal(prob, loaded_prob)


def test_classical_bundle_schema_mismatch():
    clf = LogisticRegression()
    scaler = StandardScaler()
    feature_names = ["feat_0", "feat_1"]
    bundle = ClassicalModelBundle(
        classifier=clf,
        scaler=scaler,
        feature_names=feature_names,
    )

    bad_names = ["feat_1", "feat_0"]  # inverted order
    sample = np.array([0.5, -0.5], dtype=np.float32)
    with pytest.raises(FeatureSchemaMismatchError):
        bundle.predict_proba(sample, feature_names=bad_names)


def test_classical_bundle_load_nonexistent():
    with pytest.raises(FileNotFoundError):
        ClassicalModelBundle.load("nonexistent_bundle.joblib")

