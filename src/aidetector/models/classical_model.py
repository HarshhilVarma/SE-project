"""Classical forensics classifier model bundle and inference interface."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from aidetector.classical.features import validate_feature_schema


SCHEMA_VERSION = "classical-v1"


class ClassicalModelNotFittedError(Exception):
    """Raised when making predictions with an unfitted or invalid classical model."""
    pass


@dataclass
class ClassicalModelBundle:
    """Serializable bundle containing a trained classical classifier, scaler, and schema."""

    classifier: BaseEstimator
    scaler: StandardScaler
    feature_names: List[str]
    schema_version: str = SCHEMA_VERSION
    model_type: str = "classical"
    training_metadata: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Predicts class probabilities [P(real), P(ai_generated)].

        Args:
            features: 1D array of shape (N_features,) or 2D array (N_samples, N_features).
            feature_names: Optional list of feature names to validate against trained schema.

        Returns:
            2D float array of shape (N_samples, 2).
        """
        if self.classifier is None or self.scaler is None:
            raise ClassicalModelNotFittedError("Classical model bundle is missing classifier or scaler.")

        if feature_names is not None:
            validate_feature_schema(feature_names, self.feature_names)

        feat_arr = np.asarray(features, dtype=np.float32)
        if feat_arr.ndim == 1:
            feat_arr = feat_arr.reshape(1, -1)

        if feat_arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(self.feature_names)}, got {feat_arr.shape[1]}"
            )

        scaled_features = self.scaler.transform(feat_arr)

        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(scaled_features)
            # If binary classifier only output 1 probability or 2
            if probs.shape[1] == 2:
                return probs
            elif probs.shape[1] == 1:
                return np.hstack([1.0 - probs, probs])
        elif hasattr(self.classifier, "decision_function"):
            decision = self.classifier.decision_function(scaled_features)
            # Apply sigmoid to decision function
            p1 = 1.0 / (1.0 + np.exp(-decision))
            p1 = p1.reshape(-1, 1)
            return np.hstack([1.0 - p1, p1])
        else:
            preds = self.classifier.predict(scaled_features).reshape(-1, 1)
            return np.hstack([1.0 - preds, preds])

        return probs

    def predict_score(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> float:
        """Convenience method returning AI-generated score for a single sample."""
        probs = self.predict_proba(features, feature_names=feature_names)
        return float(probs[0, 1])

    def save(self, path: Union[str, Path]) -> None:
        """Saves bundle to disk using joblib."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClassicalModelBundle":
        """Loads a model bundle from disk."""
        load_path = Path(path)
        if not load_path.is_file():
            raise FileNotFoundError(f"Model bundle not found at: {load_path}")
        bundle = joblib.load(load_path)
        if not isinstance(bundle, cls):
            raise TypeError(f"Loaded object is not a ClassicalModelBundle: {type(bundle)}")
        return bundle

