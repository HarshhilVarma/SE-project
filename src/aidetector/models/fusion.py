"""Late fusion classifier combining Classical forensics and CNN branch probabilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from aidetector.models.calibration import ProbabilityCalibrator


@dataclass
class LateFusionModel:
    """Late fusion model combining Branch A (classical) and Branch B (CNN) scores.

    Takes:
        - s_classical: classical branch probability in [0, 1]
        - s_cnn: CNN branch probability in [0, 1]
    Constructs features:
        [s_classical, s_cnn, |s_classical - s_cnn|]
    Applies calibrated logistic regression to produce unified score.
    """

    classifier: Optional[LogisticRegression] = None
    calibrator: Optional[ProbabilityCalibrator] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _build_features(
        self,
        s_classical: Union[float, np.ndarray],
        s_cnn: Union[float, np.ndarray],
    ) -> np.ndarray:
        c_arr = np.atleast_1d(np.asarray(s_classical, dtype=np.float32))
        n_arr = np.atleast_1d(np.asarray(s_cnn, dtype=np.float32))
        diff = np.abs(c_arr - n_arr)
        return np.column_stack([c_arr, n_arr, diff])

    def fit(
        self,
        s_classical: np.ndarray,
        s_cnn: np.ndarray,
        y_true: np.ndarray,
        calibrate: bool = True,
        calibration_method: str = "platt",
    ) -> "LateFusionModel":
        """Fits late fusion classifier and optional calibrator."""
        X = self._build_features(s_classical, s_cnn)
        y = np.asarray(y_true, dtype=np.int32)

        clf = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
        clf.fit(X, y)
        self.classifier = clf

        raw_probs = clf.predict_proba(X)[:, 1]

        if calibrate:
            self.calibrator = ProbabilityCalibrator(method=calibration_method)
            self.calibrator.fit(raw_probs, y)

        return self

    def predict_score(
        self,
        s_classical: Union[float, np.ndarray],
        s_cnn: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Predicts unified AI-generated probability score."""
        if self.classifier is None:
            raise RuntimeError("Late fusion model is not fitted.")

        X = self._build_features(s_classical, s_cnn)
        raw_prob = self.classifier.predict_proba(X)[:, 1]

        if self.calibrator is not None:
            cal_prob = self.calibrator.calibrate(raw_prob)
            return float(cal_prob[0]) if len(cal_prob) == 1 else cal_prob

        return float(raw_prob[0]) if len(raw_prob) == 1 else raw_prob

    def save(self, path: Union[str, Path]) -> None:
        """Saves late fusion model to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LateFusionModel":
        """Loads late fusion model from disk."""
        load_path = Path(path)
        if not load_path.is_file():
            raise FileNotFoundError(f"Late fusion model not found at: {load_path}")
        model = joblib.load(load_path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not LateFusionModel: {type(model)}")
        return model

