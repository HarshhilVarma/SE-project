"""Probability calibration using Platt scaling and Isotonic regression."""

from dataclasses import dataclass
from typing import Literal, Optional, Union
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    """Calibrates uncalibrated model scores into true empirical probabilities.

    Supports:
        - 'platt': Platt scaling (logistic sigmoid fit)
        - 'isotonic': Non-parametric isotonic regression
    """

    method: Literal["platt", "isotonic"] = "platt"
    calibrator: Optional[Union[LogisticRegression, IsotonicRegression]] = None

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """Fits calibrator on held-out validation scores."""
        scores_arr = np.asarray(scores, dtype=np.float64).ravel()
        y_arr = np.asarray(y_true, dtype=np.int32).ravel()

        if self.method == "platt":
            clf = LogisticRegression(C=1.0, solver="lbfgs")
            clf.fit(scores_arr.reshape(-1, 1), y_arr)
            self.calibrator = clf
        elif self.method == "isotonic":
            reg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            reg.fit(scores_arr, y_arr)
            self.calibrator = reg
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Transforms scores into calibrated probabilities."""
        if self.calibrator is None:
            raise RuntimeError("Calibrator is not fitted yet.")

        scores_arr = np.asarray(scores, dtype=np.float64)
        flat_scores = scores_arr.ravel()

        if self.method == "platt":
            probs = self.calibrator.predict_proba(flat_scores.reshape(-1, 1))[:, 1]
        else:
            probs = self.calibrator.predict(flat_scores)

        return np.clip(probs, 0.0, 1.0).reshape(scores_arr.shape)

