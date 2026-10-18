"""Tests for aidetector.models.fusion module."""

from pathlib import Path
import numpy as np
import pytest
from aidetector.models.fusion import LateFusionModel


def test_late_fusion_fit_predict_save_load(tmp_path: Path):
    s_c = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3], dtype=np.float32)
    s_n = np.array([0.85, 0.95, 0.15, 0.05, 0.6, 0.4], dtype=np.float32)
    y = np.array([1, 1, 0, 0, 1, 0], dtype=np.int32)

    fusion = LateFusionModel()
    fusion.fit(s_c, s_n, y, calibrate=True)

    score = fusion.predict_score(0.85, 0.9)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    save_path = tmp_path / "fusion.joblib"
    fusion.save(save_path)

    loaded = LateFusionModel.load(save_path)
    loaded_score = loaded.predict_score(0.85, 0.9)
    assert loaded_score == pytest.approx(score)


def test_late_fusion_unfitted():
    fusion = LateFusionModel()
    with pytest.raises(RuntimeError):
        fusion.predict_score(0.5, 0.5)

