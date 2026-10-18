"""Unified inference contract and three-way decision policy."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from aidetector.classical.features import extract_classical_features
from aidetector.config import ClassicalConfig
from aidetector.explain.gradcam import GradCAM
from aidetector.io import load_image
from aidetector.models.classical_model import ClassicalModelBundle
from aidetector.models.cnn import CNNModelBundle, get_inference_transform
from aidetector.models.fusion import LateFusionModel
from aidetector.quality import assess_image_quality


LIMITATIONS = [
    "This model output is a statistical estimation and not forensic or legal proof of image authenticity.",
    "Detection accuracy can vary on generative models unseen during training.",
    "Error Level Analysis is most directly informative on native JPEG images.",
]


class FullDetector:
    """Integrated dual-branch detector with classical, CNN, and fusion capabilities."""

    def __init__(
        self,
        classical_bundle: Optional[ClassicalModelBundle] = None,
        cnn_bundle: Optional[CNNModelBundle] = None,
        fusion_model: Optional[LateFusionModel] = None,
        classical_config: Optional[ClassicalConfig] = None,
        ai_threshold: float = 0.60,
        real_threshold: float = 0.40,
        disagreement_threshold: float = 0.45,
        model_version: str = "aid-ab-v1",
    ):
        self.classical_bundle = classical_bundle
        self.cnn_bundle = cnn_bundle
        self.fusion_model = fusion_model
        self.classical_config = classical_config or ClassicalConfig()
        self.ai_threshold = ai_threshold
        self.real_threshold = real_threshold
        self.disagreement_threshold = disagreement_threshold
        self.model_version = model_version

        self.gradcam: Optional[GradCAM] = None
        if self.cnn_bundle is not None:
            try:
                self.gradcam = GradCAM(self.cnn_bundle.model, device=self.cnn_bundle.device)
            except Exception:
                self.gradcam = None

    def predict_image(
        self,
        source: Union[str, Path, bytes, np.ndarray, Image.Image],
        generate_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """Classifies an image using the unified inference contract."""
        img_arr, metadata = load_image(source)
        quality_flags = assess_image_quality(metadata)

        # 1. Classical branch
        s_classical: Optional[float] = None
        visualizations: Dict[str, Any] = {}
        if self.classical_bundle is not None:
            feats, names, visual_maps = extract_classical_features(
                img_arr, config=self.classical_config, is_jpeg=metadata.is_jpeg
            )
            s_classical = self.classical_bundle.predict_score(feats, feature_names=names)
            if generate_visualizations:
                visualizations = visual_maps

        # 2. CNN branch
        s_cnn: Optional[float] = None
        gradcam_heatmap = None
        if self.cnn_bundle is not None:
            pil_img = Image.fromarray(img_arr, mode="RGB")
            transform = get_inference_transform(input_size=224)
            tensor = transform(pil_img)

            s_cnn = self.cnn_bundle.predict_score(tensor)

            if generate_visualizations and self.gradcam is not None:
                try:
                    gradcam_heatmap = self.gradcam.generate_heatmap(tensor, target_class=1)
                except Exception:
                    gradcam_heatmap = None

        # 3. Combined / Fusion score
        ai_score: Optional[float] = None
        score_type = "uncalibrated_model_score"

        if self.fusion_model is not None and s_classical is not None and s_cnn is not None:
            ai_score = float(self.fusion_model.predict_score(s_classical, s_cnn))
            score_type = "calibrated_probability"
        elif s_classical is not None and s_cnn is not None:
            ai_score = float(0.5 * (s_classical + s_cnn))
        elif s_classical is not None:
            ai_score = float(s_classical)
        elif s_cnn is not None:
            ai_score = float(s_cnn)

        # 4. Decision policy (three-way)
        prediction = "uncertain"
        if ai_score is not None:
            branch_disagreement = False
            if s_classical is not None and s_cnn is not None:
                if abs(s_classical - s_cnn) > self.disagreement_threshold:
                    branch_disagreement = True

            if branch_disagreement:
                prediction = "uncertain"
            elif "low_resolution" in quality_flags:
                prediction = "uncertain"
            elif ai_score >= self.ai_threshold:
                prediction = "likely_ai_generated"
            elif ai_score <= self.real_threshold:
                prediction = "likely_real"
            else:
                prediction = "uncertain"

        # Build response artifact references
        artifacts = {
            "ela_map": visualizations.get("ela_map"),
            "color_difference_map": visualizations.get("color_difference_map"),
            "residual_map": visualizations.get("residual_map"),
            "gradcam": gradcam_heatmap,
        }

        return {
            "prediction": prediction,
            "ai_score": round(ai_score, 4) if ai_score is not None else None,
            "score_type": score_type,
            "thresholds": {
                "real_threshold": self.real_threshold,
                "ai_threshold": self.ai_threshold,
            },
            "model_version": self.model_version,
            "quality_flags": quality_flags,
            "branch_scores": {
                "classical": round(s_classical, 4) if s_classical is not None else None,
                "cnn_rgb": round(s_cnn, 4) if s_cnn is not None else None,
                "cnn_forensic": None,
            },
            "artifacts": {k: (v is not None) for k, v in artifacts.items()},
            "limitations": LIMITATIONS,
        }

