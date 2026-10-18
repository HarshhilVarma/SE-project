"""FastAPI backend server for AI Image Detector."""

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional
import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from aidetector.inference import FullDetector
from aidetector.io import load_image
from aidetector.models.classical_model import ClassicalModelBundle
from aidetector.models.cnn import CNNModelBundle
from aidetector.models.fusion import LateFusionModel
from aidetector.standards import check_metadata, check_provenance, check_watermark


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models upon application start."""
    try:
        get_detector()
        print("FullDetector models initialized successfully.")
    except Exception as e:
        print(f"Warning during detector initialization: {e}")
    yield


app = FastAPI(
    title="AI Image Verification API",
    description="Backend API powering AI-generated image verification using Classical Forensics, CNN, and Late Fusion",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for local Vite dev server and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector: Optional[FullDetector] = None


def array_to_data_url(arr: np.ndarray) -> str:
    """Encodes a uint8 numpy array (grayscale or RGB) to a base64 PNG data URL."""
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pil_img = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def get_detector() -> FullDetector:
    """Loads or retrieves the singleton FullDetector instance."""
    global detector
    if detector is not None:
        return detector

    base_dir = Path(__file__).resolve().parents[2]
    c_path = base_dir / "artifacts" / "classical_bundle.joblib"
    n_path = base_dir / "artifacts" / "cnn_resnet18_best.pt"
    f_path = base_dir / "artifacts" / "fusion_model.joblib"

    c_bundle = ClassicalModelBundle.load(c_path) if c_path.is_file() else None
    n_bundle = CNNModelBundle.load(n_path) if n_path.is_file() else None
    f_model = LateFusionModel.load(f_path) if f_path.is_file() else None

    detector = FullDetector(
        classical_bundle=c_bundle,
        cnn_bundle=n_bundle,
        fusion_model=f_model,
        ai_threshold=0.60,
        real_threshold=0.40,
    )
    return detector




@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Returns system status and loaded model configurations."""
    det = get_detector()
    device = str(det.cnn_bundle.device) if det.cnn_bundle else "cpu"
    return {
        "status": "healthy",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": {
            "classical": det.classical_bundle is not None,
            "cnn": det.cnn_bundle is not None,
            "fusion": det.fusion_model is not None,
        },
        "model_version": det.model_version,
    }


@app.post("/api/verify")
async def verify_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Verifies an uploaded image through standards checks and AI visual detection."""
    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

        # 1. Provenance / C2PA check
        provenance = check_provenance(raw_bytes, pil_img)

        # 2. Metadata / EXIF check
        metadata = check_metadata(raw_bytes, pil_img, file.filename or "uploaded_image")

        # 3. Watermark check
        watermark = check_watermark(raw_bytes, pil_img)

        # Check if verified credentials provide conclusive authenticity
        has_usable_evidence = bool(
            (provenance.get("found") and provenance.get("status") == "verified")
            or watermark.get("status") == "found"
        )

        # 4. Visual AI Detection
        det = get_detector()
        pred_res = det.predict_image(raw_bytes, generate_visualizations=True)

        raw_pred = pred_res.get("prediction", "uncertain")
        ai_score = pred_res.get("ai_score")

        if raw_pred == "likely_ai_generated":
            classification = "AI_GENERATED"
            confidence = int(round((ai_score or 0.85) * 100))
            summary = (
                f"Visual forensic analysis and neural patterns indicate synthetic generation "
                f"({confidence}% confidence). Statistical anomalies detected across color and texture spectra."
            )
        elif raw_pred == "likely_real":
            classification = "AUTHENTIC"
            confidence = int(round((1.0 - (ai_score or 0.15)) * 100))
            summary = (
                f"Visual forensics and spectral analysis found characteristics consistent with "
                f"authentic photographic capture ({confidence}% confidence)."
            )
        else:
            classification = "INCONCLUSIVE"
            confidence = 50
            summary = (
                "The available visual evidence was insufficient for confident classification. "
                "Confidence is near threshold or branch agreement is inconclusive."
            )

        # Visual artifacts (Grad-CAM & ELA maps)
        heatmap_url = None
        ela_url = None

        if det.gradcam is not None and det.cnn_bundle is not None:
            try:
                rgb_arr, _ = load_image(raw_bytes)
                from aidetector.models.cnn import get_inference_transform
                transform = get_inference_transform(224)
                tensor = transform(Image.fromarray(rgb_arr))
                cam = det.gradcam.generate_heatmap(tensor, target_class=1)
                overlay, _ = det.gradcam.overlay_heatmap(rgb_arr, cam, alpha=0.5)
                heatmap_url = array_to_data_url(overlay)
            except Exception as e:
                print(f"Warning: Failed to render Grad-CAM overlay: {e}")

        # ELA map
        try:
            rgb_arr, meta = load_image(raw_bytes)
            from aidetector.classical.ela import compute_ela
            _, _, ela_map = compute_ela(rgb_arr, is_jpeg=meta.is_jpeg)
            ela_url = array_to_data_url(ela_map)
        except Exception as e:
            print(f"Warning: Failed to render ELA map: {e}")

        raw_branches = pred_res.get("branch_scores", {})
        branch_scores = {
            "classical": raw_branches.get("classical"),
            "classical_probability": raw_branches.get("classical"),
            "cnn_rgb": raw_branches.get("cnn_rgb"),
            "cnn_probability": raw_branches.get("cnn_rgb"),
            "fusion": pred_res.get("ai_score"),
            "fusion_probability": pred_res.get("ai_score"),
        }

        ai_result = {
            "classification": classification,
            "confidence": confidence,
            "heatmapAvailable": heatmap_url is not None,
            "heatmapUrl": heatmap_url,
            "elaUrl": ela_url,
            "summary": summary,
            "branchScores": branch_scores,
            "qualityFlags": pred_res.get("quality_flags", []),
            "modelVersion": pred_res.get("model_version", "aid-ab-v1"),
            "limitations": pred_res.get("limitations", []),
        }

        return {
            "provenance": provenance,
            "metadata": metadata,
            "watermark": watermark,
            "aiResult": ai_result,
            "hasUsableEvidence": has_usable_evidence,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal verification error: {str(e)}")


# Mount pre-built React frontend if dist/ directory exists
dist_dir = Path(__file__).resolve().parents[2] / "dist"
if dist_dir.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")


