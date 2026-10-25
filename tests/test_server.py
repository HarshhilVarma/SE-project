"""Tests for aidetector.server FastAPI application."""

import io
from PIL import Image
from starlette.testclient import TestClient
from aidetector.server import app


def test_api_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "models_loaded" in data


def test_api_verify_endpoint_with_image():
    client = TestClient(app)

    # Create dummy JPEG image in memory
    img = Image.new("RGB", (64, 64), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/verify",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    assert "provenance" in data
    assert "metadata" in data
    assert "watermark" in data
    assert "aiResult" in data
    assert "hasUsableEvidence" in data

    ai = data["aiResult"]
    assert ai["classification"] in ("AI_GENERATED", "AUTHENTIC", "INCONCLUSIVE")
    assert 0 <= ai["confidence"] <= 100
    assert "branchScores" in ai
    assert "classical_probability" in ai["branchScores"]
    assert "cnn_probability" in ai["branchScores"]
    assert "fusion_probability" in ai["branchScores"]
    assert ai["heatmapUrl"] is not None
    assert ai["heatmapUrl"].startswith("data:image/png;base64,")
    assert ai["elaUrl"] is not None
    assert ai["elaUrl"].startswith("data:image/png;base64,")


def test_api_verify_endpoint_with_real_sample():
    from pathlib import Path
    sample_path = Path("/home/harshhil/SE-project/data/images/train/train_1_midjourney_ab63ddc911d1cf51.jpeg")
    if not sample_path.is_file():
        return

    client = TestClient(app)
    with open(sample_path, "rb") as f:
        response = client.post(
            "/api/verify",
            files={"file": ("sample.jpeg", f, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["aiResult"]["classification"] == "AI_GENERATED"
    assert data["aiResult"]["confidence"] >= 60

