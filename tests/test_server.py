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


def test_uml_use_cases_auth_flow():
    """Tests Login, Register (<<include>>), and Logout."""
    client = TestClient(app)

    # 1. Register
    reg_resp = client.post("/api/auth/register", json={
        "username": "testanalyst",
        "email": "testanalyst@example.com",
        "password": "SecretPassword123!",
        "role": "User"
    })
    assert reg_resp.status_code == 200
    reg_data = reg_resp.json()
    assert reg_data["success"] is True
    assert reg_data["user"]["username"] == "testanalyst"

    # 2. Login
    login_resp = client.post("/api/auth/login", json={
        "username_or_email": "testanalyst@example.com",
        "password": "SecretPassword123!",
        "role": "User"
    })
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert login_data["success"] is True
    assert "token" in login_data

    # 3. Logout
    logout_resp = client.post("/api/auth/logout?username=testanalyst&role=User")
    assert logout_resp.status_code == 200


def test_uml_use_cases_admin_user_management():
    """Tests Admin: Add User, Suspend/Activate User, and Remove User."""
    client = TestClient(app)

    # 1. Add User
    add_resp = client.post("/api/admin/users", json={
        "username": "newmoderator",
        "email": "newmod@example.com",
        "password": "Password99!",
        "role": "Admin",
        "status": "active"
    })
    assert add_resp.status_code == 200
    user_id = add_resp.json()["user"]["id"]

    # 2. Suspend User
    suspend_resp = client.patch(f"/api/admin/users/{user_id}/status", json={"status": "suspended"})
    assert suspend_resp.status_code == 200
    assert suspend_resp.json()["user"]["status"] == "suspended"

    # Suspended user login should be blocked
    blocked_login = client.post("/api/auth/login", json={
        "username_or_email": "newmod@example.com",
        "password": "Password99!"
    })
    assert blocked_login.status_code == 403

    # Reactivate User
    activate_resp = client.patch(f"/api/admin/users/{user_id}/status", json={"status": "active"})
    assert activate_resp.status_code == 200
    assert activate_resp.json()["user"]["status"] == "active"

    # 3. Remove User
    delete_resp = client.delete(f"/api/admin/users/{user_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["removed_id"] == user_id


def test_uml_use_cases_admin_statistics_and_reports():
    """Tests Admin: View System Statistics and Generate System Reports."""
    client = TestClient(app)

    # 1. View System Statistics
    stats_resp = client.get("/api/admin/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert "overview" in stats
    assert "totalScans" in stats["overview"]
    assert "modelStatus" in stats

    # 2. Generate System Reports
    report_resp = client.get("/api/admin/report")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert "reportId" in report
    assert "systemStatistics" in report


def test_uml_use_cases_security_service_and_logs():
    """Tests Security Service: Authenticate User and View System Logs."""
    client = TestClient(app)

    # 1. View System Logs (Admin & Security Service)
    logs_resp = client.get("/api/admin/logs")
    assert logs_resp.status_code == 200
    logs = logs_resp.json()
    assert "logs" in logs
    assert len(logs["logs"]) > 0

    # 2. Security Service: Inspect & Authenticate User Session
    sessions_resp = client.get("/api/security/sessions")
    assert sessions_resp.status_code == 200
    sessions = sessions_resp.json()["sessions"]
    assert len(sessions) > 0

    target_sess = sessions[0]["id"]
    verify_resp = client.post(f"/api/security/sessions/{target_sess}/verify")
    assert verify_resp.status_code == 200
    assert verify_resp.json()["session"]["verifiedBySecurity"] is True

    # Revoke Session
    revoke_resp = client.post(f"/api/security/sessions/{target_sess}/revoke")
    assert revoke_resp.status_code == 200
    assert revoke_resp.json()["session"]["status"] == "revoked"


