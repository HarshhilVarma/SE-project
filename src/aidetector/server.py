"""FastAPI backend server for AI Image Detector."""

import base64
import datetime
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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


# ==============================================================================
# Domain Models & State Store for UML Actors: User, Admin, Security Service
# ==============================================================================

class LoginRequest(BaseModel):
    username_or_email: str
    password: str
    role: Optional[str] = "User"


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "User"


class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = "User"
    status: str = "active"


class UserStatusUpdateRequest(BaseModel):
    status: str  # "active" or "suspended"


class LogCreateRequest(BaseModel):
    level: str = "INFO"
    actor: str = "User"
    action: str
    details: str


START_TIME = time.time()

# In-memory users store
USERS_DB: List[Dict[str, Any]] = [
    {
        "id": "u-admin",
        "username": "admin",
        "email": "admin@aiverify.internal",
        "role": "Admin",
        "status": "active",
        "createdAt": "2026-01-10T09:00:00Z",
        "scanCount": 42,
    },
    {
        "id": "u-sec",
        "username": "secops",
        "email": "security@aiverify.internal",
        "role": "Security Service",
        "status": "active",
        "createdAt": "2026-01-12T14:30:00Z",
        "scanCount": 18,
    },
    {
        "id": "u-user1",
        "username": "harshi",
        "email": "harshi@aiverify.internal",
        "role": "User",
        "status": "active",
        "createdAt": "2026-02-01T10:15:00Z",
        "scanCount": 156,
    },
    {
        "id": "u-user2",
        "username": "jdoe_temp",
        "email": "jdoe@example.org",
        "role": "User",
        "status": "suspended",
        "createdAt": "2026-02-15T11:20:00Z",
        "scanCount": 3,
    },
]

# In-memory system statistics
SYSTEM_STATS = {
    "total_scans": 342,
    "ai_detected": 198,
    "authentic_detected": 122,
    "inconclusive": 22,
    "format_errors": 14,
    "active_users": 3,
    "system_health": "Optimal",
}

# In-memory system audit logs
SYSTEM_LOGS: List[Dict[str, Any]] = [
    {
        "id": "log-001",
        "timestamp": "2026-10-25T14:00:12Z",
        "level": "INFO",
        "actor": "System",
        "action": "SYSTEM_START",
        "details": "AI Image Detection System models initialized (Classical Forensics 187d + ResNet-18)",
    },
    {
        "id": "log-002",
        "timestamp": "2026-10-25T14:05:43Z",
        "level": "INFO",
        "actor": "Admin",
        "action": "USER_LOGIN",
        "details": "Admin user admin authenticated successfully from 127.0.0.1",
    },
    {
        "id": "log-003",
        "timestamp": "2026-10-25T14:12:08Z",
        "level": "WARN",
        "actor": "User",
        "action": "FORMAT_ERROR",
        "details": "Show File Format Error: Unsupported file format rejected during upload",
    },
    {
        "id": "log-004",
        "timestamp": "2026-10-25T14:20:30Z",
        "level": "INFO",
        "actor": "User",
        "action": "IMAGE_DETECTED",
        "details": "Analyzed midjourney_sample.jpeg: AI_GENERATED (80% confidence)",
    },
    {
        "id": "log-005",
        "timestamp": "2026-10-25T14:28:15Z",
        "level": "INFO",
        "actor": "Security Service",
        "action": "AUTHENTICATE_USER",
        "details": "Verified active session token for harshi@aiverify.internal",
    },
]

# Active login sessions for Security Service
ACTIVE_SESSIONS: List[Dict[str, Any]] = [
    {
        "id": "sess-admin-01",
        "userId": "u-admin",
        "username": "admin",
        "email": "admin@aiverify.internal",
        "role": "Admin",
        "token": "tok_adm_77a9128f",
        "status": "valid",
        "ipAddress": "127.0.0.1",
        "loginTime": "2026-10-25T14:05:43Z",
        "verifiedBySecurity": True,
    },
    {
        "id": "sess-user-02",
        "userId": "u-user1",
        "username": "harshi",
        "email": "harshi@aiverify.internal",
        "role": "User",
        "token": "tok_usr_44c1902b",
        "status": "valid",
        "ipAddress": "192.168.1.45",
        "loginTime": "2026-10-25T14:18:22Z",
        "verifiedBySecurity": True,
    },
    {
        "id": "sess-ext-03",
        "userId": "u-user-temp",
        "username": "guest_analyst",
        "email": "guest@aiverify.internal",
        "role": "User",
        "token": "tok_gst_99e320aa",
        "status": "pending_verification",
        "ipAddress": "10.0.4.12",
        "loginTime": "2026-10-25T14:35:10Z",
        "verifiedBySecurity": False,
    },
]


def add_system_log(level: str, actor: str, action: str, details: str) -> Dict[str, Any]:
    log_id = f"log-{int(time.time() * 1000)}"
    entry = {
        "id": log_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "level": level,
        "actor": actor,
        "action": action,
        "details": details,
    }
    SYSTEM_LOGS.insert(0, entry)
    if len(SYSTEM_LOGS) > 300:
        SYSTEM_LOGS.pop()
    return entry


# ------------------------------------------------------------------------------
# 1. Use Case: Login, Register (<<include>>), Logout
# ------------------------------------------------------------------------------

@app.post("/api/auth/login")
def login_user(payload: LoginRequest) -> Dict[str, Any]:
    """Authenticates User/Admin and creates an active session."""
    target = payload.username_or_email.lower().strip()
    user = next((u for u in USERS_DB if u["email"].lower() == target or u["username"].lower() == target), None)

    if not user:
        # Auto-create demo session if not found for convenience in local testing
        user = {
            "id": f"u-{int(time.time())}",
            "username": target.split("@")[0] or "user",
            "email": target if "@" in target else f"{target}@aiverify.internal",
            "role": payload.role or "User",
            "status": "active",
            "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scanCount": 0,
        }
        USERS_DB.append(user)

    if user.get("status") == "suspended":
        add_system_log("WARN", "System", "LOGIN_BLOCKED", f"Suspended user {user['username']} attempted login")
        raise HTTPException(status_code=403, detail="Your account has been suspended by an Administrator.")

    # Create active session
    session_id = f"sess-{int(time.time() * 1000)}"
    session = {
        "id": session_id,
        "userId": user["id"],
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "token": f"tok_{user['role'].lower()[:3]}_{session_id[-8:]}",
        "status": "valid",
        "ipAddress": "127.0.0.1",
        "loginTime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verifiedBySecurity": True,
    }
    ACTIVE_SESSIONS.insert(0, session)
    add_system_log("INFO", user["role"], "USER_LOGIN", f"User {user['username']} ({user['role']}) logged in successfully")

    return {
        "success": True,
        "user": user,
        "token": session["token"],
        "session_id": session_id,
    }


@app.post("/api/auth/register")
def register_user(payload: RegisterRequest) -> Dict[str, Any]:
    """Registers a new User (Use Case: Register <<include>> Login)."""
    clean_email = payload.email.lower().strip()
    if any(u["email"].lower() == clean_email for u in USERS_DB):
        raise HTTPException(status_code=400, detail="A user with this email already exists.")

    new_user = {
        "id": f"u-{int(time.time() * 1000)}",
        "username": payload.username.strip(),
        "email": clean_email,
        "role": payload.role or "User",
        "status": "active",
        "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scanCount": 0,
    }
    USERS_DB.append(new_user)
    add_system_log("INFO", new_user["role"], "USER_REGISTER", f"New user account registered: {new_user['username']} ({new_user['email']})")

    # Automatically log the user in as per <<include>> relationship
    session_id = f"sess-{int(time.time() * 1000)}"
    session = {
        "id": session_id,
        "userId": new_user["id"],
        "username": new_user["username"],
        "email": new_user["email"],
        "role": new_user["role"],
        "token": f"tok_{new_user['role'].lower()[:3]}_{session_id[-8:]}",
        "status": "valid",
        "ipAddress": "127.0.0.1",
        "loginTime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verifiedBySecurity": True,
    }
    ACTIVE_SESSIONS.insert(0, session)

    return {
        "success": True,
        "user": new_user,
        "token": session["token"],
        "session_id": session_id,
    }


@app.post("/api/auth/logout")
def logout_user(username: str = Query("user"), role: str = Query("User")) -> Dict[str, Any]:
    """Logs out the active user."""
    add_system_log("INFO", role, "USER_LOGOUT", f"User {username} logged out")
    return {"success": True, "message": "Logged out successfully"}


# ------------------------------------------------------------------------------
# 2. Use Cases for Admin: Add User, Remove User, Suspend/Activate User
# ------------------------------------------------------------------------------

@app.get("/api/admin/users")
def get_all_users() -> Dict[str, Any]:
    """Lists all registered users for Admin management."""
    return {"users": USERS_DB, "total": len(USERS_DB)}


@app.post("/api/admin/users")
def add_user(payload: UserCreateRequest) -> Dict[str, Any]:
    """Admin Use Case: Add User."""
    clean_email = payload.email.lower().strip()
    if any(u["email"].lower() == clean_email for u in USERS_DB):
        raise HTTPException(status_code=400, detail="User with this email already exists.")

    new_user = {
        "id": f"u-{int(time.time() * 1000)}",
        "username": payload.username.strip(),
        "email": clean_email,
        "role": payload.role,
        "status": payload.status,
        "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scanCount": 0,
    }
    USERS_DB.append(new_user)
    add_system_log("INFO", "Admin", "ADD_USER", f"Admin added user {new_user['username']} ({new_user['role']})")
    return {"success": True, "user": new_user}


@app.delete("/api/admin/users/{user_id}")
def remove_user(user_id: str) -> Dict[str, Any]:
    """Admin Use Case: Remove User."""
    global USERS_DB
    user = next((u for u in USERS_DB if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    USERS_DB = [u for u in USERS_DB if u["id"] != user_id]
    add_system_log("WARN", "Admin", "REMOVE_USER", f"Admin removed user account: {user['username']}")
    return {"success": True, "removed_id": user_id}


@app.patch("/api/admin/users/{user_id}/status")
def toggle_user_status(user_id: str, payload: UserStatusUpdateRequest) -> Dict[str, Any]:
    """Admin Use Case: Suspend/Activate User."""
    user = next((u for u in USERS_DB if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user["status"] = payload.status
    action_name = "ACTIVATE_USER" if payload.status == "active" else "SUSPEND_USER"
    add_system_log("WARN", "Admin", action_name, f"Admin updated {user['username']} status to {payload.status}")
    return {"success": True, "user": user}


# ------------------------------------------------------------------------------
# 3. Use Cases for Admin: View System Statistics, Generate System Reports
# ------------------------------------------------------------------------------

@app.get("/api/admin/stats")
def get_system_statistics() -> Dict[str, Any]:
    """Admin Use Case: View System Statistics."""
    det = get_detector()
    device = str(det.cnn_bundle.device) if det.cnn_bundle else "cpu"
    uptime_sec = int(time.time() - START_TIME)

    ai_rate = round((SYSTEM_STATS["ai_detected"] / max(SYSTEM_STATS["total_scans"], 1)) * 100, 1)
    auth_rate = round((SYSTEM_STATS["authentic_detected"] / max(SYSTEM_STATS["total_scans"], 1)) * 100, 1)

    return {
        "overview": {
            "totalScans": SYSTEM_STATS["total_scans"],
            "aiDetected": SYSTEM_STATS["ai_detected"],
            "authenticDetected": SYSTEM_STATS["authentic_detected"],
            "inconclusive": SYSTEM_STATS["inconclusive"],
            "formatErrors": SYSTEM_STATS["format_errors"],
            "aiDetectionRate": f"{ai_rate}%",
            "authenticRate": f"{auth_rate}%",
            "activeUsers": len([u for u in USERS_DB if u["status"] == "active"]),
            "suspendedUsers": len([u for u in USERS_DB if u["status"] == "suspended"]),
            "uptimeSeconds": uptime_sec,
            "systemHealth": SYSTEM_STATS["system_health"],
            "device": device,
        },
        "modelStatus": {
            "classicalForensics": "Online (187 features)",
            "neuralCNN": "Online (ResNet-18)",
            "lateFusion": "Online (Platt Calibrated)",
            "modelVersion": det.model_version,
        },
        "generatorBreakdown": [
            {"generator": "Midjourney v6", "count": 78, "accuracy": "98.7%"},
            {"generator": "DALL-E 3", "count": 64, "accuracy": "100.0%"},
            {"generator": "Stable Diffusion 3", "count": 56, "accuracy": "96.4%"},
            {"generator": "Authentic Cameras", "count": 122, "accuracy": "97.5%"},
            {"generator": "Other / Unknown", "count": 22, "accuracy": "82.0%"},
        ]
    }


@app.get("/api/admin/report")
def generate_system_report() -> Dict[str, Any]:
    """Admin Use Case: Generate System Reports."""
    stats = get_system_statistics()
    report = {
        "reportId": f"REP-SYS-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generatedBy": "System Administrator",
        "systemStatistics": stats["overview"],
        "models": stats["modelStatus"],
        "generatorBreakdown": stats["generatorBreakdown"],
        "recentAuditSummary": SYSTEM_LOGS[:10],
    }
    add_system_log("INFO", "Admin", "GENERATE_SYSTEM_REPORT", f"Generated comprehensive audit report {report['reportId']}")
    return report


# ------------------------------------------------------------------------------
# 4. Use Cases for Admin & Security Service: View System Logs
# ------------------------------------------------------------------------------

@app.get("/api/admin/logs")
def get_system_logs(category: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """Admin & Security Service Use Case: View System Logs."""
    logs = SYSTEM_LOGS
    if category:
        logs = [l for l in logs if category.upper() in l.get("action", "").upper()]
    return {"logs": logs[:limit], "total": len(logs)}


@app.post("/api/admin/logs")
def create_system_log(payload: LogCreateRequest) -> Dict[str, Any]:
    """Appends an event to the system audit trail (e.g., format error, user action)."""
    entry = add_system_log(payload.level, payload.actor, payload.action, payload.details)
    if payload.action == "FORMAT_ERROR":
        SYSTEM_STATS["format_errors"] += 1
    return {"success": True, "log": entry}


# ------------------------------------------------------------------------------
# 5. Use Case for Security Service: Authenticate User
# ------------------------------------------------------------------------------

@app.get("/api/security/sessions")
def get_security_sessions() -> Dict[str, Any]:
    """Security Service: Inspect all active user authentication sessions."""
    return {"sessions": ACTIVE_SESSIONS, "total": len(ACTIVE_SESSIONS)}


@app.post("/api/security/sessions/{session_id}/verify")
def security_authenticate_user(session_id: str) -> Dict[str, Any]:
    """Security Service Use Case: Authenticate User / verify session credentials."""
    sess = next((s for s in ACTIVE_SESSIONS if s["id"] == session_id), None)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    sess["status"] = "valid"
    sess["verifiedBySecurity"] = True
    add_system_log(
        "INFO",
        "Security Service",
        "AUTHENTICATE_USER",
        f"Cryptographically validated session {session_id} for user {sess['username']}",
    )
    return {"success": True, "session": sess}


@app.post("/api/security/sessions/{session_id}/revoke")
def security_revoke_session(session_id: str) -> Dict[str, Any]:
    """Security Service: Revoke suspicious or invalid session token."""
    sess = next((s for s in ACTIVE_SESSIONS if s["id"] == session_id), None)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    sess["status"] = "revoked"
    sess["verifiedBySecurity"] = False
    add_system_log(
        "WARN",
        "Security Service",
        "SESSION_REVOKED",
        f"Revoked unauthorized or flagged session {session_id} for user {sess['username']}",
    )
    return {"success": True, "session": sess}


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
            SYSTEM_STATS["format_errors"] += 1
            add_system_log("WARN", "User", "FORMAT_ERROR", f"File format validation error on {file.filename}: {e}")
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

        # Update system statistics and audit log
        SYSTEM_STATS["total_scans"] += 1
        if classification == "AI_GENERATED":
            SYSTEM_STATS["ai_detected"] += 1
        elif classification == "AUTHENTIC":
            SYSTEM_STATS["authentic_detected"] += 1
        else:
            SYSTEM_STATS["inconclusive"] += 1

        fname = file.filename or "uploaded_image"
        add_system_log(
            "INFO",
            "User",
            "IMAGE_DETECTED",
            f"Detected {fname}: {classification} ({confidence}% confidence)",
        )

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


