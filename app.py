"""
DeepForensics — FastAPI Backend Server
=======================================
Full-stack image forgery detection web application backend.
Provides REST API endpoints for image upload, ELA generation,
model inference, and scan history management.

Usage:
    python app.py
    # or
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.requests import Request

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from database import ScanRecord, get_db, init_db
from utils.ela import convert_to_ela_image

# Try to import TensorFlow (optional for fallback mode)
try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    keras = None


# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = Path("ela_resnet50_best.keras")
UPLOAD_DIR = Path("static/uploads")
ELA_DIR = Path("static/ela")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ELA_DIR.mkdir(parents=True, exist_ok=True)

# ── FastAPI App Setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="DeepForensics",
    description="Image Forgery Detection using Error Level Analysis & Deep Learning",
    version="1.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")


# ── Model Loader (Singleton) ─────────────────────────────────────────────────

class ModelManager:
    """Singleton model loader with fallback heuristic mode."""

    def __init__(self):
        self.model: Optional[keras.Model] = None
        self.mode: str = "uninitialized"
        self._load_model()

    def _load_model(self) -> None:
        """Try to load trained model, fallback to heuristic."""
        if not TF_AVAILABLE:
            self.mode = "heuristic"
            print("[MODEL] TensorFlow not available — using ELA variance heuristic")
            return

        if MODEL_PATH.exists():
            try:
                self.model = keras.models.load_model(MODEL_PATH)
                self.mode = "model"
                print(f"[MODEL] Loaded trained model from {MODEL_PATH}")
            except Exception as e:
                self.mode = "heuristic"
                print(f"[MODEL] Failed to load model: {e}")
                print("[MODEL] Falling back to ELA variance heuristic")
        else:
            self.mode = "heuristic"
            print(f"[MODEL] Model not found at {MODEL_PATH}")
            print("[MODEL] Using ELA variance heuristic (train model with: python train_model.py)")

    def predict(self, ela_image) -> tuple[str, float]:
        """Run inference and return (prediction, confidence).

        Args:
            ela_image: PIL Image (224x224 RGB)

        Returns:
            prediction: "Authentic" or "Tampered"
            confidence: float in [0, 1]
        """
        if self.mode == "model":
            # Model inference
            img_array = np.asarray(ela_image, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            pred_proba = self.model.predict(img_array, verbose=0)[0][0]
            confidence = float(pred_proba)

            if pred_proba >= 0.5:
                prediction = "Tampered"
            else:
                prediction = "Authentic"
                confidence = 1.0 - confidence  # Flip confidence for authentic

        else:
            # Fallback heuristic: high ELA variance indicates tampering
            img_array = np.asarray(ela_image, dtype=np.float32)
            variance = np.var(img_array)

            # Threshold tuned empirically (you can adjust)
            threshold = 1000.0

            if variance > threshold:
                prediction = "Tampered"
                confidence = min(variance / (threshold * 2), 1.0)
            else:
                prediction = "Authentic"
                confidence = 1.0 - min(variance / threshold, 1.0)

        return prediction, confidence

    def get_status(self) -> dict:
        """Return model status for UI badge."""
        return {
            "mode": self.mode,
            "model_available": self.mode == "model",
            "model_path": str(MODEL_PATH) if self.mode == "model" else None,
        }


# Initialize model manager
model_manager = ModelManager()


# ── Startup Event ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize database on app startup."""
    init_db()
    print("[APP] DeepForensics server started")


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render main dashboard."""
    model_status = model_manager.get_status()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"model_status": model_status},
    )


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    quality: int = Form(90),
    db: Session = Depends(get_db),
):
    """Analyze uploaded image for forgery.

    Args:
        file: Uploaded image file
        quality: JPEG quality for ELA (70-98, default 90)
        db: Database session

    Returns:
        JSON with scan results
    """
    # Validate quality
    if not 70 <= quality <= 98:
        raise HTTPException(status_code=400, detail="Quality must be between 70 and 98")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Generate unique filename
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            raise HTTPException(status_code=400, detail="Unsupported image format")

        unique_id = uuid.uuid4().hex[:12]
        safe_filename = f"{unique_id}_{file.filename}"
        original_path = UPLOAD_DIR / safe_filename
        ela_filename = f"{unique_id}_{Path(file.filename).stem}_ela.png"
        ela_path = ELA_DIR / ela_filename

        # Save original image
        content = await file.read()
        with open(original_path, "wb") as f:
            f.write(content)

        # Generate ELA
        ela_image = convert_to_ela_image(original_path, quality=quality)
        ela_image.save(ela_path, format="PNG")

        # Run inference
        prediction, confidence = model_manager.predict(ela_image)

        # Store in database
        scan_record = ScanRecord(
            filename=file.filename,
            original_url=f"/static/uploads/{safe_filename}",
            ela_url=f"/static/ela/{ela_filename}",
            prediction=prediction,
            confidence=confidence,
            ela_quality=quality,
            timestamp=datetime.utcnow(),
        )
        db.add(scan_record)
        db.commit()
        db.refresh(scan_record)

        return scan_record.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/history")
async def get_history(
    limit: int = 25,
    db: Session = Depends(get_db),
):
    """Get recent scan history.

    Args:
        limit: Maximum number of records to return (default 25)
        db: Database session

    Returns:
        JSON array of scan records
    """
    records = (
        db.query(ScanRecord)
        .order_by(ScanRecord.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [rec.to_dict() for rec in records]


@app.delete("/api/history/{scan_id}")
async def delete_scan(
    scan_id: int,
    db: Session = Depends(get_db),
):
    """Delete a scan record and associated files.

    Args:
        scan_id: Scan record ID
        db: Database session

    Returns:
        Success message
    """
    record = db.query(ScanRecord).filter(ScanRecord.id == scan_id).first()

    if not record:
        raise HTTPException(status_code=404, detail="Scan record not found")

    # Delete associated files
    try:
        original_path = Path(record.original_url.lstrip("/"))
        ela_path = Path(record.ela_url.lstrip("/"))

        if original_path.exists():
            original_path.unlink()

        if ela_path.exists():
            ela_path.unlink()

    except Exception as e:
        print(f"[WARN] Failed to delete files for scan {scan_id}: {e}")

    # Delete database record
    db.delete(record)
    db.commit()

    return {"message": "Scan deleted successfully", "id": scan_id}


@app.get("/api/status")
async def get_status(db: Session = Depends(get_db)):
    """Get application status.

    Returns:
        JSON with model status and database stats
    """
    total_scans = db.query(ScanRecord).count()
    authentic_scans = db.query(ScanRecord).filter(ScanRecord.prediction == "Authentic").count()
    tampered_scans = db.query(ScanRecord).filter(ScanRecord.prediction == "Tampered").count()

    return {
        "model_status": model_manager.get_status(),
        "database": {
            "total_scans": total_scans,
            "authentic": authentic_scans,
            "tampered": tampered_scans,
        },
    }


# ── Main Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  DeepForensics - Starting Server")
    print("=" * 60)
    print(f"  Model mode: {model_manager.mode}")
    print("  Visit: http://localhost:8000")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
