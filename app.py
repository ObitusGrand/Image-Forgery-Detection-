"""
DeepForensics — FastAPI Backend Server (v2 Multi-Model Ensemble + XAI)
======================================================================
Full-stack deep learning image forgery detection backend supporting:
    - Multi-Branch model (`ela_multibranch_best.keras`: Multi-ELA + BayarConv2D + FFT + CBAM)
    - Legacy ResNet50 model (`ela_resnet50_v3.keras` or `ela_resnet50_best.keras`)
    - Calibrated Weighted Ensemble (`ensemble_config.json` with Temperature Scaling)
    - Optional 4-View Lossless Test-Time Augmentation (TTA)
    - Explainable AI: Grad-CAM++ localization heatmap overlay per scan
    - Graceful fallback to ELA variance heuristic when no trained weights exist

Usage:
    python app.py
    # or
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
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
from PIL import Image
from sqlalchemy.orm import Session
from starlette.requests import Request

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from database import ScanRecord, get_db, init_db
from utils.calibration import (
    DEFAULT_ENSEMBLE_CONFIG_PATH,
    EnsembleConfig,
    apply_temperature,
)
from utils.ela import convert_to_ela_image
from utils.features import extract_multibranch_inputs

try:
    from tensorflow import keras
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
    from utils.custom_layers import get_custom_objects
    from utils.xai import compute_gradcam_pp, overlay_heatmap, predict_with_tta

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    keras = None
    resnet50_preprocess = None


# ── Configuration ────────────────────────────────────────────────────────────

NEW_MODEL_PATH = Path("ela_multibranch_best.keras")
LEGACY_MODEL_CANDIDATES = (Path("ela_resnet50_v3.keras"), Path("ela_resnet50_best.keras"))
ENSEMBLE_CONFIG_PATH = DEFAULT_ENSEMBLE_CONFIG_PATH

UPLOAD_DIR = Path("static/uploads")
ELA_DIR = Path("static/ela")
GRADCAM_DIR = Path("static/gradcam")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ELA_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


# ── FastAPI App Setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="DeepForensics",
    description="Multi-Branch Deep Learning & Calibrated Ensemble Image Forgery Detection",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Multi-Model Ensemble Manager (Singleton) ─────────────────────────────────

def _read_meta(path: Path) -> dict:
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


class ModelManager:
    """Singleton loader managing Multi-Branch, Legacy ResNet50, and Weighted Ensemble."""

    def __init__(self) -> None:
        self.new_model: Optional[keras.Model] = None
        self.new_model_path: Optional[Path] = None
        self.new_meta: dict = {}

        self.legacy_model: Optional[keras.Model] = None
        self.legacy_model_path: Optional[Path] = None
        self.legacy_meta: dict = {}

        self.ensemble_config: EnsembleConfig = EnsembleConfig.load(ENSEMBLE_CONFIG_PATH)
        self.mode: str = "heuristic"
        self._load_models()

    def _load_models(self) -> None:
        if not TF_AVAILABLE:
            self.mode = "heuristic"
            print("[MODEL] TensorFlow not available — using ELA variance heuristic")
            return

        custom_objs = get_custom_objects()

        # 1. Load New Multi-Branch Model if present (compile=False for fast inference)
        if NEW_MODEL_PATH.exists():
            try:
                self.new_model = keras.models.load_model(
                    NEW_MODEL_PATH, custom_objects=custom_objs, compile=False
                )
                self.new_model_path = NEW_MODEL_PATH
                self.new_meta = _read_meta(NEW_MODEL_PATH)
                print(f"[MODEL] Loaded Multi-Branch model: {NEW_MODEL_PATH}")
            except Exception as exc:
                print(f"[MODEL] Failed to load Multi-Branch model ({NEW_MODEL_PATH}): {exc}")

        # 2. Load Legacy ResNet50 Model (prefer fixed v3, fall back to best)
        for cand in LEGACY_MODEL_CANDIDATES:
            if cand.exists():
                try:
                    self.legacy_model = keras.models.load_model(
                        cand, custom_objects=custom_objs, compile=False
                    )
                    self.legacy_model_path = cand
                    self.legacy_meta = _read_meta(cand)
                    print(
                        f"[MODEL] Loaded Legacy ResNet50 model: {cand} "
                        f"(preprocessing={self.legacy_meta.get('preprocessing', 'rescale_255')})"
                    )
                    break
                except Exception as exc:
                    print(f"[MODEL] Failed to load Legacy model ({cand}): {exc}")

        # 3. Determine active default operating mode
        if self.new_model is not None and self.legacy_model is not None:
            self.mode = self.ensemble_config.default_mode
        elif self.new_model is not None:
            self.mode = "new_only"
        elif self.legacy_model is not None:
            self.mode = "legacy_only"
        else:
            self.mode = "heuristic"
            print("[MODEL] No trained weights found — running ELA variance heuristic.")

    def _prepare_legacy_input(self, ela_image: Image.Image) -> np.ndarray:
        arr = np.asarray(ela_image, dtype=np.float32)
        if self.legacy_meta.get("preprocessing") == "resnet50_caffe" and resnet50_preprocess is not None:
            arr = resnet50_preprocess(arr)
        else:
            arr = arr / 255.0
        return arr[None, ...]

    def _prepare_new_input(self, original_path: Path) -> dict[str, np.ndarray]:
        include_fft = bool(self.new_meta.get("include_fft", True))
        ela_only = bool(self.new_meta.get("ela_only", False))
        img_size = tuple(self.new_meta.get("img_size", (224, 224)))
        bundle = extract_multibranch_inputs(
            original_path,
            target_size=img_size,
            include_fft=(include_fft and not ela_only),
        )
        if ela_only:
            bundle = {"multi_ela_input": bundle["multi_ela_input"]}
        return {k: v[None, ...] for k, v in bundle.items()}

    def _resolve_requested_mode(self, requested: str) -> str:
        req = (requested or "auto").lower().strip()
        has_new = self.new_model is not None
        has_leg = self.legacy_model is not None

        if req in ("auto", "default"):
            return self.mode
        if req == "ensemble":
            if has_new and has_leg:
                return "ensemble"
            if has_new:
                return "new_only"
            if has_leg:
                return "legacy_only"
            return "heuristic"
        if req in ("new", "new_only", "multibranch"):
            if has_new:
                return "new_only"
            if has_leg:
                return "legacy_only"
            return "heuristic"
        if req in ("legacy", "legacy_only", "resnet50"):
            if has_leg:
                return "legacy_only"
            if has_new:
                return "new_only"
            return "heuristic"
        return self.mode

    def predict(
        self,
        original_path: Path,
        ela_image: Image.Image,
        requested_mode: str = "auto",
        use_tta: bool = False,
        gradcam_out_path: Optional[Path] = None,
    ) -> dict:
        """Run inference in the selected mode and optionally generate Grad-CAM++ overlay."""
        effective_mode = self._resolve_requested_mode(requested_mode)
        cfg = self.ensemble_config

        p_new_cal: Optional[float] = None
        p_leg_cal: Optional[float] = None
        p_ens: Optional[float] = None
        gradcam_generated = False

        new_inputs: Optional[dict[str, np.ndarray]] = None
        leg_input: Optional[np.ndarray] = None

        if effective_mode in ("ensemble", "new_only") and self.new_model is not None:
            new_inputs = self._prepare_new_input(original_path)
            p_new_raw = float(predict_with_tta(self.new_model, new_inputs, use_tta=use_tta)[0])
            p_new_cal = float(apply_temperature(np.array([p_new_raw]), cfg.temp_new)[0])

        if effective_mode in ("ensemble", "legacy_only") and self.legacy_model is not None:
            leg_input = self._prepare_legacy_input(ela_image)
            p_leg_raw = float(predict_with_tta(self.legacy_model, leg_input, use_tta=use_tta)[0])
            p_leg_cal = float(apply_temperature(np.array([p_leg_raw]), cfg.temp_legacy)[0])

        # Determine final tampered probability & threshold
        if effective_mode == "ensemble" and p_new_cal is not None and p_leg_cal is not None:
            p_ens = float(np.clip(cfg.weight_new * p_new_cal + cfg.weight_legacy * p_leg_cal, 0.0, 1.0))
            final_prob = p_ens
            threshold = cfg.threshold
        elif effective_mode == "new_only" and p_new_cal is not None:
            final_prob = p_new_cal
            threshold = 0.50
        elif effective_mode == "legacy_only" and p_leg_cal is not None:
            final_prob = p_leg_cal
            threshold = 0.50
        else:
            # Heuristic fallback
            effective_mode = "heuristic"
            img_array = np.asarray(ela_image, dtype=np.float32)
            variance = float(np.var(img_array))
            heur_thresh = 1000.0
            final_prob = float(np.clip(variance / (heur_thresh * 2.0), 0.0, 1.0))
            threshold = 0.50

        if final_prob >= threshold:
            prediction = "Tampered"
            confidence = final_prob
        else:
            prediction = "Authentic"
            confidence = 1.0 - final_prob

        # Explainable AI: compute Grad-CAM++ on the primary active deep model
        if TF_AVAILABLE and gradcam_out_path is not None and effective_mode != "heuristic":
            try:
                with Image.open(original_path) as orig_im:
                    orig_rgb = orig_im.convert("RGB")
                    if self.new_model is not None and new_inputs is not None:
                        layer_hint = self.new_meta.get("gradcam_layer")
                        hm = compute_gradcam_pp(self.new_model, new_inputs, last_conv_layer_name=layer_hint)
                    elif self.legacy_model is not None and leg_input is not None:
                        hm = compute_gradcam_pp(self.legacy_model, leg_input)
                    else:
                        hm = None

                    if hm is not None:
                        ov = overlay_heatmap(orig_rgb, hm)
                        ov.save(gradcam_out_path, format="PNG")
                        gradcam_generated = True
            except Exception as exc:
                print(f"[XAI] Grad-CAM++ skipped for {original_path.name}: {exc}")

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "effective_mode": effective_mode,
            "p_new": p_new_cal,
            "p_legacy": p_leg_cal,
            "p_ensemble": p_ens,
            "tta_used": bool(use_tta and effective_mode != "heuristic"),
            "gradcam_generated": gradcam_generated,
        }

    def get_status(self) -> dict:
        has_new = self.new_model is not None
        has_leg = self.legacy_model is not None
        return {
            "mode": self.mode,
            "model_available": has_new or has_leg,
            "new_model_available": has_new,
            "legacy_model_available": has_leg,
            "ensemble_available": has_new and has_leg,
            "new_model_path": str(self.new_model_path) if has_new else None,
            "legacy_model_path": str(self.legacy_model_path) if has_leg else None,
            "new_model_backbone": self.new_meta.get("backbone") if has_new else None,
            "legacy_preprocessing": self.legacy_meta.get("preprocessing", "rescale_255") if has_leg else None,
            "ensemble_config": {
                "weight_new": self.ensemble_config.weight_new,
                "weight_legacy": self.ensemble_config.weight_legacy,
                "threshold": self.ensemble_config.threshold,
                "temp_new": self.ensemble_config.temp_new,
                "temp_legacy": self.ensemble_config.temp_legacy,
                "calibrated": self.ensemble_config.calibrated,
                "default_mode": self.mode,
            },
        }


model_manager = ModelManager()


# ── Startup Event ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    init_db()
    print("[APP] DeepForensics v2 server started")


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
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
    mode: str = Form("auto"),
    use_tta: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Analyze uploaded image using Multi-Branch / ResNet50 / Weighted Ensemble + Grad-CAM++."""
    if not 70 <= quality <= 98:
        raise HTTPException(status_code=400, detail="Quality must be between 70 and 98")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    try:
        unique_id = uuid.uuid4().hex[:12]
        safe_filename = f"{unique_id}_{file.filename}"
        original_path = UPLOAD_DIR / safe_filename
        ela_filename = f"{unique_id}_{Path(file.filename).stem}_ela.png"
        ela_path = ELA_DIR / ela_filename
        gradcam_filename = f"{unique_id}_{Path(file.filename).stem}_gradcam.png"
        gradcam_path = GRADCAM_DIR / gradcam_filename

        content = await file.read()
        with open(original_path, "wb") as f:
            f.write(content)

        # Generate visual ELA map at user-selected quality
        ela_image = convert_to_ela_image(original_path, quality=quality)
        ela_image.save(ela_path, format="PNG")

        # Run deep learning inference + Grad-CAM++ localization
        result = model_manager.predict(
            original_path=original_path,
            ela_image=ela_image,
            requested_mode=mode,
            use_tta=use_tta,
            gradcam_out_path=gradcam_path,
        )

        gradcam_url = f"/static/gradcam/{gradcam_filename}" if result["gradcam_generated"] else None

        scan_record = ScanRecord(
            filename=file.filename,
            original_url=f"/static/uploads/{safe_filename}",
            ela_url=f"/static/ela/{ela_filename}",
            gradcam_url=gradcam_url,
            prediction=result["prediction"],
            confidence=result["confidence"],
            ela_quality=quality,
            model_mode=result["effective_mode"],
            p_new=result["p_new"],
            p_legacy=result["p_legacy"],
            p_ensemble=result["p_ensemble"],
            tta_used=1 if result["tta_used"] else 0,
            timestamp=datetime.utcnow(),
        )
        db.add(scan_record)
        db.commit()
        db.refresh(scan_record)

        return scan_record.to_dict()

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


@app.get("/api/history")
async def get_history(limit: int = 25, db: Session = Depends(get_db)):
    records = (
        db.query(ScanRecord)
        .order_by(ScanRecord.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [rec.to_dict() for rec in records]


@app.delete("/api/history/{scan_id}")
async def delete_scan(scan_id: int, db: Session = Depends(get_db)):
    record = db.query(ScanRecord).filter(ScanRecord.id == scan_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Scan record not found")

    for url_attr in (record.original_url, record.ela_url, record.gradcam_url):
        if not url_attr:
            continue
        try:
            p = Path(url_attr.lstrip("/"))
            if p.exists():
                p.unlink()
        except Exception as exc:
            print(f"[WARN] Failed to delete file {url_attr} for scan {scan_id}: {exc}")

    db.delete(record)
    db.commit()
    return {"message": "Scan deleted successfully", "id": scan_id}


@app.get("/api/status")
async def get_status(db: Session = Depends(get_db)):
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

    print("=" * 64)
    print("  DeepForensics v2 — Starting Forensic Console Server")
    print("=" * 64)
    print(f"  Default mode : {model_manager.mode}")
    print(f"  Multi-Branch : {'LOADED' if model_manager.new_model else 'not trained yet'}")
    print(f"  ResNet-50    : {'LOADED' if model_manager.legacy_model else 'not found'}")
    print("  Visit        : http://localhost:8000")
    print("=" * 64)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
