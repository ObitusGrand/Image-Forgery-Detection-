"""
DeepForensics — SQLite Database Layer (v2 Multi-Model + XAI Schema)
====================================================================
Defines the SQLAlchemy schema for storing forgery scan history, ensemble
probability breakdowns, and Grad-CAM++ localization paths, with automatic
backward-compatible SQLite column migration on startup.

CLI test:
    python database.py
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# ── Configuration ────────────────────────────────────────────────────────────

DATABASE_URL = "sqlite:///./forgery_scans.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ── Schema Definition ────────────────────────────────────────────────────────

class ScanRecord(Base):
    """Table storing image forgery scan results, ensemble scores, and XAI maps."""

    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String, nullable=False, index=True)
    original_url = Column(String, nullable=False)
    ela_url = Column(String, nullable=False)
    prediction = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    ela_quality = Column(Integer, nullable=False, default=90)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # v2 Multi-Model Ensemble & Grad-CAM++ columns (nullable for legacy rows)
    model_mode = Column(String, nullable=True, default="legacy_only")
    gradcam_url = Column(String, nullable=True)
    p_new = Column(Float, nullable=True)
    p_legacy = Column(Float, nullable=True)
    p_ensemble = Column(Float, nullable=True)
    tta_used = Column(Integer, nullable=True, default=0)

    def __repr__(self) -> str:
        return (
            f"<ScanRecord(id={self.id}, filename={self.filename}, "
            f"mode={self.model_mode}, prediction={self.prediction}, "
            f"confidence={self.confidence:.3f})>"
        )

    def to_dict(self) -> dict:
        """Convert record to JSON-serializable dict."""
        models_agree = None
        if self.p_new is not None and self.p_legacy is not None:
            models_agree = (self.p_new >= 0.5) == (self.p_legacy >= 0.5)

        return {
            "id": self.id,
            "filename": self.filename,
            "original_url": self.original_url,
            "ela_url": self.ela_url,
            "gradcam_url": self.gradcam_url,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "ela_quality": self.ela_quality,
            "model_mode": self.model_mode or "legacy_only",
            "p_new": self.p_new,
            "p_legacy": self.p_legacy,
            "p_ensemble": self.p_ensemble,
            "models_agree": models_agree,
            "tta_used": bool(self.tta_used),
            "timestamp": self.timestamp.isoformat(),
        }


# ── Auto-Migration & Utilities ───────────────────────────────────────────────

_V2_MIGRATION_COLUMNS = {
    "model_mode": "VARCHAR DEFAULT 'legacy_only'",
    "gradcam_url": "VARCHAR",
    "p_new": "FLOAT",
    "p_legacy": "FLOAT",
    "p_ensemble": "FLOAT",
    "tta_used": "INTEGER DEFAULT 0",
}


def _migrate_sqlite_schema() -> None:
    """Safely add v2 columns to an existing SQLite scan_records table."""
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(scan_records)")).fetchall()
        existing = {row[1] for row in rows}
        for col_name, col_def in _V2_MIGRATION_COLUMNS.items():
            if col_name not in existing:
                conn.execute(text(f"ALTER TABLE scan_records ADD COLUMN {col_name} {col_def}"))
                print(f"[DB] Migrated scan_records: added column '{col_name}'")


def init_db() -> None:
    """Create tables if missing and apply any backward-compatible migrations."""
    Base.metadata.create_all(bind=engine)
    _migrate_sqlite_schema()
    print(f"[DB] Initialized database: {DATABASE_URL}")


def get_db() -> Session:
    """FastAPI dependency providing a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── CLI Test ─────────────────────────────────────────────────────────────────

def _test_database() -> None:
    print("=" * 60)
    print("  DeepForensics - Database Test (v2 Schema)")
    print("=" * 60)
    init_db()

    db = SessionLocal()
    try:
        rec = ScanRecord(
            filename="test_ensemble.jpg",
            original_url="/static/uploads/test_ensemble.jpg",
            ela_url="/static/ela/test_ensemble_ela.png",
            gradcam_url="/static/gradcam/test_ensemble_gradcam.png",
            prediction="Tampered",
            confidence=0.91,
            ela_quality=90,
            model_mode="ensemble",
            p_new=0.95,
            p_legacy=0.79,
            p_ensemble=0.91,
            tta_used=1,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        data = rec.to_dict()
        assert data["models_agree"] is True
        assert data["tta_used"] is True
        print(f"  Inserted & serialized: {data}")

        db.delete(rec)
        db.commit()
        print("  Cleanup OK")
    finally:
        db.close()

    print("=" * 60)
    print("  All database tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _test_database()
