"""
DeepForensics — SQLite Database Layer
======================================
Defines the SQLAlchemy schema for storing forgery scan history and
provides database initialization utilities for FastAPI.

Usage as a library:
    from database import get_db, ScanRecord

CLI test:
    python database.py  # Creates forgery_scans.db and tests schema
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# ── Configuration ────────────────────────────────────────────────────────────

DATABASE_URL = "sqlite:///./forgery_scans.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False,  # Set to True for SQL debug logs
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ── Schema Definition ────────────────────────────────────────────────────────

class ScanRecord(Base):
    """Table storing image forgery scan results."""

    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String, nullable=False, index=True)
    original_url = Column(String, nullable=False)  # Path to uploaded image
    ela_url = Column(String, nullable=False)       # Path to ELA heatmap
    prediction = Column(String, nullable=False, index=True)  # "Authentic" or "Tampered"
    confidence = Column(Float, nullable=False)     # 0.0 to 1.0
    ela_quality = Column(Integer, nullable=False, default=90)  # JPEG quality used
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:
        return f"<ScanRecord(id={self.id}, filename={self.filename}, prediction={self.prediction}, confidence={self.confidence:.3f})>"

    def to_dict(self) -> dict:
        """Convert record to JSON-serializable dict."""
        return {
            "id": self.id,
            "filename": self.filename,
            "original_url": self.original_url,
            "ela_url": self.ela_url,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "ela_quality": self.ela_quality,
            "timestamp": self.timestamp.isoformat(),
        }


# ── Database Utilities ───────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    print(f"[DB] Initialized database: {DATABASE_URL}")


def get_db() -> Session:
    """FastAPI dependency: provides a database session per request.

    Usage in FastAPI:
        @app.get("/api/history")
        def get_history(db: Session = Depends(get_db)):
            records = db.query(ScanRecord).all()
            return records
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── CLI Test ─────────────────────────────────────────────────────────────────

def _test_database() -> None:
    """Test database creation and basic CRUD operations."""
    print("=" * 60)
    print("  DeepForensics - Database Test")
    print("=" * 60)

    # Create tables
    init_db()

    # Test insertion
    db = SessionLocal()
    try:
        print("\n[TEST 1] Inserting a test scan record...")
        test_record = ScanRecord(
            filename="test_image.jpg",
            original_url="/static/uploads/test_image.jpg",
            ela_url="/static/ela/test_image_ela.png",
            prediction="Tampered",
            confidence=0.87,
            ela_quality=90,
        )
        db.add(test_record)
        db.commit()
        db.refresh(test_record)
        print(f"  Inserted: {test_record}")

        # Test query
        print("\n[TEST 2] Querying all records...")
        all_records = db.query(ScanRecord).all()
        print(f"  Total records: {len(all_records)}")
        for rec in all_records:
            print(f"    {rec}")

        # Test filtering
        print("\n[TEST 3] Filtering tampered images...")
        tampered = db.query(ScanRecord).filter(ScanRecord.prediction == "Tampered").all()
        print(f"  Tampered records: {len(tampered)}")

        # Test to_dict
        print("\n[TEST 4] Serializing to JSON...")
        json_data = test_record.to_dict()
        print(f"  JSON: {json_data}")

        # Test deletion
        print("\n[TEST 5] Deleting test record...")
        db.delete(test_record)
        db.commit()
        print("  Deleted successfully")

        remaining = db.query(ScanRecord).count()
        print(f"  Remaining records: {remaining}")

    finally:
        db.close()

    print("\n" + "=" * 60)
    print("  All database tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _test_database()
