"""
DeepForensics — Shared Stratified Split Manifest (Zero Leakage)
===============================================================
Creates and loads a single deterministic 80/20 stratified train/val split
(`dataset/split_manifest.json`, seed=42) keyed by ``(class, stem)`` pairs.

Using the exact same split across:
    - Fixed ResNet50 (`train_model.py`)
    - Multi-Branch model (`train_multibranch.py`)
    - Ensemble calibration (`calibrate_ensemble.py`)
    - Evaluation / t-SNE / Robustness (`evaluate_model.py`)

guarantees zero train-to-validation data leakage when calibrating the
weighted ensemble and comparing ablation experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from sklearn.model_selection import train_test_split


DEFAULT_MANIFEST_PATH = Path("dataset/split_manifest.json")
DEFAULT_FEATURES_DIR = Path("dataset/features_224")
DEFAULT_ELA_DIR = Path("dataset/ela_224")
DEFAULT_RAW_DIR = Path("dataset/raw")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_TO_LABEL = {"Au": 0, "Tp": 1}


def build_or_load_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    source_dir: Path = DEFAULT_FEATURES_DIR,
    val_ratio: float = 0.2,
    seed: int = 42,
    force_rebuild: bool = False,
) -> dict:
    """Load existing split manifest, or create a stratified 80/20 manifest.

    Each entry is ``{"cls": "Au"|"Tp", "stem": "<filename_stem>", "label": 0|1}``.
    When scanning ``source_dir``, looks for ``ela_q85/`` if it is a features root,
    otherwise looks directly for ``Au/`` and ``Tp/`` subdirectories.
    """
    if manifest_path.exists() and not force_rebuild:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    scan_root = (source_dir / "ela_q85") if (source_dir / "ela_q85").is_dir() else source_dir
    if not (scan_root / "Au").is_dir() or not (scan_root / "Tp").is_dir():
        # Fall back to ela_224 or raw if features_224 hasn't been built yet
        for fallback in (DEFAULT_ELA_DIR, DEFAULT_RAW_DIR):
            if (fallback / "Au").is_dir() and (fallback / "Tp").is_dir():
                scan_root = fallback
                break

    records: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for cls, label in CLASS_TO_LABEL.items():
        cls_dir = scan_root / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                key = (cls, p.stem)
                if key not in seen:
                    seen.add(key)
                    records.append({"cls": cls, "stem": p.stem, "label": label})

    if not records:
        raise FileNotFoundError(
            f"No dataset images found under {scan_root}. Run prepare_features.py first."
        )

    labels = [r["label"] for r in records]
    train_items, val_items = train_test_split(
        records,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )

    manifest = {
        "seed": seed,
        "val_ratio": val_ratio,
        "total": len(records),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "train": train_items,
        "val": val_items,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def resolve_ela_paths(
    items: list[dict],
    ela_dir: Path = DEFAULT_ELA_DIR,
) -> tuple[list[str], list[int]]:
    """Resolve manifest entries to existing PNG paths under ``ela_dir``."""
    paths: list[str] = []
    labels: list[int] = []
    for item in items:
        p = ela_dir / item["cls"] / f"{item['stem']}.png"
        if p.exists():
            paths.append(str(p))
            labels.append(int(item["label"]))
    return paths, labels


def resolve_multibranch_paths(
    items: list[dict],
    features_dir: Path = DEFAULT_FEATURES_DIR,
    include_fft: bool = True,
) -> tuple[dict[str, list[str]], list[int]]:
    """Resolve manifest entries to synchronized paths across all feature caches."""
    out: dict[str, list[str]] = {
        "ela_q70": [],
        "ela_q85": [],
        "ela_q95": [],
        "raw_224": [],
    }
    if include_fft:
        out["fft_224"] = []
    labels: list[int] = []

    for item in items:
        cls = item["cls"]
        stem = item["stem"]
        p70 = features_dir / "ela_q70" / cls / f"{stem}.png"
        p85 = features_dir / "ela_q85" / cls / f"{stem}.png"
        p95 = features_dir / "ela_q95" / cls / f"{stem}.png"
        praw = features_dir / "raw_224" / cls / f"{stem}.png"
        pfft = features_dir / "fft_224" / cls / f"{stem}.png"

        if not (p70.exists() and p85.exists() and p95.exists() and praw.exists()):
            continue
        if include_fft and not pfft.exists():
            continue

        out["ela_q70"].append(str(p70))
        out["ela_q85"].append(str(p85))
        out["ela_q95"].append(str(p95))
        out["raw_224"].append(str(praw))
        if include_fft:
            out["fft_224"].append(str(pfft))
        labels.append(int(item["label"]))

    return out, labels


def compute_class_weights(labels: list[int]) -> dict[int, float]:
    """Compute inverse-frequency binary class weights {0: w_au, 1: w_tp}."""
    au = sum(1 for y in labels if y == 0)
    tp = sum(1 for y in labels if y == 1)
    total = au + tp
    if au == 0 or tp == 0:
        return {0: 1.0, 1: 1.0}
    return {
        0: total / (2.0 * au),
        1: total / (2.0 * tp),
    }


def find_raw_image_path(
    cls: str,
    stem: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> Optional[Path]:
    """Locate the original raw image file for a given (cls, stem)."""
    cls_dir = raw_dir / cls
    for ext in (".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".png", ".JPG", ".TIF"):
        cand = cls_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None
