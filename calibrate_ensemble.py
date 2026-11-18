"""
DeepForensics — Temperature Scaling & Weighted Ensemble Calibrator
==================================================================
Calibrates the multi-model ensemble on the shared held-out CASIA v2.0
validation split (`dataset/split_manifest.json`, seed=42):

    1. Runs both the Multi-Branch model (`ela_multibranch_best.keras`) and
       the ResNet50 model (`ela_resnet50_v3.keras` or `ela_resnet50_best.keras`)
       on identical aligned validation images (with optional 4-view TTA).
    2. Fits scalar Temperature Scaling (T_new, T_legacy) to minimize
       validation NLL and reports Expected Calibration Error (ECE) before
       and after calibration.
    3. Grid-searches ensemble weight w in [0.0, 1.0] and decision threshold
       in [0.30, 0.70] for:
           p_ensemble = w * p_new_cal + (1 - w) * p_legacy_cal
       maximizing F1-score.
    4. Persists optimal parameters to ``ensemble_config.json`` and saves
       diagnostic plots to ``static/metrics/``:
           - calibration_reliability.png
           - ensemble_weight_sweep.png

Usage:
    python calibrate_ensemble.py               # calibrate both models
    python calibrate_ensemble.py --tta         # calibrate with 4-view TTA
    python calibrate_ensemble.py --smoke-test  # synthetic verification
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

from utils.calibration import (
    DEFAULT_ENSEMBLE_CONFIG_PATH,
    EnsembleConfig,
    apply_temperature,
    binary_nll,
    compute_ece,
    fit_temperature,
)
from utils.custom_layers import get_custom_objects
from utils.dataset_split import (
    DEFAULT_ELA_DIR,
    DEFAULT_FEATURES_DIR,
    DEFAULT_MANIFEST_PATH,
    build_or_load_manifest,
)
from utils.xai import predict_with_tta


METRICS_DIR = Path("static/metrics")
DEFAULT_NEW_MODEL = Path("ela_multibranch_best.keras")
DEFAULT_LEGACY_CANDIDATES = (Path("ela_resnet50_v3.keras"), Path("ela_resnet50_best.keras"))


def _read_meta(model_path: Path) -> dict:
    meta_path = model_path.with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _resolve_legacy_model(explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None
    for cand in DEFAULT_LEGACY_CANDIDATES:
        if cand.exists():
            return cand
    return None


def _load_aligned_validation_items(
    manifest_path: Path,
    ela_dir: Path,
    features_dir: Path,
    include_fft: bool,
    ela_only: bool,
) -> list[dict]:
    """Return validation manifest items that exist in BOTH legacy and multi-modal caches."""
    manifest = build_or_load_manifest(manifest_path=manifest_path, source_dir=features_dir)
    aligned = []
    need_fft = include_fft and (not ela_only)
    for item in manifest["val"]:
        cls, stem = item["cls"], item["stem"]
        p_leg = ela_dir / cls / f"{stem}.png"
        p70 = features_dir / "ela_q70" / cls / f"{stem}.png"
        p85 = features_dir / "ela_q85" / cls / f"{stem}.png"
        p95 = features_dir / "ela_q95" / cls / f"{stem}.png"
        praw = features_dir / "raw_224" / cls / f"{stem}.png"
        pfft = features_dir / "fft_224" / cls / f"{stem}.png"

        if not (p_leg.exists() and p70.exists() and p85.exists() and p95.exists()):
            continue
        if (not ela_only) and (not praw.exists()):
            continue
        if need_fft and (not pfft.exists()):
            continue
        aligned.append(item)
    return aligned


def _decode_rgb(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_png(raw, channels=3)
    img = tf.image.resize(img, (224, 224), method="bilinear")
    return tf.cast(img, tf.float32)


def _decode_gray(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_png(raw, channels=1)
    img = tf.image.resize(img, (224, 224), method="bilinear")
    return tf.cast(img, tf.float32)


def _predict_legacy(
    model: keras.Model,
    items: list[dict],
    ela_dir: Path,
    preprocessing: str,
    batch_size: int,
    use_tta: bool,
) -> np.ndarray:
    paths = [str(ela_dir / it["cls"] / f"{it['stem']}.png") for it in items]

    def load_one(p):
        img = _decode_rgb(p)
        if preprocessing == "resnet50_caffe":
            return resnet50_preprocess(img)
        return img / 255.0

    ds = (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(load_one, num_parallel_calls=4)
        .batch(batch_size)
        .prefetch(1)
    )
    total_batches = max(1, int(np.ceil(len(items) / batch_size)))
    preds = []
    for idx, batch in enumerate(ds, start=1):
        preds.append(predict_with_tta(model, batch.numpy(), use_tta=use_tta))
        if idx == 1 or idx % 40 == 0 or idx == total_batches:
            print(f"    Legacy ResNet50 batches: {idx}/{total_batches}", flush=True)
    return np.concatenate(preds, axis=0)


def _predict_multibranch(
    model: keras.Model,
    items: list[dict],
    features_dir: Path,
    include_fft: bool,
    ela_only: bool,
    batch_size: int,
    use_tta: bool,
) -> np.ndarray:
    need_fft = include_fft and (not ela_only)
    slices = {
        "p70": [str(features_dir / "ela_q70" / it["cls"] / f"{it['stem']}.png") for it in items],
        "p85": [str(features_dir / "ela_q85" / it["cls"] / f"{it['stem']}.png") for it in items],
        "p95": [str(features_dir / "ela_q95" / it["cls"] / f"{it['stem']}.png") for it in items],
        "praw": [str(features_dir / "raw_224" / it["cls"] / f"{it['stem']}.png") for it in items],
    }
    if need_fft:
        slices["pfft"] = [str(features_dir / "fft_224" / it["cls"] / f"{it['stem']}.png") for it in items]

    def load_one(rec):
        mela = tf.concat([_decode_rgb(rec["p70"]), _decode_rgb(rec["p85"]), _decode_rgb(rec["p95"])], axis=-1)
        out = {"multi_ela_input": mela}
        if not ela_only:
            out["raw_rgb_input"] = _decode_rgb(rec["praw"])
            if need_fft:
                out["fft_input"] = _decode_gray(rec["pfft"])
        return out

    ds = (
        tf.data.Dataset.from_tensor_slices(slices)
        .map(load_one, num_parallel_calls=4)
        .batch(batch_size)
        .prefetch(1)
    )
    total_batches = max(1, int(np.ceil(len(items) / batch_size)))
    preds = []
    for idx, batch in enumerate(ds, start=1):
        batch_np = {k: v.numpy() for k, v in batch.items()}
        preds.append(predict_with_tta(model, batch_np, use_tta=use_tta))
        if idx == 1 or idx % 40 == 0 or idx == total_batches:
            print(f"    Multi-Branch batches   : {idx}/{total_batches}", flush=True)
    return np.concatenate(preds, axis=0)


# ── Grid Search & Plotting ───────────────────────────────────────────────────

def _metrics_at(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0,
        "ece": float(compute_ece(y_true, probs)["ece"]),
        "nll": float(binary_nll(y_true, probs)),
    }


def grid_search_ensemble(
    y_true: np.ndarray,
    p_new_cal: np.ndarray,
    p_leg_cal: np.ndarray,
) -> tuple[float, float, list[dict]]:
    """Grid-search weight w in [0, 1] and threshold in [0.30, 0.70] maximizing F1."""
    weights = np.round(np.linspace(0.0, 1.0, 21), 2)
    thresholds = np.round(np.linspace(0.30, 0.70, 21), 2)

    best_score = (-1.0, -1.0)  # (f1, auc)
    best_w = 1.0
    best_t = 0.50
    sweep_curve: list[dict] = []

    for w in weights:
        p_ens = w * p_new_cal + (1.0 - w) * p_leg_cal
        auc_val = float(roc_auc_score(y_true, p_ens)) if len(np.unique(y_true)) > 1 else 0.0
        best_f1_for_w = -1.0
        best_acc_for_w = 0.0
        best_t_for_w = 0.50
        for t in thresholds:
            y_pred = (p_ens >= t).astype(int)
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            if f1 > best_f1_for_w:
                best_f1_for_w = f1
                best_acc_for_w = float(accuracy_score(y_true, y_pred))
                best_t_for_w = float(t)

        sweep_curve.append(
            {
                "weight_new": float(w),
                "f1": best_f1_for_w,
                "accuracy": best_acc_for_w,
                "auc": auc_val,
                "best_threshold": best_t_for_w,
            }
        )
        if (best_f1_for_w, auc_val) > best_score:
            best_score = (best_f1_for_w, auc_val)
            best_w = float(w)
            best_t = best_t_for_w

    return best_w, best_t, sweep_curve


def plot_reliability_diagrams(
    y_true: np.ndarray,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot pre- vs post-temperature-scaling reliability diagrams."""
    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, (title, p_raw, p_cal) in zip(axes[0], series):
        ece_r = compute_ece(y_true, p_raw)
        ece_c = compute_ece(y_true, p_cal)
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
        ax.plot(
            ece_r["bin_confidences"],
            ece_r["bin_accuracies"],
            "o-",
            color="#e74c3c",
            label=f"Raw (ECE={ece_r['ece']:.4f})",
        )
        ax.plot(
            ece_c["bin_confidences"],
            ece_c["bin_accuracies"],
            "s-",
            color="#2ecc71",
            label=f"Temp-Scaled (ECE={ece_c['ece']:.4f})",
        )
        ax.set_title(title)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Empirical Positive Fraction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved reliability diagram : {output_path}")


def plot_weight_sweep(sweep: list[dict], best_w: float, output_path: Path) -> None:
    ws = [s["weight_new"] for s in sweep]
    f1s = [s["f1"] for s in sweep]
    accs = [s["accuracy"] for s in sweep]
    aucs = [s["auc"] for s in sweep]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ws, f1s, "o-", lw=2, label="F1-Score")
    ax.plot(ws, accs, "s--", lw=1.8, label="Accuracy")
    ax.plot(ws, aucs, "^:", lw=1.8, label="ROC-AUC")
    ax.axvline(best_w, color="#e74c3c", linestyle="--", lw=1.5, label=f"Optimal w = {best_w:.2f}")
    ax.set_xlabel("Weight assigned to Multi-Branch Model (w)")
    ax.set_ylabel("Validation Metric")
    ax.set_title("Calibrated Weighted Ensemble — Weight Sweep (w * p_new + (1 - w) * p_legacy)")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved weight sweep plot   : {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepForensics: Temperature Scaling & Weighted Ensemble Calibration"
    )
    parser.add_argument("--new-model", type=Path, default=DEFAULT_NEW_MODEL)
    parser.add_argument("--legacy-model", type=Path, default=None)
    parser.add_argument("--ela-dir", type=Path, default=DEFAULT_ELA_DIR)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ENSEMBLE_CONFIG_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tta", action="store_true", help="Use 4-view TTA during calibration.")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic calibration verification.")
    args = parser.parse_args()

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    print("=" * 68)
    print("  DeepForensics — Temperature Scaling & Ensemble Calibration")
    print("=" * 68)

    if args.smoke_test:
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=300)
        z_new = (y_true * 2 - 1) * 2.5 + rng.normal(0, 1.0, size=300)
        z_leg = (y_true * 2 - 1) * 1.4 + rng.normal(0, 1.2, size=300)
        p_new_raw = 1.0 / (1.0 + np.exp(-z_new))
        p_leg_raw = 1.0 / (1.0 + np.exp(-z_leg))

        t_new = fit_temperature(y_true, p_new_raw)
        t_leg = fit_temperature(y_true, p_leg_raw)
        p_new_cal = apply_temperature(p_new_raw, t_new)
        p_leg_cal = apply_temperature(p_leg_raw, t_leg)
        best_w, best_t, sweep = grid_search_ensemble(y_true, p_new_cal, p_leg_cal)

        plot_reliability_diagrams(
            y_true,
            [("Multi-Branch (Smoke)", p_new_raw, p_new_cal), ("Legacy ResNet50 (Smoke)", p_leg_raw, p_leg_cal)],
            METRICS_DIR / "tmp_smoke_reliability.png",
        )
        plot_weight_sweep(sweep, best_w, METRICS_DIR / "tmp_smoke_sweep.png")
        (METRICS_DIR / "tmp_smoke_reliability.png").unlink(missing_ok=True)
        (METRICS_DIR / "tmp_smoke_sweep.png").unlink(missing_ok=True)
        print(f"  [SMOKE TEST] T_new={t_new:.3f}, T_leg={t_leg:.3f}, best_w={best_w:.2f}, thresh={best_t:.2f}")
        print("  [SMOKE TEST] Calibration & Weighted Ensemble PASSED!")
        return

    if not args.new_model.exists():
        print(f"[ERROR] New multi-branch model not found: {args.new_model}")
        print("        Run: python train_multibranch.py")
        sys.exit(1)

    legacy_path = _resolve_legacy_model(args.legacy_model)
    if legacy_path is None:
        print("[ERROR] No legacy ResNet50 model found (checked ela_resnet50_v3.keras & ela_resnet50_best.keras).")
        sys.exit(1)

    new_meta = _read_meta(args.new_model)
    leg_meta = _read_meta(legacy_path)
    include_fft = bool(new_meta.get("include_fft", True))
    ela_only = bool(new_meta.get("ela_only", False))
    leg_preproc = leg_meta.get("preprocessing", "rescale_255")

    print(f"  Multi-Branch model : {args.new_model} (fft={include_fft}, ela_only={ela_only})")
    print(f"  Legacy model       : {legacy_path} (preprocessing={leg_preproc})")
    print(f"  4-View TTA         : {args.tta}")

    items = _load_aligned_validation_items(
        args.manifest, args.ela_dir, args.features_dir, include_fft, ela_only
    )
    if not items:
        print("[ERROR] No aligned validation images found across ela_224/ and features_224/.")
        sys.exit(1)

    y_true = np.array([int(it["label"]) for it in items], dtype=np.int32)
    print(f"  Aligned val images : {len(items)}")

    print("\n[1/4] Running Multi-Branch validation inference...")
    new_model = keras.models.load_model(args.new_model, custom_objects=get_custom_objects(), compile=False)
    p_new_raw = _predict_multibranch(
        new_model, items, args.features_dir, include_fft, ela_only, args.batch_size, args.tta
    )
    del new_model
    keras.backend.clear_session()
    gc.collect()

    print("[2/4] Running Legacy ResNet50 validation inference...")
    leg_model = keras.models.load_model(legacy_path, custom_objects=get_custom_objects(), compile=False)
    p_leg_raw = _predict_legacy(
        leg_model, items, args.ela_dir, leg_preproc, args.batch_size, args.tta
    )
    del leg_model
    keras.backend.clear_session()
    gc.collect()

    print("[3/4] Fitting Temperature Scaling (minimizing NLL)...")
    t_new = fit_temperature(y_true, p_new_raw)
    t_leg = fit_temperature(y_true, p_leg_raw)
    p_new_cal = apply_temperature(p_new_raw, t_new)
    p_leg_cal = apply_temperature(p_leg_raw, t_leg)

    print(f"  Multi-Branch T = {t_new:.3f} | ECE: {compute_ece(y_true, p_new_raw)['ece']:.4f} -> {compute_ece(y_true, p_new_cal)['ece']:.4f}")
    print(f"  Legacy R50   T = {t_leg:.3f} | ECE: {compute_ece(y_true, p_leg_raw)['ece']:.4f} -> {compute_ece(y_true, p_leg_cal)['ece']:.4f}")

    print("[4/4] Grid-searching optimal ensemble weight w and decision threshold...")
    best_w, best_t, sweep = grid_search_ensemble(y_true, p_new_cal, p_leg_cal)
    p_ens = best_w * p_new_cal + (1.0 - best_w) * p_leg_cal

    m_leg = _metrics_at(y_true, p_leg_cal, threshold=0.50)
    m_new = _metrics_at(y_true, p_new_cal, threshold=0.50)
    m_ens = _metrics_at(y_true, p_ens, threshold=best_t)

    # Honest academic default: only set ensemble as default if legacy contributes meaningfully
    default_mode = "new_only" if best_w >= 0.999 else "ensemble"

    cfg = EnsembleConfig(
        weight_new=round(best_w, 4),
        weight_legacy=round(1.0 - best_w, 4),
        threshold=round(best_t, 4),
        temp_new=round(t_new, 4),
        temp_legacy=round(t_leg, 4),
        default_mode=default_mode,
        use_tta_default=bool(args.tta),
        calibrated=True,
        metrics={
            "val_samples": int(len(items)),
            "legacy_only": m_leg,
            "new_only": m_new,
            "ensemble": m_ens,
        },
    )
    cfg.save(args.output)

    plot_reliability_diagrams(
        y_true,
        [
            ("Multi-Branch Model", p_new_raw, p_new_cal),
            ("Legacy ResNet50", p_leg_raw, p_leg_cal),
        ],
        METRICS_DIR / "calibration_reliability.png",
    )
    plot_weight_sweep(sweep, best_w, METRICS_DIR / "ensemble_weight_sweep.png")

    print("\n" + "-" * 68)
    print("  Calibration Summary (Held-Out CASIA v2.0 Validation Split)")
    print("-" * 68)
    print(f"  Legacy ResNet50 : Acc={m_leg['accuracy']:.4f} | F1={m_leg['f1']:.4f} | AUC={m_leg['roc_auc']:.4f} | ECE={m_leg['ece']:.4f}")
    print(f"  Multi-Branch    : Acc={m_new['accuracy']:.4f} | F1={m_new['f1']:.4f} | AUC={m_new['roc_auc']:.4f} | ECE={m_new['ece']:.4f}")
    print(f"  Ensemble (w={best_w:.2f}, thr={best_t:.2f}) : Acc={m_ens['accuracy']:.4f} | F1={m_ens['f1']:.4f} | AUC={m_ens['roc_auc']:.4f} | ECE={m_ens['ece']:.4f}")
    print(f"  Selected Default Mode : {default_mode}")
    print(f"  Config Saved          : {args.output.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()
