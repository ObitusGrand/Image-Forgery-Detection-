"""
DeepForensics — Deep Learning Evaluation, XAI & Robustness Suite (v2)
======================================================================
Comprehensive academic evaluation on the held-out CASIA v2.0 validation split
(`dataset/split_manifest.json`, seed=42):

    1. Multi-Model & Ensemble Metrics:
       Accuracy, Precision, Recall, F1-Score, ROC-AUC, ECE for:
         - Legacy ResNet50 (`ela_resnet50_v3.keras` or `ela_resnet50_best.keras`)
         - Multi-Branch Model (`ela_multibranch_best.keras`)
         - Calibrated Weighted Ensemble (`ensemble_config.json`)
    2. Comparison Plots (saved to ``static/metrics/``):
         - ``confusion_matrix.png`` (side-by-side per mode)
         - ``roc_curve.png`` (all models + ensemble overlaid)
         - ``tsne_embeddings.png`` (--tsne: 2D latent cluster visualization)
         - ``robustness_curves.png`` (--robustness: JPEG / Noise / Resize stress test)
         - ``gradcam_samples.png`` (--gradcam-samples: Original | ELA | Grad-CAM++)
         - ``evaluation_summary.json`` (structured academic report metrics)

Usage:
    python evaluate_model.py                # evaluate all available models
    python evaluate_model.py --all --tta    # full suite + 4-view TTA
    python evaluate_model.py --smoke-test   # synthetic end-to-end check
"""

from __future__ import annotations

import argparse
import gc
import io
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
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

from utils.calibration import (
    DEFAULT_ENSEMBLE_CONFIG_PATH,
    EnsembleConfig,
    apply_temperature,
    compute_ece,
)
from utils.custom_layers import get_custom_objects
from utils.dataset_split import (
    DEFAULT_ELA_DIR,
    DEFAULT_FEATURES_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_DIR,
    build_or_load_manifest,
    find_raw_image_path,
)
from utils.ela import convert_to_ela_image
from utils.features import extract_multibranch_inputs
from utils.xai import compute_gradcam_pp, overlay_heatmap, predict_with_tta


METRICS_DIR = Path("static/metrics")
DEFAULT_NEW_MODEL = Path("ela_multibranch_best.keras")
DEFAULT_LEGACY_CANDIDATES = (Path("ela_resnet50_v3.keras"), Path("ela_resnet50_best.keras"))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_meta(model_path: Path) -> dict:
    p = model_path.with_suffix(".meta.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _resolve_legacy_model(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    for cand in DEFAULT_LEGACY_CANDIDATES:
        if cand.exists():
            return cand
    return None


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


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute classification, AUC, ECE, and confusion matrix metrics."""
    y_t = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_pred_proba, dtype=np.float64).ravel()
    y_pred = (p >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_t, y_pred)),
        "precision": float(precision_score(y_t, y_pred, zero_division=0)),
        "recall": float(recall_score(y_t, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_t, y_pred, zero_division=0)),
        "ece": float(compute_ece(y_t, p)["ece"]),
    }

    if len(np.unique(y_t)) > 1:
        fpr, tpr, _ = roc_curve(y_t, p)
        metrics["roc_auc"] = float(auc(fpr, tpr))
        metrics["fpr"] = fpr
        metrics["tpr"] = tpr
    else:
        metrics["roc_auc"] = 0.0
        metrics["fpr"] = np.array([0.0, 1.0])
        metrics["tpr"] = np.array([0.0, 1.0])

    metrics["confusion_matrix"] = confusion_matrix(y_t, y_pred, labels=[0, 1])
    return metrics


# ── Visualization Generators ─────────────────────────────────────────────────

def plot_confusion_matrices(results: dict[str, dict], output_path: Path) -> None:
    """Plot side-by-side confusion matrices for all evaluated modes."""
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 5), squeeze=False)
    classes = ["Authentic", "Tampered"]

    for ax, name in zip(axes[0], names):
        cm = results[name]["confusion_matrix"]
        acc = results[name]["accuracy"]
        f1 = results[name]["f1_score"]
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(2),
            yticks=np.arange(2),
            xticklabels=classes,
            yticklabels=classes,
            ylabel="True Label",
            xlabel="Predicted Label",
            title=f"{name}\nAcc={acc:.3f} | F1={f1:.3f}",
        )
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]:d}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight="bold",
                )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrices : {output_path}")


def plot_roc_comparison(results: dict[str, dict], output_path: Path) -> None:
    """Overlay ROC curves for all evaluated models and the ensemble."""
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (name, res) in enumerate(results.items()):
        c = colors[idx % len(colors)]
        ax.plot(
            res["fpr"], res["tpr"],
            color=c, lw=2.2,
            label=f"{name} (AUC = {res['roc_auc']:.4f})",
        )
    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.4, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison — DeepForensics")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ROC comparison     : {output_path}")


def _extract_embeddings(
    model: keras.Model,
    dataset: tf.data.Dataset,
    max_samples: int = 1200,
) -> np.ndarray:
    """Extract penultimate dense embeddings from a model."""
    embed_layer = None
    for candidate in ("penultimate_embedding", "head_dense_1", "ela_gap"):
        try:
            embed_layer = model.get_layer(candidate)
            break
        except ValueError:
            continue
    if embed_layer is None:
        # Second-to-last layer fallback
        embed_layer = model.layers[-2]

    feat_model = keras.Model(inputs=model.inputs, outputs=embed_layer.output)
    chunks = []
    seen = 0
    for batch in dataset:
        x = batch[0] if isinstance(batch, tuple) else batch
        emb = feat_model(x, training=False).numpy()
        chunks.append(np.asarray(emb, dtype=np.float32))
        seen += len(emb)
        if seen >= max_samples:
            break
    return np.concatenate(chunks, axis=0)[:max_samples]


def plot_tsne_embeddings(
    panels: list[tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Compute and plot 2D t-SNE projections of penultimate latent embeddings."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.5), squeeze=False)
    for ax, (title, embeddings, labels) in zip(axes[0], panels):
        n = len(embeddings)
        perplexity = min(30, max(5, (n - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
        coords = tsne.fit_transform(embeddings)

        au_mask = labels == 0
        tp_mask = labels == 1
        ax.scatter(coords[au_mask, 0], coords[au_mask, 1], s=14, alpha=0.65, c="#2ecc71", label="Authentic")
        ax.scatter(coords[tp_mask, 0], coords[tp_mask, 1], s=14, alpha=0.65, c="#e74c3c", label="Tampered")
        ax.set_title(f"t-SNE Latent Space — {title}")
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved t-SNE plot         : {output_path}")


# ── Perturbation Robustness Stress Test ──────────────────────────────────────

def _perturb_image(pil_img: Image.Image, kind: str, level: float) -> Image.Image:
    """Apply synthetic post-processing degradation to a raw RGB PIL Image."""
    rgb = pil_img.convert("RGB")
    if kind == "jpeg":
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=int(level))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "noise":
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        rng = np.random.default_rng(42)
        noisy = np.clip(arr + rng.normal(0.0, float(level), size=arr.shape), 0.0, 1.0)
        return Image.fromarray((noisy * 255.0).astype(np.uint8), mode="RGB")
    if kind == "resize":
        w, h = rgb.size
        nw, nh = max(32, int(w * level)), max(32, int(h * level))
        return rgb.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return rgb


def run_robustness_suite(
    items: list[dict],
    raw_dir: Path,
    new_model_path: Path | None,
    new_meta: dict,
    leg_model_path: Path | None,
    leg_meta: dict,
    ens_cfg: EnsembleConfig,
    output_path: Path,
    max_samples: int = 200,
) -> dict:
    """Evaluate degradation under JPEG, Gaussian noise, and resize (1 model in VRAM at a time)."""
    au = [it for it in items if it["label"] == 0][: max_samples // 2]
    tp = [it for it in items if it["label"] == 1][: max_samples // 2]
    sub_items = au + tp

    loaded_raw: list[tuple[Image.Image, int]] = []
    for it in sub_items:
        rp = find_raw_image_path(it["cls"], it["stem"], raw_dir=raw_dir)
        if rp is not None:
            try:
                with Image.open(rp) as im:
                    loaded_raw.append((im.convert("RGB"), int(it["label"])))
            except Exception:
                continue

    if len(loaded_raw) < 10:
        print("  [WARN] Not enough raw images found for robustness test; skipping.")
        return {}

    y_sub = np.array([lbl for _, lbl in loaded_raw], dtype=int)
    include_fft = bool(new_meta.get("include_fft", True))
    ela_only = bool(new_meta.get("ela_only", False))
    leg_preproc = leg_meta.get("preprocessing", "rescale_255")

    scenarios = [
        ("Clean", "clean", 0.0),
        ("JPEG Q80", "jpeg", 80),
        ("JPEG Q70", "jpeg", 70),
        ("JPEG Q60", "jpeg", 60),
        ("Noise 0.02", "noise", 0.02),
        ("Noise 0.04", "noise", 0.04),
        ("Resize 0.75x", "resize", 0.75),
        ("Resize 0.50x", "resize", 0.50),
    ]

    leg_probs_by_sc: dict[str, np.ndarray] = {}
    new_probs_by_sc: dict[str, np.ndarray] = {}

    # Pass 1: Legacy ResNet50 alone in VRAM
    if leg_model_path is not None and leg_model_path.exists():
        leg_m = keras.models.load_model(leg_model_path, custom_objects=get_custom_objects(), compile=False)
        for label_name, kind, lvl in scenarios:
            probs = []
            for pil_im, _ in loaded_raw:
                pert = pil_im if kind == "clean" else _perturb_image(pil_im, kind, lvl)
                ela_im = convert_to_ela_image(pert, quality=90)
                arr = np.asarray(ela_im, dtype=np.float32)
                arr = resnet50_preprocess(arr) if leg_preproc == "resnet50_caffe" else (arr / 255.0)
                probs.append(float(leg_m(tf.convert_to_tensor(arr[None, ...]), training=False).numpy()[0][0]))
            leg_probs_by_sc[label_name] = apply_temperature(np.array(probs), ens_cfg.temp_legacy)
        del leg_m
        keras.backend.clear_session()
        gc.collect()

    # Pass 2: Multi-Branch alone in VRAM
    if new_model_path is not None and new_model_path.exists():
        new_m = keras.models.load_model(new_model_path, custom_objects=get_custom_objects(), compile=False)
        for label_name, kind, lvl in scenarios:
            probs = []
            for pil_im, _ in loaded_raw:
                pert = pil_im if kind == "clean" else _perturb_image(pil_im, kind, lvl)
                bundle = extract_multibranch_inputs(pert, include_fft=include_fft and not ela_only)
                if ela_only:
                    bundle = {"multi_ela_input": bundle["multi_ela_input"]}
                batched = {k: tf.convert_to_tensor(v[None, ...], dtype=tf.float32) for k, v in bundle.items()}
                probs.append(float(new_m(batched, training=False).numpy()[0][0]))
            new_probs_by_sc[label_name] = apply_temperature(np.array(probs), ens_cfg.temp_new)
        del new_m
        keras.backend.clear_session()
        gc.collect()

    curves: dict[str, list[float]] = {}
    if leg_probs_by_sc:
        curves["Legacy ResNet50"] = []
    if new_probs_by_sc:
        curves["Multi-Branch"] = []
    if leg_probs_by_sc and new_probs_by_sc:
        curves["Weighted Ensemble"] = []

    for label_name, _, _ in scenarios:
        if leg_probs_by_sc:
            p_l = leg_probs_by_sc[label_name]
            curves["Legacy ResNet50"].append(float(accuracy_score(y_sub, (p_l >= 0.5).astype(int))))
        if new_probs_by_sc:
            p_n = new_probs_by_sc[label_name]
            curves["Multi-Branch"].append(float(accuracy_score(y_sub, (p_n >= 0.5).astype(int))))
        if leg_probs_by_sc and new_probs_by_sc:
            p_e = ens_cfg.weight_new * new_probs_by_sc[label_name] + ens_cfg.weight_legacy * leg_probs_by_sc[label_name]
            curves["Weighted Ensemble"].append(float(accuracy_score(y_sub, (p_e >= ens_cfg.threshold).astype(int))))

        print(f"    Robustness [{label_name:12s}] -> " + " | ".join(f"{k}: {v[-1]:.3f}" for k, v in curves.items()))

    # Plot curves
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_idx = np.arange(len(scenarios))
    for name, vals in curves.items():
        ax.plot(x_idx, vals, "o-", lw=2.2, label=name)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([s[0] for s in scenarios], rotation=25, ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(f"Post-Processing Perturbation Robustness (n={len(loaded_raw)} CASIA v2.0 val samples)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved robustness curves  : {output_path}")

    return {"scenarios": [s[0] for s in scenarios], "accuracies": curves}


def plot_gradcam_sample_grid(
    model: keras.Model,
    meta: dict,
    items: list[dict],
    raw_dir: Path,
    output_path: Path,
    is_multibranch: bool,
    n_samples: int = 4,
) -> None:
    """Save an academic visualization grid: Original | ELA (Q90) | Grad-CAM++."""
    chosen = [it for it in items if it["label"] == 1][: n_samples // 2] + [
        it for it in items if it["label"] == 0
    ][: n_samples - n_samples // 2]

    rows = []
    include_fft = bool(meta.get("include_fft", True))
    ela_only = bool(meta.get("ela_only", False))
    preproc = meta.get("preprocessing", "rescale_255")
    gcam_layer = meta.get("gradcam_layer", None)

    for it in chosen:
        rp = find_raw_image_path(it["cls"], it["stem"], raw_dir=raw_dir)
        if rp is None:
            continue
        with Image.open(rp) as im:
            orig = im.convert("RGB").resize((224, 224), Image.LANCZOS)
        ela_im = convert_to_ela_image(orig, quality=90)

        if is_multibranch:
            bundle = extract_multibranch_inputs(orig, include_fft=include_fft and not ela_only)
            if ela_only:
                bundle = {"multi_ela_input": bundle["multi_ela_input"]}
            m_in = {k: v[None, ...] for k, v in bundle.items()}
        else:
            arr = np.asarray(ela_im, dtype=np.float32)
            arr = resnet50_preprocess(arr) if preproc == "resnet50_caffe" else (arr / 255.0)
            m_in = arr[None, ...]

        m_in_t = {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in m_in.items()} if isinstance(m_in, dict) else tf.convert_to_tensor(m_in, dtype=tf.float32)
        prob = float(model(m_in_t, training=False).numpy()[0][0])
        hm = compute_gradcam_pp(model, m_in, last_conv_layer_name=gcam_layer)
        ov = overlay_heatmap(orig, hm)
        rows.append((orig, ela_im, ov, "Tampered" if it["label"] == 1 else "Authentic", prob))

    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.2 * len(rows)), squeeze=False)
    for r_idx, (orig, ela_im, ov, true_lbl, prob) in enumerate(rows):
        pred_lbl = "Tampered" if prob >= 0.5 else "Authentic"
        axes[r_idx, 0].imshow(orig)
        axes[r_idx, 0].set_title(f"Original (True: {true_lbl})", fontsize=10)
        axes[r_idx, 1].imshow(ela_im)
        axes[r_idx, 1].set_title("ELA (Q=90)", fontsize=10)
        axes[r_idx, 2].imshow(ov)
        axes[r_idx, 2].set_title(f"Grad-CAM++ (Pred: {pred_lbl} {prob:.2f})", fontsize=10)
        for c_idx in range(3):
            axes[r_idx, c_idx].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved Grad-CAM++ grid    : {output_path}")


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepForensics: Comprehensive Deep Learning Evaluation Suite"
    )
    parser.add_argument("--new-model", type=Path, default=DEFAULT_NEW_MODEL)
    parser.add_argument("--legacy-model", type=Path, default=None)
    parser.add_argument("--ensemble-config", type=Path, default=DEFAULT_ENSEMBLE_CONFIG_PATH)
    parser.add_argument("--ela-dir", type=Path, default=DEFAULT_ELA_DIR)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tta", action="store_true", help="Enable 4-view Test-Time Augmentation.")
    parser.add_argument("--tsne", action="store_true", help="Generate t-SNE latent space plot.")
    parser.add_argument("--robustness", action="store_true", help="Run perturbation robustness test.")
    parser.add_argument("--gradcam-samples", action="store_true", help="Generate Grad-CAM++ grid.")
    parser.add_argument("--all", action="store_true", help="Run all academic diagnostics.")
    parser.add_argument("--smoke-test", action="store_true", help="Synthetic verification of all plots.")
    args = parser.parse_args()

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    do_tsne = args.tsne or args.all
    do_robust = args.robustness or args.all
    do_gcam = args.gradcam_samples or args.all

    print("=" * 68)
    print("  DeepForensics — Deep Learning Evaluation & XAI Suite (v2)")
    print("=" * 68)

    if args.smoke_test:
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=80)
        p1 = np.clip(y * 0.7 + rng.normal(0.15, 0.2, size=80), 0.01, 0.99)
        p2 = np.clip(y * 0.85 + rng.normal(0.08, 0.15, size=80), 0.01, 0.99)
        pe = 0.75 * p2 + 0.25 * p1
        res = {
            "Legacy ResNet50": compute_metrics(y, p1),
            "Multi-Branch": compute_metrics(y, p2),
            "Weighted Ensemble": compute_metrics(y, pe),
        }
        tmp_cm = METRICS_DIR / "tmp_smoke_cm.png"
        tmp_roc = METRICS_DIR / "tmp_smoke_roc.png"
        tmp_tsne = METRICS_DIR / "tmp_smoke_tsne.png"
        plot_confusion_matrices(res, tmp_cm)
        plot_roc_comparison(res, tmp_roc)
        emb = rng.normal(0, 1, size=(80, 32)).astype(np.float32) + y[:, None] * 1.5
        plot_tsne_embeddings([("Smoke Latent Space", emb, y)], tmp_tsne)
        for p in (tmp_cm, tmp_roc, tmp_tsne):
            p.unlink(missing_ok=True)
        print("  [SMOKE TEST] Evaluation, ROC, CM, and t-SNE plots PASSED!")
        return

    legacy_path = _resolve_legacy_model(args.legacy_model)
    has_new = args.new_model.exists()
    has_leg = legacy_path is not None

    if not has_new and not has_leg:
        print("[ERROR] No trained models found. Train with train_model.py or train_multibranch.py.")
        sys.exit(1)

    ens_cfg = EnsembleConfig.load(args.ensemble_config)
    new_meta = _read_meta(args.new_model) if has_new else {}
    leg_meta = _read_meta(legacy_path) if has_leg else {}

    manifest = build_or_load_manifest(
        manifest_path=args.manifest,
        source_dir=args.features_dir if has_new else args.ela_dir,
    )
    val_items = manifest["val"]

    # Filter to items present in all needed caches
    include_fft = bool(new_meta.get("include_fft", True))
    ela_only = bool(new_meta.get("ela_only", False))
    need_fft = include_fft and (not ela_only)

    aligned = []
    for it in val_items:
        cls, stem = it["cls"], it["stem"]
        ok_leg = (not has_leg) or (args.ela_dir / cls / f"{stem}.png").exists()
        ok_new = (not has_new) or (
            (args.features_dir / "ela_q70" / cls / f"{stem}.png").exists()
            and (args.features_dir / "ela_q85" / cls / f"{stem}.png").exists()
            and (args.features_dir / "ela_q95" / cls / f"{stem}.png").exists()
            and (ela_only or (args.features_dir / "raw_224" / cls / f"{stem}.png").exists())
            and ((not need_fft) or (args.features_dir / "fft_224" / cls / f"{stem}.png").exists())
        )
        if ok_leg and ok_new:
            aligned.append(it)

    if not aligned:
        print("[ERROR] No validation images found matching available models.")
        sys.exit(1)

    y_true = np.array([int(it["label"]) for it in aligned], dtype=int)
    print(f"  Validation samples : {len(aligned)} | TTA={args.tta}")

    results: dict[str, dict] = {}
    tsne_panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    p_leg_cal: np.ndarray | None = None
    p_new_cal: np.ndarray | None = None
    leg_model: keras.Model | None = None
    new_model: keras.Model | None = None

    # ── Evaluate Legacy ResNet50 ────────────────────────────────────────────
    if has_leg and legacy_path is not None:
        leg_preproc = leg_meta.get("preprocessing", "rescale_255")
        print(f"\n[EVAL] Legacy ResNet50 ({legacy_path}, preprocessing={leg_preproc})...")
        leg_model = keras.models.load_model(legacy_path, custom_objects=get_custom_objects(), compile=False)

        leg_paths = [str(args.ela_dir / it["cls"] / f"{it['stem']}.png") for it in aligned]

        def load_leg(p):
            im = _decode_rgb(p)
            return resnet50_preprocess(im) if leg_preproc == "resnet50_caffe" else (im / 255.0)

        leg_ds = (
            tf.data.Dataset.from_tensor_slices(leg_paths)
            .map(load_leg, num_parallel_calls=4)
            .batch(args.batch_size)
            .prefetch(1)
        )
        p_leg_raw = np.concatenate(
            [predict_with_tta(leg_model, b.numpy(), use_tta=args.tta) for b in leg_ds],
            axis=0,
        )
        p_leg_cal = apply_temperature(p_leg_raw, ens_cfg.temp_legacy)
        results["Legacy ResNet50"] = compute_metrics(y_true, p_leg_cal, threshold=0.50)

        if do_tsne:
            emb_leg = _extract_embeddings(leg_model, leg_ds)
            tsne_panels.append(("Legacy ResNet50", emb_leg, y_true[: len(emb_leg)]))

        if do_gcam and not has_new:
            print("\n[XAI] Generating Grad-CAM++ localization sample grid (Legacy ResNet50)...")
            plot_gradcam_sample_grid(
                leg_model, leg_meta, aligned, args.raw_dir,
                METRICS_DIR / "gradcam_samples.png", is_multibranch=False,
            )

        # Free Legacy model completely before loading Multi-Branch (prevents VRAM OOM)
        del leg_model
        leg_model = None
        keras.backend.clear_session()
        gc.collect()

    # ── Evaluate Multi-Branch Model ─────────────────────────────────────────
    if has_new:
        print(f"\n[EVAL] Multi-Branch Model ({args.new_model})...")
        new_model = keras.models.load_model(args.new_model, custom_objects=get_custom_objects(), compile=False)

        slices = {
            "p70": [str(args.features_dir / "ela_q70" / it["cls"] / f"{it['stem']}.png") for it in aligned],
            "p85": [str(args.features_dir / "ela_q85" / it["cls"] / f"{it['stem']}.png") for it in aligned],
            "p95": [str(args.features_dir / "ela_q95" / it["cls"] / f"{it['stem']}.png") for it in aligned],
            "praw": [str(args.features_dir / "raw_224" / it["cls"] / f"{it['stem']}.png") for it in aligned],
        }
        if need_fft:
            slices["pfft"] = [str(args.features_dir / "fft_224" / it["cls"] / f"{it['stem']}.png") for it in aligned]

        def load_mb(rec):
            mela = tf.concat([_decode_rgb(rec["p70"]), _decode_rgb(rec["p85"]), _decode_rgb(rec["p95"])], axis=-1)
            out = {"multi_ela_input": mela}
            if not ela_only:
                out["raw_rgb_input"] = _decode_rgb(rec["praw"])
                if need_fft:
                    out["fft_input"] = _decode_gray(rec["pfft"])
            return out

        mb_ds = (
            tf.data.Dataset.from_tensor_slices(slices)
            .map(load_mb, num_parallel_calls=4)
            .batch(args.batch_size)
            .prefetch(1)
        )
        p_new_raw = np.concatenate(
            [predict_with_tta(new_model, {k: v.numpy() for k, v in b.items()}, use_tta=args.tta) for b in mb_ds],
            axis=0,
        )
        p_new_cal = apply_temperature(p_new_raw, ens_cfg.temp_new)
        results["Multi-Branch"] = compute_metrics(y_true, p_new_cal, threshold=0.50)

        if do_tsne:
            emb_new = _extract_embeddings(new_model, mb_ds)
            tsne_panels.append(("Multi-Branch", emb_new, y_true[: len(emb_new)]))

        if do_gcam:
            print("\n[XAI] Generating Grad-CAM++ localization sample grid (Multi-Branch)...")
            plot_gradcam_sample_grid(
                new_model, new_meta, aligned, args.raw_dir,
                METRICS_DIR / "gradcam_samples.png", is_multibranch=True,
            )

        del new_model
        new_model = None
        keras.backend.clear_session()
        gc.collect()

    # ── Evaluate Calibrated Weighted Ensemble ───────────────────────────────
    if p_new_cal is not None and p_leg_cal is not None:
        p_ens = ens_cfg.weight_new * p_new_cal + ens_cfg.weight_legacy * p_leg_cal
        results["Weighted Ensemble"] = compute_metrics(y_true, p_ens, threshold=ens_cfg.threshold)

    # ── Print Comparison Table ──────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(f"  {'Model / Mode':22s} | {'Acc':>6s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s} | {'AUC':>6s} | {'ECE':>6s}")
    print("-" * 78)
    summary_json: dict = {"val_samples": int(len(aligned)), "tta": bool(args.tta), "models": {}}
    for name, m in results.items():
        print(
            f"  {name:22s} | {m['accuracy']:6.4f} | {m['precision']:6.4f} | "
            f"{m['recall']:6.4f} | {m['f1_score']:6.4f} | {m['roc_auc']:6.4f} | {m['ece']:6.4f}"
        )
        summary_json["models"][name] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in m.items()
            if k not in ("fpr", "tpr")
        }
    print("-" * 78)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrices(results, METRICS_DIR / "confusion_matrix.png")
    plot_roc_comparison(results, METRICS_DIR / "roc_curve.png")

    if do_tsne and tsne_panels:
        print("\n[XAI] Generating 2D t-SNE latent space embeddings...")
        plot_tsne_embeddings(tsne_panels, METRICS_DIR / "tsne_embeddings.png")

    if do_robust:
        print("\n[ROBUSTNESS] Running post-processing perturbation stress test...")
        rob = run_robustness_suite(
            aligned,
            args.raw_dir,
            args.new_model if has_new else None,
            new_meta,
            legacy_path if has_leg else None,
            leg_meta,
            ens_cfg,
            METRICS_DIR / "robustness_curves.png",
        )
        summary_json["robustness"] = rob

    summary_path = METRICS_DIR / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(f"  Saved evaluation summary : {summary_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
