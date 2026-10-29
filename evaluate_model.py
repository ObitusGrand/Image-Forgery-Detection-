"""
DeepForensics — Model Evaluation & Metrics Generator
=====================================================
Loads the trained model and computes academic-grade evaluation metrics
on the validation set. Generates confusion matrix and ROC curve plots
for inclusion in the web UI and project report.

Usage:
    python evaluate_model.py                    # evaluate on real ELA dataset
    python evaluate_model.py --smoke-test       # evaluate on synthetic data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = Path("ela_resnet50_best.keras")
ELA_DIR = Path("dataset/ela_224")
METRICS_DIR = Path("static/metrics")
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 42


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_validation_dataset(
    ela_dir: Path,
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
    seed: int = SEED,
) -> tf.data.Dataset:
    """Load validation split from ELA dataset."""
    val_ds = keras.utils.image_dataset_from_directory(
        ela_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        color_mode="rgb",
    )

    # Normalize to [0, 1]
    normalization = keras.layers.Rescaling(1.0 / 255.0)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    return val_ds


def load_smoke_test_dataset(
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
) -> tf.data.Dataset:
    """Create synthetic validation dataset."""
    # 16 authentic + 16 tampered
    auth_images = []
    for i in range(16):
        c = (i * 16, 80, 160)
        img = np.full((*img_size, 3), c, dtype=np.uint8)
        auth_images.append(img)

    tamp_images = []
    for i in range(16):
        img = np.full((*img_size, 3), (30, 90, 160), dtype=np.uint8)
        patch_c = (255, 80 + i * 10, 80)
        img[50:150, 50:150] = patch_c
        tamp_images.append(img)

    x = np.vstack([auth_images, tamp_images]).astype(np.float32) / 255.0
    y = np.array([0] * 16 + [1] * 16, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    return ds


# ── Metrics Computation ──────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute classification metrics."""
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
    y_true = y_true.astype(int).flatten()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    # ROC-AUC
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics["roc_auc"] = auc(fpr, tpr)
        metrics["fpr"] = fpr
        metrics["tpr"] = tpr
    else:
        metrics["roc_auc"] = 0.0
        metrics["fpr"] = np.array([0, 1])
        metrics["tpr"] = np.array([0, 1])

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return metrics


# ── Visualization ────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Generate and save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Labels
    classes = ["Authentic", "Tampered"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to: {output_path}")


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_path: Path) -> None:
    """Generate and save ROC curve plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved to: {output_path}")


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepForensics: Evaluate trained model and generate metrics"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Evaluate on synthetic data instead of real dataset.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  DeepForensics - Model Evaluation")
    print("=" * 60)

    # Check model exists
    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Model not found: {MODEL_PATH.resolve()}")
        print("        Run: python train_model.py")
        sys.exit(1)

    # Load model
    print(f"\n[1/4] Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)

    # Load validation data
    if args.smoke_test:
        print("[2/4] Loading smoke test dataset...")
        val_ds = load_smoke_test_dataset()
    else:
        if not ELA_DIR.exists() or not any(ELA_DIR.iterdir()):
            print(f"\n[ERROR] ELA dataset not found: {ELA_DIR.resolve()}")
            print("        Run: python prepare_dataset.py")
            sys.exit(1)
        print(f"[2/4] Loading validation dataset from {ELA_DIR}...")
        val_ds = load_validation_dataset(ELA_DIR)

    # Run predictions
    print("[3/4] Running predictions on validation set...")
    y_true_list = []
    y_pred_list = []

    for x_batch, y_batch in val_ds:
        y_pred_batch = model.predict(x_batch, verbose=0)
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(y_pred_batch)

    y_true = np.concatenate(y_true_list)
    y_pred_proba = np.concatenate(y_pred_list)

    # Compute metrics
    print("[4/4] Computing metrics and generating plots...")
    metrics = compute_metrics(y_true, y_pred_proba)

    # Print results
    print("\n" + "-" * 60)
    print("  Evaluation Results")
    print("-" * 60)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("-" * 60)
    print(f"  Confusion Matrix:")
    print(f"    {metrics['confusion_matrix']}")
    print("-" * 60)

    # Save plots
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    cm_path = METRICS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], cm_path)

    roc_path = METRICS_DIR / "roc_curve.png"
    plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["roc_auc"], roc_path)

    print("\n" + "=" * 60)
    print("  Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
