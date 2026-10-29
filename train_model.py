"""
DeepForensics — Two-Stage ResNet50 + ANN Training Pipeline
===========================================================
Trains a forgery detector on pre-computed ELA images using a frozen-then-
fine-tuned ResNet50 backbone with a custom classification head.

Hardware: NVIDIA RTX 3060 (12GB VRAM) + 16GB DDR4 RAM
Strategy: mixed_float16, pre-cached ELA on disk, AUTOTUNE prefetching.

Usage:
    python train_model.py                 # full two-stage training
    python train_model.py --smoke-test    # 1 epoch on tiny synthetic data
    python train_model.py --stage 1       # only Stage 1 (frozen backbone)
    python train_model.py --stage 2       # only Stage 2 (fine-tune)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# TensorFlow imports after env config (mixed precision must be set first)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduce TF verbosity

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision

from utils.ela import convert_to_ela_image


# ── Configuration ────────────────────────────────────────────────────────────

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 42
ELA_DIR = Path("dataset/ela_224")
MODEL_PATH = Path("ela_resnet50_best.keras")

# Stage 1: frozen backbone
STAGE1_EPOCHS = 10
STAGE1_LR = 1e-3

# Stage 2: fine-tune top layers (increased for better adaptation to ELA)
STAGE2_EPOCHS = 25  # Increased from 20
STAGE2_LR = 5e-6  # Reduced from 1e-5 for more careful fine-tuning
FINE_TUNE_LAYERS = 100  # Increased from 50 to adapt more layers to ELA
EARLY_STOPPING_PATIENCE = 7  # Increased from 5


# ── Data Pipeline ────────────────────────────────────────────────────────────

def build_datasets(
    ela_dir: Path,
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
    seed: int = SEED,
    validation_split: float = 0.2,
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[int, float]]:
    """Build train/val datasets from directory structure using AUTOTUNE.

    Returns:
        train_ds, val_ds, class_weights
    """
    train_ds = keras.utils.image_dataset_from_directory(
        ela_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        color_mode="rgb",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        ela_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        color_mode="rgb",
    )

    # Calculate class weights to handle imbalance (7,492 Au vs 5,125 Tp)
    # Count samples per class from class_names
    # Assuming directories are alphabetically ordered: Au=0, Tp=1
    au_count = len(list((ela_dir / "Au").glob("*")))
    tp_count = len(list((ela_dir / "Tp").glob("*")))
    total = au_count + tp_count

    # Class weights inversely proportional to frequency
    class_weight = {
        0: total / (2.0 * au_count),  # Authentic
        1: total / (2.0 * tp_count),  # Tampered
    }

    print(f"  Dataset: {au_count} Authentic, {tp_count} Tampered")
    print(f"  Class weights: Au={class_weight[0]:.3f}, Tp={class_weight[1]:.3f}")

    # Data augmentation for training only
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),  # Increased from 0.1 for more variety
        layers.RandomZoom(0.1),  # Add zoom augmentation
    ])

    # Normalize to [0, 1] (models expect float32 in [0,1])
    normalization_layer = layers.Rescaling(1.0 / 255.0)

    # Apply augmentation only to training set
    train_ds = train_ds.map(
        lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Validation set gets normalization only (no augmentation)
    val_ds = val_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch for GPU overlap
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_weight


def build_smoke_test_dataset(
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create tiny synthetic dataset for pipeline verification."""
    # 16 synthetic authentic (solid colors)
    auth_images = []
    for i in range(16):
        c = (i * 16, 80, 160)
        img = np.full((*img_size, 3), c, dtype=np.uint8)
        auth_images.append(img)

    # 16 synthetic tampered (solid + patch)
    tamp_images = []
    for i in range(16):
        img = np.full((*img_size, 3), (30, 90, 160), dtype=np.uint8)
        patch_c = (255, 80 + i * 10, 80)
        img[50:150, 50:150] = patch_c
        tamp_images.append(img)

    x = np.vstack([auth_images, tamp_images]).astype(np.float32) / 255.0
    y = np.array([0] * 16 + [1] * 16, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(32, seed=SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # Use same dataset for train/val in smoke test
    return ds, ds


# ── Model Architecture ───────────────────────────────────────────────────────

def build_model(
    img_size: tuple[int, int] = IMG_SIZE,
    fine_tune_at: int | None = None,
) -> keras.Model:
    """Build ResNet50 + ANN head model.

    Args:
        img_size: Input resolution (H, W).
        fine_tune_at: If int, unfreeze layers from this index onwards
                      (i.e., backbone.layers[fine_tune_at:] become trainable).
                      If None, all backbone layers frozen.
    """
    inputs = keras.Input(shape=(*img_size, 3))

    # ResNet50 backbone (ImageNet weights, no top)
    # Note: Data augmentation is now applied in the dataset pipeline,
    # not here, so it only runs during training (not validation)
    backbone = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        input_shape=(*img_size, 3),
    )

    # Freeze / unfreeze logic
    if fine_tune_at is None:
        backbone.trainable = False
    else:
        backbone.trainable = True
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False

    # Classification head (ANN)
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dense(512, activation="relu")(x)  # Increased from 256
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)  # Reduced from 0.5 to allow more learning
    x = layers.Dense(128, activation="relu")(x)  # Added second dense layer
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    # Final sigmoid kept in float32 for numerical stability in BCE
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = keras.Model(inputs, outputs)
    return model


def compile_model(model: keras.Model, learning_rate: float) -> None:
    """Compile model with mixed-precision-aware optimizer and BCE loss."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    # With mixed_float16, use LossScaleOptimizer automatically via compile
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


# ── Callbacks ────────────────────────────────────────────────────────────────

def get_callbacks(model_path: Path, patience: int = EARLY_STOPPING_PATIENCE) -> list[keras.callbacks.Callback]:
    """Return standard callbacks for training."""
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger("training_log.csv", append=True),
    ]


# ── Training Stages ──────────────────────────────────────────────────────────

def run_stage1(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_path: Path,
    epochs: int,
    class_weight: dict = None,
) -> keras.Model:
    """Stage 1: Train ANN head with frozen ResNet50 backbone."""
    print("\n" + "=" * 60)
    print("  STAGE 1: Frozen Backbone — Training ANN Head")
    print("=" * 60)
    print(f"  Epochs: {epochs} | LR: {STAGE1_LR} | Batch: {BATCH_SIZE}")
    print(f"  Mixed precision: {mixed_precision.global_policy().name}")

    model = build_model(fine_tune_at=None)
    compile_model(model, STAGE1_LR)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=get_callbacks(model_path),
        verbose=1,
    )
    return model


def run_stage2(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_path: Path,
    epochs: int,
    class_weight: dict = None,
) -> keras.Model:
    """Stage 2: Fine-tune top 50 layers of ResNet50 + ANN head."""
    print("\n" + "=" * 60)
    print(f"  STAGE 2: Fine-Tune Top {FINE_TUNE_LAYERS} Layers")
    print("=" * 60)
    print(f"  Epochs: {epochs} | LR: {STAGE2_LR} | Batch: {BATCH_SIZE}")

    # Load best weights from Stage 1
    model = keras.models.load_model(model_path)

    # Unfreeze top FINE_TUNE_LAYERS of backbone
    # In Keras 3.x, the saved model flattens ResNet50 layers into the
    # top-level model (no nested "resnet50" sub-model). Identify backbone
    # layers by their conv/bn naming pattern and freeze all but the last N.
    try:
        backbone = model.get_layer("resnet50")
        backbone.trainable = True
        backbone_layers = backbone.layers
    except ValueError:
        # Keras 3.x flattened model — collect backbone layers by name
        backbone_layers = [
            l for l in model.layers
            if l.name.startswith(("conv", "pool")) and not l.name.startswith("global")
        ]
        for layer in backbone_layers:
            layer.trainable = True

    # Freeze everything except the last FINE_TUNE_LAYERS backbone layers
    for layer in backbone_layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    trainable_count = sum(1 for l in backbone_layers if l.trainable)
    frozen_count = len(backbone_layers) - trainable_count
    print(f"  Backbone layers: {len(backbone_layers)} total, "
          f"{frozen_count} frozen, {trainable_count} trainable")

    # Recompile with lower LR (critical for fine-tuning)
    compile_model(model, STAGE2_LR)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=get_callbacks(model_path),
        verbose=1,
    )
    return model


# ── Main Entry Point ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepForensics: Two-Stage ResNet50 + ANN Training"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run 1 epoch on tiny synthetic data to verify pipeline.",
    )
    parser.add_argument(
        "--stage", type=int, choices=[1, 2],
        help="Run only a specific stage (1 or 2).",
    )
    parser.add_argument(
        "--epochs1", type=int, default=STAGE1_EPOCHS,
        help=f"Stage 1 epochs (default: {STAGE1_EPOCHS}).",
    )
    parser.add_argument(
        "--epochs2", type=int, default=STAGE2_EPOCHS,
        help=f"Stage 2 epochs (default: {STAGE2_EPOCHS}).",
    )
    args = parser.parse_args()

    # Mixed precision MUST be set before any other TF operations
    mixed_precision.set_global_policy("mixed_float16")
    print(f"[TF] Mixed precision policy: {mixed_precision.global_policy().name}")
    print(f"[TF] GPU devices: {tf.config.list_physical_devices('GPU')}")

    # Use local variables from args (no need to modify module constants)
    epochs1 = args.epochs1
    epochs2 = args.epochs2

    # Verify GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # ── Prepare datasets ─────────────────────────────────────────────────
    class_weight = None
    if args.smoke_test:
        print("\n[SMOKE TEST] Using synthetic dataset...")
        train_ds, val_ds = build_smoke_test_dataset()
    else:
        if not ELA_DIR.exists() or not any(ELA_DIR.iterdir()):
            print(f"\n[ERROR] ELA dataset not found at {ELA_DIR.resolve()}")
            print("        Run: python prepare_dataset.py")
            sys.exit(1)
        print(f"\n[DATA] Loading from {ELA_DIR.resolve()}")
        train_ds, val_ds, class_weight = build_datasets(ELA_DIR)

    # ── Training ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    if args.stage == 1 or (args.stage is None and not args.smoke_test):
        run_stage1(train_ds, val_ds, MODEL_PATH, epochs=epochs1, class_weight=class_weight)

    if args.stage == 2:
        if not MODEL_PATH.exists():
            print(f"[ERROR] {MODEL_PATH} not found. Run Stage 1 first.")
            sys.exit(1)
        run_stage2(train_ds, val_ds, MODEL_PATH, epochs=epochs2, class_weight=class_weight)

    elif args.stage is None and not args.smoke_test:
        if MODEL_PATH.exists():
            run_stage2(train_ds, val_ds, MODEL_PATH, epochs=epochs2, class_weight=class_weight)
        else:
            print(f"[WARN] {MODEL_PATH} not found after Stage 1 — skipping Stage 2")

    elif args.smoke_test:
        # For smoke test, just run a quick Stage 1 equivalent
        model = build_model(fine_tune_at=None)
        compile_model(model, STAGE1_LR)
        model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
        model.save(MODEL_PATH)
        print(f"\n[SMOKE TEST] Model saved to {MODEL_PATH}")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"  Total training time: {elapsed/60:.1f} minutes")
    print(f"  Best model saved to: {MODEL_PATH.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()