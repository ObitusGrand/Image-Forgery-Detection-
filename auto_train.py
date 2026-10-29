"""
DeepForensics — Auto-Tuning Training Pipeline (Keras Tuner)
=============================================================
Automatically searches for the best hyperparameters using Bayesian
optimization, then trains the final model with the optimal configuration.

Tunes: dropout rate, dense layer sizes, fine-tune depth, learning rates,
       augmentation intensity, and batch size.

Hardware: NVIDIA RTX 3060 (12GB VRAM) + 16GB DDR4 RAM
Strategy: mixed_float16, Bayesian search, auto-retrain on best params.

Features:
    - Checkpoint/resume: pipeline saves progress after each phase and
      resumes from the last completed phase on restart.
    - Temp files written to project-local directory (not C: drive).
    - Phase 2 grid reduced for reliability on 12GB VRAM.

Usage:
    python auto_train.py                    # full auto-tuning + final training
    python auto_train.py --max-trials 10    # limit search to 10 trials
    python auto_train.py --retrain-only     # skip search, train with best found params
    python auto_train.py --target-acc 0.95  # stop when accuracy >= 95%
    python auto_train.py --clean            # wipe checkpoints and start fresh
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision

# ── Configuration ────────────────────────────────────────────────────────────

IMG_SIZE = (224, 224)
SEED = 42
ELA_DIR = Path("dataset/ela_224")
MODEL_PATH = Path("ela_resnet50_best.keras")
TUNER_DIR = Path("tuner_results")
BEST_PARAMS_PATH = Path("best_hyperparams.json")

# Local temp directory — avoids filling up C: drive on Windows
CHECKPOINT_DIR = Path("checkpoints/auto_train")
TEMP_DIR = CHECKPOINT_DIR / "tmp"
PROGRESS_PATH = CHECKPOINT_DIR / "progress.json"

# Search space defaults
DEFAULT_MAX_TRIALS = 8
DEFAULT_TARGET_ACC = 0.94
SEARCH_EPOCHS_STAGE1 = 5   # Fewer epochs during search (faster)
SEARCH_EPOCHS_STAGE2 = 5   # Fewer epochs during search (was 8, reduced for reliability)
FINAL_EPOCHS_STAGE1 = 10   # Full epochs for final training
FINAL_EPOCHS_STAGE2 = 25   # Full epochs for final training


# ── Progress Tracking (Checkpoint / Resume) ──────────────────────────────────

def _ensure_dirs() -> None:
    """Create checkpoint and temp directories on the project drive."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def _load_progress() -> dict:
    """Load pipeline progress from disk, or return a blank state."""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {
        "phase1_complete": False,
        "best_hp_values": None,
        "phase2_results": [],
        "phase2_complete": False,
        "best_params": None,
        "final_stage1_complete": False,
        "final_stage2_complete": False,
    }


def _save_progress(progress: dict) -> None:
    """Persist pipeline progress to disk."""
    _ensure_dirs()
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def _clear_progress() -> None:
    """Wipe all checkpoint state so the next run starts fresh."""
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
    print("  Checkpoints cleared.")


def _temp_path(name: str) -> Path:
    """Return a path inside the project-local temp directory (D: drive)."""
    _ensure_dirs()
    return TEMP_DIR / name


# ── Data Pipeline ────────────────────────────────────────────────────────────

def get_class_weights(ela_dir: Path) -> dict[int, float]:
    """Calculate class weights for imbalanced dataset."""
    au_count = len(list((ela_dir / "Au").glob("*")))
    tp_count = len(list((ela_dir / "Tp").glob("*")))
    total = au_count + tp_count

    class_weight = {
        0: total / (2.0 * au_count),
        1: total / (2.0 * tp_count),
    }

    print(f"  Dataset: {au_count} Authentic, {tp_count} Tampered")
    print(f"  Class weights: Au={class_weight[0]:.3f}, Tp={class_weight[1]:.3f}")
    return class_weight


def build_datasets(
    ela_dir: Path,
    batch_size: int = 32,
    img_size: tuple[int, int] = IMG_SIZE,
    seed: int = SEED,
    rotation_factor: float = 0.15,
    zoom_factor: float = 0.1,
    flip_horizontal: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build train/val datasets with configurable augmentation."""
    train_ds = keras.utils.image_dataset_from_directory(
        ela_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        color_mode="rgb",
    )

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

    # Configurable augmentation (training only)
    aug_layers = []
    if flip_horizontal:
        aug_layers.append(layers.RandomFlip("horizontal"))
    aug_layers.append(layers.RandomRotation(rotation_factor))
    aug_layers.append(layers.RandomZoom(zoom_factor))
    data_augmentation = keras.Sequential(aug_layers)

    normalization_layer = layers.Rescaling(1.0 / 255.0)

    train_ds = train_ds.map(
        lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


# ── Tunable Model Builder ────────────────────────────────────────────────────

def build_tunable_model(hp) -> keras.Model:
    """Build model with hyperparameters from Keras Tuner.

    Hyperparameters tuned:
        - dense_1_units: First dense layer size [256, 512, 1024]
        - dense_2_units: Second dense layer size [64, 128, 256]
        - dropout_rate: Dropout rate [0.2, 0.3, 0.4, 0.5]
        - stage1_lr: Stage 1 learning rate [1e-4 to 1e-2]
    """
    dense_1 = hp.Choice("dense_1_units", values=[256, 512, 1024], default=512)
    dense_2 = hp.Choice("dense_2_units", values=[64, 128, 256], default=128)
    dropout_rate = hp.Choice("dropout_rate", values=[0.2, 0.3, 0.4], default=0.3)
    learning_rate = hp.Float("stage1_lr", min_value=1e-4, max_value=1e-2, sampling="log", default=1e-3)

    inputs = keras.Input(shape=(*IMG_SIZE, 3))

    backbone = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        input_shape=(*IMG_SIZE, 3),
    )
    backbone.trainable = False  # Stage 1: frozen

    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dense(dense_1, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_2, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
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
    return model


# ── Stage 2 Hyperparameter Search ────────────────────────────────────────────

def fine_tune_with_params(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    fine_tune_layers: int,
    stage2_lr: float,
    epochs: int,
    class_weight: dict,
    model_path: Path,
    patience: int = 7,
    use_temp_checkpoint: bool = False,
) -> tuple[keras.Model, dict]:
    """Fine-tune model with given hyperparameters and return metrics."""
    # Use project-local temp directory (D: drive) instead of system temp (C: drive)
    if use_temp_checkpoint:
        checkpoint_path = _temp_path(f"checkpoint_{id(model)}.keras")
    else:
        checkpoint_path = model_path
    # Unfreeze backbone layers (Keras 3.x compatible)
    try:
        backbone = model.get_layer("resnet50")
        backbone.trainable = True
        backbone_layers = backbone.layers
    except ValueError:
        backbone_layers = [
            l for l in model.layers
            if l.name.startswith(("conv", "pool")) and not l.name.startswith("global")
        ]
        for layer in backbone_layers:
            layer.trainable = True

    # Freeze all but top N layers
    for layer in backbone_layers[:-fine_tune_layers]:
        layer.trainable = False

    trainable_count = sum(1 for l in backbone_layers if l.trainable)
    print(f"  Fine-tuning {trainable_count}/{len(backbone_layers)} backbone layers")

    optimizer = keras.optimizers.Adam(learning_rate=stage2_lr)
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

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=patience, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max",
            factor=0.5, patience=3, min_lr=1e-7, verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Extract best metrics
    best_epoch = np.argmax(history.history["val_auc"])
    metrics = {
        "val_accuracy": float(history.history["val_accuracy"][best_epoch]),
        "val_precision": float(history.history["val_precision"][best_epoch]),
        "val_recall": float(history.history["val_recall"][best_epoch]),
        "val_auc": float(history.history["val_auc"][best_epoch]),
        "best_epoch": int(best_epoch) + 1,
    }

    # Copy from temp to final location if needed
    if use_temp_checkpoint and checkpoint_path.exists():
        shutil.copy2(str(checkpoint_path), str(model_path))
        checkpoint_path.unlink()  # Clean up temp file

    return model, metrics


# ── Auto-Tuning Orchestrator ─────────────────────────────────────────────────

def run_auto_tuning(
    ela_dir: Path,
    max_trials: int,
    target_acc: float,
) -> dict:
    """Run Bayesian hyperparameter search across configurable dimensions.

    Search Strategy:
    1. Use Keras Tuner Bayesian search for Stage 1 params (head architecture + LR)
    2. For each top candidate, run Stage 2 with multiple fine-tune configs
    3. Pick the combination that maximizes val_auc
    4. Retrain from scratch with the winning config at full epoch count

    Supports resume: reads progress.json and skips already-completed phases.
    """
    print("\n" + "=" * 70)
    print("  DeepForensics — AUTO-TUNING PIPELINE")
    print("=" * 70)
    print(f"  Max trials     : {max_trials}")
    print(f"  Target accuracy: {target_acc * 100:.0f}%")
    print(f"  GPU devices    : {tf.config.list_physical_devices('GPU')}")
    print(f"  Temp directory : {TEMP_DIR.resolve()}")
    print("=" * 70)

    progress = _load_progress()
    class_weight = get_class_weights(ela_dir)

    try:
        import keras_tuner as kt
    except ImportError:
        print("[ERROR] keras-tuner not installed. Install with:")
        print("        pip install keras-tuner")
        sys.exit(1)

    # ── Phase 1: Bayesian Search for Stage 1 (Head Architecture) ─────────
    if progress["phase1_complete"] and progress.get("best_hp_values"):
        print("\n" + "-" * 70)
        print("  PHASE 1: SKIPPED (already complete — resuming from checkpoint)")
        print("-" * 70)

        # Reconstruct best_hp from saved values
        best_hp_values = progress["best_hp_values"]
        print(f"  Restored: dense_1={best_hp_values['dense_1_units']}, "
              f"dense_2={best_hp_values['dense_2_units']}, "
              f"dropout={best_hp_values['dropout_rate']}, "
              f"lr={best_hp_values['stage1_lr']:.6f}")

    else:
        print("\n" + "-" * 70)
        print("  PHASE 1: Searching Head Architecture (Bayesian Optimization)")
        print("-" * 70)

        # Build datasets with default augmentation for search
        train_ds, val_ds = build_datasets(ela_dir, batch_size=32)

        tuner = kt.BayesianOptimization(
            hypermodel=build_tunable_model,
            objective=kt.Objective("val_auc", direction="max"),
            max_trials=max_trials,
            seed=SEED,
            directory=str(TUNER_DIR),
            project_name="deepforensics_v3",
            overwrite=False,  # Resume previous search if it exists
        )

        # Search callbacks
        search_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max",
                patience=3, restore_best_weights=True,
            ),
        ]

        print(f"\n  Running {max_trials} trials, {SEARCH_EPOCHS_STAGE1} epochs each...")
        tuner.search(
            train_ds,
            validation_data=val_ds,
            epochs=SEARCH_EPOCHS_STAGE1,
            class_weight=class_weight,
            callbacks=search_callbacks,
            verbose=1,
        )

        # Get top 3 results
        top_hps = tuner.get_best_hyperparameters(num_trials=min(3, max_trials))
        best_hp = top_hps[0]

        print("\n" + "-" * 70)
        print("  PHASE 1 RESULTS: Top Hyperparameters")
        print("-" * 70)
        for i, hp in enumerate(top_hps):
            print(f"  #{i+1}: dense_1={hp.get('dense_1_units')}, "
                  f"dense_2={hp.get('dense_2_units')}, "
                  f"dropout={hp.get('dropout_rate')}, "
                  f"lr={hp.get('stage1_lr'):.6f}")

        # Save Phase 1 results to progress
        best_hp_values = {
            "dense_1_units": best_hp.get("dense_1_units"),
            "dense_2_units": best_hp.get("dense_2_units"),
            "dropout_rate": best_hp.get("dropout_rate"),
            "stage1_lr": best_hp.get("stage1_lr"),
        }
        progress["phase1_complete"] = True
        progress["best_hp_values"] = best_hp_values
        _save_progress(progress)
        print("  ✓ Phase 1 checkpoint saved")

    # ── Phase 2: Fine-Tune Search ────────────────────────────────────────

    if progress.get("phase2_complete") and progress.get("best_params"):
        print("\n" + "-" * 70)
        print("  PHASE 2: SKIPPED (already complete — resuming from checkpoint)")
        print("-" * 70)
        best_params = progress["best_params"]
        print(f"  Restored best: AUC={best_params.get('search_val_auc', '?'):.4f}, "
              f"Acc={best_params.get('search_val_accuracy', '?'):.4f}")

    else:
        print("\n" + "-" * 70)
        print("  PHASE 2: Searching Fine-Tune Configuration")
        print("-" * 70)

        best_hp_values = progress["best_hp_values"]

        # Fine-tune search space (reduced: removed 150-layer config for VRAM safety)
        ft_configs = [
            {"fine_tune_layers": 50,  "stage2_lr": 1e-5},
            {"fine_tune_layers": 100, "stage2_lr": 5e-6},
        ]

        # Augmentation configs (reduced: removed batch_size=16 for OOM safety)
        aug_configs = [
            {"rotation": 0.10, "zoom": 0.05, "batch_size": 32},
            {"rotation": 0.15, "zoom": 0.10, "batch_size": 32},
        ]

        # Restore any previously completed Phase 2 trials
        completed_trials = {r["trial"] for r in progress.get("phase2_results", [])}
        all_results = list(progress.get("phase2_results", []))
        best_overall = {"val_auc": 0.0}
        for r in all_results:
            if r["val_auc"] > best_overall["val_auc"]:
                best_overall = r.copy()

        if completed_trials:
            print(f"  Resuming: {len(completed_trials)} trials already done")

        trial_num = 0

        for aug_cfg in aug_configs:
            # Rebuild datasets with this augmentation config
            train_ds_aug, val_ds_aug = build_datasets(
                ela_dir,
                batch_size=aug_cfg["batch_size"],
                rotation_factor=aug_cfg["rotation"],
                zoom_factor=aug_cfg["zoom"],
            )

            for ft_cfg in ft_configs:
                trial_num += 1

                # Skip already-completed trials
                if trial_num in completed_trials:
                    print(f"\n  Trial {trial_num}: SKIPPED (already complete)")
                    continue

                trial_name = (f"aug(rot={aug_cfg['rotation']},zoom={aug_cfg['zoom']},"
                              f"bs={aug_cfg['batch_size']}) + "
                              f"ft(layers={ft_cfg['fine_tune_layers']},"
                              f"lr={ft_cfg['stage2_lr']})")

                print(f"\n  Trial {trial_num}: {trial_name}")

                # Build fresh model with best Stage 1 params — use an HP-like
                # object that Keras Tuner's build function accepts.
                trial_hp = kt.HyperParameters()
                trial_hp.Choice("dense_1_units", values=[256, 512, 1024],
                                default=best_hp_values["dense_1_units"])
                trial_hp.Choice("dense_2_units", values=[64, 128, 256],
                                default=best_hp_values["dense_2_units"])
                trial_hp.Choice("dropout_rate", values=[0.2, 0.3, 0.4],
                                default=best_hp_values["dropout_rate"])
                trial_hp.Float("stage1_lr", min_value=1e-4, max_value=1e-2,
                               sampling="log", default=best_hp_values["stage1_lr"])
                # Force the values to the saved best
                trial_hp.values = best_hp_values.copy()

                model = build_tunable_model(trial_hp)

                # Train Stage 1 (quick — now visible with verbose=1)
                print("    Stage 1 (head only):")
                model.fit(
                    train_ds_aug,
                    validation_data=val_ds_aug,
                    epochs=SEARCH_EPOCHS_STAGE1,
                    class_weight=class_weight,
                    verbose=1,
                )

                # Fine-tune Stage 2 (use project-local temp checkpoint)
                tmp_path = _temp_path(f"tmp_trial_{trial_num}.keras")
                model, metrics = fine_tune_with_params(
                    model, train_ds_aug, val_ds_aug,
                    fine_tune_layers=ft_cfg["fine_tune_layers"],
                    stage2_lr=ft_cfg["stage2_lr"],
                    epochs=SEARCH_EPOCHS_STAGE2,
                    class_weight=class_weight,
                    model_path=tmp_path,
                    patience=4,
                    use_temp_checkpoint=True,
                )

                result = {
                    "trial": trial_num,
                    "config": trial_name,
                    "dense_1": best_hp_values["dense_1_units"],
                    "dense_2": best_hp_values["dense_2_units"],
                    "dropout": best_hp_values["dropout_rate"],
                    "stage1_lr": best_hp_values["stage1_lr"],
                    **ft_cfg,
                    **aug_cfg,
                    **metrics,
                }
                all_results.append(result)

                print(f"    → Accuracy: {metrics['val_accuracy']:.4f} | "
                      f"AUC: {metrics['val_auc']:.4f} | "
                      f"Precision: {metrics['val_precision']:.4f} | "
                      f"Recall: {metrics['val_recall']:.4f}")

                # Track best
                if metrics["val_auc"] > best_overall["val_auc"]:
                    best_overall = result.copy()
                    print(f"    ★ NEW BEST! AUC={metrics['val_auc']:.4f}")

                # Clean up temp model
                if tmp_path.exists():
                    tmp_path.unlink()

                # Checkpoint after each trial
                progress["phase2_results"] = all_results
                _save_progress(progress)
                print(f"    ✓ Trial {trial_num} checkpoint saved")

                # Early stop if target met
                if metrics["val_accuracy"] >= target_acc:
                    print(f"\n  ✅ Target accuracy {target_acc*100:.0f}% reached! Stopping search.")
                    break

            if best_overall.get("val_accuracy", 0) >= target_acc:
                break

        # ── Print Search Results ─────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  AUTO-TUNING SEARCH COMPLETE")
        print("=" * 70)
        print(f"\n  Total trials: {len(all_results)}")
        print(f"\n  All results (sorted by AUC):")
        print(f"  {'#':<4} {'AUC':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'Config'}")
        print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*40}")

        for r in sorted(all_results, key=lambda x: x["val_auc"], reverse=True):
            marker = " ★" if r["trial"] == best_overall["trial"] else ""
            print(f"  {r['trial']:<4} {r['val_auc']:.4f}   {r['val_accuracy']:.4f}   "
                  f"{r['val_precision']:.4f}   {r['val_recall']:.4f}   "
                  f"{r['config'][:40]}{marker}")

        # Save best params
        best_params = {
            "dense_1_units": best_overall["dense_1"],
            "dense_2_units": best_overall["dense_2"],
            "dropout_rate": best_overall["dropout"],
            "stage1_lr": best_overall["stage1_lr"],
            "stage2_lr": best_overall["stage2_lr"],
            "fine_tune_layers": best_overall["fine_tune_layers"],
            "batch_size": best_overall["batch_size"],
            "rotation_factor": best_overall["rotation"],
            "zoom_factor": best_overall["zoom"],
            "search_val_accuracy": best_overall["val_accuracy"],
            "search_val_auc": best_overall["val_auc"],
            "search_val_precision": best_overall["val_precision"],
            "search_val_recall": best_overall["val_recall"],
        }

        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n  Best params saved to: {BEST_PARAMS_PATH}")

        # Mark Phase 2 complete
        progress["phase2_complete"] = True
        progress["best_params"] = best_params
        _save_progress(progress)
        print("  ✓ Phase 2 checkpoint saved")

    return progress["best_params"]


# ── Final Training with Best Params ──────────────────────────────────────────

def train_final_model(params: dict, ela_dir: Path) -> None:
    """Train the final production model with the best hyperparameters.

    Supports resume: checks progress.json and skips completed stages.
    Does NOT delete the old model until the new one is verified.
    """
    print("\n" + "=" * 70)
    print("  FINAL TRAINING — Best Hyperparameters")
    print("=" * 70)
    print(f"  Dense layers   : {params['dense_1_units']} → {params['dense_2_units']} → 1")
    print(f"  Dropout rate   : {params['dropout_rate']}")
    print(f"  Stage 1 LR     : {params['stage1_lr']:.6f}")
    print(f"  Stage 2 LR     : {params['stage2_lr']:.6f}")
    print(f"  Fine-tune layers: {params['fine_tune_layers']}")
    print(f"  Batch size     : {params['batch_size']}")
    print(f"  Augmentation   : rotation={params['rotation_factor']}, zoom={params['zoom_factor']}")
    print("=" * 70)

    progress = _load_progress()
    class_weight = get_class_weights(ela_dir)

    # Build datasets with best augmentation
    train_ds, val_ds = build_datasets(
        ela_dir,
        batch_size=params["batch_size"],
        rotation_factor=params["rotation_factor"],
        zoom_factor=params["zoom_factor"],
    )

    # Intermediate model path — don't overwrite production model yet
    stage1_model_path = _temp_path("final_stage1.keras")

    # ── Stage 1: Train head ──────────────────────────────────────────────
    if progress.get("final_stage1_complete") and stage1_model_path.exists():
        print("\n" + "-" * 70)
        print(f"  STAGE 1: SKIPPED (already complete — resuming from checkpoint)")
        print("-" * 70)
    else:
        print("\n" + "-" * 70)
        print(f"  STAGE 1: Training Head ({FINAL_EPOCHS_STAGE1} epochs)")
        print("-" * 70)

        inputs = keras.Input(shape=(*IMG_SIZE, 3))
        backbone = keras.applications.ResNet50(
            include_top=False, weights="imagenet",
            input_tensor=inputs, input_shape=(*IMG_SIZE, 3),
        )
        backbone.trainable = False

        x = layers.GlobalAveragePooling2D()(backbone.output)
        x = layers.Dense(params["dense_1_units"], activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(params["dropout_rate"])(x)
        x = layers.Dense(params["dense_2_units"], activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(params["dropout_rate"])(x)
        outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params["stage1_lr"]),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )

        # Checkpoint to project-local temp (D: drive)
        stage1_callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(stage1_model_path), monitor="val_auc",
                mode="max", save_best_only=True, verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=5,
                restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", factor=0.5,
                patience=3, min_lr=1e-7, verbose=1,
            ),
            keras.callbacks.CSVLogger("training_log.csv", append=False),
        ]

        model.fit(
            train_ds, validation_data=val_ds,
            epochs=FINAL_EPOCHS_STAGE1,
            class_weight=class_weight,
            callbacks=stage1_callbacks,
            verbose=1,
        )

        # Save checkpoint (ModelCheckpoint already wrote to stage1_model_path)
        if not stage1_model_path.exists():
            model.save(stage1_model_path)

        progress["final_stage1_complete"] = True
        _save_progress(progress)
        print(f"  ✓ Stage 1 checkpoint saved to: {stage1_model_path}")

    # ── Stage 2: Fine-tune ───────────────────────────────────────────────
    print("\n" + "-" * 70)
    print(f"  STAGE 2: Fine-Tuning Top {params['fine_tune_layers']} Layers "
          f"({FINAL_EPOCHS_STAGE2} epochs)")
    print("-" * 70)

    model = keras.models.load_model(stage1_model_path)

    # Fine-tune writes to a second temp path, not the production model
    stage2_model_path = _temp_path("final_stage2.keras")

    model, final_metrics = fine_tune_with_params(
        model, train_ds, val_ds,
        fine_tune_layers=params["fine_tune_layers"],
        stage2_lr=params["stage2_lr"],
        epochs=FINAL_EPOCHS_STAGE2,
        class_weight=class_weight,
        model_path=stage2_model_path,
        patience=7,
        use_temp_checkpoint=True,
    )

    # ── Promote to production ─────────────────────────────────────────────
    # Only replace the old model AFTER the new one is verified on disk
    if stage2_model_path.exists():
        # Back up old model if it exists
        if MODEL_PATH.exists():
            backup_path = MODEL_PATH.with_suffix(".keras.bak")
            shutil.copy2(str(MODEL_PATH), str(backup_path))
            print(f"  Old model backed up to: {backup_path}")

        shutil.copy2(str(stage2_model_path), str(MODEL_PATH))
        print(f"  ✅ New model promoted to: {MODEL_PATH.resolve()}")
    else:
        # Stage 2 didn't produce a file — save the in-memory model directly
        model.save(MODEL_PATH)
        print(f"  ✅ Model saved to: {MODEL_PATH.resolve()}")

    progress["final_stage2_complete"] = True
    _save_progress(progress)

    # ── Final Results ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL MODEL RESULTS")
    print("=" * 70)
    print(f"  Accuracy  : {final_metrics['val_accuracy']:.4f}")
    print(f"  Precision : {final_metrics['val_precision']:.4f}")
    print(f"  Recall    : {final_metrics['val_recall']:.4f}")
    print(f"  AUC       : {final_metrics['val_auc']:.4f}")
    print(f"  Best epoch: {final_metrics['best_epoch']}")
    print(f"\n  Model saved to: {MODEL_PATH.resolve()}")
    print("=" * 70)

    # Save final results
    final_results = {**params, **final_metrics, "model_path": str(MODEL_PATH)}
    with open("final_training_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"  Results saved to: final_training_results.json")

    # Clean up temp files
    for tmp in [stage1_model_path, stage2_model_path]:
        if tmp.exists():
            tmp.unlink()
    print("  Temp files cleaned up")


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepForensics: Auto-Tuning Training Pipeline"
    )
    parser.add_argument(
        "--max-trials", type=int, default=DEFAULT_MAX_TRIALS,
        help=f"Maximum Bayesian search trials (default: {DEFAULT_MAX_TRIALS}).",
    )
    parser.add_argument(
        "--target-acc", type=float, default=DEFAULT_TARGET_ACC,
        help=f"Stop search when accuracy >= this (default: {DEFAULT_TARGET_ACC}).",
    )
    parser.add_argument(
        "--retrain-only", action="store_true",
        help="Skip search, train with previously saved best params.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Wipe all checkpoints and start fresh.",
    )
    args = parser.parse_args()

    # Handle --clean flag first
    if args.clean:
        _clear_progress()
        return

    # Mixed precision
    mixed_precision.set_global_policy("mixed_float16")
    print(f"[TF] Mixed precision: {mixed_precision.global_policy().name}")
    print(f"[TF] GPU devices: {tf.config.list_physical_devices('GPU')}")

    # GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # Validate dataset
    if not ELA_DIR.exists() or not any(ELA_DIR.iterdir()):
        print(f"\n[ERROR] ELA dataset not found at {ELA_DIR.resolve()}")
        print("        Run: python prepare_dataset.py")
        sys.exit(1)

    t0 = time.perf_counter()

    if args.retrain_only:
        # Load previously saved best params
        if not BEST_PARAMS_PATH.exists():
            print(f"[ERROR] {BEST_PARAMS_PATH} not found. Run search first (without --retrain-only).")
            sys.exit(1)
        with open(BEST_PARAMS_PATH) as f:
            best_params = json.load(f)
        print(f"\n[RETRAIN] Loading best params from {BEST_PARAMS_PATH}")
    else:
        # Run auto-tuning search
        best_params = run_auto_tuning(ELA_DIR, args.max_trials, args.target_acc)

    # Train final model with best params (does NOT delete old model until new one ready)
    train_final_model(best_params, ELA_DIR)

    elapsed = time.perf_counter() - t0
    print(f"\n  Total pipeline time: {elapsed / 60:.1f} minutes")

    # Final cleanup — progress is complete, remove checkpoint state
    _clear_progress()
    print("  ✓ Pipeline complete — checkpoints cleared")


if __name__ == "__main__":
    main()
