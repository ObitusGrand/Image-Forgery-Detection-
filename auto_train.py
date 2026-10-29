"""
DeepForensics — Auto-Tuning Training Pipeline (Keras Tuner)
=============================================================
Automatically searches for the best hyperparameters using Bayesian
optimization, then trains the final model with the optimal configuration.

Tunes: dropout rate, dense layer sizes, fine-tune depth, learning rates,
       augmentation intensity, and batch size.

Hardware: NVIDIA RTX 3060 (12GB VRAM) + 16GB DDR4 RAM
Strategy: mixed_float16, Bayesian search, auto-retrain on best params.

Usage:
    python auto_train.py                    # full auto-tuning + final training
    python auto_train.py --max-trials 10    # limit search to 10 trials
    python auto_train.py --retrain-only     # skip search, train with best found params
    python auto_train.py --target-acc 0.95  # stop when accuracy >= 95%
    # Re-run the same command after an interruption to resume checkpoints
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import tempfile
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision

# ── Configuration ────────────────────────────────────────────────────────────

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
DATA_PIPELINE_PARALLELISM = 1
PREFETCH_BUFFER = 1
SEED = 42
ELA_DIR = Path("dataset/ela_224")
MODEL_PATH = Path("ela_resnet50_best.keras")
TUNER_DIR = Path("tuner_results")
BEST_PARAMS_PATH = Path("best_hyperparams.json")
CHECKPOINT_DIR = Path("checkpoints") / "auto_train"
STATE_PATH = CHECKPOINT_DIR / "progress.json"

# Search space defaults
DEFAULT_MAX_TRIALS = 8
DEFAULT_TARGET_ACC = 0.94
SEARCH_EPOCHS_STAGE1 = 5   # Fewer epochs during search (faster)
SEARCH_EPOCHS_STAGE2 = 8   # Fewer epochs during search
FINAL_EPOCHS_STAGE1 = 10   # Full epochs for final training
FINAL_EPOCHS_STAGE2 = 25   # Full epochs for final training
SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


# ── Checkpoint State ─────────────────────────────────────────────────────────

def write_json_atomic(path: Path, value: dict) -> None:
    """Write JSON through a temporary file so interruptions cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def save_state(state: dict) -> None:
    """Atomically persist progress so an interrupted run can resume safely."""
    write_json_atomic(STATE_PATH, state)


def load_state() -> dict:
    """Load saved progress, returning a fresh state when none exists."""
    if not STATE_PATH.exists():
        return {
            "phase1_complete": False,
            "phase2_results": [],
            "final_stage1_complete": False,
            "final_stage2_complete": False,
        }
    try:
        with STATE_PATH.open(encoding="utf-8") as file:
            state = json.load(file)
        if not isinstance(state, dict):
            raise ValueError("checkpoint state must be a JSON object")
        return state
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Unable to read checkpoint state {STATE_PATH}: {exc}") from exc


def params_fingerprint(params: dict) -> str:
    """Create a stable identifier for final-training checkpoint compatibility."""
    payload = json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def dataset_fingerprint(ela_dir: Path) -> str:
    """Identify the exact image set used by a resumable training run."""
    digest = hashlib.sha256()
    for class_name in ("Au", "Tp"):
        class_dir = ela_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected dataset class directory: {class_dir}")
        for path in sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ):
            stat = path.stat()
            digest.update(
                f"{class_name}/{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
            )
    return digest.hexdigest()[:16]


# ── Data Pipeline ────────────────────────────────────────────────────────────

def get_class_weights(ela_dir: Path) -> dict[int, float]:
    """Calculate class weights for imbalanced dataset."""
    au_dir = ela_dir / "Au"
    tp_dir = ela_dir / "Tp"
    if not au_dir.is_dir() or not tp_dir.is_dir():
        raise FileNotFoundError(
            f"Expected dataset class directories: {au_dir} and {tp_dir}"
        )
    au_count = sum(
        1 for path in au_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    tp_count = sum(
        1 for path in tp_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if au_count == 0 or tp_count == 0:
        raise ValueError(
            f"Both dataset classes must contain images: Au={au_count}, Tp={tp_count}"
        )
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
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
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
        num_parallel_calls=DATA_PIPELINE_PARALLELISM,
    )
    val_ds = val_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=DATA_PIPELINE_PARALLELISM,
    )

    train_ds = train_ds.prefetch(PREFETCH_BUFFER)
    val_ds = val_ds.prefetch(PREFETCH_BUFFER)

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
    backup_dir: Path | None = None,
) -> tuple[keras.Model, dict]:
    """Fine-tune model with given hyperparameters and return metrics."""
    # Use temp directory for checkpoint if on Windows mount to avoid I/O errors
    if use_temp_checkpoint:
        temp_dir = tempfile.gettempdir()
        checkpoint_path = Path(temp_dir) / f"checkpoint_{id(model)}.keras"
    else:
        checkpoint_path = model_path
    if fine_tune_layers < 1:
        raise ValueError("fine_tune_layers must be at least 1")

    # Unfreeze backbone layers (Keras 3.x compatible)
    try:
        backbone = model.get_layer("resnet50")
        backbone.trainable = True
        backbone_layers = backbone.layers
    except ValueError:
        pooling_index = next(
            (
                index
                for index, layer in enumerate(model.layers)
                if isinstance(layer, layers.GlobalAveragePooling2D)
            ),
            None,
        )
        if pooling_index is None:
            raise RuntimeError("Unable to identify ResNet50 backbone layers for fine-tuning")
        backbone_layers = [
            layer for layer in model.layers[:pooling_index]
            if not isinstance(layer, keras.layers.InputLayer)
        ]
        if not backbone_layers:
            raise RuntimeError("Unable to identify ResNet50 backbone layers for fine-tuning")
        for layer in backbone_layers:
            layer.trainable = True

    # Freeze all but top N layers
    fine_tune_layers = min(fine_tune_layers, len(backbone_layers))
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
    if backup_dir is not None:
        callbacks.append(
            keras.callbacks.BackupAndRestore(
                backup_dir=str(backup_dir),
                delete_checkpoint=True,
            )
        )

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
        try:
            shutil.copy(str(checkpoint_path), str(model_path))
        finally:
            checkpoint_path.unlink(missing_ok=True)

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
    4. Retrain with the winning config at full epoch count

    Keras Tuner resumes Phase 1 from its project directory. Phase 2 stores
    each completed trial in STATE_PATH and retries only an interrupted trial.
    """
    if max_trials < 1:
        raise ValueError("max_trials must be at least 1")
    if not 0.0 < target_acc <= 1.0:
        raise ValueError("target_acc must be greater than 0 and at most 1")
    print("\n" + "=" * 70)
    print("  DeepForensics — AUTO-TUNING PIPELINE")
    print("=" * 70)
    print(f"  Max trials     : {max_trials}")
    print(f"  Target accuracy: {target_acc * 100:.0f}%")
    print(f"  GPU devices    : {tf.config.list_physical_devices('GPU')}")
    print("=" * 70)

    class_weight = get_class_weights(ela_dir)
    run_fingerprint = params_fingerprint({
        "dataset": dataset_fingerprint(ela_dir),
        "batch_size": BATCH_SIZE,
        "search_epochs": (SEARCH_EPOCHS_STAGE1, SEARCH_EPOCHS_STAGE2),
    })
    state = load_state()
    if state.get("run_fingerprint") != run_fingerprint:
        state = {
            "run_fingerprint": run_fingerprint,
            "phase1_complete": False,
            "phase2_results": [],
            "final_stage1_complete": False,
            "final_stage2_complete": False,
        }
        save_state(state)

    # ── Phase 1: Bayesian Search for Stage 1 (Head Architecture) ─────────
    print("\n" + "-" * 70)
    print("  PHASE 1: Searching Head Architecture (Bayesian Optimization)")
    print("-" * 70)

    # Build datasets with default augmentation for search
    train_ds, val_ds = build_datasets(ela_dir, batch_size=BATCH_SIZE)

    try:
        import keras_tuner as kt
    except ImportError:
        print("[ERROR] keras-tuner not installed. Install with:")
        print("        pip install keras-tuner")
        sys.exit(1)

    tuner = kt.BayesianOptimization(
        hypermodel=build_tunable_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=max_trials,
        seed=SEED,
        directory=str(TUNER_DIR),
        project_name=f"deepforensics_v3_{run_fingerprint}",
        overwrite=False,
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
    state["phase1_complete"] = True
    save_state(state)

    # Get top 3 results
    top_hps = tuner.get_best_hyperparameters(num_trials=min(3, max_trials))
    if not top_hps:
        raise RuntimeError(
            "Keras Tuner produced no successful trials. Check the first trial error above."
        )
    best_hp = top_hps[0]

    print("\n" + "-" * 70)
    print("  PHASE 1 RESULTS: Top Hyperparameters")
    print("-" * 70)
    for i, hp in enumerate(top_hps):
        print(f"  #{i+1}: dense_1={hp.get('dense_1_units')}, "
              f"dense_2={hp.get('dense_2_units')}, "
              f"dropout={hp.get('dropout_rate')}, "
              f"lr={hp.get('stage1_lr'):.6f}")

    # ── Phase 2: Fine-Tune Search ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  PHASE 2: Searching Fine-Tune Configuration")
    print("-" * 70)

    # Fine-tune search space
    ft_configs = [
        {"fine_tune_layers": 50,  "stage2_lr": 1e-5},
        {"fine_tune_layers": 100, "stage2_lr": 5e-6},
        {"fine_tune_layers": 100, "stage2_lr": 1e-5},
        {"fine_tune_layers": 150, "stage2_lr": 5e-6},
    ]

    # Augmentation configs to try
    aug_configs = [
        {"rotation": 0.10, "zoom": 0.05, "batch_size": BATCH_SIZE},
        {"rotation": 0.15, "zoom": 0.10, "batch_size": BATCH_SIZE},
        {"rotation": 0.20, "zoom": 0.15, "batch_size": max(1, BATCH_SIZE // 2)},
    ]

    saved_results = state.get("phase2_results", [])
    results_by_trial = {
        int(result["trial"]): result for result in saved_results
    }
    best_overall = max(
        saved_results,
        key=lambda result: result["val_auc"],
        default=None,
    )
    all_results = list(results_by_trial.values())
    target_reached = False

    for aug_index, aug_cfg in enumerate(aug_configs):
        # Rebuild datasets with this augmentation config
        train_ds_aug, val_ds_aug = build_datasets(
            ela_dir,
            batch_size=aug_cfg["batch_size"],
            rotation_factor=aug_cfg["rotation"],
            zoom_factor=aug_cfg["zoom"],
        )

        for ft_index, ft_cfg in enumerate(ft_configs):
            trial_num = aug_index * len(ft_configs) + ft_index + 1
            if trial_num in results_by_trial:
                print(f"\n  Trial {trial_num}: already complete; skipping")
                if results_by_trial[trial_num]["val_accuracy"] >= target_acc:
                    target_reached = True
                    break
                continue
            trial_name = (f"aug(rot={aug_cfg['rotation']},zoom={aug_cfg['zoom']},"
                          f"bs={aug_cfg['batch_size']}) + "
                          f"ft(layers={ft_cfg['fine_tune_layers']},"
                          f"lr={ft_cfg['stage2_lr']})")

            print(f"\n  Trial {trial_num}: {trial_name}")

            # Build fresh model with best Stage 1 params
            model = build_tunable_model(best_hp)

            # Train Stage 1 (quick)
            model.fit(
                train_ds_aug,
                validation_data=val_ds_aug,
                epochs=SEARCH_EPOCHS_STAGE1,
                class_weight=class_weight,
                verbose=0,
            )

            # Fine-tune Stage 2 (use temp checkpoint to avoid I/O errors)
            import tempfile
            tmp_path = Path(tempfile.gettempdir()) / f"tmp_trial_{trial_num}.keras"
            trial_backup_dir = CHECKPOINT_DIR / f"trial_{trial_num}_backup"
            try:
                model, metrics = fine_tune_with_params(
                    model, train_ds_aug, val_ds_aug,
                    fine_tune_layers=ft_cfg["fine_tune_layers"],
                    stage2_lr=ft_cfg["stage2_lr"],
                    epochs=SEARCH_EPOCHS_STAGE2,
                    class_weight=class_weight,
                    model_path=tmp_path,
                    patience=4,
                    use_temp_checkpoint=True,
                    backup_dir=trial_backup_dir,
                )
            finally:
                keras.backend.clear_session()
                del model
                gc.collect()

            result = {
                "trial": trial_num,
                "config": trial_name,
                "dense_1": best_hp.get("dense_1_units"),
                "dense_2": best_hp.get("dense_2_units"),
                "dropout": best_hp.get("dropout_rate"),
                "stage1_lr": best_hp.get("stage1_lr"),
                **ft_cfg,
                **aug_cfg,
                **metrics,
            }
            all_results.append(result)
            results_by_trial[trial_num] = result
            state["phase2_results"] = list(results_by_trial.values())
            save_state(state)

            print(f"    → Accuracy: {metrics['val_accuracy']:.4f} | "
                  f"AUC: {metrics['val_auc']:.4f} | "
                  f"Precision: {metrics['val_precision']:.4f} | "
                  f"Recall: {metrics['val_recall']:.4f}")

            # Track best
            if best_overall is None or metrics["val_auc"] > best_overall["val_auc"]:
                best_overall = result.copy()
                print(f"    ★ NEW BEST! AUC={metrics['val_auc']:.4f}")

            # Clean up temp model
            if tmp_path.exists():
                tmp_path.unlink()
            if trial_backup_dir.exists():
                shutil.rmtree(trial_backup_dir)

            # Early stop if target met
            if metrics["val_accuracy"] >= target_acc:
                print(f"\n  ✅ Target accuracy {target_acc*100:.0f}% reached! Stopping search.")
                target_reached = True
                break

        if target_reached:
            break

    # ── Print Search Results ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  AUTO-TUNING SEARCH COMPLETE")
    print("=" * 70)
    print(f"\n  Total trials: {trial_num}")
    print(f"\n  All results (sorted by AUC):")
    print(f"  {'#':<4} {'AUC':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'Config'}")
    print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*40}")

    if best_overall is None:
        raise RuntimeError("Auto-tuning completed without a successful trial")

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

    write_json_atomic(BEST_PARAMS_PATH, best_params)
    print(f"\n  Best params saved to: {BEST_PARAMS_PATH}")

    return best_params


# ── Final Training with Best Params ──────────────────────────────────────────

def train_final_model(params: dict, ela_dir: Path) -> None:
    """Train the final production model with the best hyperparameters."""
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

    class_weight = get_class_weights(ela_dir)
    state = load_state()
    fingerprint = params_fingerprint(params)
    if state.get("final_params_fingerprint") != fingerprint:
        state["final_params_fingerprint"] = fingerprint
        state["final_stage1_complete"] = False
        state["final_stage2_complete"] = False
        save_state(state)

    # Build datasets with best augmentation
    train_ds, val_ds = build_datasets(
        ela_dir,
        batch_size=params["batch_size"],
        rotation_factor=params["rotation_factor"],
        zoom_factor=params["zoom_factor"],
    )

    # ── Stage 1: Train head ──────────────────────────────────────────────
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

    stage1_checkpoint = CHECKPOINT_DIR / f"final_stage1_{fingerprint}.keras"
    stage1_backup_dir = CHECKPOINT_DIR / f"final_stage1_{fingerprint}_backup"

    stage1_callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(stage1_checkpoint), monitor="val_auc",
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
        keras.callbacks.BackupAndRestore(
            backup_dir=str(stage1_backup_dir),
            delete_checkpoint=True,
        ),
    ]

    if state.get("final_stage1_complete", False) and stage1_checkpoint.exists():
        print("  Resuming: Stage 1 checkpoint already complete; skipping Stage 1.")
    else:
        model.fit(
            train_ds, validation_data=val_ds,
            epochs=FINAL_EPOCHS_STAGE1,
            class_weight=class_weight,
            callbacks=stage1_callbacks,
            verbose=1,
        )
        if not stage1_checkpoint.exists():
            raise RuntimeError("Stage 1 finished without creating a model checkpoint")
        state["final_stage1_complete"] = True
        save_state(state)

    if not stage1_checkpoint.exists():
        raise RuntimeError("Stage 1 finished without creating a model checkpoint")
    shutil.copy(str(stage1_checkpoint), str(MODEL_PATH))
    print(f"  Saved Stage 1 model to: {MODEL_PATH}")

    # ── Stage 2: Fine-tune ───────────────────────────────────────────────
    print("\n" + "-" * 70)
    print(f"  STAGE 2: Fine-Tuning Top {params['fine_tune_layers']} Layers "
          f"({FINAL_EPOCHS_STAGE2} epochs)")
    print("-" * 70)

    stage2_checkpoint = CHECKPOINT_DIR / f"final_stage2_{fingerprint}.keras"
    stage2_backup_dir = CHECKPOINT_DIR / f"final_stage2_{fingerprint}_backup"
    if (
        state.get("final_stage2_complete", False)
        and stage2_checkpoint.exists()
        and isinstance(state.get("final_metrics"), dict)
    ):
        print("  Resuming: Stage 2 checkpoint already complete; skipping Stage 2.")
        final_metrics = state["final_metrics"]
        shutil.copy(str(stage2_checkpoint), str(MODEL_PATH))
    else:
        model = keras.models.load_model(MODEL_PATH)
        model, final_metrics = fine_tune_with_params(
            model, train_ds, val_ds,
            fine_tune_layers=params["fine_tune_layers"],
            stage2_lr=params["stage2_lr"],
            epochs=FINAL_EPOCHS_STAGE2,
            class_weight=class_weight,
            model_path=stage2_checkpoint,
            patience=7,
            backup_dir=stage2_backup_dir,
        )
        if not stage2_checkpoint.exists():
            raise RuntimeError("Stage 2 finished without creating a model checkpoint")
        shutil.copy(str(stage2_checkpoint), str(MODEL_PATH))
        state["final_stage2_complete"] = True
        state["final_metrics"] = final_metrics
        save_state(state)

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
    write_json_atomic(Path("final_training_results.json"), final_results)
    print(f"  Results saved to: final_training_results.json")


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main() -> None:
    global BATCH_SIZE

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
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Dataset batch size (default: {BATCH_SIZE}; use 8 if memory is limited).",
    )
    args = parser.parse_args()
    if args.max_trials < 1:
        parser.error("--max-trials must be at least 1")
    if not 0.0 < args.target_acc <= 1.0:
        parser.error("--target-acc must be greater than 0 and at most 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    BATCH_SIZE = args.batch_size

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

    # Train final model with best params
    train_final_model(best_params, ELA_DIR)

    elapsed = time.perf_counter() - t0
    print(f"\n  Total pipeline time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
