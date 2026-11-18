"""
DeepForensics — Bayesian Auto-Tuned & Crash-Resumable ResNet50 Pipeline (v3)
=============================================================================
Automatically searches for the best hyperparameters via Keras Tuner Bayesian
Optimization and then trains the final ResNet50 + ANN forgery detector in ONE
GO, with full crash-safe auto-resume across every phase and every epoch:

    Phase 0A — Bayesian Optimization (Stage 1 head architecture, LR, focal gamma)
               Resumes unfinished trials automatically via `overwrite=False`.
    Phase 0B — Stage 2 Fine-Tune Search (unfreeze depth & Stage 2 LR)
               Checkpoints each candidate trial in `state.json`.
    Phase 1  — Full Stage 1 Training (frozen backbone + winning head)
               Atomic per-epoch checkpointing (`latest_stage1.keras`).
    Phase 2  — Full Stage 2 Fine-Tuning (top conv layers, BN frozen)
               Atomic per-epoch checkpointing (`latest_stage2.keras`).

Key Deep Learning Fixes (v3):
    - Canonical `keras.applications.resnet50.preprocess_input` (Caffe BGR +
      ImageNet mean subtraction) replacing buggy `Rescaling(1/255)`.
    - Shared stratified split (`dataset/split_manifest.json`, seed=42) for
      zero train/val leakage during multi-model ensemble calibration.
    - Frozen BatchNormalization during fine-tuning + BinaryFocalCrossentropy
      + AdamW + CosineDecay warmup.
    - Sidecar metadata (`<model>.meta.json`) recording exact preprocessing.

Hardware: NVIDIA RTX 3060 (12GB VRAM) + 16GB DDR4 RAM (mixed_float16).

Usage:
    python train_model.py                        # Bayesian tune + train in one go (auto-resumes)
    python train_model.py --max-trials 8         # customize Bayesian trial budget
    python train_model.py --no-tune              # skip search, train with defaults
    python train_model.py --retrain-only         # reuse saved best_hyperparams_*.json
    python train_model.py --fresh                # wipe checkpoints & tuner, start from scratch
    python train_model.py --smoke-test           # fast synthetic verification
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Use CUDA native async allocator so VRAM is actually released between tuner trials & stages
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

from utils.dataset_split import (
    DEFAULT_ELA_DIR,
    DEFAULT_MANIFEST_PATH,
    build_or_load_manifest,
    compute_class_weights,
    resolve_ela_paths,
)
from utils.resume import (
    ResumableTrainingCallback,
    TrainingResumeManager,
    clear_gpu_memory,
    create_vram_safe_bayesian_tuner,
    load_resumable_model,
)


# ── Configuration ────────────────────────────────────────────────────────────

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 42
ELA_DIR = DEFAULT_ELA_DIR
MODEL_PATH = Path("ela_resnet50_v3.keras")

# Default hyperparameters (used when --no-tune is passed)
DEFAULT_HYPERPARAMS = {
    "dense_1_units": 512,
    "dense_2_units": 128,
    "dropout_1": 0.35,
    "dropout_2": 0.25,
    "stage1_lr": 1e-3,
    "stage2_lr": 3e-5,
    "fine_tune_layers": 100,
    "focal_gamma": 2.0,
}

# Epochs & tuning defaults
DEFAULT_MAX_TRIALS = 6
DEFAULT_SEARCH_EPOCHS = 4
DEFAULT_TARGET_ACC = 0.95
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 7


# ── Data Pipeline ────────────────────────────────────────────────────────────

def _decode_png_rgb(path: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]):
    raw_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, img_size, method="bilinear")
    return tf.cast(img, tf.float32), tf.cast(label, tf.float32)


def build_datasets(
    ela_dir: Path = ELA_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[int, float], int]:
    """Build train/val tf.data.Datasets using the shared stratified manifest."""
    manifest = build_or_load_manifest(
        manifest_path=manifest_path,
        source_dir=ela_dir,
        seed=seed,
    )
    train_paths, train_labels = resolve_ela_paths(manifest["train"], ela_dir=ela_dir)
    val_paths, val_labels = resolve_ela_paths(manifest["val"], ela_dir=ela_dir)

    if not train_paths or not val_paths:
        raise FileNotFoundError(
            f"Could not resolve ELA PNGs in {ela_dir}. Run prepare_dataset.py first."
        )

    class_weight = compute_class_weights(train_labels)
    au_count = sum(1 for y in train_labels if y == 0)
    tp_count = sum(1 for y in train_labels if y == 1)
    print(f"  Split manifest : {manifest_path} (seed={manifest['seed']})")
    print(f"  Train samples  : {len(train_paths)} (Au={au_count}, Tp={tp_count})")
    print(f"  Val samples    : {len(val_paths)}")
    print(f"  Class weights  : Au={class_weight[0]:.3f}, Tp={class_weight[1]:.3f}")

    def augment_train(img: tf.Tensor, label: tf.Tensor):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k=k)
        return resnet50_preprocess(img), label

    def preprocess_val(img: tf.Tensor, label: tf.Tensor):
        return resnet50_preprocess(img), label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .shuffle(buffer_size=min(len(train_paths), 4096), seed=seed, reshuffle_each_iteration=True)
        .map(lambda p, y: _decode_png_rgb(p, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment_train, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(1)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(lambda p, y: _decode_png_rgb(p, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        .map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(1)
    )

    steps_per_epoch = max(1, int(np.ceil(len(train_paths) / batch_size)))
    return train_ds, val_ds, class_weight, steps_per_epoch


def build_smoke_test_dataset(
    batch_size: int = 8,
    img_size: tuple[int, int] = IMG_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Create tiny synthetic dataset preprocessed with resnet50.preprocess_input."""
    auth_images = [np.full((*img_size, 3), (i * 15, 80, 160), dtype=np.float32) for i in range(8)]
    tamp_images = []
    for i in range(8):
        img = np.full((*img_size, 3), (30, 90, 160), dtype=np.float32)
        img[50:150, 50:150] = (255, 80 + i * 10, 80)
        tamp_images.append(img)

    x = resnet50_preprocess(np.stack(auth_images + tamp_images, axis=0))
    y = np.array([0] * 8 + [1] * 8, dtype=np.float32)

    ds = (
        tf.data.Dataset.from_tensor_slices((x, y))
        .shuffle(16, seed=SEED)
        .batch(batch_size)
        .prefetch(2)
    )
    return ds, ds, max(1, len(y) // batch_size)


# ── Model Architecture ───────────────────────────────────────────────────────

def build_model(
    img_size: tuple[int, int] = IMG_SIZE,
    weights: str | None = "imagenet",
    dense_1_units: int = 512,
    dense_2_units: int = 128,
    dropout_1: float = 0.35,
    dropout_2: float = 0.25,
) -> keras.Model:
    """Build ResNet50 + tunable ANN head with named penultimate_embedding layer."""
    inputs = keras.Input(shape=(*img_size, 3), name="ela_input")

    backbone = keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=(*img_size, 3),
    )
    backbone.trainable = False

    # Call backbone with training=False so pretrained ImageNet BN stats stay frozen.
    # Expose spatial output on the outer Functional graph for clean Grad-CAM++.
    x = backbone(inputs, training=False)
    x = layers.Activation("linear", name="ela_backbone_out")(x)
    x = layers.GlobalAveragePooling2D(name="ela_gap")(x)
    x = layers.Dense(int(dense_1_units), activation="gelu", name="head_dense_1")(x)
    x = layers.BatchNormalization(name="head_bn_1")(x)
    x = layers.Dropout(float(dropout_1), name="head_drop_1")(x)
    x = layers.Dense(int(dense_2_units), activation="gelu", name="penultimate_embedding")(x)
    x = layers.BatchNormalization(name="head_bn_2")(x)
    x = layers.Dropout(float(dropout_2), name="head_drop_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="tampered_prob")(x)

    return keras.Model(inputs, outputs, name="deepforensics_resnet50_v3")


def compile_model(
    model: keras.Model,
    learning_rate: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
) -> None:
    """Compile model with AdamW + CosineDecay warmup + BinaryFocalCrossentropy."""
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    decay_steps = max(warmup_steps + 1, total_steps)

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate * 0.1,
        decay_steps=decay_steps,
        alpha=0.01,
        warmup_target=learning_rate,
        warmup_steps=warmup_steps,
    )
    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    if use_focal_loss:
        loss_fn = keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=False,
            gamma=float(focal_gamma),
            label_smoothing=0.05,
        )
    else:
        loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=0.05)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def _unfreeze_top_backbone_layers(model: keras.Model, fine_tune_layers: int) -> None:
    """Unfreeze top N backbone layers while keeping BatchNormalization frozen."""
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

    cutoff = max(0, len(backbone_layers) - int(fine_tune_layers))
    for idx, layer in enumerate(backbone_layers):
        if idx < cutoff or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum(1 for l in backbone_layers if l.trainable)
    frozen_count = len(backbone_layers) - trainable_count
    print(
        f"  Backbone layers: {len(backbone_layers)} total | "
        f"{frozen_count} frozen (incl. BN) | {trainable_count} trainable"
    )


def get_callbacks(
    resume_mgr: TrainingResumeManager,
    stage: int,
    log_path: str = "training_resnet50_v3_log.csv",
    patience: int = EARLY_STOPPING_PATIENCE,
) -> list[keras.callbacks.Callback]:
    """Return crash-safe atomic checkpoint + CSV logger callbacks."""
    return [
        ResumableTrainingCallback(
            manager=resume_mgr,
            stage=stage,
            monitor="val_auc",
            patience=patience,
        ),
        keras.callbacks.CSVLogger(log_path, append=True),
    ]


def save_metadata(model_path: Path, use_focal_loss: bool, params: dict) -> None:
    """Write sidecar JSON metadata describing exact preprocessing & winning hyperparams."""
    meta = {
        "model_type": "resnet50_ela_v3",
        "preprocessing": "resnet50_caffe",
        "img_size": list(IMG_SIZE),
        "loss": "binary_focal_crossentropy" if use_focal_loss else "binary_crossentropy",
        "manifest": str(DEFAULT_MANIFEST_PATH),
        "hyperparams": params,
    }
    meta_path = model_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  [META] Saved sidecar metadata to {meta_path}")


# ── Phase 0: Resumable Bayesian Hyperparameter Optimization ──────────────────

def run_bayesian_tuning(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weight: dict,
    steps_per_epoch: int,
    resume_mgr: TrainingResumeManager,
    max_trials: int = DEFAULT_MAX_TRIALS,
    search_epochs: int = DEFAULT_SEARCH_EPOCHS,
    target_acc: float = DEFAULT_TARGET_ACC,
    use_focal_loss: bool = True,
) -> dict:
    """Run resumable 2-phase Bayesian optimization (Head -> Fine-Tune depth/LR).

    Phase 0A: Keras Tuner BayesianOptimization over dense units, dropouts,
              Stage 1 LR, and focal gamma (resumes trials via `overwrite=False`).
    Phase 0B: Evaluates VRAM-safe Stage 2 fine-tune configs on top of the
              winning head (checkpoints each candidate in `state.json`).
    """
    if resume_mgr.state.get("tuning_phase2_completed") and resume_mgr.load_saved_hyperparams():
        saved = resume_mgr.load_saved_hyperparams()
        print("\n[RESUME] Bayesian tuning already COMPLETE — restoring winning hyperparameters:")
        print(f"  {json.dumps(saved, indent=2)}")
        return saved

    # ── Phase 0A: Bayesian Search for Stage 1 Head & LR ─────────────────────
    if resume_mgr.state.get("tuning_phase1_completed") and resume_mgr.state.get("tuning_phase1_best_hp"):
        best_p1 = dict(resume_mgr.state["tuning_phase1_best_hp"])
        print("\n[RESUME] Bayesian Phase 0A already COMPLETE — skipping to Phase 0B:")
        print(f"  Restored Phase 0A winner: {best_p1}")
    else:
        print("\n" + "=" * 68)
        print("  PHASE 0A: Bayesian Hyperparameter Optimization (ResNet50 Head)")
        print("=" * 68)
        print(f"  Max trials: {max_trials} | Epochs/trial: {search_epochs} | Objective: val_auc")

        def build_tunable(hp) -> keras.Model:
            d1 = hp.Choice("dense_1_units", values=[256, 512, 1024], default=512)
            d2 = hp.Choice("dense_2_units", values=[64, 128, 256], default=128)
            dr1 = hp.Choice("dropout_1", values=[0.25, 0.35, 0.45], default=0.35)
            dr2 = hp.Choice("dropout_2", values=[0.15, 0.25, 0.35], default=0.25)
            lr1 = hp.Float("stage1_lr", min_value=2e-4, max_value=3e-3, sampling="log", default=1e-3)
            gamma = hp.Choice("focal_gamma", values=[1.5, 2.0], default=2.0) if use_focal_loss else 2.0

            model = build_model(
                img_size=IMG_SIZE,
                weights="imagenet",
                dense_1_units=d1,
                dense_2_units=d2,
                dropout_1=dr1,
                dropout_2=dr2,
            )
            compile_model(
                model,
                learning_rate=lr1,
                total_steps=max(1, search_epochs * steps_per_epoch),
                use_focal_loss=use_focal_loss,
                focal_gamma=gamma,
            )
            return model

        try:
            tuner = create_vram_safe_bayesian_tuner(
                hypermodel=build_tunable,
                directory=resume_mgr.tuner_dir,
                project_name="resnet50_v3_bayes",
                max_trials=max_trials,
                seed=SEED,
                objective_name="val_auc",
            )
            tuner.search(
                train_ds,
                validation_data=val_ds,
                epochs=search_epochs,
                class_weight=class_weight,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor="val_auc", mode="max", patience=2, restore_best_weights=True
                    )
                ],
                verbose=1,
            )
            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_p1 = {
                "dense_1_units": int(best_hp.get("dense_1_units")),
                "dense_2_units": int(best_hp.get("dense_2_units")),
                "dropout_1": float(best_hp.get("dropout_1")),
                "dropout_2": float(best_hp.get("dropout_2")),
                "stage1_lr": float(best_hp.get("stage1_lr")),
                "focal_gamma": float(best_hp.get("focal_gamma")) if use_focal_loss else 2.0,
            }
        except (KeyboardInterrupt, Exception) as exc:
            resume_mgr.record_tuning_interruption("phase0a", exc)
            sys.exit(1)

        resume_mgr.state["tuning_phase1_completed"] = True
        resume_mgr.state["tuning_phase1_best_hp"] = best_p1
        resume_mgr.save_state()
        clear_gpu_memory()
        print(f"\n  [TUNER] Phase 0A Winner saved: {best_p1}")

    # ── Phase 0B: Fine-Tune Depth & Stage 2 LR Search (Per-Trial Resumable) ─
    print("\n" + "=" * 68)
    print("  PHASE 0B: Stage 2 Fine-Tune Depth & Learning Rate Selection")
    print("=" * 68)

    ft_candidates = [
        {"trial": 1, "fine_tune_layers": 60,  "stage2_lr": 5e-5},
        {"trial": 2, "fine_tune_layers": 100, "stage2_lr": 3e-5},
    ]

    p2_results = list(resume_mgr.state.get("tuning_phase2_results", []))
    done_ids = {int(r["trial"]) for r in p2_results}

    for cand in ft_candidates:
        tid = cand["trial"]
        if tid in done_ids:
            print(f"  [RESUME] Phase 0B Trial {tid} already done — skipping.")
            continue

        print(
            f"\n  Trial {tid}/{len(ft_candidates)}: "
            f"fine_tune_layers={cand['fine_tune_layers']}, stage2_lr={cand['stage2_lr']}"
        )
        try:
            model = build_model(
                img_size=IMG_SIZE,
                weights="imagenet",
                dense_1_units=best_p1["dense_1_units"],
                dense_2_units=best_p1["dense_2_units"],
                dropout_1=best_p1["dropout_1"],
                dropout_2=best_p1["dropout_2"],
            )
            compile_model(
                model,
                learning_rate=best_p1["stage1_lr"],
                total_steps=max(1, search_epochs * steps_per_epoch),
                use_focal_loss=use_focal_loss,
                focal_gamma=best_p1["focal_gamma"],
            )
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=search_epochs,
                class_weight=class_weight,
                verbose=1,
            )

            _unfreeze_top_backbone_layers(model, cand["fine_tune_layers"])
            compile_model(
                model,
                learning_rate=cand["stage2_lr"],
                total_steps=max(1, search_epochs * steps_per_epoch),
                use_focal_loss=use_focal_loss,
                focal_gamma=best_p1["focal_gamma"],
            )
            hist = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=search_epochs,
                class_weight=class_weight,
                verbose=1,
            )
            best_ep = int(np.argmax(hist.history["val_auc"]))
            rec = {
                **cand,
                "val_auc": float(hist.history["val_auc"][best_ep]),
                "val_accuracy": float(hist.history["val_accuracy"][best_ep]),
            }
            p2_results.append(rec)
            resume_mgr.state["tuning_phase2_results"] = p2_results
            resume_mgr.save_state()
            print(f"  -> Trial {tid} result: val_auc={rec['val_auc']:.4f}, val_acc={rec['val_accuracy']:.4f}")
        except (KeyboardInterrupt, Exception) as exc:
            resume_mgr.record_tuning_interruption(f"phase0b_trial{tid}", exc)
            sys.exit(1)
        finally:
            clear_gpu_memory()

        if rec["val_accuracy"] >= target_acc:
            print(f"  [TUNER] Target accuracy {target_acc:.2%} reached — ending search early.")
            break

    winner_p2 = max(p2_results, key=lambda r: (r["val_auc"], r["val_accuracy"]))
    best_params = {
        **best_p1,
        "fine_tune_layers": int(winner_p2["fine_tune_layers"]),
        "stage2_lr": float(winner_p2["stage2_lr"]),
        "search_val_auc": float(winner_p2["val_auc"]),
        "search_val_accuracy": float(winner_p2["val_accuracy"]),
    }
    resume_mgr.save_best_hyperparams(best_params)
    print("\n" + "=" * 68)
    print("  BAYESIAN AUTO-TUNING COMPLETE — WINNING HYPERPARAMETERS")
    print("=" * 68)
    print(json.dumps(best_params, indent=2))
    print(f"  Saved to: {resume_mgr.best_hp_path}")
    return best_params


# ── Production Training Stages (Per-Epoch Atomic Crash Resume) ───────────────

def run_stage1(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_path: Path,
    resume_mgr: TrainingResumeManager,
    params: dict,
    epochs: int,
    steps_per_epoch: int,
    class_weight: dict | None = None,
    use_focal_loss: bool = True,
) -> keras.Model | None:
    initial_epoch = resume_mgr.get_initial_epoch(stage=1)
    lr1 = float(params["stage1_lr"])
    gamma = float(params.get("focal_gamma", 2.0))

    print("\n" + "=" * 68)
    print("  STAGE 1: Frozen ResNet50 Backbone (Caffe Preprocessing) — Head")
    print("=" * 68)
    print(
        f"  Epochs: {initial_epoch} -> {epochs} | Peak LR: {lr1:.6f} | "
        f"Head: {params['dense_1_units']}->{params['dense_2_units']} "
        f"(drop={params['dropout_1']}/{params['dropout_2']})"
    )

    if resume_mgr.has_resumable_checkpoint(stage=1):
        ckpt_path = resume_mgr.latest_checkpoint_path(stage=1)
        print(f"  [RESUME] Restoring Stage 1 checkpoint from {ckpt_path} (epoch {initial_epoch})")
        try:
            model = load_resumable_model(ckpt_path)
            if model.optimizer is None:
                raise ValueError("Optimizer state missing")
        except Exception as exc:
            print(f"  [RESUME] Reloaded weights without optimizer ({exc}); recompiling remaining steps.")
            model = keras.models.load_model(ckpt_path, compile=False)
            remaining = max(1, (epochs - initial_epoch) * steps_per_epoch)
            compile_model(
                model, learning_rate=lr1, total_steps=remaining,
                use_focal_loss=use_focal_loss, focal_gamma=gamma,
            )
    else:
        model = build_model(
            weights="imagenet",
            dense_1_units=params["dense_1_units"],
            dense_2_units=params["dense_2_units"],
            dropout_1=params["dropout_1"],
            dropout_2=params["dropout_2"],
        )
        compile_model(
            model,
            learning_rate=lr1,
            total_steps=max(1, epochs * steps_per_epoch),
            use_focal_loss=use_focal_loss,
            focal_gamma=gamma,
        )

    save_metadata(model_path, use_focal_loss, params)
    try:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            initial_epoch=initial_epoch,
            class_weight=class_weight,
            callbacks=get_callbacks(resume_mgr, stage=1),
            verbose=1,
        )
    except (KeyboardInterrupt, Exception) as exc:
        resume_mgr.record_interruption(stage=1, exc=exc, model=model)
        sys.exit(1)

    resume_mgr.mark_stage_completed(stage=1)
    save_metadata(model_path, use_focal_loss, params)
    del model
    clear_gpu_memory()
    return None


def run_stage2(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_path: Path,
    resume_mgr: TrainingResumeManager,
    params: dict,
    epochs: int,
    steps_per_epoch: int,
    class_weight: dict | None = None,
    use_focal_loss: bool = True,
) -> keras.Model | None:
    initial_epoch = resume_mgr.get_initial_epoch(stage=2)
    ft_layers = int(params["fine_tune_layers"])
    lr2 = float(params["stage2_lr"])
    gamma = float(params.get("focal_gamma", 2.0))

    print("\n" + "=" * 68)
    print(f"  STAGE 2: Fine-Tune Top {ft_layers} Conv Layers (BN Frozen)")
    print("=" * 68)
    print(f"  Epochs: {initial_epoch} -> {epochs} | Peak LR: {lr2:.6f}")

    if resume_mgr.has_resumable_checkpoint(stage=2):
        ckpt_path = resume_mgr.latest_checkpoint_path(stage=2)
        print(f"  [RESUME] Restoring Stage 2 checkpoint from {ckpt_path} (epoch {initial_epoch})")
        try:
            model = load_resumable_model(ckpt_path)
            if model.optimizer is None:
                raise ValueError("Optimizer state missing")
        except Exception as exc:
            print(f"  [RESUME] Reloaded weights without optimizer ({exc}); recompiling remaining steps.")
            model = keras.models.load_model(ckpt_path, compile=False)
            _unfreeze_top_backbone_layers(model, ft_layers)
            remaining = max(1, (epochs - initial_epoch) * steps_per_epoch)
            compile_model(
                model, learning_rate=lr2, total_steps=remaining,
                use_focal_loss=use_focal_loss, focal_gamma=gamma,
            )
    else:
        model = keras.models.load_model(model_path, compile=False)
        _unfreeze_top_backbone_layers(model, ft_layers)
        compile_model(
            model,
            learning_rate=lr2,
            total_steps=max(1, epochs * steps_per_epoch),
            use_focal_loss=use_focal_loss,
            focal_gamma=gamma,
        )

    try:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            initial_epoch=initial_epoch,
            class_weight=class_weight,
            callbacks=get_callbacks(resume_mgr, stage=2),
            verbose=1,
        )
    except (KeyboardInterrupt, Exception) as exc:
        resume_mgr.record_interruption(stage=2, exc=exc, model=model)
        sys.exit(1)

    resume_mgr.mark_stage_completed(stage=2)
    save_metadata(model_path, use_focal_loss, params)
    return model


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepForensics: Bayesian Auto-Tuned & Crash-Resumable ResNet50 Pipeline (v3)"
    )
    parser.add_argument("--smoke-test", action="store_true", help="1-epoch synthetic verification.")
    parser.add_argument("--stage", type=int, choices=[1, 2], help="Run only Stage 1 or 2.")
    parser.add_argument("--epochs1", type=int, default=STAGE1_EPOCHS)
    parser.add_argument("--epochs2", type=int, default=STAGE2_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--no-focal", action="store_true", help="Use standard BCE instead of Focal Loss.")

    # Bayesian Optimization & Resume controls
    parser.add_argument("--no-tune", action="store_true", help="Skip Bayesian search and use default hyperparams.")
    parser.add_argument("--retrain-only", action="store_true", help="Skip search and reuse saved best_hyperparams JSON.")
    parser.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS, help=f"Bayesian trials (default: {DEFAULT_MAX_TRIALS}).")
    parser.add_argument("--search-epochs", type=int, default=DEFAULT_SEARCH_EPOCHS, help=f"Epochs per search trial (default: {DEFAULT_SEARCH_EPOCHS}).")
    parser.add_argument("--target-acc", type=float, default=DEFAULT_TARGET_ACC, help=f"Early stop tuning accuracy (default: {DEFAULT_TARGET_ACC}).")
    parser.add_argument("--fresh", action="store_true", help="Wipe checkpoints & tuner history and start from scratch.")
    args = parser.parse_args()

    use_focal = not args.no_focal
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        mixed_precision.set_global_policy("mixed_float16")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        print("[TF] No GPU detected — running in float32 (CPU mode).")

    print(f"[TF] Policy: {mixed_precision.global_policy().name} | GPUs: {gpus}")

    t0 = time.perf_counter()

    if args.smoke_test:
        print("\n[SMOKE TEST] Running synthetic Stage 1 + Stage 2 + Save/Load verification...")
        train_ds, val_ds, steps = build_smoke_test_dataset(batch_size=8)
        model = build_model(weights=None, dense_1_units=256, dense_2_units=64)
        compile_model(model, learning_rate=1e-3, total_steps=steps, use_focal_loss=use_focal)
        model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
        _unfreeze_top_backbone_layers(model, 20)
        compile_model(model, learning_rate=3e-5, total_steps=steps, use_focal_loss=use_focal)
        model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
        smoke_out = Path("tmp_smoke_resnet50.keras")
        model.save(smoke_out)
        reloaded = keras.models.load_model(smoke_out, compile=False)
        assert reloaded is not None
        smoke_out.unlink(missing_ok=True)
        print("[SMOKE TEST] ResNet50 v3 build/train/unfreeze/save/load PASSED!")
        return

    if not ELA_DIR.exists() or not any(ELA_DIR.iterdir()):
        print(f"[ERROR] ELA dataset not found at {ELA_DIR.resolve()}. Run prepare_dataset.py.")
        sys.exit(1)

    resume_mgr = TrainingResumeManager(args.output, fresh=args.fresh)
    if args.stage is None and resume_mgr.state.get("completed") and args.output.exists():
        print(
            f"\n[RESUME] Training for {args.output} is already COMPLETE "
            f"(best val_auc={resume_mgr.get_best_val_auc():.4f})."
        )
        print("         Pass --fresh to retrain from scratch.")
        return

    train_ds, val_ds, class_weight, steps = build_datasets(
        ela_dir=ELA_DIR,
        batch_size=args.batch_size,
    )

    # ── Phase 0: Resolve Hyperparameters (Bayesian Tune / Saved / Defaults) ─
    if args.no_tune:
        params = dict(DEFAULT_HYPERPARAMS)
        print("\n[CONFIG] --no-tune enabled: using curated default hyperparameters.")
    elif args.retrain_only:
        saved = resume_mgr.load_saved_hyperparams()
        if not saved:
            print(f"[ERROR] No saved hyperparameters found at {resume_mgr.best_hp_path}. Run without --retrain-only first.")
            sys.exit(1)
        params = saved
        print(f"\n[CONFIG] --retrain-only: loaded hyperparameters from {resume_mgr.best_hp_path}")
    elif args.stage == 2:
        params = resume_mgr.load_saved_hyperparams() or dict(DEFAULT_HYPERPARAMS)
    else:
        params = run_bayesian_tuning(
            train_ds=train_ds,
            val_ds=val_ds,
            class_weight=class_weight,
            steps_per_epoch=steps,
            resume_mgr=resume_mgr,
            max_trials=args.max_trials,
            search_epochs=args.search_epochs,
            target_acc=args.target_acc,
            use_focal_loss=use_focal,
        )

    # ── Phase 1: Production Stage 1 ─────────────────────────────────────────
    if args.stage == 1 or args.stage is None:
        if args.stage is None and resume_mgr.state.get("stage1_completed") and args.output.exists():
            print(
                f"\n[RESUME] Stage 1 already completed "
                f"(best val_auc={resume_mgr.get_best_val_auc():.4f}) — skipping directly to Stage 2."
            )
        else:
            run_stage1(
                train_ds, val_ds, args.output, resume_mgr, params,
                epochs=args.epochs1,
                steps_per_epoch=steps,
                class_weight=class_weight,
                use_focal_loss=use_focal,
            )

    # ── Phase 2: Production Stage 2 ─────────────────────────────────────────
    can_run_s2 = args.output.exists() or resume_mgr.has_resumable_checkpoint(stage=2)
    if args.stage == 2:
        if not can_run_s2:
            print(f"[ERROR] {args.output} not found. Run Stage 1 first.")
            sys.exit(1)
        run_stage2(
            train_ds, val_ds, args.output, resume_mgr, params,
            epochs=args.epochs2,
            steps_per_epoch=steps,
            class_weight=class_weight,
            use_focal_loss=use_focal,
        )
        resume_mgr.mark_all_completed(cleanup_latest=True)
    elif args.stage is None and can_run_s2:
        run_stage2(
            train_ds, val_ds, args.output, resume_mgr, params,
            epochs=args.epochs2,
            steps_per_epoch=steps,
            class_weight=class_weight,
            use_focal_loss=use_focal,
        )
        resume_mgr.mark_all_completed(cleanup_latest=True)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 68)
    print(f"  ResNet50 v3 pipeline complete in {elapsed / 60:.1f} min")
    print(f"  Best val_auc : {resume_mgr.get_best_val_auc():.4f}")
    print(f"  Saved model  : {args.output.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()
