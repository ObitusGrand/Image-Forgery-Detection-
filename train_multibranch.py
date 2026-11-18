"""
DeepForensics — Bayesian Auto-Tuned & Crash-Resumable Multi-Branch Pipeline
===========================================================================
Automatically searches for the best hyperparameters via Keras Tuner Bayesian
Optimization and then trains the final Multi-Branch forensic detector in ONE
GO, with full crash-safe auto-resume across every phase and every epoch:

    Phase 0A — Bayesian Optimization (Fusion dim, head sizes, dropouts, LR, focal gamma)
               Resumes unfinished trials automatically via `overwrite=False`.
    Phase 0B — Stage 2 Fine-Tune Search (backbone unfreeze fraction & Stage 2 LR)
               Checkpoints each candidate trial in `state.json`.
    Phase 1  — Full Stage 1 Training (frozen backbone + winning multi-branch head)
               Atomic per-epoch checkpointing (`latest_stage1.keras`).
    Phase 2  — Full Stage 2 End-to-End Fine-Tuning (top backbone layers, BN frozen)
               Atomic per-epoch checkpointing (`latest_stage2.keras`).

Architecture Streams (CASIA v2.0 disk-cached `dataset/features_224/`):
    Branch A (Compression Domain):
        Multi-Quality ELA (Q=70,85,95 -> 224x224x9)
        -> Learnable 9->3 Channel Adapter
        -> ConvNeXt-Tiny (or EfficientNetV2-S) ImageNet Backbone
        -> CBAM Attention (Channel + Spatial) -> GAP

    Branch B (Learned Noise Residual Domain):
        Raw RGB (224x224x3)
        -> BayarConv2D (constrained 5x5 high-pass noise extractor, center=-1)
           [or StaticSRMConv2D via --static-srm for controlled ablation]
        -> 4-Stage Lightweight Noise CNN (32->64->128->256)
        -> CBAM Attention -> GAP

    Branch C (Frequency Domain, toggleable via --no-fft):
        2D FFT Log-Magnitude Spectrum (224x224x1)
        -> 4-Stage Spectral CNN (16->32->64->128) -> GAP

    Fusion & Classifier Head:
        GatedModalityFusion (learned per-sample attention weights across streams)
        -> Dense(d1, GELU) -> BN -> Dropout(dr1)
        -> Dense(d2, GELU, name="penultimate_embedding") -> BN -> Dropout(dr2)
        -> Dense(1, Sigmoid, dtype="float32")

Hardware: NVIDIA RTX 3060 (12GB VRAM) + 16GB DDR4 RAM (mixed_float16, batch_size=16).

Usage:
    python train_multibranch.py                              # Bayesian tune + train in one go
    python train_multibranch.py --max-trials 8               # customize Bayesian trial budget
    python train_multibranch.py --no-tune                    # skip search, train with defaults
    python train_multibranch.py --retrain-only               # reuse saved best_hyperparams_*.json
    python train_multibranch.py --fresh                      # wipe checkpoints & tuner, restart
    python train_multibranch.py --backbone efficientnetv2_s  # alternate ImageNet backbone
    python train_multibranch.py --no-fft                     # ablation: disable 2D FFT stream
    python train_multibranch.py --static-srm                 # ablation: StaticSRM vs BayarConv2D
    python train_multibranch.py --no-cbam                    # ablation: disable CBAM attention
    python train_multibranch.py --ela-only                   # ablation: Multi-ELA stream only
    python train_multibranch.py --smoke-test                 # fast synthetic verification
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

from utils.custom_layers import (
    BayarConv2D,
    CBAMBlock,
    GatedModalityFusion,
    StaticSRMConv2D,
    get_custom_objects,
)
from utils.dataset_split import (
    DEFAULT_FEATURES_DIR,
    DEFAULT_MANIFEST_PATH,
    build_or_load_manifest,
    compute_class_weights,
    resolve_multibranch_paths,
)
from utils.resume import (
    ResumableTrainingCallback,
    TrainingResumeManager,
    clear_gpu_memory,
    create_vram_safe_bayesian_tuner,
    load_resumable_model,
)


# ── Default Configuration (RTX 3060 12GB Safe) ───────────────────────────────

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
DEFAULT_MODEL_PATH = Path("ela_multibranch_best.keras")

DEFAULT_HYPERPARAMS = {
    "fusion_embed_dim": 256,
    "dense_1_units": 512,
    "dense_2_units": 128,
    "dropout_1": 0.35,
    "dropout_2": 0.25,
    "stage1_lr": 8e-4,
    "stage2_lr": 3e-5,
    "fine_tune_fraction": 0.55,
    "focal_gamma": 2.0,
}

DEFAULT_MAX_TRIALS = 6
DEFAULT_SEARCH_EPOCHS = 4
DEFAULT_TARGET_ACC = 0.96
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 7


# ── Synchronized Multi-Modal Data Pipeline ───────────────────────────────────

def _decode_rgb(path: tf.Tensor, img_size: tuple[int, int]) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_png(raw, channels=3)
    img = tf.image.resize(img, img_size, method="bilinear")
    return tf.cast(img, tf.float32)


def _decode_gray(path: tf.Tensor, img_size: tuple[int, int]) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_png(raw, channels=1)
    img = tf.image.resize(img, img_size, method="bilinear")
    return tf.cast(img, tf.float32)


def _apply_synchronized_augmentation(inputs_dict: dict[str, tf.Tensor], label: tf.Tensor):
    """Apply identical lossless spatial transforms across all input streams."""
    do_hflip = tf.random.uniform([]) < 0.5
    do_vflip = tf.random.uniform([]) < 0.5
    rot_k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)

    out: dict[str, tf.Tensor] = {}
    for key, tensor in inputs_dict.items():
        t = tf.cond(do_hflip, lambda: tf.image.flip_left_right(tensor), lambda: tensor)
        t = tf.cond(do_vflip, lambda: tf.image.flip_up_down(t), lambda: t)
        t = tf.image.rot90(t, k=rot_k)
        out[key] = t
    return out, label


def build_multibranch_datasets(
    features_dir: Path = DEFAULT_FEATURES_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
    include_fft: bool = True,
    ela_only: bool = False,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[int, float], int]:
    """Stream synchronized multi-modal PNGs from SSD via tf.data (low RAM)."""
    manifest = build_or_load_manifest(
        manifest_path=manifest_path,
        source_dir=features_dir,
        seed=seed,
    )
    need_fft = include_fft and (not ela_only)
    tr_paths, tr_labels = resolve_multibranch_paths(
        manifest["train"], features_dir=features_dir, include_fft=need_fft
    )
    va_paths, va_labels = resolve_multibranch_paths(
        manifest["val"], features_dir=features_dir, include_fft=need_fft
    )

    if not tr_paths["ela_q70"] or not va_paths["ela_q70"]:
        raise FileNotFoundError(
            f"Multi-modal feature caches not found in {features_dir}. "
            "Run: python prepare_features.py"
        )

    class_weight = compute_class_weights(tr_labels)
    au_n = sum(1 for y in tr_labels if y == 0)
    tp_n = sum(1 for y in tr_labels if y == 1)
    print(f"  Split manifest : {manifest_path} (seed={manifest['seed']})")
    print(f"  Train samples  : {len(tr_labels)} (Au={au_n}, Tp={tp_n})")
    print(f"  Val samples    : {len(va_labels)}")
    print(f"  Class weights  : Au={class_weight[0]:.3f}, Tp={class_weight[1]:.3f}")

    def make_ds(path_dict: dict[str, list[str]], labels: list[int], is_train: bool) -> tf.data.Dataset:
        slices: dict[str, list] = {
            "p70": path_dict["ela_q70"],
            "p85": path_dict["ela_q85"],
            "p95": path_dict["ela_q95"],
            "praw": path_dict["raw_224"],
        }
        if need_fft:
            slices["pfft"] = path_dict["fft_224"]

        ds = tf.data.Dataset.from_tensor_slices((slices, labels))
        if is_train:
            ds = ds.shuffle(
                buffer_size=min(len(labels), 4096),
                seed=seed,
                reshuffle_each_iteration=True,
            )

        def load_sample(rec: dict[str, tf.Tensor], y: tf.Tensor):
            e70 = _decode_rgb(rec["p70"], img_size)
            e85 = _decode_rgb(rec["p85"], img_size)
            e95 = _decode_rgb(rec["p95"], img_size)
            multi_ela = tf.concat([e70, e85, e95], axis=-1)

            feat_dict: dict[str, tf.Tensor] = {"multi_ela_input": multi_ela}
            if not ela_only:
                feat_dict["raw_rgb_input"] = _decode_rgb(rec["praw"], img_size)
                if need_fft:
                    feat_dict["fft_input"] = _decode_gray(rec["pfft"], img_size)
            return feat_dict, tf.cast(y, tf.float32)

        # Cap parallel decode threads to 4 to stay safely under WSL2's 8GB RAM limit
        ds = ds.map(load_sample, num_parallel_calls=4)
        if is_train:
            ds = ds.map(_apply_synchronized_augmentation, num_parallel_calls=4)
        ds = ds.batch(batch_size).prefetch(1)
        return ds

    train_ds = make_ds(tr_paths, tr_labels, is_train=True)
    val_ds = make_ds(va_paths, va_labels, is_train=False)
    steps_per_epoch = max(1, int(np.ceil(len(tr_labels) / batch_size)))
    return train_ds, val_ds, class_weight, steps_per_epoch


def build_smoke_test_datasets(
    include_fft: bool = True,
    ela_only: bool = False,
    batch_size: int = 4,
    img_size: tuple[int, int] = (128, 128),
) -> tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Tiny synthetic multi-modal dataset for rapid end-to-end verification."""
    rng = np.random.default_rng(SEED)
    n = 8
    inputs: dict[str, np.ndarray] = {
        "multi_ela_input": rng.uniform(0, 255, size=(n, *img_size, 9)).astype(np.float32),
    }
    if not ela_only:
        inputs["raw_rgb_input"] = rng.uniform(0, 255, size=(n, *img_size, 3)).astype(np.float32)
        if include_fft:
            inputs["fft_input"] = rng.uniform(0, 255, size=(n, *img_size, 1)).astype(np.float32)

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((inputs, y)).batch(batch_size).prefetch(2)
    return ds, ds, max(1, n // batch_size)


# ── Model Builders ───────────────────────────────────────────────────────────

def _create_backbone(
    backbone_name: str,
    img_size: tuple[int, int],
    weights: str | None = "imagenet",
) -> keras.Model:
    """Create a modern ImageNet backbone with built-in [0, 255] preprocessing."""
    if backbone_name == "convnext_tiny":
        return keras.applications.ConvNeXtTiny(
            include_top=False,
            include_preprocessing=True,
            weights=weights,
            input_shape=(*img_size, 3),
        )
    if backbone_name == "efficientnetv2_s":
        return keras.applications.EfficientNetV2S(
            include_top=False,
            include_preprocessing=True,
            weights=weights,
            input_shape=(*img_size, 3),
        )
    raise ValueError(f"Unsupported backbone: {backbone_name}")


def _conv_block(x: tf.Tensor, filters: int, name_prefix: str, pool: bool = True) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name_prefix}_conv")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Activation("gelu", name=f"{name_prefix}_gelu")(x)
    if pool:
        x = layers.MaxPooling2D(2, name=f"{name_prefix}_pool")(x)
    return x


def build_multibranch_model(
    img_size: tuple[int, int] = IMG_SIZE,
    backbone_name: str = "convnext_tiny",
    include_fft: bool = True,
    use_cbam: bool = True,
    use_static_srm: bool = False,
    ela_only: bool = False,
    weights: str | None = "imagenet",
    fusion_embed_dim: int = 256,
    dense_1_units: int = 512,
    dense_2_units: int = 128,
    dropout_1: float = 0.35,
    dropout_2: float = 0.25,
) -> tuple[keras.Model, str]:
    """Construct the tunable multi-stream DeepForensics architecture."""
    model_inputs: dict[str, keras.Input] = {}
    branch_vectors: list[tf.Tensor] = []

    # ── Branch A: Multi-Quality ELA (9 channels -> Adapter -> Backbone -> CBAM)
    ela_in = keras.Input(shape=(*img_size, 9), name="multi_ela_input")
    model_inputs["multi_ela_input"] = ela_in

    x_ela = layers.Rescaling(1.0 / 255.0, name="ela_scale_in")(ela_in)
    x_ela = layers.Conv2D(16, 3, padding="same", use_bias=False, name="ela_adapter_conv1")(x_ela)
    x_ela = layers.BatchNormalization(name="ela_adapter_bn1")(x_ela)
    x_ela = layers.Activation("gelu", name="ela_adapter_gelu1")(x_ela)
    x_ela = layers.Conv2D(3, 1, padding="same", activation="sigmoid", name="ela_adapter_proj")(x_ela)
    x_ela = layers.Rescaling(255.0, name="ela_scale_out")(x_ela)

    backbone = _create_backbone(backbone_name, img_size=img_size, weights=weights)
    backbone.trainable = False
    backbone_layer_name = backbone.name

    x_ela = backbone(x_ela, training=False)
    if use_cbam:
        x_ela = CBAMBlock(reduction_ratio=8, name="ela_cbam")(x_ela)
    else:
        x_ela = layers.Activation("linear", name="ela_backbone_out")(x_ela)
    ela_vec = layers.GlobalAveragePooling2D(name="ela_gap")(x_ela)
    branch_vectors.append(ela_vec)

    # ── Branch B: Learned Noise Residual Stream (BayarConv2D or StaticSRM) ──
    if not ela_only:
        raw_in = keras.Input(shape=(*img_size, 3), name="raw_rgb_input")
        model_inputs["raw_rgb_input"] = raw_in

        x_noise = layers.Rescaling(1.0 / 255.0, name="noise_scale_in")(raw_in)
        if use_static_srm:
            x_noise = StaticSRMConv2D(name="static_srm_conv")(x_noise)
        else:
            x_noise = BayarConv2D(filters=12, kernel_size=5, name="bayar_conv")(x_noise)

        x_noise = layers.BatchNormalization(name="noise_in_bn")(x_noise)
        x_noise = _conv_block(x_noise, 32, "noise_b1", pool=True)
        x_noise = _conv_block(x_noise, 64, "noise_b2", pool=True)
        x_noise = _conv_block(x_noise, 128, "noise_b3", pool=True)
        x_noise = _conv_block(x_noise, 256, "noise_b4", pool=True)
        if use_cbam:
            x_noise = CBAMBlock(reduction_ratio=8, name="noise_cbam")(x_noise)
        noise_vec = layers.GlobalAveragePooling2D(name="noise_gap")(x_noise)
        branch_vectors.append(noise_vec)

        # ── Branch C: 2D FFT Log-Magnitude Frequency Spectrum Stream ────────
        if include_fft:
            fft_in = keras.Input(shape=(*img_size, 1), name="fft_input")
            model_inputs["fft_input"] = fft_in

            x_fft = layers.Rescaling(1.0 / 255.0, name="fft_scale_in")(fft_in)
            x_fft = _conv_block(x_fft, 16, "fft_b1", pool=True)
            x_fft = _conv_block(x_fft, 32, "fft_b2", pool=True)
            x_fft = _conv_block(x_fft, 64, "fft_b3", pool=True)
            x_fft = _conv_block(x_fft, 128, "fft_b4", pool=True)
            fft_vec = layers.GlobalAveragePooling2D(name="fft_gap")(x_fft)
            branch_vectors.append(fft_vec)

    # ── Fusion + Classification Head ────────────────────────────────────────
    if len(branch_vectors) > 1:
        fused = GatedModalityFusion(embed_dim=int(fusion_embed_dim), name="gated_fusion")(branch_vectors)
    else:
        fused = branch_vectors[0]

    x = layers.Dense(int(dense_1_units), activation="gelu", name="head_dense_1")(fused)
    x = layers.BatchNormalization(name="head_bn_1")(x)
    x = layers.Dropout(float(dropout_1), name="head_drop_1")(x)
    x = layers.Dense(int(dense_2_units), activation="gelu", name="penultimate_embedding")(x)
    x = layers.BatchNormalization(name="head_bn_2")(x)
    x = layers.Dropout(float(dropout_2), name="head_drop_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="tampered_prob")(x)

    model = keras.Model(inputs=model_inputs, outputs=outputs, name="deepforensics_multibranch")
    return model, backbone_layer_name


def compile_model(
    model: keras.Model,
    learning_rate: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
) -> None:
    """Compile with AdamW + CosineDecay warmup + BinaryFocalCrossentropy."""
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


def _unfreeze_backbone_for_stage2(
    model: keras.Model,
    backbone_layer_name: str,
    fine_tune_fraction: float = 0.55,
) -> None:
    """Unfreeze top fraction of the ELA backbone while keeping BN layers frozen."""
    backbone = None
    if backbone_layer_name:
        try:
            backbone = model.get_layer(backbone_layer_name)
        except ValueError:
            backbone = None
    if backbone is None:
        for layer in model.layers:
            if isinstance(layer, keras.Model) and any(
                k in layer.name.lower() for k in ("convnext", "efficientnet", "resnet")
            ):
                backbone = layer
                break

    if backbone is None:
        print("  [WARN] Could not locate nested backbone sub-model; unfreezing non-BN layers.")
        for layer in model.layers:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        return

    backbone.trainable = True
    b_layers = backbone.layers
    cutoff = int(len(b_layers) * (1.0 - float(fine_tune_fraction)))
    for idx, layer in enumerate(b_layers):
        if idx < cutoff or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    n_train = sum(1 for l in b_layers if l.trainable)
    print(
        f"  Backbone '{backbone.name}': {len(b_layers)} layers | "
        f"{len(b_layers) - n_train} frozen (incl. BN) | {n_train} trainable"
    )


def get_callbacks(
    resume_mgr: TrainingResumeManager,
    stage: int,
    log_path: str = "training_multibranch_log.csv",
    patience: int = EARLY_STOPPING_PATIENCE,
) -> list[keras.callbacks.Callback]:
    return [
        ResumableTrainingCallback(
            manager=resume_mgr,
            stage=stage,
            monitor="val_auc",
            patience=patience,
        ),
        keras.callbacks.CSVLogger(log_path, append=True),
    ]


def save_metadata(
    model_path: Path,
    backbone_name: str,
    backbone_layer_name: str,
    include_fft: bool,
    use_cbam: bool,
    use_static_srm: bool,
    ela_only: bool,
    use_focal_loss: bool,
    params: dict,
) -> None:
    input_names = ["multi_ela_input"]
    if not ela_only:
        input_names.append("raw_rgb_input")
        if include_fft:
            input_names.append("fft_input")

    meta = {
        "model_type": "multibranch_forensics",
        "backbone": backbone_name,
        "backbone_layer_name": backbone_layer_name,
        "img_size": list(IMG_SIZE),
        "input_names": input_names,
        "include_fft": bool(include_fft and not ela_only),
        "use_cbam": bool(use_cbam),
        "noise_extractor": "none" if ela_only else ("static_srm" if use_static_srm else "bayar_conv2d"),
        "ela_only": bool(ela_only),
        "gradcam_layer": "ela_cbam" if use_cbam else "ela_backbone_out",
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
    backbone_name: str,
    include_fft: bool,
    use_cbam: bool,
    use_static_srm: bool,
    ela_only: bool,
    use_focal_loss: bool,
    max_trials: int = DEFAULT_MAX_TRIALS,
    search_epochs: int = DEFAULT_SEARCH_EPOCHS,
    target_acc: float = DEFAULT_TARGET_ACC,
) -> dict:
    """Resumable 2-phase Bayesian hyperparameter optimization for Multi-Branch model."""
    if resume_mgr.state.get("tuning_phase2_completed") and resume_mgr.load_saved_hyperparams():
        saved = resume_mgr.load_saved_hyperparams()
        print("\n[RESUME] Bayesian tuning already COMPLETE — restoring winning hyperparameters:")
        print(f"  {json.dumps(saved, indent=2)}")
        return saved

    # ── Phase 0A: Bayesian Search for Fusion Dim, Head Sizes, Dropouts & LR ─
    if resume_mgr.state.get("tuning_phase1_completed") and resume_mgr.state.get("tuning_phase1_best_hp"):
        best_p1 = dict(resume_mgr.state["tuning_phase1_best_hp"])
        print("\n[RESUME] Bayesian Phase 0A already COMPLETE — skipping to Phase 0B:")
        print(f"  Restored Phase 0A winner: {best_p1}")
    else:
        print("\n" + "=" * 68)
        print("  PHASE 0A: Bayesian Hyperparameter Optimization (Multi-Branch Head)")
        print("=" * 68)
        print(f"  Max trials: {max_trials} | Epochs/trial: {search_epochs} | Objective: val_auc")

        def build_tunable(hp) -> keras.Model:
            f_dim = hp.Choice("fusion_embed_dim", values=[128, 256], default=256) if not ela_only else 256
            d1 = hp.Choice("dense_1_units", values=[256, 512], default=512)
            d2 = hp.Choice("dense_2_units", values=[64, 128], default=128)
            dr1 = hp.Choice("dropout_1", values=[0.25, 0.35, 0.45], default=0.35)
            dr2 = hp.Choice("dropout_2", values=[0.15, 0.25, 0.35], default=0.25)
            lr1 = hp.Float("stage1_lr", min_value=2e-4, max_value=2e-3, sampling="log", default=8e-4)
            gamma = hp.Choice("focal_gamma", values=[1.5, 2.0], default=2.0) if use_focal_loss else 2.0

            model, _ = build_multibranch_model(
                img_size=IMG_SIZE,
                backbone_name=backbone_name,
                include_fft=include_fft,
                use_cbam=use_cbam,
                use_static_srm=use_static_srm,
                ela_only=ela_only,
                weights="imagenet",
                fusion_embed_dim=f_dim,
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
                project_name=f"multibranch_{backbone_name}_bayes",
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
                "fusion_embed_dim": int(best_hp.get("fusion_embed_dim")) if not ela_only else 256,
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

    # ── Phase 0B: Stage 2 Fine-Tune Fraction & LR Selection (Resumable) ─────
    print("\n" + "=" * 68)
    print("  PHASE 0B: Stage 2 Fine-Tune Depth & Learning Rate Selection")
    print("=" * 68)

    ft_candidates = [
        {"trial": 1, "fine_tune_fraction": 0.45, "stage2_lr": 4e-5},
        {"trial": 2, "fine_tune_fraction": 0.60, "stage2_lr": 2e-5},
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
            f"fine_tune_fraction={cand['fine_tune_fraction']}, stage2_lr={cand['stage2_lr']}"
        )
        try:
            model, b_name = build_multibranch_model(
                img_size=IMG_SIZE,
                backbone_name=backbone_name,
                include_fft=include_fft,
                use_cbam=use_cbam,
                use_static_srm=use_static_srm,
                ela_only=ela_only,
                weights="imagenet",
                fusion_embed_dim=best_p1["fusion_embed_dim"],
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

            _unfreeze_backbone_for_stage2(model, b_name, fine_tune_fraction=cand["fine_tune_fraction"])
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
        "fine_tune_fraction": float(winner_p2["fine_tune_fraction"]),
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


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepForensics: Bayesian Auto-Tuned & Crash-Resumable Multi-Branch Trainer"
    )
    parser.add_argument(
        "--backbone",
        choices=["convnext_tiny", "efficientnetv2_s"],
        default="convnext_tiny",
        help="ImageNet backbone for Multi-Quality ELA stream (default: convnext_tiny).",
    )
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs1", type=int, default=STAGE1_EPOCHS)
    parser.add_argument("--epochs2", type=int, default=STAGE2_EPOCHS)
    parser.add_argument("--stage", type=int, choices=[1, 2], help="Run only Stage 1 or 2.")

    # Controlled DL Ablation Flags
    parser.add_argument("--no-fft", action="store_true", help="Ablation: disable 2D FFT stream.")
    parser.add_argument("--no-cbam", action="store_true", help="Ablation: disable CBAM attention.")
    parser.add_argument("--static-srm", action="store_true", help="Ablation: StaticSRM instead of BayarConv2D.")
    parser.add_argument("--ela-only", action="store_true", help="Ablation: Multi-ELA stream only.")
    parser.add_argument("--no-focal", action="store_true", help="Ablation: standard BCE instead of Focal Loss.")

    # Bayesian Optimization & Resume controls
    parser.add_argument("--no-tune", action="store_true", help="Skip Bayesian search and use default hyperparams.")
    parser.add_argument("--retrain-only", action="store_true", help="Skip search and reuse saved best_hyperparams JSON.")
    parser.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS, help=f"Bayesian trials (default: {DEFAULT_MAX_TRIALS}).")
    parser.add_argument("--search-epochs", type=int, default=DEFAULT_SEARCH_EPOCHS, help=f"Epochs per search trial (default: {DEFAULT_SEARCH_EPOCHS}).")
    parser.add_argument("--target-acc", type=float, default=DEFAULT_TARGET_ACC, help=f"Early stop tuning accuracy (default: {DEFAULT_TARGET_ACC}).")
    parser.add_argument("--fresh", action="store_true", help="Wipe checkpoints & tuner history and start from scratch.")
    parser.add_argument("--smoke-test", action="store_true", help="Fast synthetic build/train/save/load test.")
    args = parser.parse_args()

    include_fft = not args.no_fft
    use_cbam = not args.no_cbam
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

    print("=" * 68)
    print("  DeepForensics — Multi-Branch Auto-Tuned Deep Learning Trainer")
    print("=" * 68)
    print(f"  Backbone    : {args.backbone}")
    print(
        f"  Streams     : Multi-ELA(Q70/85/95)"
        f"{'' if args.ela_only else (' + StaticSRM' if args.static_srm else ' + BayarConv2D')}"
        f"{' + 2D-FFT' if (include_fft and not args.ela_only) else ''}"
    )
    print(f"  Attention   : {'CBAM + GatedFusion' if use_cbam else 'GatedFusion (no CBAM)'}")
    print(f"  Loss        : {'BinaryFocalCrossentropy' if use_focal else 'BinaryCrossentropy'}")
    print(f"  Bayes Tune  : {'OFF (--no-tune)' if args.no_tune else ('REUSE (--retrain-only)' if args.retrain_only else f'ON ({args.max_trials} trials)')}")
    print(f"  Precision   : {mixed_precision.global_policy().name} | Batch: {args.batch_size}")
    print("=" * 68)

    if args.smoke_test:
        print("\n[SMOKE TEST] Running synthetic 1-epoch Stage1 + Stage2 + Save/Load + Grad-CAM++...")
        smoke_size = (128, 128)
        train_ds, val_ds, steps = build_smoke_test_datasets(
            include_fft=include_fft,
            ela_only=args.ela_only,
            batch_size=4,
            img_size=smoke_size,
        )
        model, b_name = build_multibranch_model(
            img_size=smoke_size,
            backbone_name=args.backbone,
            include_fft=include_fft,
            use_cbam=use_cbam,
            use_static_srm=args.static_srm,
            ela_only=args.ela_only,
            weights=None,
            fusion_embed_dim=128,
            dense_1_units=256,
            dense_2_units=64,
        )
        compile_model(model, 8e-4, total_steps=steps, use_focal_loss=use_focal)
        model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)

        _unfreeze_backbone_for_stage2(model, b_name, fine_tune_fraction=0.5)
        compile_model(model, 3e-5, total_steps=steps, use_focal_loss=use_focal)
        model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)

        smoke_path = Path("tmp_smoke_multibranch.keras")
        model.save(smoke_path)
        reloaded = keras.models.load_model(smoke_path, custom_objects=get_custom_objects(), compile=False)

        from utils.xai import compute_gradcam_pp
        sample_x, _ = next(iter(val_ds.take(1)))
        sample_1 = {k: v[:1].numpy() for k, v in sample_x.items()}
        pred = reloaded.predict(sample_1, verbose=0)
        hm = compute_gradcam_pp(
            reloaded, sample_1, last_conv_layer_name="ela_cbam" if use_cbam else "ela_backbone_out"
        )
        assert pred.shape == (1, 1) and hm.ndim == 2
        smoke_path.unlink(missing_ok=True)
        print(f"[SMOKE TEST] Prediction={float(pred[0, 0]):.4f}, Grad-CAM++ shape={hm.shape}")
        print("[SMOKE TEST] Multi-Branch full pipeline PASSED!")
        return

    t0 = time.perf_counter()
    resume_mgr = TrainingResumeManager(args.output, fresh=args.fresh)
    if args.stage is None and resume_mgr.state.get("completed") and args.output.exists():
        print(
            f"\n[RESUME] Training for {args.output} is already COMPLETE "
            f"(best val_auc={resume_mgr.get_best_val_auc():.4f})."
        )
        print("         Pass --fresh to retrain from scratch.")
        return

    train_ds, val_ds, class_weight, steps = build_multibranch_datasets(
        features_dir=args.features_dir,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        img_size=IMG_SIZE,
        include_fft=include_fft,
        ela_only=args.ela_only,
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
            backbone_name=args.backbone,
            include_fft=include_fft,
            use_cbam=use_cbam,
            use_static_srm=args.static_srm,
            ela_only=args.ela_only,
            use_focal_loss=use_focal,
            max_trials=args.max_trials,
            search_epochs=args.search_epochs,
            target_acc=args.target_acc,
        )

    backbone_layer_name = ""
    meta_path = args.output.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            backbone_layer_name = json.loads(meta_path.read_text(encoding="utf-8")).get("backbone_layer_name", "")
        except Exception:
            backbone_layer_name = ""

    lr1 = float(params["stage1_lr"])
    lr2 = float(params["stage2_lr"])
    ft_frac = float(params["fine_tune_fraction"])
    gamma = float(params.get("focal_gamma", 2.0))

    # ── Phase 1: Production Stage 1 (Per-Epoch Atomic Resume) ───────────────
    if args.stage == 1 or args.stage is None:
        if args.stage is None and resume_mgr.state.get("stage1_completed") and args.output.exists():
            print(
                f"\n[RESUME] Stage 1 already completed "
                f"(best val_auc={resume_mgr.get_best_val_auc():.4f}) — skipping directly to Stage 2."
            )
        else:
            initial_epoch = resume_mgr.get_initial_epoch(stage=1)
            print(
                f"\n--- STAGE 1: Frozen Backbone (Epoch {initial_epoch} -> {args.epochs1}, LR={lr1:.6f}) ---"
            )
            if resume_mgr.has_resumable_checkpoint(stage=1):
                ckpt_path = resume_mgr.latest_checkpoint_path(stage=1)
                print(f"  [RESUME] Restoring Stage 1 checkpoint from {ckpt_path} (epoch {initial_epoch})")
                try:
                    model = load_resumable_model(ckpt_path, custom_objects=get_custom_objects())
                    if model.optimizer is None:
                        raise ValueError("Optimizer state missing")
                except Exception as exc:
                    print(f"  [RESUME] Reloaded weights without optimizer ({exc}); recompiling remaining steps.")
                    model = keras.models.load_model(ckpt_path, compile=False, custom_objects=get_custom_objects())
                    remaining = max(1, (args.epochs1 - initial_epoch) * steps)
                    compile_model(
                        model, learning_rate=lr1, total_steps=remaining,
                        use_focal_loss=use_focal, focal_gamma=gamma,
                    )
            else:
                model, backbone_layer_name = build_multibranch_model(
                    img_size=IMG_SIZE,
                    backbone_name=args.backbone,
                    include_fft=include_fft,
                    use_cbam=use_cbam,
                    use_static_srm=args.static_srm,
                    ela_only=args.ela_only,
                    weights="imagenet",
                    fusion_embed_dim=params["fusion_embed_dim"],
                    dense_1_units=params["dense_1_units"],
                    dense_2_units=params["dense_2_units"],
                    dropout_1=params["dropout_1"],
                    dropout_2=params["dropout_2"],
                )
                compile_model(
                    model,
                    learning_rate=lr1,
                    total_steps=max(1, args.epochs1 * steps),
                    use_focal_loss=use_focal,
                    focal_gamma=gamma,
                )

            save_metadata(
                args.output, args.backbone, backbone_layer_name,
                include_fft, use_cbam, args.static_srm, args.ela_only, use_focal, params,
            )
            try:
                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=args.epochs1,
                    initial_epoch=initial_epoch,
                    class_weight=class_weight,
                    callbacks=get_callbacks(resume_mgr, stage=1),
                    verbose=1,
                )
            except (KeyboardInterrupt, Exception) as exc:
                resume_mgr.record_interruption(stage=1, exc=exc, model=model)
                sys.exit(1)

            resume_mgr.mark_stage_completed(stage=1)
            del model
            clear_gpu_memory()

    # ── Phase 2: Production Stage 2 (Per-Epoch Atomic Resume) ───────────────
    can_run_stage2 = args.output.exists() or resume_mgr.has_resumable_checkpoint(stage=2)
    if args.stage == 2 or (args.stage is None and can_run_stage2):
        if not can_run_stage2:
            print(f"[ERROR] {args.output} not found. Run Stage 1 first.")
            sys.exit(1)

        initial_epoch = resume_mgr.get_initial_epoch(stage=2)
        print(
            f"\n--- STAGE 2: End-to-End Fine-Tuning "
            f"(Epoch {initial_epoch} -> {args.epochs2}, frac={ft_frac:.2f}, LR={lr2:.6f}) ---"
        )
        if not backbone_layer_name and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            backbone_layer_name = meta.get("backbone_layer_name", "")

        if resume_mgr.has_resumable_checkpoint(stage=2):
            ckpt_path = resume_mgr.latest_checkpoint_path(stage=2)
            print(f"  [RESUME] Restoring Stage 2 checkpoint from {ckpt_path} (epoch {initial_epoch})")
            try:
                model = load_resumable_model(ckpt_path, custom_objects=get_custom_objects())
                if model.optimizer is None:
                    raise ValueError("Optimizer state missing")
            except Exception as exc:
                print(f"  [RESUME] Reloaded weights without optimizer ({exc}); recompiling remaining steps.")
                model = keras.models.load_model(ckpt_path, compile=False, custom_objects=get_custom_objects())
                _unfreeze_backbone_for_stage2(model, backbone_layer_name, fine_tune_fraction=ft_frac)
                remaining = max(1, (args.epochs2 - initial_epoch) * steps)
                compile_model(
                    model, learning_rate=lr2, total_steps=remaining,
                    use_focal_loss=use_focal, focal_gamma=gamma,
                )
        else:
            model = keras.models.load_model(args.output, compile=False, custom_objects=get_custom_objects())
            _unfreeze_backbone_for_stage2(model, backbone_layer_name, fine_tune_fraction=ft_frac)
            compile_model(
                model,
                learning_rate=lr2,
                total_steps=max(1, args.epochs2 * steps),
                use_focal_loss=use_focal,
                focal_gamma=gamma,
            )

        try:
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs2,
                initial_epoch=initial_epoch,
                class_weight=class_weight,
                callbacks=get_callbacks(resume_mgr, stage=2),
                verbose=1,
            )
        except (KeyboardInterrupt, Exception) as exc:
            resume_mgr.record_interruption(stage=2, exc=exc, model=model)
            sys.exit(1)

        resume_mgr.mark_stage_completed(stage=2)
        resume_mgr.mark_all_completed(cleanup_latest=True)
        save_metadata(
            args.output, args.backbone, backbone_layer_name,
            include_fft, use_cbam, args.static_srm, args.ela_only, use_focal, params,
        )

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 68)
    print(f"  Multi-Branch pipeline finished in {elapsed / 60:.1f} minutes")
    print(f"  Best val_auc : {resume_mgr.get_best_val_auc():.4f}")
    print(f"  Best model   : {args.output.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()
