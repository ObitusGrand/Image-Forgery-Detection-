"""
DeepForensics — Explainable Deep Learning (Grad-CAM++) & TTA Utilities
======================================================================
Provides two core deep learning inference & interpretability modules:

    1. Grad-CAM++ (Chattopadhay et al., 2018):
       Weakly-supervised visual localization of forged regions using ONLY
       image-level binary labels via ``tf.GradientTape``. Automatically
       falls back to standard Grad-CAM when higher-order gradients vanish.

    2. 4-View Lossless Test-Time Augmentation (TTA):
       Averages model predictions over 4 orientation-preserving lossless
       views (original, horizontal flip, vertical flip, 90-degree rotation)
       to reduce prediction variance without distorting JPEG/noise traces.

CLI self-test:
    python -m utils.xai
"""

from __future__ import annotations

from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


# ── Grad-CAM++ Localization ──────────────────────────────────────────────────

def find_last_conv_layer(
    model: keras.Model,
    preferred_name: Optional[str] = None,
) -> Optional[str]:
    """Locate the best 4D spatial feature layer for Grad-CAM++.

    Priority:
        1. Explicit ``preferred_name`` if present and 4D.
        2. Named ``ela_cbam`` / ``conv5_block3_out`` / ``top_activation``.
        3. Last layer in the model graph whose output has rank 4 and H, W > 1.
    """
    if preferred_name:
        try:
            layer = model.get_layer(preferred_name)
            shape = getattr(layer.output, "shape", None)
            if shape is not None and len(shape) == 4:
                return layer.name
        except ValueError:
            pass

    for candidate in ("ela_cbam", "ela_backbone_out", "conv5_block3_out", "top_activation"):
        try:
            layer = model.get_layer(candidate)
            shape = getattr(layer.output, "shape", None)
            if shape is not None and len(shape) == 4:
                return layer.name
        except ValueError:
            continue

    # Generic fallback: walk backwards for last non-Model 4D feature map first,
    # then allow nested keras.Model layers if no outer 4D layer exists.
    nested_fallback: Optional[str] = None
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.InputLayer):
            continue
        try:
            out = layer.get_output_at(-1) if isinstance(layer, keras.Model) else layer.output
        except Exception:
            continue
        if isinstance(out, (list, tuple)):
            continue
        shape = getattr(out, "shape", None)
        if shape is not None and len(shape) == 4:
            h, w = shape[1], shape[2]
            if (h is None or h > 1) and (w is None or w > 1):
                if isinstance(layer, keras.Model):
                    if nested_fallback is None:
                        nested_fallback = layer.name
                else:
                    return layer.name

    return nested_fallback


def _to_tensor_inputs(
    model_inputs: Union[np.ndarray, tf.Tensor, dict[str, np.ndarray], list[np.ndarray]],
):
    """Convert numpy/dict/list inputs into float32 tf.Tensors with batch dim."""
    if isinstance(model_inputs, dict):
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in model_inputs.items()}
    if isinstance(model_inputs, (list, tuple)):
        return [tf.convert_to_tensor(v, dtype=tf.float32) for v in model_inputs]
    return tf.convert_to_tensor(model_inputs, dtype=tf.float32)


def compute_gradcam_pp(
    model: keras.Model,
    model_inputs: Union[np.ndarray, tf.Tensor, dict[str, np.ndarray], list[np.ndarray]],
    last_conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """Compute a normalized [0, 1] 2D Grad-CAM++ localization heatmap.

    Targets the binary "Tampered" logit (using log-odds when output is sigmoid
    so gradients remain strong even for high-confidence predictions).
    """
    layer_name = find_last_conv_layer(model, preferred_name=last_conv_layer_name)
    if layer_name is None:
        return np.zeros((7, 7), dtype=np.float32)

    target_layer = model.get_layer(layer_name)
    target_tensor = target_layer.get_output_at(-1) if isinstance(target_layer, keras.Model) else target_layer.output
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_tensor, model.output],
    )

    tensors = _to_tensor_inputs(model_inputs)

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(tensors, training=False)
        conv_outputs_f32 = tf.cast(conv_outputs, tf.float32)
        preds_f32 = tf.cast(preds, tf.float32)
        # Use pre-sigmoid log-odds to avoid sigmoid saturation vanishing gradients
        p_clip = tf.clip_by_value(preds_f32[:, 0], 1e-6, 1.0 - 1e-6)
        target_score = tf.math.log(p_clip / (1.0 - p_clip))

    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        return np.zeros((7, 7), dtype=np.float32)

    grads = tf.cast(grads[0], tf.float32)           # (H, W, C)
    activations = conv_outputs_f32[0]               # (H, W, C)

    # Grad-CAM++ closed-form pixel-wise weights:
    # alpha_kc = g^2 / (2 * g^2 + sum(A * g^3) + eps)
    grads_sq = tf.square(grads)
    grads_cu = grads_sq * grads
    sum_act_g3 = tf.reduce_sum(activations * grads_cu, axis=(0, 1), keepdims=True)
    denom = 2.0 * grads_sq + sum_act_g3 + 1e-7
    alpha = tf.where(grads != 0.0, grads_sq / denom, tf.zeros_like(grads))

    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(0, 1))  # (C,)
    cam = tf.reduce_sum(activations * weights[None, None, :], axis=-1)
    cam = tf.nn.relu(cam).numpy()

    # Fallback to standard Grad-CAM if Grad-CAM++ yields an all-zero map
    if float(cam.max()) < 1e-8:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.nn.relu(tf.reduce_sum(activations * pooled_grads[None, None, :], axis=-1)).numpy()

    max_val = float(cam.max())
    if max_val > 1e-8:
        cam = (cam - float(cam.min())) / (max_val - float(cam.min()) + 1e-8)
    else:
        cam = np.zeros_like(cam, dtype=np.float32)

    return cam.astype(np.float32)


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.48,
    colormap_name: str = "inferno",
) -> Image.Image:
    """Blend a normalized [0, 1] Grad-CAM++ heatmap onto an RGB PIL Image."""
    base = original_image.convert("RGB")
    w, h = base.size

    # Upsample coarse CAM smoothly to image dimensions
    hm_uint8 = np.clip(heatmap * 255.0, 0.0, 255.0).astype(np.uint8)
    hm_resized = np.asarray(
        Image.fromarray(hm_uint8, mode="L").resize((w, h), Image.BICUBIC),
        dtype=np.float32,
    ) / 255.0

    # Map scalar intensity to RGB via matplotlib colormap
    cmap = matplotlib.colormaps[colormap_name]
    colored = (cmap(hm_resized)[..., :3] * 255.0).astype(np.float32)

    base_arr = np.asarray(base, dtype=np.float32)
    # Modulate blend by local activation intensity so background stays legible
    local_alpha = (alpha * (0.25 + 0.75 * hm_resized))[..., np.newaxis]
    blended = base_arr * (1.0 - local_alpha) + colored * local_alpha
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8), mode="RGB")


# ── 4-View Lossless Test-Time Augmentation (TTA) ─────────────────────────────

def _transform_4d(arr: np.ndarray, view_idx: int) -> np.ndarray:
    """Apply lossless spatial view transform to a (B, H, W, C) numpy array."""
    if view_idx == 0:
        return arr
    if view_idx == 1:
        return arr[:, :, ::-1, :]           # horizontal flip
    if view_idx == 2:
        return arr[:, ::-1, :, :]           # vertical flip
    if view_idx == 3:
        return np.rot90(arr, k=1, axes=(1, 2))  # 90-degree rotation
    raise ValueError(f"Unsupported TTA view_idx: {view_idx}")


def _forward_probs(
    model: keras.Model,
    inputs: Union[np.ndarray, dict[str, np.ndarray]],
) -> np.ndarray:
    """Direct eager/graph forward pass (zero DataAdapter leak across batch loops)."""
    tensors = _to_tensor_inputs(inputs)
    preds = model(tensors, training=False)
    return np.asarray(preds.numpy(), dtype=np.float64).reshape(-1)


def predict_with_tta(
    model: keras.Model,
    inputs: Union[np.ndarray, dict[str, np.ndarray]],
    use_tta: bool = True,
) -> np.ndarray:
    """Run model prediction with optional 4-view lossless TTA.

    Uses direct ``model(x, training=False)`` instead of ``model.predict()``
    to prevent Keras DataAdapter memory leaks when invoked inside batch loops.

    Supports both single-tensor inputs ``(B, H, W, C)`` and multi-branch
    dict inputs ``{"multi_ela_input": ..., "raw_rgb_input": ..., ...}``.
    Returns 1D float64 array of predicted probabilities of shape ``(B,)``.
    """
    if not use_tta:
        return _forward_probs(model, inputs)

    view_preds = []
    for v in range(4):
        if isinstance(inputs, dict):
            aug_in = {k: _transform_4d(np.asarray(arr), v) for k, arr in inputs.items()}
        else:
            aug_in = _transform_4d(np.asarray(inputs), v)
        view_preds.append(_forward_probs(model, aug_in))

    return np.mean(view_preds, axis=0)


# ── CLI Self-Test ────────────────────────────────────────────────────────────

def _self_test() -> None:
    print("=" * 60)
    print("  DeepForensics XAI (Grad-CAM++ & TTA) Self-Test")
    print("=" * 60)

    inp = keras.Input(shape=(64, 64, 3))
    x = keras.layers.Conv2D(8, 3, padding="same", activation="relu", name="last_conv")(inp)
    x = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    m = keras.Model(inp, out)

    dummy = np.random.rand(1, 64, 64, 3).astype(np.float32)
    layer_name = find_last_conv_layer(m)
    assert layer_name == "last_conv", f"Expected last_conv, got {layer_name}"
    print(f"  Detected conv layer : {layer_name} OK")

    hm = compute_gradcam_pp(m, dummy)
    assert hm.shape == (64, 64)
    assert 0.0 <= float(hm.min()) <= float(hm.max()) <= 1.0
    print(f"  Grad-CAM++ heatmap  : shape={hm.shape}, range=[{hm.min():.3f}, {hm.max():.3f}] OK")

    orig_pil = Image.new("RGB", (224, 224), color=(50, 100, 150))
    ov = overlay_heatmap(orig_pil, hm)
    assert ov.size == (224, 224) and ov.mode == "RGB"
    print(f"  Overlay PIL image   : size={ov.size}, mode={ov.mode} OK")

    p_tta = predict_with_tta(m, dummy, use_tta=True)
    assert p_tta.shape == (1,) and 0.0 <= p_tta[0] <= 1.0
    print(f"  4-view TTA output   : {p_tta[0]:.4f} OK")

    print("-" * 60)
    print("  All XAI & TTA checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
