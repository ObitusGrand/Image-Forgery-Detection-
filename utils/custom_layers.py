"""
DeepForensics — Custom Deep Learning Layers (Keras 3.x Serializable)
====================================================================
Implements pure deep learning building blocks specifically designed for
image forgery detection on RTX 3060 (FP16 mixed precision):

    1. BayarConstraint & BayarConv2D (Bayar & Stamm, 2018):
       Trainable constrained 5x5 high-pass convolution where the center weight
       is fixed to -1 and surrounding weights are constrained to sum to +1.
       Forces the network to suppress image content and learn forensic noise
       residuals end-to-end via backpropagation.

    2. StaticSRMConv2D:
       Fixed 3-kernel Spatial Rich Model (SRM) high-pass filter bank
       (included for controlled ablation against trainable BayarConv2D).

    3. CBAMBlock (Woo et al., 2018):
       Convolutional Block Attention Module (Channel Attention + Spatial
       Attention) that focuses feature pooling on small tampered regions.

    4. GatedModalityFusion:
       Learned attention-weighted fusion across multiple forensic streams
       (Multi-Quality ELA, Bayar Noise Residual, 2D FFT Spectrum).

All layers are registered under package ``DeepForensics`` so
``keras.models.load_model()`` restores them cleanly once this module is imported.

CLI self-test:
    python -m utils.custom_layers
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Compatible across standalone Keras 3 (`keras.saving`) and `tensorflow.keras` (`keras.utils`)
try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable


# ── 1. Bayar Constrained Convolution ─────────────────────────────────────────

@register_serializable(package="DeepForensics")
class BayarConstraint(keras.constraints.Constraint):
    """Constrained high-pass kernel projection (Bayar & Stamm, 2018).

    Enforces on every 2D kernel ``w`` of shape ``(kh, kw, in_c, out_c)``:
        - center weight ``w[kh//2, kw//2] = -1``
        - sum of all surrounding weights = ``+1``
        - total filter sum = ``0`` (strict prediction-error residual filter)

    Computations run in float32 for numerical stability under mixed_float16.
    """

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        orig_dtype = w.dtype
        w32 = tf.cast(w, tf.float32)

        shape = tf.shape(w32)
        kh = shape[0]
        kw = shape[1]
        cy = kh // 2
        cx = kw // 2

        # Build center mask: 1 at (cy, cx), 0 elsewhere
        ys = tf.range(kh)[:, None]
        xs = tf.range(kw)[None, :]
        center_2d = tf.cast(tf.logical_and(tf.equal(ys, cy), tf.equal(xs, cx)), tf.float32)
        center_mask = center_2d[:, :, None, None]
        surround_mask = 1.0 - center_mask

        # Zero out center and normalize surrounding weights to sum to +1
        w_surr = w32 * surround_mask
        surr_sum = tf.reduce_sum(w_surr, axis=[0, 1], keepdims=True)
        num_surr = tf.cast(kh * kw - 1, tf.float32)

        # Safe normalization: divide when |sum| is healthy, else shift evenly
        safe_div = w_surr / tf.where(tf.abs(surr_sum) >= 1e-3, surr_sum, tf.ones_like(surr_sum))
        safe_shift = (w_surr + (1.0 - surr_sum) / num_surr) * surround_mask
        w_norm = tf.where(tf.abs(surr_sum) >= 1e-3, safe_div, safe_shift)

        # Set center weight to -1
        constrained = w_norm - center_mask
        return tf.cast(constrained, orig_dtype)

    def get_config(self) -> dict:
        return {}


@register_serializable(package="DeepForensics")
class BayarConv2D(layers.Layer):
    """Trainable constrained high-pass convolution layer for noise extraction.

    Applies a 5x5 (or odd-sized) constrained convolution with ``BayarConstraint``
    to suppress semantic scene content and expose camera/splicing residuals.
    """

    def __init__(
        self,
        filters: int = 12,
        kernel_size: int = 5,
        padding: str = "same",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if kernel_size % 2 == 0:
            raise ValueError(f"BayarConv2D kernel_size must be odd, got {kernel_size}")
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.padding = padding
        self.constraint = BayarConstraint()

    def build(self, input_shape) -> None:
        in_channels = int(input_shape[-1])
        # Initialize surround weights around +1/(k^2 - 1) so initial sum ~ +1
        base_val = 1.0 / float(self.kernel_size * self.kernel_size - 1)
        initializer = keras.initializers.RandomUniform(
            minval=base_val * 0.5,
            maxval=base_val * 1.5,
        )
        self.kernel = self.add_weight(
            name="bayar_kernel",
            shape=(self.kernel_size, self.kernel_size, in_channels, self.filters),
            initializer=initializer,
            constraint=self.constraint,
            trainable=True,
        )
        # Apply constraint immediately on initial weights
        self.kernel.assign(self.constraint(self.kernel))
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Enforce constraint cleanly in forward graph as well
        k = self.constraint(self.kernel)
        return tf.nn.conv2d(
            inputs,
            filters=tf.cast(k, inputs.dtype),
            strides=[1, 1, 1, 1],
            padding=self.padding.upper(),
        )

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
            }
        )
        return cfg


# ── 2. Static SRM High-Pass Filter Bank (For Ablation vs BayarConv2D) ────────

def _build_srm_kernels() -> np.ndarray:
    """Return 3 canonical 5x5 SRM high-pass filters (shape: 5, 5, 3, 3)."""
    # 1st-order / 2nd-order / 3rd-order (KV) SRM filters normalized by denominator
    f1 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ) / 4.0

    f2 = np.array(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=np.float32,
    ) / 12.0

    f3 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ) / 2.0

    # Stack across 3 RGB input channels -> 3 output maps
    kernels = np.zeros((5, 5, 3, 3), dtype=np.float32)
    for c in range(3):
        kernels[:, :, c, 0] = f1 / 3.0
        kernels[:, :, c, 1] = f2 / 3.0
        kernels[:, :, c, 2] = f3 / 3.0
    return kernels


@register_serializable(package="DeepForensics")
class StaticSRMConv2D(layers.Layer):
    """Non-trainable 3-filter Spatial Rich Model (SRM) high-pass layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape) -> None:
        init_kernels = _build_srm_kernels()
        self.srm_kernel = self.add_weight(
            name="srm_kernel",
            shape=(5, 5, 3, 3),
            initializer=keras.initializers.Constant(init_kernels),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.nn.conv2d(
            inputs,
            filters=tf.cast(self.srm_kernel, inputs.dtype),
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

    def get_config(self) -> dict:
        return super().get_config()


# ── 3. CBAM Attention Module (Channel + Spatial Attention) ───────────────────

@register_serializable(package="DeepForensics")
class CBAMBlock(layers.Layer):
    """Convolutional Block Attention Module (Woo et al., 2018).

    Sequentially applies:
        1. Channel Attention (shared bottleneck MLP over AvgPool + MaxPool)
        2. Spatial Attention (7x7 Conv2D over channel-wise Mean + Max maps)
    """

    def __init__(self, reduction_ratio: int = 8, spatial_kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = int(reduction_ratio)
        self.spatial_kernel_size = int(spatial_kernel_size)

    def build(self, input_shape) -> None:
        channels = int(input_shape[-1])
        hidden = max(channels // self.reduction_ratio, 8)

        self.mlp_down = layers.Dense(hidden, activation="relu", use_bias=True, name="cbam_fc1")
        self.mlp_up = layers.Dense(channels, use_bias=True, name="cbam_fc2")
        self.spatial_conv = layers.Conv2D(
            filters=1,
            kernel_size=self.spatial_kernel_size,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name="cbam_spatial_conv",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # ── Channel Attention ────────────────────────────────────────────
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=False)

        avg_out = self.mlp_up(self.mlp_down(avg_pool))
        max_out = self.mlp_up(self.mlp_down(max_pool))
        ch_att = tf.nn.sigmoid(avg_out + max_out)
        ch_att = tf.reshape(ch_att, [-1, 1, 1, tf.shape(inputs)[-1]])
        x = inputs * tf.cast(ch_att, inputs.dtype)

        # ── Spatial Attention ────────────────────────────────────────────
        sp_avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        sp_max = tf.reduce_max(x, axis=-1, keepdims=True)
        sp_cat = tf.concat([sp_avg, sp_max], axis=-1)
        sp_att = self.spatial_conv(sp_cat)
        return x * tf.cast(sp_att, x.dtype)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update(
            {
                "reduction_ratio": self.reduction_ratio,
                "spatial_kernel_size": self.spatial_kernel_size,
            }
        )
        return cfg


# ── 4. Gated Modality Fusion ─────────────────────────────────────────────────

@register_serializable(package="DeepForensics")
class GatedModalityFusion(layers.Layer):
    """Learned attention-weighted fusion across N feature-vector modalities.

    Projects every branch vector into a shared ``embed_dim`` latent space,
    computes per-sample softmax attention weights over the branches, and
    returns ``concat([weighted_sum, g_1 * h_1, ..., g_N * h_N])``.
    """

    def __init__(self, embed_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.proj_layers: list[layers.Layer] = []
        self.norm_layers: list[layers.Layer] = []

    def build(self, input_shape) -> None:
        if not isinstance(input_shape, (list, tuple)) or not isinstance(input_shape[0], (list, tuple)):
            raise ValueError("GatedModalityFusion expects a list of 2D feature tensors.")
        self.num_modalities = len(input_shape)
        self.proj_layers = [
            layers.Dense(self.embed_dim, activation="gelu", name=f"mod_proj_{i}")
            for i in range(self.num_modalities)
        ]
        self.norm_layers = [
            layers.LayerNormalization(name=f"mod_norm_{i}")
            for i in range(self.num_modalities)
        ]
        self.gate_hidden = layers.Dense(64, activation="relu", name="gate_hidden")
        self.gate_logits = layers.Dense(
            self.num_modalities,
            activation="softmax",
            dtype="float32",
            name="gate_weights",
        )
        super().build(input_shape)

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        projected = [
            norm(proj(feat))
            for feat, proj, norm in zip(inputs, self.proj_layers, self.norm_layers)
        ]
        cat_proj = tf.concat(projected, axis=-1)
        gates = self.gate_logits(self.gate_hidden(cat_proj))  # (B, M) float32

        gated_branches = []
        weighted_sum = None
        for i, h in enumerate(projected):
            g_i = tf.cast(gates[:, i : i + 1], h.dtype)
            gh = h * g_i
            gated_branches.append(gh)
            weighted_sum = gh if weighted_sum is None else (weighted_sum + gh)

        return tf.concat([weighted_sum, *gated_branches], axis=-1)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim})
        return cfg


# ── Custom Objects Registry Helper ───────────────────────────────────────────

def get_custom_objects() -> dict[str, type]:
    """Return mapping of custom classes for explicit Keras deserialization."""
    return {
        "BayarConstraint": BayarConstraint,
        "BayarConv2D": BayarConv2D,
        "StaticSRMConv2D": StaticSRMConv2D,
        "CBAMBlock": CBAMBlock,
        "GatedModalityFusion": GatedModalityFusion,
    }


# ── CLI Self-Test ────────────────────────────────────────────────────────────

def _self_test() -> None:
    print("=" * 60)
    print("  DeepForensics Custom DL Layers Self-Test")
    print("=" * 60)

    # 1. BayarConv2D constraint invariant check
    inp_rgb = keras.Input(shape=(32, 32, 3))
    bayar_layer = BayarConv2D(filters=6, kernel_size=5)
    out_bayar = bayar_layer(inp_rgb)
    assert out_bayar.shape == (None, 32, 32, 6)

    k_val = bayar_layer.constraint(bayar_layer.kernel).numpy()
    center_vals = k_val[2, 2, :, :]
    total_sums = np.sum(k_val, axis=(0, 1))
    assert np.allclose(center_vals, -1.0, atol=1e-5), "Bayar center != -1"
    assert np.allclose(total_sums, 0.0, atol=1e-4), "Bayar total kernel sum != 0"
    print("  BayarConv2D constraint invariant (center=-1, sum=0) : OK")

    # 2. StaticSRMConv2D check
    out_srm = StaticSRMConv2D()(inp_rgb)
    assert out_srm.shape == (None, 32, 32, 3)
    print("  StaticSRMConv2D shape (None, 32, 32, 3)            : OK")

    # 3. CBAMBlock check
    out_cbam = CBAMBlock(reduction_ratio=4)(out_bayar)
    assert out_cbam.shape == (None, 32, 32, 6)
    print("  CBAMBlock shape preservation                       : OK")

    # 4. GatedModalityFusion check
    f1 = keras.Input(shape=(64,))
    f2 = keras.Input(shape=(32,))
    fused = GatedModalityFusion(embed_dim=48)([f1, f2])
    assert fused.shape == (None, 48 * 3)
    print("  GatedModalityFusion output shape                   : OK")

    # 5. Full save/load round-trip check
    import tempfile
    from pathlib import Path

    g1 = layers.GlobalAveragePooling2D()(out_cbam)
    g2 = layers.GlobalAveragePooling2D()(out_srm)
    mf = GatedModalityFusion(embed_dim=32)([g1, g2])
    pred = layers.Dense(1, activation="sigmoid", dtype="float32")(mf)
    m = keras.Model(inp_rgb, pred)
    x_dummy = np.random.rand(2, 32, 32, 3).astype(np.float32)
    y_before = m.predict(x_dummy, verbose=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test_custom.keras"
        m.save(p)
        m_loaded = keras.models.load_model(p, custom_objects=get_custom_objects())
        y_after = m_loaded.predict(x_dummy, verbose=0)
        assert np.allclose(y_before, y_after, atol=1e-5), "Save/load mismatch!"
    print("  Keras 3 save/load round-trip                       : OK")
    print("-" * 60)
    print("  All custom layer checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
