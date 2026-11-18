"""
DeepForensics — Multi-Modal Forensic Feature Extractors
=======================================================
Provides unified, deterministic preprocessing utilities shared across
offline dataset preparation (`prepare_features.py`), model training
(`train_multibranch.py`), evaluation (`evaluate_model.py`), and
real-time FastAPI inference (`app.py`).

Modalities extracted from a raw RGB image:
    1. Multi-Quality ELA (Q=70, 85, 95) -> (H, W, 9) float32 in [0, 255]
    2. Raw RGB tensor (for BayarConv2D learned noise residuals) -> (H, W, 3) float32 in [0, 255]
    3. 2D FFT Log-Magnitude Spectrum -> (H, W, 1) float32 in [0, 255]

CLI self-test:
    python -m utils.features
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image

from utils.ela import convert_to_ela_image


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_TARGET_SIZE: tuple[int, int] = (224, 224)
DEFAULT_ELA_QUALITIES: tuple[int, int, int] = (70, 85, 95)
DEFAULT_ELA_SCALE: float = 10.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _open_rgb(image_source: Union[str, Path, Image.Image]) -> Image.Image:
    """Open an image path or convert a PIL Image to RGB."""
    if isinstance(image_source, (str, Path)):
        with Image.open(image_source) as img:
            return img.convert("RGB")
    return image_source.convert("RGB")


# ── Modality 1: Multi-Quality ELA ────────────────────────────────────────────

def generate_multi_ela_images(
    image_source: Union[str, Path, Image.Image],
    qualities: Sequence[int] = DEFAULT_ELA_QUALITIES,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    scale: float = DEFAULT_ELA_SCALE,
) -> dict[int, Image.Image]:
    """Generate ELA PIL images at multiple JPEG compression quality levels.

    Using multiple qualities (e.g., 70, 85, 95) reveals compression
    discrepancies across different primary JPEG quantization tables that
    a single quality=90 ELA map can miss.
    """
    original = _open_rgb(image_source)
    return {
        q: convert_to_ela_image(original, quality=q, target_size=target_size, scale=scale)
        for q in qualities
    }


def generate_multi_ela_array(
    image_source: Union[str, Path, Image.Image],
    qualities: Sequence[int] = DEFAULT_ELA_QUALITIES,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    scale: float = DEFAULT_ELA_SCALE,
) -> np.ndarray:
    """Return stacked multi-quality ELA tensor of shape (H, W, 3 * len(qualities)) float32 in [0, 255]."""
    ela_map = generate_multi_ela_images(
        image_source,
        qualities=qualities,
        target_size=target_size,
        scale=scale,
    )
    arrays = [np.asarray(ela_map[q], dtype=np.float32) for q in qualities]
    return np.concatenate(arrays, axis=-1)


# ── Modality 2: Raw RGB (for BayarConv2D Learned Noise Residual Branch) ──────

def generate_raw_rgb_image(
    image_source: Union[str, Path, Image.Image],
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> Image.Image:
    """Resize raw RGB image cleanly for the constrained BayarConv2D noise branch."""
    original = _open_rgb(image_source)
    return original.resize(target_size, Image.LANCZOS)


def generate_raw_rgb_array(
    image_source: Union[str, Path, Image.Image],
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """Return raw RGB float32 array of shape (H, W, 3) in [0, 255]."""
    img = generate_raw_rgb_image(image_source, target_size=target_size)
    return np.asarray(img, dtype=np.float32)


# ── Modality 3: 2D FFT Log-Magnitude Frequency Spectrum ──────────────────────

def generate_fft_spectrum_image(
    image_source: Union[str, Path, Image.Image],
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> Image.Image:
    """Compute centered 2D FFT log-magnitude spectrum as a grayscale PIL Image ('L').

    Natural photographs exhibit smooth radial high-frequency spectral decay.
    Splicing, resampling, and double-JPEG recompression introduce distinct
    periodic spikes and high-frequency discontinuities in Fourier space.
    """
    original = _open_rgb(image_source)
    # Resize first so frequency grid dimensions are consistent across dataset
    gray = original.resize(target_size, Image.LANCZOS).convert("L")
    arr = np.asarray(gray, dtype=np.float32)

    # 2D FFT shifted so DC component sits at center
    fft_coeffs = np.fft.fft2(arr)
    fft_shifted = np.fft.fftshift(fft_coeffs)
    magnitude = np.log1p(np.abs(fft_shifted))

    # Min-max normalize to [0, 255] uint8
    mag_min = float(magnitude.min())
    mag_max = float(magnitude.max())
    if mag_max > mag_min:
        norm = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        norm = np.zeros_like(magnitude)

    uint8_spec = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(uint8_spec, mode="L")


def generate_fft_spectrum_array(
    image_source: Union[str, Path, Image.Image],
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """Return 2D FFT log-magnitude spectrum array of shape (H, W, 1) float32 in [0, 255]."""
    spec_img = generate_fft_spectrum_image(image_source, target_size=target_size)
    arr = np.asarray(spec_img, dtype=np.float32)
    return arr[..., np.newaxis]


# ── Combined Inference Bundle ────────────────────────────────────────────────

def extract_multibranch_inputs(
    image_source: Union[str, Path, Image.Image],
    qualities: Sequence[int] = DEFAULT_ELA_QUALITIES,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    include_fft: bool = True,
) -> dict[str, np.ndarray]:
    """Extract all synchronized model inputs for a single image (unbatched).

    Returns a dict keyed by Keras Input layer names:
        - ``multi_ela_input``: (H, W, 9) float32 in [0, 255]
        - ``raw_rgb_input``:   (H, W, 3) float32 in [0, 255]
        - ``fft_input``:       (H, W, 1) float32 in [0, 255] (if include_fft)
    """
    original = _open_rgb(image_source)
    inputs: dict[str, np.ndarray] = {
        "multi_ela_input": generate_multi_ela_array(
            original, qualities=qualities, target_size=target_size
        ),
        "raw_rgb_input": generate_raw_rgb_array(original, target_size=target_size),
    }
    if include_fft:
        inputs["fft_input"] = generate_fft_spectrum_array(original, target_size=target_size)
    return inputs


# ── CLI Self-Test ────────────────────────────────────────────────────────────

def _self_test() -> None:
    print("=" * 60)
    print("  DeepForensics Multi-Modal Feature Self-Test")
    print("=" * 60)

    synth = Image.new("RGB", (512, 512), color=(40, 100, 170))
    patch = Image.new("RGB", (180, 180), color=(240, 75, 75))
    synth.paste(patch, (160, 160))

    bundle = extract_multibranch_inputs(synth, include_fft=True)
    assert bundle["multi_ela_input"].shape == (224, 224, 9)
    assert bundle["raw_rgb_input"].shape == (224, 224, 3)
    assert bundle["fft_input"].shape == (224, 224, 1)

    for k, v in bundle.items():
        assert v.dtype == np.float32
        assert 0.0 <= float(v.min()) <= float(v.max()) <= 255.0
        print(f"  {k:16s} : shape={v.shape}, range=[{v.min():.1f}, {v.max():.1f}] OK")

    print("-" * 60)
    print("  All multi-modal feature checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
