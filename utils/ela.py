"""
DeepForensics — Error Level Analysis (ELA) Utility
===================================================
Generates ELA heatmaps by re-saving an image at a specified JPEG quality
and computing the per-pixel difference against the original.  Tampered
regions compress differently and appear as bright spots in the ELA map.

Usage as a library:
    from utils.ela import convert_to_ela_image
    ela_pil = convert_to_ela_image("photo.jpg", quality=90)

Usage from CLI (self-test):
    python -m utils.ela
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


# ── Public API ───────────────────────────────────────────────────────────────

def convert_to_ela_image(
    image_source: Union[str, Path, Image.Image],
    quality: int = 90,
    target_size: tuple[int, int] = (224, 224),
    scale: float = 10.0,
) -> Image.Image:
    """Return a brightness-enhanced ELA heatmap as an RGB PIL Image.

    Parameters
    ----------
    image_source : str | Path | PIL.Image.Image
        File path to an image *or* an already-opened PIL Image.
    quality : int, default 90
        JPEG re-compression quality (1-100).  Lower values amplify the
        ELA signal but also introduce more noise.
    target_size : (width, height), default (224, 224)
        Output size — matches ResNet50 input expectations.
    scale : float, default 10.0
        Brightness multiplier applied to the raw difference image so
        subtle compression artefacts become visible.

    Returns
    -------
    PIL.Image.Image
        RGB image of shape ``target_size`` with pixel values in [0, 255].
    """
    # ── 1. Load & normalise to RGB ───────────────────────────────────────
    if isinstance(image_source, (str, Path)):
        original = Image.open(image_source).convert("RGB")
    else:
        original = image_source.convert("RGB")

    # ── 2. Re-save at the chosen JPEG quality into an in-memory buffer ──
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    # ── 3. Pixel-wise absolute difference  ──────────────────────────────
    diff = ImageChops.difference(original, resaved)

    # ── 4. Enhance brightness so forensic artefacts are visible ─────────
    enhancer = ImageEnhance.Brightness(diff)
    ela_image = enhancer.enhance(scale)

    # ── 5. Resize to model-ready dimensions ─────────────────────────────
    ela_image = ela_image.resize(target_size, Image.LANCZOS)

    return ela_image


def ela_to_array(ela_image: Image.Image) -> np.ndarray:
    """Convert an ELA PIL Image to a float32 NumPy array normalised to [0, 1].

    Useful for feeding directly into a TensorFlow/Keras model's
    ``model.predict()`` after adding a batch dimension.
    """
    arr = np.asarray(ela_image, dtype=np.float32) / 255.0
    return arr


# ── CLI Self-Test ────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Create a synthetic image, run ELA, and verify output shape & range."""
    print("=" * 60)
    print("  DeepForensics ELA Self-Test")
    print("=" * 60)

    # Build a 512x512 synthetic image with two distinct colour blocks
    # to simulate a naive splice (different compression characteristics).
    synth = Image.new("RGB", (512, 512), color=(30, 90, 160))
    # Paste a bright block to mimic a tampered region
    tampered_region = Image.new("RGB", (200, 200), color=(255, 80, 80))
    synth.paste(tampered_region, (150, 150))

    print(f"  Synthetic source size : {synth.size}")

    # Run ELA at default quality=90
    ela_result = convert_to_ela_image(synth, quality=90)

    # Verify type
    assert isinstance(ela_result, Image.Image), "Output must be a PIL Image"
    print(f"  ELA output size       : {ela_result.size}")

    # Verify dimensions
    assert ela_result.size == (224, 224), (
        f"Expected (224, 224), got {ela_result.size}"
    )
    print("  Dimensions (224x224)  : OK")

    # Verify pixel range [0, 255]
    arr = np.asarray(ela_result)
    assert arr.shape == (224, 224, 3), f"Expected (224,224,3), got {arr.shape}"
    assert arr.min() >= 0, f"Pixel min {arr.min()} < 0"
    assert arr.max() <= 255, f"Pixel max {arr.max()} > 255"
    print(f"  Pixel range [{arr.min()}, {arr.max()}] in [0, 255] : OK")

    # Verify normalised array helper
    norm = ela_to_array(ela_result)
    assert norm.dtype == np.float32, f"Expected float32, got {norm.dtype}"
    assert 0.0 <= norm.min() <= norm.max() <= 1.0, "Normalised range failed"
    print(f"  Normalised range [{norm.min():.3f}, {norm.max():.3f}] in [0,1] : OK")

    # Verify it also works with a file path (round-trip through temp file)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        synth.save(tmp, format="JPEG", quality=95)
        tmp_path = tmp.name
    ela_from_path = convert_to_ela_image(tmp_path, quality=90)
    assert ela_from_path.size == (224, 224), "File-path input failed"
    Path(tmp_path).unlink(missing_ok=True)
    print("  File-path input       : OK")

    print("-" * 60)
    print("  All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
