"""
DeepForensics — Multi-Modal Feature Cache Builder (CASIA v2.0)
==============================================================
Pre-computes disk-cached lossless PNG forensic modalities from
``dataset/raw/{Au,Tp}`` into ``dataset/features_224/`` so training on
RTX 3060 (12GB VRAM) + 16GB RAM streams cleanly from SSD without CPU
bottlenecks or RAM OOM:

    dataset/features_224/
    ├── ela_q70/{Au,Tp}/*.png   (224x224 RGB ELA @ JPEG Q=70)
    ├── ela_q85/{Au,Tp}/*.png   (224x224 RGB ELA @ JPEG Q=85)
    ├── ela_q95/{Au,Tp}/*.png   (224x224 RGB ELA @ JPEG Q=95)
    ├── raw_224/{Au,Tp}/*.png   (224x224 RGB for BayarConv2D noise branch)
    └── fft_224/{Au,Tp}/*.png   (224x224 Grayscale 2D FFT log-magnitude)

Also builds the shared deterministic stratified 80/20 split manifest at
``dataset/split_manifest.json`` so ResNet50, Multi-Branch training,
ensemble calibration, and evaluation share the EXACT same train/val split.

Usage:
    python prepare_features.py                  # build all caches + manifest
    python prepare_features.py --workers 4      # parallel CPU preprocessing
    python prepare_features.py --dry-run        # inspect counts without writing
    python prepare_features.py --limit 50       # quick test on first 50/class
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from utils.dataset_split import (
    DEFAULT_FEATURES_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_DIR,
    SUPPORTED_EXTENSIONS,
    build_or_load_manifest,
)
from utils.features import (
    DEFAULT_ELA_QUALITIES,
    DEFAULT_TARGET_SIZE,
    generate_fft_spectrum_image,
    generate_multi_ela_images,
    generate_raw_rgb_image,
)


CLASSES = ("Au", "Tp")


def _process_single_image(args: tuple[str, str, str]) -> tuple[bool, str]:
    """Worker function: generate all 5 PNG feature maps for one raw image."""
    src_str, cls, out_root_str = args
    src_path = Path(src_str)
    out_root = Path(out_root_str)
    stem = src_path.stem

    dst_q70 = out_root / "ela_q70" / cls / f"{stem}.png"
    dst_q85 = out_root / "ela_q85" / cls / f"{stem}.png"
    dst_q95 = out_root / "ela_q95" / cls / f"{stem}.png"
    dst_raw = out_root / "raw_224" / cls / f"{stem}.png"
    dst_fft = out_root / "fft_224" / cls / f"{stem}.png"

    # Resume-friendly skip if all 5 cached outputs already exist
    if (
        dst_q70.exists()
        and dst_q85.exists()
        and dst_q95.exists()
        and dst_raw.exists()
        and dst_fft.exists()
    ):
        return True, "skipped"

    try:
        ela_map = generate_multi_ela_images(
            src_path,
            qualities=DEFAULT_ELA_QUALITIES,
            target_size=DEFAULT_TARGET_SIZE,
        )
        raw_img = generate_raw_rgb_image(src_path, target_size=DEFAULT_TARGET_SIZE)
        fft_img = generate_fft_spectrum_image(src_path, target_size=DEFAULT_TARGET_SIZE)

        ela_map[70].save(dst_q70, format="PNG")
        ela_map[85].save(dst_q85, format="PNG")
        ela_map[95].save(dst_q95, format="PNG")
        raw_img.save(dst_raw, format="PNG")
        fft_img.save(dst_fft, format="PNG")
        return True, "converted"
    except Exception as exc:
        return False, f"{src_path.name}: {exc}"


def collect_raw_files(raw_dir: Path, limit: int | None = None) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for cls in CLASSES:
        cls_dir = raw_dir / cls
        if not cls_dir.is_dir():
            print(f"  [WARN] Missing directory: {cls_dir}")
            result[cls] = []
            continue
        files = sorted(
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if limit is not None:
            files = files[:limit]
        result[cls] = files
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepForensics: Build disk-cached Multi-ELA + Raw + 2D FFT features."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--workers", type=int, default=4, help="Parallel CPU workers (default: 4).")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-class cap for testing.")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing.")
    parser.add_argument("--rebuild-manifest", action="store_true", help="Force regenerate split manifest.")
    args = parser.parse_args()

    print("=" * 64)
    print("  DeepForensics - Multi-Modal Feature Cache Builder")
    print("=" * 64)
    print(f"  Raw dataset : {args.raw_dir.resolve()}")
    print(f"  Output root : {args.out_dir.resolve()}")
    print(f"  Modalities  : ELA(Q=70,85,95) + Raw RGB + 2D FFT Spectrum")
    print(f"  Workers     : {args.workers} | Dry-run: {args.dry_run}")
    print("-" * 64)

    class_files = collect_raw_files(args.raw_dir, limit=args.limit)
    total_files = sum(len(v) for v in class_files.values())
    if total_files == 0:
        print("  [ERROR] No raw images found. Expected dataset/raw/Au and dataset/raw/Tp.")
        sys.exit(1)

    for cls, files in class_files.items():
        print(f"    {cls}: {len(files)} images")

    if args.dry_run:
        print("  [DRY-RUN] Exiting without modifying disk.")
        return

    # Create output directories upfront
    for sub in ("ela_q70", "ela_q85", "ela_q95", "raw_224", "fft_224"):
        for cls in CLASSES:
            (args.out_dir / sub / cls).mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, str]] = []
    for cls, files in class_files.items():
        for src in files:
            tasks.append((str(src), cls, str(args.out_dir)))

    t0 = time.perf_counter()
    converted = 0
    skipped = 0
    errors = 0

    if args.workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            ok, msg = _process_single_image(task)
            if ok:
                if msg == "skipped":
                    skipped += 1
                else:
                    converted += 1
            else:
                errors += 1
                print(f"    [ERROR] {msg}")
            if idx % 500 == 0 or idx == len(tasks):
                elapsed = time.perf_counter() - t0
                print(f"    Progress: {idx}/{len(tasks)} | conv={converted} skip={skipped} err={errors} ({elapsed:.1f}s)")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_process_single_image, t) for t in tasks]
            for idx, fut in enumerate(as_completed(futures), start=1):
                ok, msg = fut.result()
                if ok:
                    if msg == "skipped":
                        skipped += 1
                    else:
                        converted += 1
                else:
                    errors += 1
                    print(f"    [ERROR] {msg}")
                if idx % 500 == 0 or idx == len(tasks):
                    elapsed = time.perf_counter() - t0
                    print(f"    Progress: {idx}/{len(tasks)} | conv={converted} skip={skipped} err={errors} ({elapsed:.1f}s)")

    # Build/verify shared stratified split manifest
    manifest = build_or_load_manifest(
        manifest_path=args.manifest,
        source_dir=args.out_dir,
        force_rebuild=args.rebuild_manifest,
    )
    elapsed = time.perf_counter() - t0
    print("-" * 64)
    print(f"  Done in {elapsed:.1f}s | Converted: {converted} | Skipped: {skipped} | Errors: {errors}")
    print(f"  Split manifest: {args.manifest.resolve()}")
    print(f"    Train: {manifest['train_count']} | Val: {manifest['val_count']} (seed={manifest['seed']})")
    print("=" * 64)


if __name__ == "__main__":
    main()
