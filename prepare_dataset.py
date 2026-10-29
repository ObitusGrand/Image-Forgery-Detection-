"""
DeepForensics — Disk-Cached ELA Dataset Builder
================================================
Scans the CASIA v2.0 directory structure::

    dataset/raw/Au/   (authentic images)
    dataset/raw/Tp/   (tampered images)

Converts every image to a 224x224 ELA heatmap and saves the result to::

    dataset/ela_224/Au/
    dataset/ela_224/Tp/

This one-time preprocessing step avoids recomputing ELA on every training
epoch and keeps peak RAM usage low on the 16GB machine.

Usage:
    python prepare_dataset.py                  # process all images
    python prepare_dataset.py --quality 85     # custom JPEG quality
    python prepare_dataset.py --dry-run        # list files without converting
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from utils.ela import convert_to_ela_image

# ── Constants ────────────────────────────────────────────────────────────────

RAW_DIR = Path("dataset/raw")
ELA_DIR = Path("dataset/ela_224")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASSES = ["Au", "Tp"]


# ── Core Logic ───────────────────────────────────────────────────────────────

def collect_image_paths(raw_dir: Path) -> dict[str, list[Path]]:
    """Return a dict mapping class name -> sorted list of image file paths."""
    result = {}
    for cls in CLASSES:
        cls_dir = raw_dir / cls
        if not cls_dir.is_dir():
            print(f"  [WARN] Directory not found: {cls_dir}  -- skipping class '{cls}'")
            result[cls] = []
            continue
        files = sorted(
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        result[cls] = files
    return result


def process_class(
    cls: str,
    file_paths: list[Path],
    ela_dir: Path,
    quality: int,
    dry_run: bool,
) -> tuple[int, int]:
    """Convert all images for one class. Returns (success_count, error_count)."""
    out_dir = ela_dir / cls
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(file_paths)
    success = 0
    errors = 0

    for idx, src_path in enumerate(file_paths, start=1):
        # Output filename — always .png to avoid double JPEG compression
        dst_path = out_dir / (src_path.stem + ".png")

        # Skip already-processed images (resume-friendly)
        if dst_path.exists():
            success += 1
            if idx % 500 == 0 or idx == total:
                print(f"    [{cls}] {idx}/{total}  (skipped, already exists)")
            continue

        if dry_run:
            if idx <= 5 or idx == total:
                print(f"    [DRY-RUN] Would convert: {src_path.name} -> {dst_path.name}")
            elif idx == 6:
                print(f"    [DRY-RUN] ... ({total - 5} more files) ...")
            success += 1
            continue

        try:
            ela_img = convert_to_ela_image(src_path, quality=quality)
            ela_img.save(dst_path, format="PNG")
            success += 1

            if idx % 500 == 0 or idx == total:
                print(f"    [{cls}] {idx}/{total}  converted")

        except Exception as exc:
            errors += 1
            print(f"    [{cls}] ERROR on {src_path.name}: {exc}")

    return success, errors


# ── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepForensics: Convert raw images to ELA heatmaps on disk."
    )
    parser.add_argument(
        "--quality", type=int, default=90,
        help="JPEG re-compression quality for ELA (default: 90).",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR,
        help=f"Path to raw dataset root (default: {RAW_DIR}).",
    )
    parser.add_argument(
        "--ela-dir", type=Path, default=ELA_DIR,
        help=f"Path to ELA output root (default: {ELA_DIR}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files without actually converting.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  DeepForensics - ELA Dataset Builder")
    print("=" * 60)
    print(f"  Raw directory : {args.raw_dir.resolve()}")
    print(f"  ELA directory : {args.ela_dir.resolve()}")
    print(f"  JPEG quality  : {args.quality}")
    print(f"  Dry run       : {args.dry_run}")
    print("-" * 60)

    # Collect files
    class_files = collect_image_paths(args.raw_dir)
    grand_total = sum(len(v) for v in class_files.values())

    if grand_total == 0:
        print("\n  [!] No images found in raw directories.")
        print(f"      Place CASIA v2.0 authentic images in: {args.raw_dir / 'Au'}")
        print(f"      Place CASIA v2.0 tampered  images in: {args.raw_dir / 'Tp'}")
        print("      Then re-run this script.")
        sys.exit(0)

    print(f"\n  Total images found: {grand_total}")
    for cls, files in class_files.items():
        print(f"    {cls}: {len(files)} images")
    print()

    # Process each class
    t0 = time.perf_counter()
    total_success = 0
    total_errors = 0

    for cls, files in class_files.items():
        if not files:
            continue
        print(f"  Processing class '{cls}' ({len(files)} images)...")
        s, e = process_class(cls, files, args.ela_dir, args.quality, args.dry_run)
        total_success += s
        total_errors += e

    elapsed = time.perf_counter() - t0

    # Summary
    print("-" * 60)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Success: {total_success} | Errors: {total_errors}")
    if total_errors:
        print(f"  [WARN] {total_errors} images failed -- check logs above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
