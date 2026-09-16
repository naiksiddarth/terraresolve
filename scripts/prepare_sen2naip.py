"""Reorganizes an extracted SEN2NAIP cross-sensor folder into the
mirrored lr_dir/hr_dir layout `paired_real` expects (same filename in
both directories).

Verified real layout of cross-sensor.zip once unzipped:
    <source>/ROI_0000/lr.tif   (Sentinel-2, 4-band RGBNIR, ~121x121)
    <source>/ROI_0000/hr.tif   (NAIP-derived, 4-band RGBNIR, ~484x484)
    <source>/ROI_0001/...
(exactly 4x scale, matching this project's default `scale: 4`.)

This script hardlinks files when possible so you don't duplicate
several GB of imagery on disk; it falls back to a plain copy if
hardlinking isn't available (e.g. destination on a different drive).

Usage:
    python scripts/prepare_sen2naip.py --source "C:/path/to/extracted/cross-sensor" --dest data/sen2naip
"""
import argparse
import os
import shutil
from pathlib import Path


def _link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink: zero extra disk space, same-volume only
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                         help="extracted cross-sensor folder (contains ROI_0000/, ROI_0001/, ...)")
    parser.add_argument("--dest", default="data/sen2naip")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    lr_dir, hr_dir = dest / "lr", dest / "hr"

    if not source.exists():
        raise FileNotFoundError(f"--source path does not exist: {source}")

    roi_dirs = sorted(p for p in source.iterdir() if p.is_dir() and p.name.startswith("ROI_"))
    if not roi_dirs:
        raise FileNotFoundError(
            f"No ROI_* subfolders found directly under {source}. Point --source at the "
            f"extracted 'cross-sensor' folder itself (the one containing ROI_0000/, ROI_0001/, ...), "
            f"not the zip file and not a parent folder above it."
        )

    total = len(roi_dirs)
    print(f"Found {total} ROI folders under {source}. Linking into {dest} ...")
    copied, skipped = 0, 0
    for i, roi in enumerate(roi_dirs, start=1):
        lr_src, hr_src = roi / "lr.tif", roi / "hr.tif"
        if not lr_src.exists() or not hr_src.exists():
            print(f"[skip] {roi.name}: missing lr.tif or hr.tif")
            skipped += 1
            continue
        _link_or_copy(lr_src, lr_dir / f"{roi.name}.tif")
        _link_or_copy(hr_src, hr_dir / f"{roi.name}.tif")
        copied += 1
        if i % 100 == 0 or i == total:
            print(f"  ... {i}/{total} processed ({copied} linked, {skipped} skipped)", flush=True)

    print(f"\nDone: {copied} pairs ready in {lr_dir} / {hr_dir} ({skipped} skipped).")
    print("Now train against them with:")
    print("  python scripts/train.py --config configs/sen2naip.yaml")


if __name__ == "__main__":
    main()
