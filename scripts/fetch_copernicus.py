"""
Pull one real Sentinel-2 L2A scene directly from the Copernicus Data
Space Ecosystem for a given area of interest, and stack its B02/B03/
B04/B08 (10m) bands into a single georeferenced GeoTIFF -- in the
exact 4-band layout the rest of TerraResolve (EDSR, SwinIR, the
normalize="sentinel2_l2a" dataset mode) already expects.

This is intentionally separate from the SEN2NAIP / SEN2VENuS training
pipelines: those give you ready-made LR-HR pairs, this proves the
pipeline can ingest a live scene straight from the platform named in
the problem statement, for any place and date you choose.

Requires a free Copernicus Data Space account:
    https://dataspace.copernicus.eu/  (Register)
and CDSE_USERNAME / CDSE_PASSWORD set as environment variables
(see terraresolve/ingest/copernicus.py docstring for exact commands).

Examples
--------
Bangalore, last 2 months, up to 20% cloud cover:

    python scripts/fetch_copernicus.py \\
        --bbox 77.55 12.90 77.65 13.00 \\
        --start 2026-07-01 --end 2026-09-01 \\
        --max-cloud 20 \\
        --output data/copernicus_raw/bangalore.tif

Any other AOI: swap --bbox for (min_lon min_lat max_lon max_lat).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terraresolve.ingest.copernicus import (
    download_product,
    find_band_files,
    get_access_token,
    search_products,
)

try:
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rasterio and numpy are required: pip install rasterio numpy"
    ) from exc


def stack_bands_to_geotiff(bands: dict, output_path: str) -> None:
    """Read B02, B03, B04, B08 jp2 files and write them as one 4-band
    uint16 GeoTIFF in that order (matches the SEN2NAIP/EDSR channel
    convention used elsewhere: B02=Blue, B03=Green, B04=Red, B08=NIR).
    """
    order = ["B02", "B03", "B04", "B08"]
    with rasterio.open(bands[order[0]]) as ref:
        profile = ref.profile.copy()
        ref_shape = (ref.height, ref.width)

    arrays = []
    for band in order:
        with rasterio.open(bands[band]) as src:
            if (src.height, src.width) != ref_shape:
                data = src.read(
                    1,
                    out_shape=ref_shape,
                    resampling=Resampling.bilinear,
                )
            else:
                data = src.read(1)
            arrays.append(data)

    stacked = np.stack(arrays, axis=0)
    profile.update(driver="GTiff", count=4, dtype=stacked.dtype, compress="deflate")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stacked)
        dst.descriptions = ("B02_Blue", "B03_Green", "B04_Red", "B08_NIR")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--max-cloud", type=float, default=30.0)
    parser.add_argument("--output", required=True, help="Path to write the stacked 4-band GeoTIFF")
    parser.add_argument("--raw-dir", default="data/copernicus_raw", help="Where to download/extract the .SAFE product")
    parser.add_argument("--keep-safe", action="store_true", help="Don't delete the extracted .SAFE folder after stacking")
    args = parser.parse_args()

    print("Authenticating with Copernicus Data Space...")
    token = get_access_token()

    print(f"Searching Sentinel-2 L2A over bbox={tuple(args.bbox)}, "
          f"{args.start} to {args.end}, cloud<={args.max_cloud}%...")
    products = search_products(
        bbox=tuple(args.bbox),
        start_date=args.start,
        end_date=args.end,
        max_cloud_cover=args.max_cloud,
    )
    best = products[0]
    print(f"Found {len(products)} match(es). Using: {best['Name']}")

    print("Re-authenticating right before download (tokens expire fast)...")
    token = get_access_token()
    safe_dir = download_product(best, token, dest_dir=args.raw_dir)
    print(f"Extracted to {safe_dir}")

    print("Locating B02/B03/B04/B08 (10m) bands...")
    bands = find_band_files(safe_dir, resolution="10m")

    print(f"Stacking bands into {args.output} ...")
    stack_bands_to_geotiff(bands, args.output)
    print(f"Done. Wrote {args.output}")

    if not args.keep_safe:
        import shutil
        shutil.rmtree(safe_dir, ignore_errors=True)
        print(f"Removed raw .SAFE folder ({safe_dir}) to save disk space. "
              f"Pass --keep-safe to retain all original bands/metadata.")


if __name__ == "__main__":
    main()
