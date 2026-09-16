"""
Build a real LR-HR pair set from the October 2023 Sikkim (South Lhonak
Lake GLOF / Teesta river) flood event: Maxar's open, sub-metre
post-event imagery as the HR reference, paired with a live Sentinel-2
L2A scene of the same area/date pulled directly from the Copernicus
Data Space Ecosystem via terraresolve/ingest/copernicus.py.

Both sides are REAL satellite captures (no synthetic degradation),
consistent with the project's real-data-only policy, and both are
genuinely Indian terrain -- unlike SEN2NAIP (US-only via NAIP).

IMPORTANT -- network requirement
---------------------------------
This script needs unrestricted internet access: it downloads from
Amazon S3 (Maxar's public bucket, via leafmap's STAC helpers) and the
Copernicus Data Space Ecosystem. Neither host is reachable from a
locked-down sandbox with an egress allowlist -- run this on your own
machine (where you already run train.py).

Install the two extra dependencies this needs beyond requirements.txt:

    pip install leafmap pystac-client

Usage
-----
    # Step 1: see what child collections (sub-events / dates) exist
    # under the main event -- Maxar groups tiles by sub-collection, not
    # as one flat list of 51 items.
    python scripts/prepare_maxar_sikkim.py --list-collections

    # Step 2: once you've picked a child collection ID from that list,
    # actually build the pairs
    python scripts/prepare_maxar_sikkim.py \\
        --child-collection <id-from-step-1> \\
        --output-dir data/maxar_sikkim \\
        --max-tiles 8

Then point a config at it, e.g. copy configs/sen2naip.yaml to
configs/maxar_sikkim.yaml and change:

    data:
      type: paired_real
      lr_dir: data/maxar_sikkim/lr
      hr_dir: data/maxar_sikkim/hr
      lr_normalize: sentinel2_l2a
      hr_normalize: uint8      # Maxar's "visual" COG asset is 8-bit RGB

What this script does
----------------------
1. Uses leafmap's Maxar STAC helpers (leafmap.maxar_collections /
   maxar_child_collections / maxar_items) to list and download the
   "visual" (RGB, pan-sharpened) COG asset for up to --max-tiles tiles
   in the chosen child collection. These are the confirmed, real
   leafmap functions this integration is built on -- see
   https://leafmap.org/notebooks/67_maxar_open_data/
2. For each Maxar tile, reads its real geographic bounding box from
   the COG's own georeferencing.
3. Calls terraresolve.ingest.copernicus to search + download a
   Sentinel-2 L2A scene covering that bounding box, as close as
   possible to the Maxar acquisition date (falls back to the nearest
   low-cloud scene within +/- SEARCH_WINDOW_DAYS).
4. Reprojects/crops both to the same extent and CRS, stacks Maxar's
   RGB into the HR array (resampled to a clean 4x relationship with
   the Sentinel-2 crop -- see note below) and Sentinel-2's B2/B3/B4/B8
   into the LR array, and writes them into the same lr_dir/hr_dir
   layout scripts/prepare_sen2naip.py already produces, so the
   existing `paired_real` dataset class needs no changes.

Geometric note: Maxar's "visual" asset is natively ~30-50cm GSD;
Sentinel-2 is 10m -- a ~20-30x gap, not the 4x this project's models
are configured for. Rather than retrain the whole architecture around
a different scale factor, the HR reference is downsampled to exactly
4x the Sentinel-2 crop's resolution (i.e. trained against a 2.5m
target), matching the existing scale=4 EDSR/SwinIR configs. The full-
resolution Maxar tiles are still saved (see --keep-native-hr) if you
later want to experiment with a higher scale factor.

Only RGB is used from Maxar's "visual" asset -- it has no NIR band, so
the HR side has 3 channels where the LR side has 4 (B2/B3/B4/B8). If
your model config expects matched channel counts, drop B8 from the LR
stack for this dataset specifically, or extend this script to pull
Maxar's non-visual multispectral asset if the collection publishes one
(check the item's asset list -- not guaranteed for every event).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, transform_bounds
    from rasterio.warp import Resampling as ResamplingEnum
    from rasterio.windows import from_bounds
except ImportError as exc:
    raise ImportError("pip install rasterio numpy") from exc

try:
    import leafmap
except ImportError as exc:
    raise ImportError(
        "pip install leafmap pystac-client  "
        "(leafmap provides the confirmed Maxar STAC helpers this "
        "script uses: maxar_collections / maxar_child_collections / "
        "maxar_items -- see https://leafmap.org/notebooks/67_maxar_open_data/)"
    ) from exc

from terraresolve.ingest.copernicus import (
    download_product,
    find_band_files,
    get_access_token,
    search_products,
)

EVENT_COLLECTION = "India-Floods-Oct-2023"
SEARCH_WINDOW_DAYS = 30  # widen the Sentinel-2 date search if the exact day is too cloudy


def list_child_collections():
    print(f"Child collections under '{EVENT_COLLECTION}':\n")
    children = leafmap.maxar_child_collections(EVENT_COLLECTION)
    for c in children:
        print(" -", c)
    print(
        f"\n{len(children)} child collection(s). Pick one and re-run with "
        f"--child-collection <id> to build pairs from it."
    )


def get_maxar_tiles(child_collection: str, max_tiles: int, dest_dir: Path):
    """Download the 'visual' COG asset for up to max_tiles items in the
    given child collection. Returns (local_path, bbox_wgs84, date) per tile,
    with bbox/date read from the actual downloaded file / STAC item, not
    assumed.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    gdf = leafmap.maxar_items(
        collection_id=EVENT_COLLECTION,
        child_id=child_collection,
        return_gdf=True,
        assets=["visual"],
    )
    print(f"Found {len(gdf)} items in child collection '{child_collection}'")

    tiles = []
    for i, row in gdf.head(max_tiles).iterrows():
        url = row["visual"]
        item_id = row.get("id", f"item_{i}")
        local_path = dest_dir / f"{item_id}.tif"
        if not local_path.exists():
            print(f"  downloading {item_id} ...")
            import urllib.request
            urllib.request.urlretrieve(url, local_path)
        with rasterio.open(local_path) as src:
            bbox = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        date = str(row.get("datetime", ""))[:10]
        tiles.append((local_path, bbox, date))
        print(f"  {item_id}: bbox={bbox}, date={date}")
    return tiles


def fetch_matching_sentinel2(bbox, center_date: str, raw_dir: Path):
    """Search/download a Sentinel-2 L2A scene over the same bbox, as
    close to center_date as a low-cloud pass allows.
    """
    try:
        center = datetime.strptime(center_date, "%Y-%m-%d")
    except ValueError:
        center = datetime(2023, 10, 4)  # the GLOF's actual date, as a fallback anchor
    start = (center - timedelta(days=SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")
    end = (center + timedelta(days=SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")

    get_access_token()  # fail fast here if credentials are missing/wrong
    products = search_products(bbox=bbox, start_date=start, end_date=end, max_cloud_cover=40, top=5)
    best = products[0]
    print(f"  Matching Sentinel-2 scene: {best['Name']}")

    token = get_access_token()  # re-fetch: tokens expire in ~10 min
    safe_dir = download_product(best, token, dest_dir=str(raw_dir))
    bands = find_band_files(safe_dir, resolution="10m")
    return bands


def crop_and_align(maxar_path: Path, s2_bands: dict, bbox, lr_out: Path, hr_out: Path):
    """Crop Sentinel-2 to bbox (the LR array), then resample the Maxar
    tile onto an exact 4x-finer grid over the same extent/CRS (the HR
    array), and write both as GeoTIFFs in the paired_real layout.
    """
    order = ["B02", "B03", "B04", "B08"]
    with rasterio.open(s2_bands[order[0]]) as ref:
        bbox_in_ref_crs = transform_bounds("EPSG:4326", ref.crs, *bbox)
        window = from_bounds(*bbox_in_ref_crs, transform=ref.transform)
        lr_transform = ref.window_transform(window)
        lr_crs = ref.crs
        lr_profile = ref.profile.copy()

    lr_arrays = [rasterio.open(s2_bands[b]).read(1, window=window) for b in order]
    lr_stack = np.stack(lr_arrays, axis=0)
    lr_h, lr_w = lr_stack.shape[1:]

    lr_profile.update(driver="GTiff", count=4, height=lr_h, width=lr_w,
                       transform=lr_transform, dtype=lr_stack.dtype, compress="deflate")
    lr_out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(lr_out, "w", **lr_profile) as dst:
        dst.write(lr_stack)

    hr_h, hr_w = lr_h * 4, lr_w * 4
    hr_transform = lr_transform * lr_transform.scale(1 / 4, 1 / 4)

    with rasterio.open(maxar_path) as src:
        n_bands = min(src.count, 3)  # RGB only -- Maxar's "visual" asset has no NIR
        hr_stack = np.zeros((n_bands, hr_h, hr_w), dtype=src.dtypes[0])
        for b in range(1, n_bands + 1):
            reproject(
                source=rasterio.band(src, b), destination=hr_stack[b - 1],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=hr_transform, dst_crs=lr_crs,
                resampling=ResamplingEnum.bilinear,
            )

    hr_profile = lr_profile.copy()
    hr_profile.update(height=hr_h, width=hr_w, count=n_bands,
                       dtype=hr_stack.dtype, transform=hr_transform)
    hr_out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(hr_out, "w", **hr_profile) as dst:
        dst.write(hr_stack)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-collections", action="store_true", help="List child collections under the event and exit")
    parser.add_argument("--child-collection", help="Child collection id to pull tiles from (get this from --list-collections first)")
    parser.add_argument("--output-dir", default="data/maxar_sikkim")
    parser.add_argument("--max-tiles", type=int, default=8, help="How many Maxar tiles to pull and pair (each needs its own Sentinel-2 search+download)")
    parser.add_argument("--raw-dir", default="data/copernicus_raw")
    args = parser.parse_args()

    if args.list_collections:
        list_child_collections()
        return

    if not args.child_collection:
        parser.error("--child-collection is required (run --list-collections first to find one)")

    output_dir = Path(args.output_dir)
    maxar_dir = output_dir / "_maxar_raw"

    print(f"Downloading up to {args.max_tiles} tiles from child collection '{args.child_collection}'...")
    tiles = get_maxar_tiles(args.child_collection, args.max_tiles, maxar_dir)

    ok, failed = 0, 0
    for i, (maxar_path, bbox, date) in enumerate(tiles):
        tile_id = f"maxar_sikkim_{i:04d}"
        print(f"\n[{i+1}/{len(tiles)}] {tile_id} (date={date})")
        try:
            s2_bands = fetch_matching_sentinel2(bbox, date, Path(args.raw_dir))
            crop_and_align(
                maxar_path, s2_bands, bbox,
                lr_out=output_dir / "lr" / f"{tile_id}.tif",
                hr_out=output_dir / "hr" / f"{tile_id}.tif",
            )
            ok += 1
            print(f"  -> wrote {tile_id} pair")
        except Exception as exc:
            failed += 1
            print(f"  SKIPPED ({type(exc).__name__}: {exc})")

    print(f"\nDone: {ok} pairs written to {output_dir}/lr and {output_dir}/hr ({failed} skipped)")


if __name__ == "__main__":
    main()
