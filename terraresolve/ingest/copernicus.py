"""
Direct ingestion from the Copernicus Data Space Ecosystem (CDSE) --
the official distribution platform for Sentinel-2 imagery named in the
problem statement (dataspace.copernicus.eu).

This module is deliberately independent of the LR-HR training-pair
datasets (SEN2NAIP, SEN2VENuS) used elsewhere in the project. Those
give you ready-made training pairs; this gives you the ability to
pull a *live, arbitrary* Sentinel-2 L2A scene for any place and date,
proving the pipeline can ingest straight from the mandated source
rather than only from pre-packaged research datasets.

Auth: CDSE uses OAuth2 (Keycloak) client-credentials-style token
exchange against your own free account -- there is no way around
creating one yourself at https://dataspace.copernicus.eu/ (Register).
Once you have an account:

    1. Log in at https://dataspace.copernicus.eu/
    2. Just use your normal username/password below (no separate API
       key needed) -- CDSE's public OAuth client "cdse-public" accepts
       password-grant login for the same credentials you log in with.
    3. Set them as environment variables so they never end up in
       config files or git history:

           setx CDSE_USERNAME "you@example.com"
           setx CDSE_PASSWORD "your-password"

       (setx persists for future terminals; use `set` for just the
       current session, or a `.env` file loaded by python-dotenv.)

Usage (see scripts/fetch_copernicus.py for the CLI wrapper):

    from terraresolve.ingest.copernicus import (
        get_access_token, search_products, download_product,
    )

    token = get_access_token()
    products = search_products(
        bbox=(77.55, 12.90, 77.65, 13.00),  # Bangalore, India
        start_date="2026-01-01",
        end_date="2026-03-01",
        max_cloud_cover=20,
    )
    download_product(products[0], token, dest_dir="data/copernicus_raw")
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({id})/$value"


def get_access_token(username: Optional[str] = None, password: Optional[str] = None) -> str:
    """Exchange CDSE username/password for a short-lived OAuth2 access token.

    Reads CDSE_USERNAME / CDSE_PASSWORD from the environment if not
    passed explicitly. Tokens expire in ~10 minutes, so call this
    again (it's cheap) rather than caching it for long-running jobs.
    """
    username = username or os.environ.get("CDSE_USERNAME")
    password = password or os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Missing Copernicus Data Space credentials. Set CDSE_USERNAME "
            "and CDSE_PASSWORD as environment variables (see the module "
            "docstring in terraresolve/ingest/copernicus.py), or pass "
            "username=/password= explicitly. Register a free account at "
            "https://dataspace.copernicus.eu/ if you don't have one."
        )

    resp = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"CDSE authentication failed ({resp.status_code}): {resp.text[:500]}"
        )
    return resp.json()["access_token"]


def search_products(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    collection: str = "SENTINEL-2",
    product_type: str = "S2MSI2A",
    max_cloud_cover: float = 30.0,
    top: int = 10,
) -> List[Dict[str, Any]]:
    """Search the CDSE OData catalog for Sentinel-2 products.

    bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84 degrees.
    product_type: "S2MSI2A" = Level-2A surface reflectance (the
        atmospherically-corrected product this project standardizes
        on). Use "S2MSI1C" for Level-1C top-of-atmosphere if you
        specifically need that instead.

    Returns a list of raw OData product dicts (name, Id, ContentLength,
    cloud cover, etc.), cheapest/least-cloudy first.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    polygon = (
        f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    )
    filter_clauses = [
        f"Collection/Name eq '{collection}'",
        f"contains(Name,'MSIL2A')",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')",
        f"ContentDate/Start gt {start_date}T00:00:00.000Z",
        f"ContentDate/Start lt {end_date}T00:00:00.000Z",
        (
            "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq "
            f"'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {max_cloud_cover})"
        ),
    ]
    params = {
        "$filter": " and ".join(filter_clauses),
        "$orderby": "ContentDate/Start desc",
        "$top": str(top),
    }
    resp = requests.get(CATALOG_URL, params=params, timeout=60)
    resp.raise_for_status()
    results = resp.json().get("value", [])
    if not results:
        raise RuntimeError(
            "No Sentinel-2 products matched that bbox/date range/cloud "
            "threshold. Try widening the date range or raising "
            "max_cloud_cover."
        )
    return results


def download_product(
    product: Dict[str, Any],
    token: str,
    dest_dir: str = "data/copernicus_raw",
    extract: bool = True,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download one product (as returned by search_products) and, by
    default, extract the .SAFE product from its zip.

    Returns the path to the extracted .SAFE directory (or the raw
    .zip path if extract=False).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    product_id = product["Id"]
    name = product["Name"]
    zip_path = dest_dir / f"{name}.zip"

    url = DOWNLOAD_URL.format(id=product_id)
    headers = {"Authorization": f"Bearer {token}"}

    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        if resp.status_code == 401:
            raise RuntimeError(
                "CDSE download rejected the access token (401) -- tokens "
                "expire after ~10 minutes, call get_access_token() again "
                "right before downloading."
            )
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r  downloading {name}: {pct:5.1f}%", end="", flush=True)
        print()

    if not extract:
        return zip_path

    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    safe_dirs = list(dest_dir.glob("*.SAFE"))
    zip_path.unlink()
    if not safe_dirs:
        raise RuntimeError(f"Extraction succeeded but no .SAFE folder found under {dest_dir}")
    return safe_dirs[-1]


def find_band_files(safe_dir: Path, resolution: str = "10m") -> Dict[str, Path]:
    """Locate the B02/B03/B04/B08 (10m) jp2 band files inside a Level-2A
    .SAFE product, so they can be stacked into a single multiband
    GeoTIFF (see scripts/fetch_copernicus.py for the stacking step).
    """
    safe_dir = Path(safe_dir)
    granule_dirs = list(safe_dir.glob("GRANULE/*/IMG_DATA"))
    if not granule_dirs:
        raise RuntimeError(f"No GRANULE/*/IMG_DATA folder found under {safe_dir}")
    img_data = granule_dirs[0]
    res_dir = img_data / f"R{resolution}"
    search_dir = res_dir if res_dir.exists() else img_data

    bands = {}
    for band in ("B02", "B03", "B04", "B08"):
        matches = list(search_dir.rglob(f"*_{band}_*.jp2")) or list(search_dir.rglob(f"*{band}.jp2"))
        if matches:
            bands[band] = matches[0]
    missing = {"B02", "B03", "B04", "B08"} - bands.keys()
    if missing:
        raise RuntimeError(f"Could not locate band(s) {missing} under {search_dir}")
    return bands
