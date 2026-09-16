import rasterio
import numpy as np
from backend.config import MAX_PIXELS, MIN_DIMENSION, MIN_BANDS

class ValidationError(Exception):
    pass

def validate_upload(file_path: str):
    try:
        src = rasterio.open(file_path)
    except Exception as e:
        raise ValidationError(f"Not a valid GeoTIFF: {e}")

    with src:
        if src.crs is None:
            raise ValidationError("No CRS defined")
        if src.count < MIN_BANDS:
            raise ValidationError(f"Need at least {MIN_BANDS} bands, got {src.count}")
        if src.width < MIN_DIMENSION or src.height < MIN_DIMENSION:
            raise ValidationError(f"Minimum dimension is {MIN_DIMENSION}px")
        if src.width * src.height > MAX_PIXELS:
            raise ValidationError("Image too large — exceeds max pixel count")
        if src.dtypes[0] not in ("uint8", "uint16", "float32"):
            raise ValidationError(f"Unsupported dtype: {src.dtypes[0]}")
    return True

def validate_output_geotiff(file_path: str, expected_aoi_bounds=None):
    """Run against every generated sr.tif/output.tif before trusting it — catches wrong-tile/misalignment bugs."""
    with rasterio.open(file_path) as src:
        if src.crs is None:
            raise ValidationError("Output has no CRS")
        if src.transform.is_identity:
            raise ValidationError("Output has identity/null transform")
        arr = src.read()
        if not np.isfinite(arr.astype("float64")).all():
            raise ValidationError("Output contains NaN/Inf")
        if expected_aoi_bounds:
            b = src.bounds
            if not (expected_aoi_bounds.left <= b.left and b.right <= expected_aoi_bounds.right and
                    expected_aoi_bounds.bottom <= b.bottom and b.top <= expected_aoi_bounds.top):
                raise ValidationError("Output bounds fall outside expected AOI — possible wrong-tile bug")
    return True
