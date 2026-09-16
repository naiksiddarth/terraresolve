import os
import rasterio
from rasterio.mask import mask
import pyproj
def build_utm_polygon(center_lon, center_lat, src_crs):
    transformer = pyproj.Transformer.from_crs("epsg:4326", src_crs, always_xy=True)
    cx, cy = transformer.transform(center_lon, center_lat)
    
    # Create a clean +/- 1000m axis-aligned bounding box in UTM
    min_x, max_x = cx - 1000, cx + 1000
    min_y, max_y = cy - 1000, cy + 1000
    
    return {
        "type": "Polygon",
        "coordinates": [[
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y)
        ]]
    }

scenes = {
    "yelahanka-lake": (77.595, 13.101),       # Center of lake
    "yelahanka-newtown": (77.584, 13.097),    # Center of radial street grid
    "yelahanka-afs-buffer": (77.590, 13.130), # North side near AFS (Air Force Station)
    "yelahanka-mixed": (77.610, 13.080)       # East side, mixed agriculture/urban (Jakkur Lake)
}

src_path = "data/copernicus_raw/yelahanka.tif"

with rasterio.open(src_path) as src:
    crs = src.crs
    for scene_id, center in scenes.items():
        print(f"\nProcessing {scene_id}...")
        print(f"Target Lon/Lat center: {center}")
        
        geom = build_utm_polygon(*center, crs)
        
        out_image, out_transform = mask(src, [geom], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        out_dir = f"data/raw/{scene_id}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/input.tif"
        
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)
            
        print(f"Saved: {out_path}")
        print(f"CRS: {crs}")
        print(f"Dimensions: {out_image.shape[1]}x{out_image.shape[2]}")
        print(f"UTM Bounds: {dest.bounds}")
