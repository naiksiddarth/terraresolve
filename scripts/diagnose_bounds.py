import rasterio
from rasterio.warp import transform_bounds
import json

scenes = ['yelahanka-lake', 'yelahanka-newtown', 'yelahanka-afs-buffer', 'yelahanka-mixed']

for s in scenes:
    print(f'\n=== {s} ===')
    with open(f'demo/scenes/{s}/metadata.json') as f:
        meta = json.load(f)
    b = meta['bounds']
    print(f"  metadata.json:  W={b['west']:.5f}, S={b['south']:.5f}, E={b['east']:.5f}, N={b['north']:.5f}")

    with rasterio.open(f'demo/scenes/{s}/input.tif') as src:
        inp_wgs = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
        print(f"  input.tif real: W={inp_wgs[0]:.5f}, S={inp_wgs[1]:.5f}, E={inp_wgs[2]:.5f}, N={inp_wgs[3]:.5f}")
        print(f"  input.tif shape: {src.height}x{src.width}")

    with rasterio.open(f'demo/scenes/{s}/sr.tif') as src:
        sr_wgs = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
        print(f"  sr.tif real:    W={sr_wgs[0]:.5f}, S={sr_wgs[1]:.5f}, E={sr_wgs[2]:.5f}, N={sr_wgs[3]:.5f}")
        print(f"  sr.tif shape: {src.height}x{src.width}")

        west_ok = abs(sr_wgs[0] - b['west']) < 0.0001
        south_ok = abs(sr_wgs[1] - b['south']) < 0.0001
        east_ok = abs(sr_wgs[2] - b['east']) < 0.0001
        north_ok = abs(sr_wgs[3] - b['north']) < 0.0001
        print(f"  metadata matches sr.tif: W={west_ok}, S={south_ok}, E={east_ok}, N={north_ok}")
