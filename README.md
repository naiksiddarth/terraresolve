# TerraResolve -- concept skeleton

Modular deep-learning super-resolution pipeline for SIH 2026 PS 26142
(medium-resolution -> <4m multispectral satellite super-resolution).
This is a working *skeleton*: every stage is a small, real, runnable
implementation, wired together so you can swap any one piece without
touching the rest.

## Why it's built this way

Everything -- model, loss recipe, dataset source, evaluation metric,
downstream task -- is resolved **by name** from `configs/*.yaml`
through the registries in `terraresolve/registry.py`. Nothing imports
a concrete class directly outside of an `__init__.py`. That means:

- Swap the model: change one line in the config (`model.name: edsr`
  -> `swinir`). Add a new architecture by writing a class in
  `terraresolve/models/` and decorating it with
  `@MODEL_REGISTRY.register("your_name")`.
- Swap or reweight the loss: edit `loss.terms` in the config (any
  combination of `l1`, `l2`, `ssim`, `sam`, each with its own
  weight). Add a new loss term the same way as a model.
- Swap the dataset: `data.type: synthetic_degrade` (only needs HR
  imagery, degrades it on the fly -- today's fallback while you don't
  have real paired data) or `paired_real` (two mirrored LR/HR
  directories, e.g. an unpacked SEN2NAIP export).
- Swap evaluation metrics: list any of `psnr`, `ssim`, `cc`, `ergas`,
  `sam` in `eval.metrics`.
- Swap the downstream "does this actually help" task: `ndwi_water` or
  `ndvi_vegetation` today; add building/road extraction the same way.

None of these swaps require touching `terraresolve/engine/trainer.py`
or `terraresolve/engine/inference.py` -- the training loop and the
full-scene inference pipeline are written against the registries, not
against any specific model/loss/dataset.

## Layout

```
configs/            default.yaml (EDSR) and swinir.yaml (transformer) -- copy/edit for more
terraresolve/
  registry.py        the plug-and-play mechanism itself
  data/              dataset plugins (paired_real, synthetic_degrade)
  models/            edsr.py (CNN baseline), swinir_lite.py (windowed-attention transformer)
  losses/            pixel.py (l1/l2), structural.py (ssim), spectral.py (sam), composite.py (combiner)
  metrics/           psnr, ssim, cc, ergas, sam -- spatial fidelity + spectral fidelity, split on purpose
  downstream/        ndwi_water / ndvi_vegetation masks + IoU-style mask_agreement
  engine/            trainer.py (generic loop), inference.py (tiled inference + GeoTIFF export)
  ingest/            copernicus.py -- direct OAuth2 ingestion from the Copernicus Data Space
                     Ecosystem (search + download a live Sentinel-2 L2A scene for any AOI/date)
scripts/             train.py, infer.py, evaluate.py, fetch_copernicus.py -- CLI entry points
webapp/              standalone map-style viewer (FastAPI + Leaflet). Reads real LR/SR/HR
                     tiles through the same registries, zero coupling to the training code.
                     main.py (backend + tile endpoints), static/index.html (frontend)
tests/smoke_test.py  no-data sanity check: builds every model, runs a forward + backward pass
data/hr_reference/   drop HR imagery here for synthetic_degrade training
checkpoints/         model weights land here during training
```

## Quickstart

```bash
pip install -r requirements.txt

# 0. sanity-check the wiring (no data needed)
python tests/smoke_test.py

# 1. put some HR reference imagery (GeoTIFF, or PNG/JPG for a quick
#    test) into data/hr_reference/, then train the EDSR baseline
python scripts/train.py --config configs/default.yaml

# 2. train the SwinIR-lite alternative instead -- same data, same loss
python scripts/train.py --config configs/swinir.yaml

# 2b. once you've downloaded SEN2NAIP's cross-sensor.zip and unzipped it,
#     reorganise it into the lr_dir/hr_dir layout paired_real expects:
python scripts/prepare_sen2naip.py --source path/to/extracted/cross-sensor --dest data/sen2naip
python scripts/train.py --config configs/sen2naip.yaml

# 3. or override individual settings from the CLI without a new file
python scripts/train.py --config configs/default.yaml \
    --set loss.terms.sam=0.3 train.epochs=5

# 4. run a trained checkpoint over a full georeferenced scene
python scripts/infer.py --config configs/default.yaml \
    --checkpoint checkpoints/edsr_epoch50.pt \
    --input scene.tif --output scene_sr.tif

# 5. score the result against a real HR reference
python scripts/evaluate.py --pred scene_sr.tif --ref hr_reference.tif
```

## What's genuinely implemented vs. deliberately simplified

- **EDSR**: a real, working residual-CNN SR network (generalised to 4
  bands instead of RGB).
- **SwinIR-lite**: a real, working local-window self-attention SR
  network -- but a simplified stand-in for the full SwinIR paper (no
  shifted windows or relative position bias yet). Good enough to
  prove out the CNN-vs-transformer comparison; swap in the official
  SwinIR implementation later for the final benchmark numbers.
- **Degradation model** (`synthetic_degrade`): plain Gaussian blur +
  strided downsampling. This is the well-known weak point of
  synthetic LR-HR training (real Sentinel-2 degradation isn't just
  bicubic decimation) -- swap `_degrade()` in
  `terraresolve/data/datasets.py` for something closer to the real
  sensor PSF/noise when you're ready, or switch `data.type` to
  `paired_real` once you have actual paired imagery.
- **Metrics**: PSNR/SSIM/CC (spatial fidelity) and ERGAS/SAM (spectral
  fidelity) are full, correct implementations, not stubs.
- **Georeferencing**: `infer.py` reads a real GeoTIFF's CRS/transform,
  rescales it by the SR factor, and writes a proper GeoTIFF -- this
  part is not a placeholder.

## Next concrete steps

1. Drop a handful of HR images into `data/hr_reference/` and run
   `scripts/train.py` end-to-end once, even before real Sentinel-2
   pairs are ready, to confirm the loop runs on your machine/GPU.
2. Once SEN2NAIP (or a subset of it) is unpacked locally, point
   `data.type: paired_real` at its LR/HR directories.
3. Wire a real Sentinel-2 scene into `scripts/infer.py` to produce a
   first GeoTIFF end-to-end, then score it with `evaluate.py`.
4. Add a building/road-extraction downstream plugin alongside
   `ndwi_water` once you pick your second differentiator task.

## Live Copernicus ingestion (proving the pipeline touches the mandated source)

`SEN2NAIP` / `SEN2VENuS` give you ready-made LR-HR training pairs, but they're
research datasets, not the Copernicus Data Space platform the problem
statement names directly. `terraresolve/ingest/copernicus.py` +
`scripts/fetch_copernicus.py` close that gap: given a bounding box and a
date range, they authenticate to the real Copernicus Data Space Ecosystem,
search its Sentinel-2 L2A catalog, download the matching scene, and stack
its B02/B03/B04/B08 bands into the same 4-band GeoTIFF layout the rest of
the project expects -- independent of any training-pair dataset.

Requires a free account at https://dataspace.copernicus.eu/ (Register),
then set credentials as environment variables (never in a config file):

```powershell
setx CDSE_USERNAME "you@example.com"
setx CDSE_PASSWORD "your-password"
```

```bash
pip install requests   # if not already installed via requirements.txt

python scripts/fetch_copernicus.py \
    --bbox 77.55 12.90 77.65 13.00 \
    --start 2026-07-01 --end 2026-09-01 \
    --max-cloud 20 \
    --output data/copernicus_raw/bangalore.tif
```

The output GeoTIFF can be fed straight into `scripts/infer.py` to
super-resolve a real, freshly-downloaded scene end to end.

## Map-style viewer

`webapp/` is a standalone FastAPI + Leaflet app that serves the LR
(original Sentinel-2), SR (model output), and HR (reference) imagery as
real zoomable XYZ map tiles -- the same tiling scheme Google Maps uses --
with a draggable side-by-side comparison slider, so you can pan and zoom
into a real scene and see exactly how much detail the model recovered.

It only *reads* from `data/` and `checkpoints/` through the same
registries as everything else, so it never has to know which model,
dataset, or checkpoint is currently in use -- swap the checkpoint and the
viewer follows.

```bash
pip install fastapi uvicorn mercantile rio-tiler

python webapp/main.py
# open http://127.0.0.1:8000
```

By default it loads `checkpoints/edsr_epoch50.pt` against
`configs/sen2naip.yaml` and lists every real SEN2NAIP LR/HR pair
available to jump to on the map. To point it at a different checkpoint
without restarting:

```bash
curl -X POST "http://127.0.0.1:8000/api/model?config=configs/swinir.yaml&checkpoint=checkpoints/swinir_epoch50.pt"
```
