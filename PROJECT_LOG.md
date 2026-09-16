# TerraResolve -- Project Log

A running record of what's actually been built and decided, for whenever
things feel like they've gotten away from you. Read this top to bottom
once and you're caught up.

## 1. What this project actually is

SIH 2026, Problem Statement 26142 (National Technical Research
Organisation): take medium-resolution satellite imagery (Sentinel-2,
10m) and reconstruct a sharper version (target: under 4m) using deep
learning, without inventing detail that isn't really there, and without
distorting the spectral (color/reflectance) information scientists
actually rely on for things like vegetation and water mapping.

The team's project name is TerraResolve. The core idea in one sentence:
train a neural network on many examples of "here's the blurry version,
here's the sharp version of the same place," so it learns the general
pattern well enough to sharpen a brand-new image where no sharp version
exists to check against.

## 2. The codebase -- what each piece does

Everything lives in this folder, organized so any single piece (model,
loss, dataset, metric) can be swapped by editing a config file instead
of touching code. The mechanism behind that is `terraresolve/registry.py`
-- a name-to-class lookup table that every other file asks for what it
needs, instead of hard-importing a specific implementation.

**Models** (`terraresolve/models/`)
- `edsr.py` -- a residual CNN (Enhanced Deep Residual Network). This is
  what's actually been trained so far. Learns local spatial patterns
  (edges, textures) via stacked convolutions.
- `swinir_lite.py` -- a simplified window-attention transformer, the
  alternative architecture for the "we benchmarked CNN vs transformer"
  comparison. Not yet trained on real data.

**Losses** (`terraresolve/losses/`) -- three judges scoring the model's
output against the true HR image during training, combined with
weights in the config:
- `l1` -- raw pixel-value closeness.
- `ssim` -- structural similarity (are edges/shapes right, not just
  colors).
- `sam` -- spectral angle mapper, checks that the *relationship*
  between the four bands (Blue/Green/Red/NIR) at each pixel is
  preserved, independent of brightness. This is what protects things
  like NDVI-relevant band ratios from getting distorted.

**Data pipeline** (`terraresolve/data/datasets.py`)
- `paired_real` -- loads real LR/HR pairs from two mirrored folders.
  This is what's in use now, pointed at the SEN2NAIP data.
- `synthetic_degrade` -- manufactures LR by blurring/downsampling HR
  images, for when no real pairs exist. Used earlier for the very
  first pipeline smoke-test (12 procedurally-generated placeholder
  tiles, before any real data was downloaded).
- A real bug was caught and fixed here: Sentinel-2 reflectance needed
  an explicit normalization mode (`sentinel2_l2a`, divide by 10000)
  instead of a generic guessed heuristic, or the LR/HR brightness
  scales would silently mismatch.

**Metrics** (`terraresolve/metrics/metrics.py`) -- for scoring a
*finished* result (not used during training itself): PSNR, SSIM, CC
(spatial/structural accuracy) and ERGAS, SAM (spectral accuracy, the
pansharpening-literature standards). Not yet run against the trained
model -- this is a next step.

**Downstream tasks** (`terraresolve/downstream/water_index.py`) --
NDWI (water) and NDVI (vegetation) mask functions, meant to answer "did
the SR output actually improve a real mapping task," not just "did the
picture get sharper." Not yet wired into an actual comparison run.

**Engine** (`terraresolve/engine/`)
- `trainer.py` -- the training loop. Model/loss/dataset-agnostic, reads
  everything from the registry via config.
- `inference.py` -- tiles a full-size real scene into overlapping
  patches, runs the model, blends and mosaics the patches back
  together, and writes a proper georeferenced GeoTIFF. Not yet run on
  a real full scene -- only small training-patch previews so far.

**Scripts** (`scripts/`)
- `train.py`, `infer.py`, `evaluate.py` -- the three main entry points.
- `make_placeholder_data.py` -- generated the very first synthetic test
  tiles (numpy+PIL, no download needed) to prove the pipeline ran
  before any real dataset was in hand.
- `prepare_sen2naip.py` -- reorganizes a raw SEN2NAIP download into the
  `lr_dir`/`hr_dir` layout `paired_real` expects.
- `visualize.py` -- generates side-by-side [blurry input | model
  output | true HR] comparison PNGs from any checkpoint, with a
  display-only contrast stretch (real Sentinel-2 reflectance is
  legitimately dim; this doesn't touch the actual training data,
  purely makes it visible to human eyes).

**Configs** (`configs/`) -- `default.yaml` (EDSR + synthetic
placeholder data), `sen2naip.yaml` (EDSR + real SEN2NAIP data, this is
what's actually been used for real training), `swinir.yaml` (transformer,
not yet run against real data).

## 3. The data situation -- what's in hand and what's genuinely missing

**In hand:** SEN2NAIP's "cross-sensor" subset -- 2,851 real paired
tiles of Sentinel-2 (LR, ~121x121, 4 bands) and NAIP aerial imagery
(HR, ~484x484, 4 bands), a clean 4x scale match. Downloaded, unpacked,
and reorganized into `data/sen2naip/lr` and `/hr`.

**The real gap, worth being honest about:** NAIP is a *United States
only* aerial imagery program. Every single one of those 2,851 pairs is
somewhere in the US -- different vegetation, different urban layout,
different field geometry than India. This was flagged early on as a
genuine research gap: as of this project, there is no published dataset
pairing Sentinel-2 with a real Indian high-resolution reference for
super-resolution training. Cartosat (ISRO's own high-res satellite
series) was investigated as the obvious India-specific alternative and
ruled out as a *training* source for concrete reasons: it's
priced/gated for non-government users under India's 2023 space policy,
it's mostly panchromatic (no NIR band, so no full spectral-consistency
check against it), and there's no ready-made paired LR-HR dataset the
way SEN2NAIP provides -- it would need to be manually assembled scene
by scene from the Bhoonidhi archive.

**So the current model has only ever seen U.S. landscapes.** That's a
legitimate limitation to flag honestly in any presentation of this
work, not something to gloss over. 2,851 pairs is also a fairly small
training set by deep-learning standards, even before the geography
question.

**Realistic next moves on data**, roughly in order of effort:
1. Pull SEN2NAIP's larger "synthetic" subset (17,657 more pairs, still
   US-only, but meaningfully more training volume) to get the model
   past its current very-early training level.
2. Revisit Bhoonidhi/Cartosat specifically as a small held-out
   *validation* set (not training data) over an Indian region, purely
   to test generalization -- this was the plan discussed earlier,
   still pending actual data access.
3. Check whether other published Sentinel-2 SR datasets exist with
   broader geographic coverage (e.g. SEN2VENuS, WorldStrat) that might
   include non-US regions -- this hasn't been researched yet and is a
   reasonable next task.
4. Squeeze more out of the current 2,851 pairs with heavier
   augmentation before assuming more raw data is strictly required.

## 4. Training run history

1. **First smoke test** -- 12 synthetic procedurally-generated tiles
   (no download needed), confirmed the full loop (data -> model ->
   loss -> optimizer -> checkpoint) actually executes.
2. **First real-data run** -- ran on CPU by accident (the default `pip
   install torch` grabbed a CPU-only build). Got to epoch 18 before
   being stopped once this was noticed.
3. **GPU fixed** -- root cause was Python 3.14 (very new) not having
   CUDA wheels on the default PyTorch index; fixed by installing from
   the `cu128` index specifically, which does publish 3.14 wheels.
4. **Real training run, EDSR, real SEN2NAIP data, GPU** -- completed
   all 50 epochs. `checkpoints/edsr_epoch50.pt` is the current final
   model. Not yet scored with the real metrics or compared to earlier
   epochs beyond a quick visual check.

## 5. Honest state of things right now

A real model has been trained end-to-end on real (if geographically
narrow) paired satellite imagery, on your own GPU, with a
scientifically-motivated loss recipe and a working evaluation toolkit
that hasn't been pointed at the result yet. That's a genuine, working
prototype -- not a mockup. What's still open: actually scoring it with
PSNR/SSIM/ERGAS/SAM, running the downstream water/vegetation
comparison, training SwinIR for the comparison story, running
inference on one real full-size scene to get an actual GeoTIFF output,
and the dataset diversity question above.
