# Datasets

Real satellite-pair sources evaluated or used for TerraResolve training.
Per project decision: **no synthetic/simulated degradation pairs** --
every dataset here is a genuine paired capture (or a real medium-res
scene we ingest ourselves), not an artificially downsampled one.

## In use

### SEN2NAIP (cross-sensor subset) -- currently trained on

Real Sentinel-2 (10m) paired with real NAIP aerial imagery (~1m),
reprojected/aligned by the dataset authors. Used for the completed
50-epoch EDSR training run (`checkpoints/edsr_epoch50.pt`).

- Dataset: https://huggingface.co/datasets/isp-uv-es/SEN2NAIP
- Paper: https://doi.org/10.5281/zenodo.10078966
- Coverage: continental USA only (NAIP is a US aerial program) --
  this is the project's main geographic-coverage gap.
- Local layout after `scripts/prepare_sen2naip.py`: `data/sen2naip/lr/`,
  `data/sen2naip/hr/` (2,851 pairs, 4-band B2/B3/B4/B8, HR upsampled x4).

## Investigated, not yet integrated

### SEN2VENuS -- real, includes a confirmed Indian site (KUDALIAR)

Real Sentinel-2 paired with real VENuS satellite captures (same-day
acquisitions), 29 global sites, 132,955 patches total. The KUDALIAR
site (Telangana, India -- confirmed via the dataset's own site list
and independent cross-reference to published Kudaliar-watershed
remote-sensing studies) is the most direct fix available for the
project's US-only coverage gap, since it's real Sentinel-2 imagery
over Indian terrain, not synthetic.

- Dataset (original, `.pt` tensors, per-site `.7z` archives):
  https://zenodo.org/records/6514159
  -- `KUDALIAR.7z` alone is ~5.0 GB, 20 pairs / 7,269 patches.
- Mirror (GeoTIFF via TACO/tacoreader, ~104GB full dataset -- avoid
  unless you specifically want the TACO tooling):
  https://huggingface.co/datasets/tacofoundation/sen2venus
- Paper: https://doi.org/10.3390/data7070096
- Licensing: Sentinel-2 side is Etalab Open Licence 2.0; VENuS/HR
  side is **CC-BY-NC 4.0 (non-commercial)** -- fine for an SIH
  prototype, flag before any commercial claim.
- Next step if adopted: write `scripts/prepare_sen2venus.py` (loads
  the `.pt` tensors, reorganizes into `paired_real`'s `lr_dir`/`hr_dir`
  layout) + `configs/sen2venus.yaml`.

## In progress

### Maxar Open Data -- confirmed real sub-metre HR reference over India

Maxar's open, disaster-response-triggered VHR satellite imagery
(sub-metre, real captures, MIT-licensed). Confirmed: the
**India-Floods-Oct-2023** collection (51 tiles) covers the October
2023 South Lhonak Lake glacial-lake-outburst flood / Teesta river
disaster in **Sikkim, India** -- genuine Indian terrain, genuinely
real (not simulated), and at a much finer native resolution than
SEN2VENuS's 5m VENuS reference.

- Event page: https://www.maxar.com/open-data/india-floods-oct-2023
- STAC catalog root: https://maxar-opendata.s3.amazonaws.com/events/catalog.json
- Collection id: `India-Floods-Oct-2023` (51 items total, grouped into
  child collections by sub-date/area -- run
  `python scripts/prepare_maxar_sikkim.py --list-collections` to see them)
- Access library: **leafmap** (`pip install leafmap pystac-client`) --
  confirmed real API: `leafmap.maxar_collections()`,
  `leafmap.maxar_child_collections()`, `leafmap.maxar_items()`. (Note:
  there is no separate `maxar-open-data` pip package, despite the repo
  name -- the actual Python access path is through leafmap's STAC
  helpers; verified against https://leafmap.org/notebooks/67_maxar_open_data/
  before writing the integration, not assumed.)
- License: MIT (Maxar Open Data Program terms)
- Integration: `scripts/prepare_maxar_sikkim.py` -- downloads Maxar's
  "visual" (RGB) COG tiles, pulls the matching Sentinel-2 L2A scene
  for the same bbox/date via `terraresolve/ingest/copernicus.py`,
  reprojects/crops both to a common grid (Maxar resampled to exactly
  4x the Sentinel-2 crop's resolution, i.e. trained against a ~2.5m
  target, to match the project's scale=4 configs), and writes the
  pair into the same `lr_dir`/`hr_dir` layout as SEN2NAIP. Needs to be
  run on a machine with normal (non-sandboxed) internet access --
  requires both an AWS S3 connection and a Copernicus account.
- Caveat: Maxar's "visual" asset is RGB only (no NIR), so this
  dataset's HR side has 3 channels vs. SEN2NAIP's 4 -- see the
  script's docstring for how that's handled.

### SEN2VENuS -- real, includes a confirmed Indian site (KUDALIAR)

Real Sentinel-2 paired with real VENuS satellite captures (same-day
acquisitions), 29 global sites, 132,955 patches total. The KUDALIAR
site (Telangana, India -- confirmed via the dataset's own site list
and independent cross-reference to published Kudaliar-watershed
remote-sensing studies) is a second, complementary fix for the
project's US-only coverage gap.

- Dataset (original, `.pt` tensors, per-site `.7z` archives):
  https://zenodo.org/records/6514159
  -- `KUDALIAR.7z` alone is ~5.0 GB, 20 pairs / 7,269 patches.
- Mirror (GeoTIFF via TACO/tacoreader, ~104GB full dataset -- avoid
  unless you specifically want the TACO tooling):
  https://huggingface.co/datasets/tacofoundation/sen2venus
- Paper: https://doi.org/10.3390/data7070096
- Licensing: Sentinel-2 side is Etalab Open Licence 2.0; VENuS/HR
  side is **CC-BY-NC 4.0 (non-commercial)** -- fine for an SIH
  prototype, flag before any commercial claim.
- Next step if adopted: write `scripts/prepare_sen2venus.py` (loads
  the `.pt` tensors, reorganizes into `paired_real`'s `lr_dir`/`hr_dir`
  layout) + `configs/sen2venus.yaml`.

## Checked and ruled out

### Zenodo record 3923841 -- wrong task, not usable

Real Sentinel-2 L1C imagery, but this is a **ship-detection** dataset
(Danish waters, paired with AIS vessel-tracking data for object
detection, not with any higher-resolution reference image). No
LR-HR pairing of any kind, no Indian coverage, CC-BY-NC-SA licensed.
Not applicable to super-resolution training -- ruled out.
https://zenodo.org/records/3923841

## Live ingestion (not a training-pair dataset)

### Copernicus Data Space Ecosystem -- direct Sentinel-2 access

The official distribution platform named in the problem statement.
Used via `terraresolve/ingest/copernicus.py` +
`scripts/fetch_copernicus.py` to pull a live Sentinel-2 L2A scene for
any AOI/date directly (not through a pre-packaged research dataset) --
proves the pipeline ingests from the mandated source. Requires a free
account: https://dataspace.copernicus.eu/
