# TerraResolve — Task Breakdown & Implementation Instructions

This is Hour 0–2 of the 24-hour plan: the phase where everyone can work in parallel because nothing hard-blocks yet, as long as two contracts get agreed on **right now, before anyone writes code**:

## Lock these two contracts first (5 minutes, whole team)

1. **`run_edsr_inference()` signature** (Shawn defines, everyone reads):
```python
def run_edsr_inference(input_tif_path: str, output_tif_path: str) -> dict:
    """Returns: {"duration_sec": float, "shape": [H, W], "scale": int}"""
```
Bharath's job worker and Harish's Docker image both need to know this exists and what it returns, even before Shawn finishes the real implementation.

2. **API JSON shapes** — Bharath posts these in the team chat once `/health` is up, so Mayasah/you aren't guessing at field names later.

Two things changed from the earlier plan that everyone should know:
- **Deployment target is HuggingFace Spaces**, not Render/Vercel. HF Spaces runs Docker containers and can give you a **GPU** (T4, depending on which Space tier you pick) — this is actually better than the original CPU-only Render assumption, so ask Shawn to check if the Space has CUDA before assuming CPU-only inference speeds.
- **Scope is "4 Bengaluru scenes"** per Siddarth's task, not strictly Yelahanka-only. Decide as a team in the next hour whether you're keeping the tight Yelahanka AOI story (recommended — it's a stronger, more specific demo) or broadening to general Bengaluru. Siddarth should pick chips that work either way (see his instructions below) until that's settled.

---

## SHAWN — Verify EDSR loads with CUDA · Run one Sentinel-2 patch · Record GPU benchmark

**What this task actually means:** You are proving the single most important technical claim of the whole project — that a real deep learning super-resolution model can take a real satellite image patch and produce a sharper, valid output — before anyone else builds anything on top of it. Nothing else in the pipeline can be trusted until this works.

**Why it's first:** Backend (Bharath), the demo dataset, and the metrics pipeline all call your inference function. If this doesn't work, the whole architecture doesn't work, so you get first priority and no dependencies.

### Step-by-step

**1. Environment setup**
```bash
pip install torch torchvision rasterio numpy pillow
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
If `torch.cuda.is_available()` returns `False` on your dev machine, that's fine for now — you're verifying CUDA works on whatever machine will actually run inference for the demo (likely the HF Space, once Harish has it up, or your own GPU if you have one).

**2. Get a checkpoint**
Use a pretrained EDSR checkpoint trained on natural images (DIV2K) — do NOT try to train your own in 24 hours. Look for a public PyTorch EDSR repo (e.g. `sanghyun-son/EDSR-PyTorch`) and download a pretrained `.pt`/`.pth` file. Record which repo and which exact checkpoint file — you'll need to disclose this honestly in the demo (see plan §6.1 and §9.2 — judges respect honesty about checkpoint provenance far more than a vague claim).

**3. Verify the model's actual contract before writing any pipeline code**
Open the checkpoint's own repo and confirm, in writing (put this in a shared doc or Slack):
- Scale factor (2x, 3x, or 4x — EDSR checkpoints are locked to one scale, you cannot mix)
- Input channels (almost certainly 3 = RGB)
- Expected input dtype/range (uint8 0–255, or normalized float — check the repo's own preprocessing script, don't guess)
- Any mean/std normalization it subtracts

**4. Load the model correctly**
```python
import torch

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path, device):
    model = EDSR(...)  # architecture class from the repo you're using
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
```
Load it **once**, not per-request — this matters later when Bharath wires it into the FastAPI startup event.

**5. Get one real Sentinel-2 patch to test on**
Don't wait for Siddarth's full 4-scene download — grab any single small Sentinel-2 L2A chip yourself right now (even a rough manual crop from Copernicus Browser) just to unblock your own testing. Swap in Siddarth's real chips once he delivers them.

**6. Run one patch through the model**
```python
import rasterio
import numpy as np

with rasterio.open("test_chip.tif") as src:
    arr = src.read([4, 3, 2])  # B04, B03, B02 -> R, G, B (check band order!)
    profile = src.profile

# normalize per the checkpoint's expected range
tensor = torch.from_numpy(arr).float().unsqueeze(0).to(device)
with torch.no_grad():
    output = model(tensor)
# denormalize, clip to [0,255], convert back to uint8 numpy array
```

**7. Write the output back as a valid GeoTIFF**
This is the part that's easy to get wrong — the output needs the pixel size divided by the scale factor, not just a bigger array:
```python
from rasterio.transform import Affine

new_transform = Affine(profile['transform'].a / scale, profile['transform'].b, profile['transform'].c,
                        profile['transform'].d, profile['transform'].e / scale, profile['transform'].f)

new_profile = profile.copy()
new_profile.update(transform=new_transform, width=arr.shape[2]*scale, height=arr.shape[1]*scale,
                    count=3, dtype='uint8')

with rasterio.open("test_sr.tif", "w", **new_profile) as dst:
    dst.write(output_array)
```

**8. Record the benchmark**
Time the inference call, note peak memory if you can (`torch.cuda.max_memory_allocated()` on GPU), and record: input dimensions, patch size used, inference duration, output dimensions. This number directly determines whether patch-based tiling (128×128 patches with overlap, per the plan) is necessary or whether you can get away with running a small full scene at once for the demo.

### Definition of done
- [ ] CUDA availability confirmed on the actual runtime environment
- [ ] Checkpoint contract documented (scale, channels, normalization) and shared with the team
- [ ] One real Sentinel-2 patch produces a visually sharper output
- [ ] Output GeoTIFF opens correctly in QGIS with valid CRS/transform (get Siddarth or yourself to sanity-check this in QGIS, not just "the file exists")
- [ ] Benchmark numbers (time, memory) written down and shared

### What you hand off next
Once this works on one patch, immediately wrap it into the `run_edsr_inference()` function signature the team agreed on, so Bharath can call it from the job worker without needing to understand your internals.

---

## HARISH — Set up repo structure · Write Dockerfile · Create HuggingFace Spaces account + push skeleton

**What this task actually means:** You're building the scaffolding everyone else's code lands in, and you're making sure that scaffolding actually deploys successfully as early as possible — deployment failures discovered on Hour 20 are catastrophic; discovered on Hour 1 they're a non-event.

### Step-by-step

**1. Create the repo structure**
Set this up exactly (mirrors the plan so nobody has to guess where their code goes):
```
terraresolve/
├── backend/
│   ├── main.py
│   ├── api/
│   ├── inference/
│   ├── geospatial/
│   ├── services/
│   ├── db/
│   ├── worker.py
│   ├── config.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/
│   ├── js/
│   └── assets/
├── models/
├── demo/scenes/
├── scripts/
├── docs/
└── README.md
```
Push this empty skeleton immediately so everyone has somewhere to commit to. An empty folder with a `.gitkeep` is fine — the point is the structure exists.

**2. Write the Dockerfile**
This needs GDAL system dependencies, which is the #1 thing that breaks Python geospatial Docker builds if you don't get it right:
```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY models/ ./models/
COPY demo/scenes/ ./demo/scenes/

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```
Note: **port 7860** — HuggingFace Spaces expects apps to listen on 7860 by default for Docker Spaces, not the usual 8000. Confirm this against HF's current docs when you set up the Space, since defaults can vary by SDK.

**3. Create the HuggingFace Spaces account and push the skeleton**
- Create the Space, choose **Docker** as the SDK (not Gradio/Streamlit).
- Check what compute tiers are available — if a free GPU tier exists, that changes Shawn's benchmark numbers significantly (GPU vs CPU inference time is often 10-50x different for a model like EDSR).
- Push a minimal working Dockerfile first — even before `main.py` has real routes, get a "hello world" FastAPI app deploying successfully. This proves the deployment pipeline itself works, independent of anyone's actual feature code.
- Document the exact push/deploy process (HF Spaces usually deploys via `git push` to the Space's own remote, similar to Heroku) so the whole team can redeploy without you being the single point of failure on demo day.

**4. requirements.txt**
Start this now even though it'll grow — at minimum:
```
fastapi
uvicorn
python-multipart
numpy
pillow
torch
torchvision
rasterio
```

### Definition of done
- [ ] Repo structure pushed, matches the layout above
- [ ] Dockerfile builds successfully locally (`docker build -t terraresolve .`)
- [ ] HF Space created, Docker SDK selected
- [ ] Minimal skeleton app deployed and reachable at the Space's public URL
- [ ] GPU/CPU tier of the Space documented and shared with Shawn
- [ ] Deploy process documented in `docs/ARCHITECTURE.md` or README so anyone can redeploy

### What you hand off next
Once Bharath has real routes in `main.py`, redeploy the Space with the real app. Keep an eye on Docker image size — `models/` + `demo/scenes/` baked into the image (per the plan, so the demo never depends on live data) can get large; watch HF Spaces' storage limits.

---

## BHARATH — Set up FastAPI skeleton · GET /health · GET /health/ready · SQLite DB models

**What this task actually means:** You're building the backend's nervous system — the part that proves the server is alive and the part that tracks every job's state. Nothing async (uploads, jobs) works without the DB models existing first.

### Step-by-step

**1. FastAPI skeleton (`backend/main.py`)**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TerraResolve API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your actual frontend domain before demo day
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None  # loaded on startup

@app.on_event("startup")
async def startup_event():
    global model
    # model = load_model(...)  # wire in once Shawn's function is ready
    pass
```

**2. Health endpoints**
This is the one part of the plan every judge and every deployment platform (HF Spaces included) cares about — get it right:
```python
@app.get("/health")
async def health():
    """Liveness check — must respond instantly, no model check."""
    return {"status": "ok"}

@app.get("/health/ready")
async def ready():
    """Readiness — actually checks the model is loaded and DB is reachable."""
    return {
        "status": "ok" if model is not None else "not_ready",
        "model_loaded": model is not None,
    }
```
The distinction matters: `/health` should never fail just because the model is slow to load — that would cause the platform to keep restarting your container in a crash loop. `/health/ready` is what the frontend should poll before enabling the Upload page.

**3. SQLite DB models**
Set up the schema exactly as the plan specifies, since Bharath's job worker (built later in the timeline) depends on these fields existing:
```python
# backend/db/database.py
import sqlite3

def init_db(db_path="terraresolve.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            stage TEXT,
            progress_pct INTEGER DEFAULT 0,
            source_type TEXT NOT NULL,
            model_version TEXT,
            input_path TEXT,
            output_path TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn
```
```python
# backend/db/models.py
from pydantic import BaseModel
from typing import Optional

class Job(BaseModel):
    id: str
    status: str  # QUEUED/RUNNING/PREPROCESSING/INFERENCE/POSTPROCESSING/COMPLETED/FAILED
    stage: Optional[str] = None
    progress_pct: int = 0
    source_type: str  # 'upload' | 'live'
    model_version: Optional[str] = None
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str
```
Wire `init_db()` into the startup event alongside the model load.

**4. Restart recovery logic (write now, even though the worker doesn't exist yet)**
On every startup, any job left `RUNNING`/`PREPROCESSING`/`INFERENCE`/`POSTPROCESSING` from before a crash/redeploy gets reset to `FAILED`. Stub this now so it's not forgotten later:
```python
@app.on_event("startup")
async def recover_interrupted_jobs():
    conn.execute("""
        UPDATE jobs SET status='FAILED', error_message='interrupted by restart'
        WHERE status IN ('RUNNING','PREPROCESSING','INFERENCE','POSTPROCESSING')
    """)
    conn.commit()
```

### Definition of done
- [ ] FastAPI app runs locally (`uvicorn main:app --reload`)
- [ ] `GET /health` returns `{"status": "ok"}` instantly
- [ ] `GET /health/ready` correctly reflects whether the model is loaded
- [ ] `jobs` table created on startup with the exact schema above
- [ ] Pydantic `Job` model matches the DB schema field-for-field
- [ ] Restart-recovery logic stubbed in
- [ ] JSON response shapes posted to the team so Mayasah/you can build the frontend against real contracts instead of guesses

### What you hand off next
Once Shawn's `run_edsr_inference()` exists, wire it into the startup model load and start building `/scenes` and `/upload`/`/jobs` routes (plan §8).

---

## YOU (TARAN) — Set up frontend/ folder · Write styles.css design system

**What this task actually means:** You're defining the entire visual language of the product before a single page of content exists. Every page Mayasah and future frontend work builds afterward inherits these tokens — get the system right once instead of every page reinventing colors/spacing ad hoc.

### Step-by-step

**1. Set up the folder** (Harish will have already pushed the skeleton — fill it in)
```
frontend/
├── index.html
├── viewer.html
├── upload.html
├── metrics.html
├── live.html
├── css/
│   └── styles.css
├── js/
│   ├── api.js
│   ├── viewer.js
│   ├── upload.js
│   ├── live.js
│   └── ui.js
└── assets/
```

**2. Define the design tokens as CSS custom properties**
This is the actual "design system" part — everything downstream (Mayasah's landing page, the Explorer viewer, the Upload page) should reference these variables, never hardcode a color or font size:

```css
:root {
  /* --- Color palette --- */
  /* Satellite/geospatial theme: deep space blues, terrain greens, alert amber */
  --color-bg-primary: #0a0e17;
  --color-bg-secondary: #111827;
  --color-bg-elevated: #1a2233;

  --color-accent-primary: #3b82f6;   /* sharpening blue — used for "SR" state */
  --color-accent-secondary: #10b981; /* terrain green — used for water/vegetation data */
  --color-accent-warning: #f59e0b;   /* used sparingly — errors, fallback states */

  --color-text-primary: #f8fafc;
  --color-text-secondary: #94a3b8;
  --color-text-muted: #64748b;

  --color-border: rgba(255, 255, 255, 0.08);
  --color-border-strong: rgba(255, 255, 255, 0.16);

  /* --- Glassmorphism tokens --- */
  --glass-bg: rgba(255, 255, 255, 0.05);
  --glass-bg-elevated: rgba(255, 255, 255, 0.08);
  --glass-border: rgba(255, 255, 255, 0.12);
  --glass-blur: blur(12px);
  --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);

  /* --- Typography --- */
  --font-display: 'Space Grotesk', sans-serif;   /* headings — geometric, technical feel */
  --font-body: 'Inter', sans-serif;               /* body text — highly legible */
  --font-mono: 'JetBrains Mono', monospace;       /* coordinates, metrics, technical readouts */

  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-base: 1rem;
  --text-lg: 1.125rem;
  --text-xl: 1.5rem;
  --text-2xl: 2rem;
  --text-3xl: 3rem;

  /* --- Spacing scale --- */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 1rem;
  --space-4: 1.5rem;
  --space-5: 2rem;
  --space-6: 3rem;
  --space-8: 4rem;

  /* --- Radius --- */
  --radius-sm: 6px;
  --radius-md: 12px;
  --radius-lg: 20px;

  /* --- Transitions --- */
  --transition-fast: 150ms ease;
  --transition-base: 250ms ease;
}
```

**3. The glassmorphism component class**
This is the actual reusable pattern — panels, metric cards, the metadata sidebar in the Explorer all use this:
```css
.glass-panel {
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--glass-shadow);
}

.glass-panel--elevated {
  background: var(--glass-bg-elevated);
}
```

**4. Base resets and typography**
```css
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  font-family: var(--font-body);
  line-height: 1.5;
}

h1, h2, h3 {
  font-family: var(--font-display);
  font-weight: 600;
}

.mono {
  font-family: var(--font-mono);
}
```

**5. Import the fonts**
Since this deploys as HTML/CSS/JS (no build step), pull from Google Fonts in the `<head>` of every page:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

**6. Core utility classes** (buttons, containers — keep minimal, expand as pages need them)
```css
.container {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 var(--space-4);
}

.btn {
  font-family: var(--font-body);
  font-weight: 500;
  padding: var(--space-2) var(--space-4);
  border-radius: var(--radius-md);
  border: none;
  cursor: pointer;
  transition: var(--transition-fast);
}

.btn--primary {
  background: var(--color-accent-primary);
  color: white;
}
.btn--primary:hover {
  filter: brightness(1.1);
}
```

### Definition of done
- [ ] Folder structure matches Harish's repo layout
- [ ] `styles.css` has full token set (color, type, spacing, radius, glassmorphism) as CSS custom properties
- [ ] At least one `.glass-panel` component class works visually (test with a throwaway div)
- [ ] Fonts load correctly
- [ ] Design tokens documented (a short comment block at the top of the file explaining what each token is for) so Mayasah doesn't have to guess what `--color-accent-secondary` means when building the landing page

### What you hand off next
Once this exists, tell Mayasah the exact class names and variables available so her `index.html` uses your tokens instead of inline styles. This is also the foundation for the Explorer's before/after slider UI later.

---

## SIDDARTH — Download 4 Bengaluru Sentinel-2 scenes from Copernicus · Verify bands

**What this task actually means:** You're sourcing the real-world data that makes this project credible instead of a toy demo. Every other piece of scientific validation, the downstream water analysis, and the Explorer viewer all depend on you delivering real, correctly-banded Sentinel-2 imagery.

### Step-by-step

**1. Decide the AOI with the team first**
Before downloading, confirm with the team whether you're going tight-Yelahanka (recommended per the plan — gives a stronger, more specific story) or general Bengaluru. If unresolved, pick 4 chips that are all comfortably inside the Yelahanka bounding box below — that way you satisfy both options:
```
bbox_wgs84: [77.550, 13.070, 77.620, 13.150]
```
This covers Yelahanka New Town, Old Town, Yelahanka Lake, the AFS periphery, and the Jakkur/Attur tank edge.

**2. Access Copernicus Data Space**
Go to **Copernicus Browser** (`browser.dataspace.copernicus.eu`) — this is the official PS-provided dataset link. Create an account if you don't have one (needed for downloads).

**3. Search for imagery**
- Product type: **Sentinel-2 L2A** (Level-2A = atmospherically corrected, ready to use — do NOT grab L1C, which needs extra correction)
- AOI: draw or paste the bbox above
- Cloud cover filter: **under 10%**
- Sort by most recent, pick the cleanest scene

**4. Pick 4 distinct chips within your AOI** matching these categories (this is what makes the "before/after" story visually interesting instead of 4 near-identical patches):
| Chip | What to look for |
|---|---|
| **Water** | Yelahanka Lake itself — clear water body, ideally with some visible shoreline vegetation for contrast |
| **Dense urban** | Yelahanka New Town — the planned grid street pattern is very visually obvious in satellite imagery, good for showing SR sharpening |
| **Vegetation** | Airbase periphery green belt — tree cover, open green space |
| **Mixed** | Old Town / peri-urban edge — mix of buildings and farmland; this is deliberately your "honest limitations" scene, so it doesn't need to look perfect |

**5. Download the required bands — this part is critical, don't skip it**
For each scene you MUST get:
- **B02** (Blue, 10m)
- **B03** (Green, 10m)
- **B04** (Red, 10m)
- **B08** (NIR, 10m) — this is required later for the water/NDWI analysis, even though it's not fed through EDSR directly

Download the specific granule/bands, not the entire massive product bundle if you can avoid it — Sentinel-2 tiles are large (~1GB+) and you only need a small AOI clip out of them.

**6. Verify the bands are actually correct**
This is the "verify bands" part of your task — don't just assume the download worked:
```python
import rasterio

with rasterio.open("scene_band04.tif") as src:
    print("CRS:", src.crs)
    print("Shape:", src.shape)
    print("Dtype:", src.dtypes)
    print("Bounds:", src.bounds)
    print("Nodata:", src.nodata)
```
Check for each band:
- [ ] File opens without error in rasterio
- [ ] CRS is defined (should be something like EPSG:32643 or EPSG:4326 depending on the product — note which one, the team will reproject to EPSG:32643 UTM 43N later)
- [ ] Resolution is actually ~10m (check pixel size in the transform)
- [ ] No huge blocks of NaN/nodata covering your AOI
- [ ] Open it visually in **QGIS** (free, install it) and confirm it actually looks like Yelahanka — this catches "wrong tile" mistakes that are otherwise invisible until much later

**7. Clip to your 4 chip bboxes**
Don't hand off entire Sentinel-2 tiles — clip each to a ~1–2 km² chip around your target feature (lake, new town, green belt, old town) so inference stays fast:
```python
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd

chip_bbox = box(77.595, 13.095, 77.605, 13.105)  # example — adjust per chip
geo = gpd.GeoDataFrame({'geometry': [chip_bbox]}, crs="EPSG:4326")
# reproject geo to match your raster's CRS, then use rasterio.mask.mask()
```

**8. Package as `input.tif` per scene** — stack B04/B03/B02 (and keep B08 available separately) into the naming convention the team is using:
```
demo/scenes/yelahanka-lake/input.tif
demo/scenes/yelahanka-newtown/input.tif
demo/scenes/yelahanka-afs-buffer/input.tif
demo/scenes/yelahanka-mixed/input.tif
```

**9. Commit these to the repo** (or hand to Harish to bake into the Docker image) — this is what guarantees the demo never depends on live internet access during judging.

### Definition of done
- [ ] 4 scenes identified, all cloud cover < 10%
- [ ] B02, B03, B04, B08 downloaded for each
- [ ] Bands verified in Python (CRS, resolution, no major nodata gaps)
- [ ] Visually confirmed in QGIS to actually be Yelahanka/Bengaluru, not a wrong-tile mistake
- [ ] Clipped to 4 distinct ~1-2km² chips matching the water/urban/vegetation/mixed categories
- [ ] Files named and placed per the repo convention, ready for Shawn to run through EDSR and for the metrics/downstream teammates to use later

### What you hand off next
Give Shawn your first working chip as soon as you have even one clean one — don't wait until all 4 are perfect. This unblocks his benchmark work immediately.

---

## MAYASAH — Write index.html landing page structure + navbar + hero section

**What this task actually means:** You're building the first thing a judge sees. This page has one job: communicate what TerraResolve is and where to go next, in under 10 seconds of looking at it.

### Step-by-step

**1. Wait for (or build alongside) your design tokens** — check with Taran/you for the CSS variable names so you're not hardcoding colors/fonts that get thrown away later. If the design system isn't ready yet, build the structure now with placeholder styling and swap classes in once it lands.

**2. Page structure**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TerraResolve — Satellite Super-Resolution for Yelahanka</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="css/styles.css">
</head>
<body>

  <nav class="navbar glass-panel">
    <div class="container navbar__inner">
      <a href="index.html" class="navbar__brand">TerraResolve</a>
      <div class="navbar__links">
        <a href="viewer.html">Explorer</a>
        <a href="metrics.html">Metrics</a>
        <a href="upload.html">Upload</a>
        <a href="live.html">Live</a>
      </div>
    </div>
  </nav>

  <section class="hero">
    <div class="container hero__inner">
      <h1 class="hero__title">AI-Powered Satellite Super-Resolution</h1>
      <p class="hero__subtitle">Yelahanka, North Bengaluru — a pilot study area.</p>
      <p class="hero__description">
        TerraResolve reconstructs high-resolution satellite imagery from real
        Sentinel-2 data, validated scientifically and applied to real
        civic infrastructure — starting with Yelahanka Lake.
      </p>
      <div class="hero__actions">
        <a href="viewer.html" class="btn btn--primary">Explore Yelahanka</a>
        <a href="upload.html" class="btn btn--secondary">Upload GeoTIFF</a>
      </div>
    </div>
  </section>

  <!-- Optional below-the-fold sections you can add once the core structure works -->
  <section class="how-it-works container">
    <!-- Sentinel-2 -> EDSR -> GeoTIFF pipeline explainer, 3-step visual -->
  </section>

  <script src="js/ui.js"></script>
</body>
</html>
```

**3. Naming discipline — this matters for the whole project's credibility**
The subtitle line above is not arbitrary — per the plan, "Bengaluru" is mentioned **exactly once**, right there in the hero subtitle, and nowhere else. Every other reference on the page and site should say "Yelahanka." This is a deliberate judge-facing consistency signal — don't let it drift.

**4. What NOT to build yet**
Don't wire up real data, don't build the actual before/after slider (that's the Explorer page, a separate and much bigger task later in the timeline). Your job right now is structure: navbar that links to all 5 pages (even if they're empty stubs), a hero section that states the pitch clearly, and two clear CTAs.

**5. Navbar behavior**
Keep it simple — a fixed/sticky glass-panel bar at the top linking to the other 4 pages (`viewer.html`, `metrics.html`, `upload.html`, `live.html`). Even if those pages don't exist yet, create empty placeholder files for them now so the links don't 404 when someone clicks around.

### Definition of done
- [ ] `index.html` exists with navbar + hero section
- [ ] Navbar links to all 5 pages (stub files created even if empty)
- [ ] Hero states the pitch in one sentence, has the Yelahanka/Bengaluru naming exactly as specified
- [ ] Two clear CTAs (Explore Yelahanka → Explorer, Upload → Upload page)
- [ ] Uses Taran's design tokens (glass-panel navbar, correct fonts/colors) once available — flag it to Taran directly if you're blocked waiting on the CSS file
- [ ] Renders correctly opened locally in a browser (no build step needed — this is plain HTML/CSS/JS)

### What you hand off next
Once this works, you're the natural person to move onto `viewer.html`'s static structure (before Person C — whoever that ends up being on the actual 24-hour team — wires in the real slider JS and API calls), since you'll already understand the page/nav pattern.

---

## How this all connects — the critical path today

```
Shawn (EDSR proof)  ──┐
                       ├──► Bharath wires model into FastAPI startup + /jobs worker
Siddarth (data)     ──┘         │
                                 ▼
Harish (Docker/HF Space) ──► deploys Bharath's backend
                                 │
Taran (design system) ──► Mayasah's landing page + later Explorer UI
```

The two people with zero dependencies right now — **Shawn and Siddarth** — should sync with each other constantly, since Shawn needs Siddarth's real chip as soon as possible to validate the pipeline end-to-end instead of testing on a throwaway crop. Everyone else (Harish, Bharath, Taran, Mayasah) can build entirely against contracts/mocks for the next several hours without being blocked.
