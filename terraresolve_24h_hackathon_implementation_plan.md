# TerraResolve --- 24-Hour Hackathon Implementation Plan

## 1. Mission

Build **TerraResolve**, a production-style Bengaluru satellite
super-resolution platform that demonstrates:

1.  Real Sentinel-2 satellite data.
2.  A real PyTorch EDSR super-resolution model.
3.  Geospatially correct GeoTIFF output.
4.  A polished before/after satellite viewer.
5.  Quantitative scientific validation.
6.  One downstream geospatial application.
7.  Real asynchronous user inference.
8.  Optional live Copernicus acquisition.
9.  A reliable deployed demo with graceful fallbacks.

### The key strategy

Do **not** try to make every part equally complex.

The system has one guaranteed path:

``` text
Preloaded Bengaluru Scene
        ↓
Before / After Viewer
        ↓
EDSR Result
        ↓
Scientific Metrics
        ↓
Downstream Analysis
```

Then prove that it is a real system with:

``` text
User GeoTIFF
      ↓
Async Job
      ↓
EDSR
      ↓
GeoTIFF Result
```

And finally add:

``` text
Bengaluru AOI
      ↓
Copernicus Sentinel-2
      ↓
EDSR
      ↓
Result
```

The **first path must work even if everything live fails**.

------------------------------------------------------------------------

# 2. Final Architecture

``` text
                         ┌────────────────────┐
                         │      USER          │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Vercel Frontend    │
                         │ HTML/CSS/JS        │
                         └─────────┬──────────┘
                                   │ HTTPS
                                   ▼
                    ┌──────────────────────────────┐
                    │ Render Docker Container      │
                    │                              │
                    │ FastAPI                      │
                    │ ├── Tiles API                │
                    │ ├── Jobs API                 │
                    │ ├── Metrics API              │
                    │ ├── Copernicus API           │
                    │ └── Health API               │
                    │                              │
                    │ Job Manager                   │
                    │       ↓                      │
                    │ Single Inference Worker       │
                    │       ↓                      │
                    │ PyTorch EDSR                  │
                    │       ↓                      │
                    │ Rasterio / rio-tiler          │
                    └─────────────┬────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             Demo Scene Store            Runtime Storage
             Precomputed assets           Inputs / outputs
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                           Frontend Viewer
```

## Why this architecture

### Frontend: Vercel

Use Vercel for the static web experience.

### Backend: Render

Use one Dockerized FastAPI service for:

-   PyTorch
-   GDAL/rasterio
-   geospatial processing
-   long-running inference
-   API routes

### Worker model

Use a **single controlled worker** initially.

Do not run multiple CPU-heavy inference jobs simultaneously unless
benchmarking proves it is safe.

### Storage

Use:

-   packaged model weights
-   precomputed demo assets
-   persistent runtime storage when available
-   an abstraction layer so object storage can be added later

### Database

Use SQLite for hackathon-scale job metadata.

Do not introduce PostgreSQL/PostGIS unless there is a concrete feature
requiring it.

------------------------------------------------------------------------

# 3. What NOT to Build

These technologies are explicitly outside the initial scope:

-   Kubernetes
-   Kafka
-   Celery
-   Redis
-   microservices
-   service mesh
-   authentication
-   complex user accounts
-   PostGIS
-   distributed GPU cluster
-   multiple ML models
-   a second downstream computer-vision system
-   custom satellite data archive
-   complicated analytics warehouse

These can be mentioned as a **future scaling architecture**, but they
should not become implementation dependencies.

------------------------------------------------------------------------

# 4. Repository Structure

``` text
terraresolve/
│
├── frontend/
│   ├── index.html
│   ├── viewer.html
│   ├── upload.html
│   ├── live.html
│   ├── metrics.html
│   ├── about.html
│   │
│   ├── css/
│   │   └── styles.css
│   │
│   ├── js/
│   │   ├── api.js
│   │   ├── viewer.js
│   │   ├── upload.js
│   │   ├── live.js
│   │   └── ui.js
│   │
│   └── assets/
│
├── backend/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── health.py
│   │   ├── scenes.py
│   │   ├── jobs.py
│   │   ├── upload.py
│   │   ├── metrics.py
│   │   └── copernicus.py
│   │
│   ├── inference/
│   │   ├── model.py
│   │   ├── runner.py
│   │   ├── tiling.py
│   │   └── postprocess.py
│   │
│   ├── geospatial/
│   │   ├── validation.py
│   │   ├── preprocessing.py
│   │   ├── geotiff.py
│   │   └── metrics.py
│   │
│   ├── services/
│   │   ├── job_service.py
│   │   ├── storage_service.py
│   │   └── copernicus_service.py
│   │
│   └── db/
│       ├── database.py
│       └── models.py
│
├── models/
│   └── edsr_epoch50.pt
│
├── demo/
│   └── scenes/
│       ├── scene_01/
│       ├── scene_02/
│       ├── scene_03/
│       └── ...
│
├── scripts/
│   ├── prepare_scene.py
│   ├── benchmark.py
│   └── smoke_test.py
│
├── tests/
│   ├── test_inference.py
│   ├── test_geotiff.py
│   ├── test_metrics.py
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .env.example
├── render.yaml
├── vercel.json
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

# 5. 24-Hour Execution Order

The order below is intentionally strict.

``` text
PHASE 1
Model proof
   ↓
PHASE 2
Demo dataset
   ↓
PHASE 3
Backend core
   ↓
PHASE 4
Viewer
   ↓
PHASE 5
Async user inference
   ↓
PHASE 6
Scientific validation
   ↓
PHASE 7
Downstream application
   ↓
PHASE 8
Live Copernicus
   ↓
PHASE 9
Deployment + hardening
   ↓
PHASE 10
Final demo rehearsal
```

Every later phase depends on something earlier.

------------------------------------------------------------------------

# PHASE 1 --- PROVE THE MODEL

## Goal

Get a single real Sentinel-2 image through EDSR and produce a valid
output.

## Step 1 --- Environment

Create:

``` text
requirements.txt
```

Minimum stack:

``` text
fastapi
uvicorn
pydantic
python-multipart
numpy
pillow
torch
torchvision
rasterio
rio-tiler
requests
```

Add any exact libraries already required by the existing EDSR
implementation.

------------------------------------------------------------------------

## Step 2 --- Load the checkpoint

Create:

``` text
backend/inference/model.py
```

Responsibilities:

``` text
load_model()
get_device()
load_weights()
model_ready()
```

The model should load **once during application startup**, not once per
request.

------------------------------------------------------------------------

## Step 3 --- Confirm model contract

Document:

``` text
Input bands:
Input dtype:
Normalization:
Expected dimensions:
Scale factor:
Output bands:
Output range:
Checkpoint:
Device:
```

Do not leave any of these implicit.

------------------------------------------------------------------------

## Step 4 --- Implement patch-based inference

Large satellite images should not be passed through the model as one
giant tensor.

Use:

``` text
Image
 ↓
tile into patches
 ↓
model inference
 ↓
overlap / stitching
 ↓
full-resolution output
```

Use overlap to reduce visible seams.

------------------------------------------------------------------------

## Step 5 --- Preserve geospatial metadata

Input:

``` text
CRS
transform
width
height
bounds
dtype
```

Output must have the correct corresponding geospatial metadata.

The SR image must remain geographically aligned with the source.

------------------------------------------------------------------------

## Step 6 --- Run the first benchmark

Record:

``` text
Input dimensions
Patch size
Overlap
Number of patches
Inference time
Peak memory
Output dimensions
```

### Done condition

One real satellite scene produces:

``` text
valid input
   ↓
EDSR
   ↓
valid image
   ↓
valid GeoTIFF
```

Do not continue until this works.

------------------------------------------------------------------------

# PHASE 2 --- CREATE THE GUARANTEED DEMO DATASET

## Goal

Build a reliable demo that requires no live inference.

Choose **3--5 Bengaluru scenes**.

Recommended scene categories:

``` text
1. Dense urban
2. Water
3. Vegetation
4. Mixed urban
5. Optional peri-urban
```

------------------------------------------------------------------------

## Step 1 --- Prepare each scene

For each scene save:

``` text
input.tif
sr.tif
input_preview.jpg/png
sr_preview.jpg/png
metadata.json
metrics.json
downstream.png
```

Example:

``` text
demo/scenes/scene_01/
├── input.tif
├── sr.tif
├── input_preview.png
├── sr_preview.png
├── metadata.json
├── metrics.json
└── downstream.png
```

------------------------------------------------------------------------

## Step 2 --- Create metadata

Example:

``` json
{
  "id": "scene_01",
  "location": "Bengaluru",
  "date": "YYYY-MM-DD",
  "cloud_cover": 4.3,
  "model": "EDSR",
  "scale_factor": 2,
  "input_resolution": "10m",
  "output_resolution": "5m"
}
```

Only include values that are actually known.

------------------------------------------------------------------------

## Step 3 --- Benchmark metrics

If a valid high-resolution reference exists:

``` text
PSNR
SSIM
SAM
ERGAS
```

If it does not:

**Do not invent a score.**

Use a controlled evaluation workflow:

``` text
HR reference
     ↓
synthetic degradation
     ↓
LR
     ↓
EDSR
     ↓
SR
     ↓
compare with HR
```

------------------------------------------------------------------------

## Step 4 --- Create a scene manifest

``` text
demo/scenes/index.json
```

Example:

``` json
[
  {
    "id": "scene_01",
    "name": "Central Bengaluru",
    "category": "urban"
  },
  {
    "id": "scene_02",
    "name": "Bengaluru Water Network",
    "category": "water"
  }
]
```

### Done condition

Opening a scene locally immediately shows:

``` text
input
SR output
metadata
metrics
downstream result
```

No external service is required.

------------------------------------------------------------------------

# PHASE 3 --- BUILD THE BACKEND CORE

## Goal

Expose the guaranteed dataset through FastAPI.

------------------------------------------------------------------------

## Step 1 --- Health endpoints

Implement:

``` http
GET /health
GET /health/ready
```

Example:

``` json
{
  "status": "ok",
  "model_loaded": true
}
```

Readiness must indicate whether the model is actually ready.

------------------------------------------------------------------------

## Step 2 --- Scene endpoints

Implement:

``` http
GET /api/scenes
GET /api/scenes/{scene_id}
```

Return:

-   scene metadata
-   preview URLs
-   metric data
-   download URL

------------------------------------------------------------------------

## Step 3 --- Result endpoints

Implement:

``` http
GET /api/scenes/{scene_id}/input
GET /api/scenes/{scene_id}/sr
GET /api/scenes/{scene_id}/metrics
GET /api/scenes/{scene_id}/downstream
```

------------------------------------------------------------------------

## Step 4 --- Static/demo strategy

The demo API should never execute EDSR.

It should read precomputed artifacts.

This makes the primary judging path:

``` text
fast
stable
repeatable
```

------------------------------------------------------------------------

## Step 5 --- Test

Use:

``` text
curl
Postman
browser
```

Verify every endpoint before starting frontend integration.

### Done condition

The entire demo dataset is accessible through the API.

------------------------------------------------------------------------

# PHASE 4 --- BUILD THE FRONTEND VIEWER

## Goal

Make the project look like a real geospatial application.

------------------------------------------------------------------------

# Page 1 --- Landing

Show:

``` text
TerraResolve

AI-Powered Satellite Super Resolution
for Bengaluru
```

Main CTA:

``` text
Explore Bengaluru
```

Secondary CTA:

``` text
Upload GeoTIFF
```

------------------------------------------------------------------------

# Page 2 --- Viewer

The viewer is the most important UI.

Required:

``` text
scene selector
map
before/after slider
zoom
pan
metadata
metrics
download
```

------------------------------------------------------------------------

## Before/after implementation

Basic version:

``` text
┌─────────────────────────────┐
│      Original | SR          │
│             │               │
│             │               │
│─────────────┼───────────────│
│             │               │
└─────────────────────────────┘
              ▲
            slider
```

Advanced visual behavior:

-   synchronized zoom
-   synchronized pan
-   draggable divider
-   fullscreen
-   reset view
-   scale indicator
-   coordinates

------------------------------------------------------------------------

# Page 3 --- Metrics

Show:

``` text
PSNR
SSIM
SAM
ERGAS
```

Also show:

``` text
Input resolution
Output resolution
Scale factor
Model
Inference time
Scene date
```

------------------------------------------------------------------------

# Page 4 --- Upload

UI:

``` text
Drop GeoTIFF
     ↓
Validate
     ↓
Start processing
```

Show job status:

``` text
Queued
Processing
Completed
Failed
```

------------------------------------------------------------------------

# Page 5 --- Live

UI:

``` text
Bengaluru AOI
Date
Cloud threshold
[Fetch latest suitable scene]
```

This page can initially be disabled behind a feature flag.

### Done condition

A judge can explore the demo without seeing any broken backend
functionality.

------------------------------------------------------------------------

# PHASE 5 --- BUILD REAL ASYNC USER INFERENCE

## Goal

Prove the system is not merely a static gallery.

------------------------------------------------------------------------

# API Design

Implement:

``` http
POST /api/jobs
GET  /api/jobs/{id}
GET  /api/jobs/{id}/result
DELETE /api/jobs/{id}
```

------------------------------------------------------------------------

# Job lifecycle

``` text
QUEUED
  ↓
RUNNING
  ↓
PREPROCESSING
  ↓
INFERENCE
  ↓
POSTPROCESSING
  ↓
COMPLETED
```

Failure:

``` text
ANY STAGE
   ↓
FAILED
```

------------------------------------------------------------------------

# Job database

Minimum schema:

``` text
id
status
stage
progress
created_at
started_at
completed_at
input_path
output_path
error
model_version
```

------------------------------------------------------------------------

# Worker

Use one controlled worker.

Pseudo-architecture:

``` text
HTTP request
     ↓
create job record
     ↓
return job_id
     ↓
worker picks job
     ↓
process
     ↓
update database
     ↓
frontend polls status
```

Do not let the request thread perform the whole inference.

------------------------------------------------------------------------

# Progress reporting

Example:

``` text
QUEUED           0%
PREPROCESSING   10%
INFERENCE       20–85%
POSTPROCESSING  90%
COMPLETE       100%
```

These percentages must be deterministic enough to be meaningful.

------------------------------------------------------------------------

# Recovery logic

On backend restart:

``` text
RUNNING jobs
      ↓
mark interrupted
      ↓
optionally requeue
```

Completed jobs must not be lost.

------------------------------------------------------------------------

# Result flow

``` text
Completed job
      ↓
result.tif
      ↓
preview.png
      ↓
metrics.json
      ↓
frontend result page
```

### Done condition

A user can upload a valid GeoTIFF and eventually obtain a real SR
GeoTIFF.

------------------------------------------------------------------------

# PHASE 6 --- SCIENTIFIC VALIDATION

## Goal

Make the project defensible to technical judges.

------------------------------------------------------------------------

# Metric definitions

Document the evaluation protocol.

### PSNR

Measures pixel-level reconstruction fidelity.

### SSIM

Measures structural similarity.

### SAM

Measures spectral-angle difference.

### ERGAS

Measures relative global spectral reconstruction error.

Do not present these as generic "AI accuracy".

------------------------------------------------------------------------

# Evaluation protocol

Use:

``` text
Reference HR image
      ↓
controlled degradation
      ↓
LR input
      ↓
EDSR
      ↓
SR result
      ↓
metric calculation
```

Store:

``` text
metrics.json
```

Example:

``` json
{
  "psnr": 31.4,
  "ssim": 0.91,
  "sam": 4.7,
  "ergas": 8.2
}
```

Only display numbers actually produced by the evaluation.

------------------------------------------------------------------------

# Live/user imagery

For arbitrary user uploads:

``` text
No HR reference
      ↓
No valid reconstruction metric
```

Display:

> Quantitative reconstruction metrics require a reference image.

This is scientifically safer than fabricating a score.

------------------------------------------------------------------------

# PHASE 7 --- ONE DOWNSTREAM APPLICATION

## Goal

Show why super-resolution matters.

Choose **one** application.

Recommended:

``` text
Water-body analysis
```

Alternative:

``` text
Vegetation analysis
```

Only choose the alternative if it fits the available bands and existing
pipeline better.

------------------------------------------------------------------------

# Pipeline

``` text
Original image
       ↓
Downstream analysis
       ↓
Map/result

SR image
       ↓
Same downstream analysis
       ↓
Map/result

           ↓

Compare outputs
```

------------------------------------------------------------------------

# Example

For water detection:

``` text
Input imagery
    ↓
water index / classifier
    ↓
water mask

SR imagery
    ↓
same method
    ↓
SR water mask
```

Show:

``` text
detected area
boundary detail
difference
```

Do not claim SR "creates real water information."

Explain that the experiment evaluates whether increased spatial detail
improves the downstream representation under a controlled methodology.

------------------------------------------------------------------------

# PHASE 8 --- LIVE COPERNICUS PIPELINE

## Goal

Make TerraResolve capable of obtaining fresh Sentinel-2 data.

This is an enhancement to the guaranteed demo.

------------------------------------------------------------------------

# Step 1 --- Input

User selects:

``` text
AOI
date/date range
maximum cloud cover
```

For the hackathon, keep the AOI constrained to Bengaluru.

------------------------------------------------------------------------

# Step 2 --- Search

Use the Copernicus Data Space STAC interface.

Search using:

``` text
bbox
datetime
cloud cover
Sentinel-2 L2A
```

------------------------------------------------------------------------

# Step 3 --- Select scene

Choose the best suitable scene.

Basic ranking:

``` text
lower cloud cover
+
valid required assets
+
closest to requested date
```

------------------------------------------------------------------------

# Step 4 --- Download required assets

Retrieve only the bands required by the EDSR preprocessing pipeline.

Do not download an entire unnecessary product if the exact assets are
available separately.

------------------------------------------------------------------------

# Step 5 --- Preprocess

``` text
Copernicus data
      ↓
band alignment
      ↓
normalization
      ↓
crop
      ↓
patch generation
```

------------------------------------------------------------------------

# Step 6 --- EDSR

``` text
preprocessed Sentinel-2
          ↓
         EDSR
          ↓
          SR
```

------------------------------------------------------------------------

# Step 7 --- Output

Generate:

``` text
GeoTIFF
preview
metadata
job result
```

------------------------------------------------------------------------

# Failure handling

Any failure should produce:

``` text
Live acquisition unavailable.
Showing preloaded Bengaluru scenes instead.
```

Never allow the live integration to break the entire demo.

------------------------------------------------------------------------

# PHASE 9 --- SECURITY + RESOURCE PROTECTION

## GeoTIFF validation

Validate:

``` text
extension
MIME/content
file size
band count
width
height
pixel count
dtype
CRS
dimensions
```

Reject malformed or excessive inputs.

------------------------------------------------------------------------

# Rate limiting

Protect:

``` text
POST /api/jobs
POST /api/fetch
```

Example conceptual limit:

``` text
10 requests/minute/IP
```

Tune based on actual deployment constraints.

------------------------------------------------------------------------

# Filenames

Never use user-provided filenames directly as filesystem paths.

Use:

``` text
UUID
```

Example:

``` text
a7f6.../input.tif
```

------------------------------------------------------------------------

# CORS

Allow only the deployed frontend origin.

Do not use:

``` text
*
```

in the production deployment unless there is a deliberate reason.

------------------------------------------------------------------------

# Secrets

Store:

``` text
Copernicus credentials
Sentry DSN
storage credentials
other API secrets
```

in environment variables.

Never ship secrets to the frontend.

------------------------------------------------------------------------

# PHASE 10 --- DOCKERIZE

## Goal

Make the backend reproducible.

------------------------------------------------------------------------

# Docker image should contain

``` text
Python
PyTorch
FastAPI
rasterio/GDAL
rio-tiler
model weights
application code
```

------------------------------------------------------------------------

# Startup

Container startup should:

``` text
start FastAPI
       ↓
load model
       ↓
validate model
       ↓
mark readiness
```

The process should not accept inference requests before the model is
ready.

------------------------------------------------------------------------

# Health checks

Implement:

``` http
GET /health
GET /health/ready
```

Docker can use readiness/health behavior to identify unhealthy
instances.

------------------------------------------------------------------------

# PHASE 11 --- DEPLOY

## Frontend

Deploy:

``` text
frontend/
```

to Vercel.

Environment variable:

``` text
API_BASE_URL=https://<render-service>
```

------------------------------------------------------------------------

# Backend

Deploy Dockerized FastAPI to Render.

Environment variables:

``` text
MODEL_PATH
COPERNICUS_CLIENT_ID
COPERNICUS_CLIENT_SECRET
STORAGE_PATH
ALLOWED_ORIGIN
```

Use the actual names chosen in the project.

------------------------------------------------------------------------

# Production startup sequence

``` text
Render starts container
        ↓
Python starts
        ↓
Model loads
        ↓
/health = OK
        ↓
/health/ready = true
        ↓
frontend sends requests
```

------------------------------------------------------------------------

# Storage warning

Render's default filesystem is not durable across restarts/deploys.

Therefore:

### Safe to package in image

``` text
model
small demo assets
application code
```

### Should use persistent/object storage

``` text
user uploads
generated GeoTIFFs
large runtime artifacts
long-lived job results
```

For a no-budget or constrained hackathon deployment, keep the
precomputed demo path independent from runtime storage.

------------------------------------------------------------------------

# PHASE 12 --- TEST THE COMPLETE SYSTEM

Run the following test matrix.

## Test A --- Demo

``` text
Open website
↓
Select scene
↓
View before/after
↓
View metrics
↓
View downstream
↓
Download GeoTIFF
```

Must pass.

------------------------------------------------------------------------

## Test B --- Upload

``` text
Upload valid GeoTIFF
↓
Job created
↓
Job queued
↓
Job running
↓
Job complete
↓
Download result
```

Must pass.

------------------------------------------------------------------------

## Test C --- Bad file

``` text
Upload random file
```

Expected:

``` text
clean validation error
```

No server crash.

------------------------------------------------------------------------

## Test D --- Large file

Expected:

``` text
rejected cleanly
```

------------------------------------------------------------------------

## Test E --- Copernicus failure

Simulate API/network failure.

Expected:

``` text
graceful error
+
demo fallback
```

------------------------------------------------------------------------

## Test F --- Worker failure

Simulate model/inference exception.

Expected:

``` text
job = FAILED
error stored
server remains healthy
```

------------------------------------------------------------------------

## Test G --- Restart

Restart backend.

Verify:

``` text
model reloads
health becomes ready
completed demo remains available
job database behaves correctly
```

------------------------------------------------------------------------

# 6. Final Demo Flow

The presentation should follow this exact order.

------------------------------------------------------------------------

## 1. Problem

Show:

``` text
Traditional satellite imagery
↓
limited spatial detail
↓
harder fine-scale analysis
```

Then:

> TerraResolve uses deep-learning-based super-resolution to generate
> higher-resolution representations while preserving geospatial
> alignment.

------------------------------------------------------------------------

## 2. Bengaluru

Show the Bengaluru map.

Select an interesting scene.

------------------------------------------------------------------------

## 3. Before / After

Use the slider.

Zoom into:

``` text
roads
buildings
water edges
vegetation
urban boundaries
```

Do not zoom into areas with obvious hallucinations or artifacts.

------------------------------------------------------------------------

## 4. Explain the model

Briefly show:

``` text
Sentinel-2
↓
preprocessing
↓
patch-based EDSR
↓
reconstruction
↓
geospatial output
```

------------------------------------------------------------------------

## 5. Scientific validation

Show:

``` text
PSNR
SSIM
SAM
ERGAS
```

Immediately explain the reference-based evaluation protocol.

------------------------------------------------------------------------

## 6. Downstream application

Show:

``` text
Original
vs
SR
```

through the selected analysis.

------------------------------------------------------------------------

## 7. Prove it is live

Upload a GeoTIFF.

Show:

``` text
QUEUED
↓
RUNNING
↓
COMPLETE
```

Download the result.

------------------------------------------------------------------------

## 8. Optional live satellite acquisition

Demonstrate:

``` text
AOI
↓
Copernicus
↓
scene
↓
EDSR
↓
result
```

Only do this after the main story already works.

------------------------------------------------------------------------

# 7. 24-Hour Checklist

## Hour 0--2 --- ML Proof

-   [ ] Environment works
-   [ ] EDSR loads
-   [ ] Model checkpoint verified
-   [ ] One real image inferred
-   [ ] GeoTIFF output valid
-   [ ] inference benchmark recorded

------------------------------------------------------------------------

## Hour 2--4 --- Demo Dataset

-   [ ] 3--5 Bengaluru scenes selected
-   [ ] inputs processed
-   [ ] SR outputs generated
-   [ ] previews created
-   [ ] metadata created
-   [ ] valid benchmark metrics created

------------------------------------------------------------------------

## Hour 4--7 --- Backend

-   [ ] FastAPI running
-   [ ] health endpoint
-   [ ] readiness endpoint
-   [ ] scenes API
-   [ ] result API
-   [ ] demo assets served

------------------------------------------------------------------------

## Hour 7--10 --- Frontend

-   [ ] landing
-   [ ] viewer
-   [ ] slider
-   [ ] metadata
-   [ ] metrics
-   [ ] download
-   [ ] upload page
-   [ ] live page

------------------------------------------------------------------------

## Hour 10--13 --- Async Inference

-   [ ] job database
-   [ ] POST /api/jobs
-   [ ] worker
-   [ ] job status
-   [ ] progress
-   [ ] output generation
-   [ ] download result

------------------------------------------------------------------------

## Hour 13--15 --- Science

-   [ ] metric calculations verified
-   [ ] evaluation protocol documented
-   [ ] metrics UI polished
-   [ ] no invalid live-image claims

------------------------------------------------------------------------

## Hour 15--17 --- Downstream

-   [ ] one application
-   [ ] original result
-   [ ] SR result
-   [ ] comparison
-   [ ] visualization

------------------------------------------------------------------------

## Hour 17--19 --- Copernicus

-   [ ] STAC search
-   [ ] scene selection
-   [ ] required assets
-   [ ] preprocessing
-   [ ] live job
-   [ ] fallback

------------------------------------------------------------------------

## Hour 19--21 --- Deployment

-   [ ] Docker builds
-   [ ] Render deployed
-   [ ] Vercel deployed
-   [ ] CORS fixed
-   [ ] environment variables set
-   [ ] health checks pass

------------------------------------------------------------------------

## Hour 21--23 --- Hardening

-   [ ] bad file test
-   [ ] oversized file test
-   [ ] worker crash test
-   [ ] restart test
-   [ ] Copernicus failure test
-   [ ] logs checked
-   [ ] demo fallback checked

------------------------------------------------------------------------

## Hour 23--24 --- Rehearsal

Run the presentation exactly as judges will see it:

``` text
Landing
↓
Bengaluru
↓
Before/After
↓
Metrics
↓
Downstream
↓
Upload
↓
Async job
↓
Download
↓
Live Copernicus
↓
Architecture
↓
Future scaling
```

------------------------------------------------------------------------

# 8. MUST / SHOULD / OPTIONAL

## MUST

These define the successful submission:

-   [ ] Real EDSR inference
-   [ ] real Sentinel-2 data
-   [ ] 3--5 Bengaluru demo scenes
-   [ ] before/after viewer
-   [ ] correct GeoTIFF output
-   [ ] scientifically valid evaluation
-   [ ] one downstream application
-   [ ] async upload inference
-   [ ] Dockerized backend
-   [ ] deployed frontend
-   [ ] deployed backend
-   [ ] failure-safe demo

------------------------------------------------------------------------

## SHOULD

-   [ ] live Copernicus search
-   [ ] persistent job metadata
-   [ ] rate limiting
-   [ ] structured logs
-   [ ] polished progress UI
-   [ ] readiness health check
-   [ ] graceful job failure
-   [ ] clean README
-   [ ] automated smoke test

------------------------------------------------------------------------

## OPTIONAL

Only do these after everything above works:

-   [ ] GPU inference
-   [ ] object storage
-   [ ] result expiration
-   [ ] cancellation
-   [ ] multiple model comparison
-   [ ] advanced tile caching
-   [ ] PostGIS
-   [ ] Kubernetes prototype

------------------------------------------------------------------------

# 9. Future Production Architecture

This is how the hackathon architecture can evolve.

## Hackathon

``` text
Vercel
   ↓
FastAPI
   ↓
single worker
   ↓
PyTorch
   ↓
persistent/local storage
```

## Production

``` text
Vercel/CDN
      ↓
API
      ↓
Queue
      ↓
GPU workers
      ↓
Object Storage
      ↓
PostgreSQL/PostGIS
```

## Large-scale production

``` text
                  API Gateway
                       ↓
                 FastAPI services
                       ↓
                    Queue
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       GPU worker   GPU worker   GPU worker
          └────────────┼────────────┘
                       ▼
                Object Storage
                       +
                 PostGIS DB
                       +
                 Monitoring
```

Kubernetes becomes useful at this stage for:

-   worker orchestration
-   autoscaling
-   GPU scheduling
-   resource limits
-   health management
-   rolling deployments

It is **not** required to prove the hackathon concept.

------------------------------------------------------------------------

# 10. Key Technical Risks

## Risk 1 --- EDSR is too slow

Mitigation:

``` text
patch-based inference
+
small concurrency
+
precomputed demo
```

------------------------------------------------------------------------

## Risk 2 --- Out-of-memory

Mitigation:

``` text
limit patch size
limit upload dimensions
process sequentially
release tensors
```

------------------------------------------------------------------------

## Risk 3 --- Fake metrics

Mitigation:

Only calculate reconstruction metrics when a valid HR reference exists.

------------------------------------------------------------------------

## Risk 4 --- Hallucinated detail

Mitigation:

Use a conservative presentation.

Do not claim:

> The model discovered real objects.

Prefer:

> The model reconstructs a higher-resolution representation learned from
> the training distribution.

------------------------------------------------------------------------

## Risk 5 --- Copernicus outage

Mitigation:

``` text
live path
   ↓
failure
   ↓
cached demo
```

------------------------------------------------------------------------

## Risk 6 --- Server restart

Mitigation:

-   persistent job metadata
-   persistent storage where available
-   reproducible model startup
-   demo assets independent of runtime state

------------------------------------------------------------------------

## Risk 7 --- Upload abuse

Mitigation:

-   size limits
-   pixel limits
-   band limits
-   MIME/content validation
-   rate limiting
-   cleanup
-   UUID paths

------------------------------------------------------------------------

# 11. Final Definition of Done

TerraResolve is complete when a judge can perform this journey:

``` text
                    BENGALURU SCENE
                          ↓
                  SENTINEL-2 INPUT
                          ↓
                      EDSR MODEL
                          ↓
                  SUPER-RESOLVED
                          ↓
             ┌────────────┴────────────┐
             ▼                         ▼
       VISUAL COMPARISON         SCIENTIFIC TEST
             │                         │
             ▼                         ▼
      BEFORE / AFTER           PSNR / SSIM / SAM
                                      / ERGAS
             │                         │
             └────────────┬────────────┘
                          ▼
                  DOWNSTREAM ANALYSIS
                          ▼
                   GEOSPATIAL VALUE
```

Then:

``` text
USER UPLOAD
    ↓
ASYNC JOB
    ↓
EDSR
    ↓
GEOTIFF
    ↓
DOWNLOAD
```

Then, if stable:

``` text
BENGALURU AOI
    ↓
COPERNICUS STAC
    ↓
SENTINEL-2
    ↓
EDSR
    ↓
GEOTIFF
```

------------------------------------------------------------------------

# 12. The One Rule for the Team

> **Never sacrifice the working demo to add a more advanced
> technology.**

The winning implementation is:

``` text
Scientifically credible
        +
Visually obvious
        +
Technically real
        +
Failure tolerant
        +
Easy to explain
```

Not:

``` text
Maximum number of services
```

The architecture should look scalable because the boundaries are clean:

``` text
Frontend
   ↓
API
   ↓
Job system
   ↓
Inference
   ↓
Geospatial pipeline
   ↓
Storage
```

That is enough for the hackathon.
