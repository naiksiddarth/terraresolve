  # TerraResolve — Product-Level Website Plan

**Project:** TerraResolve — Satellite Super-Resolution Platform  
**Competition:** SIH 2026, Problem Statement 26142 (NTRO)  
**Date:** September 2026  
**Status:** Planning → Execution

---

## 1. Executive Summary

TerraResolve is a deep-learning satellite super-resolution system that upscales Sentinel-2 imagery (10 m/pixel) to under 4 m/pixel using a trained EDSR neural network. This document outlines the plan to transform the existing local research pipeline into a fully deployed, publicly accessible web product focused on **Bengaluru, India** as the demonstration region.

---

## 2. Final Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Frontend hosting** | Vercel (static) | Free global CDN, instant deploys |
| **Backend hosting** | HuggingFace Spaces (Docker) | Free NVIDIA T4 GPU |
| **Inference engine** | PyTorch EDSR on CUDA 12.8 | 2–5s/tile vs 30–60s on CPU |
| **Authentication** | None — fully public | Open access for SIH demo |
| **Data scope** | Bengaluru only | Focused demo, manageable compute |
| **User upload** | Yes — browse pre-loaded + upload own GeoTIFFs | Interactive experience |
| **Containerisation** | Docker (pytorch/pytorch:2.11.0-cuda12.8) | Solves all CUDA/PyTorch install issues |
| **Future scaling** | Kubernetes (HPA) | Auto-scale GPU pods under traffic |

---

## 3. System Architecture

### 3.1 Current Architecture

`
USER BROWSER
  |-- loads pages --> VERCEL (Static Frontend)
  |                   landing.html, viewer.html, upload.html, fetch.html, metrics.html
  |
  |-- API calls ----> HUGGINGFACE SPACES (Docker Container, GPU)
                      pytorch/pytorch:2.11.0-cuda12.8
                      FastAPI + PyTorch EDSR + rasterio + NVIDIA T4
`

### 3.2 Future: Kubernetes Scaling

`
VERCEL --> Load Balancer (K8s Ingress)
              |-- GPU Pod 1 (Docker + EDSR)
              |-- GPU Pod 2 (Docker + EDSR)
              |-- GPU Pod N (auto-scaled by HPA)
`

### 3.3 Bengaluru Bounding Box
- Coordinates: 77.45E, 12.85N → 77.75E, 13.10N
- Pre-fetched tiles: 10–20 Sentinel-2 L2A scenes
- All tiles pre-inferred before demo for instant display

---

## 4. Build Phases

### Phase 1 — Landing Page & Branding (1 day)
- webapp/static/landing.html
- Hero: "Bengaluru from Space — Super-Resolved"
- Animated before/after comparison, feature cards, CTA button
- Google Fonts: Outfit + Inter

### Phase 2 — SR Viewer Overhaul (1–2 days)
- webapp/static/viewer.html
- Map locked to Bengaluru bbox
- Before/after slider + keyboard scrub (arrow keys)
- Metric overlay: PSNR / SSIM / SAM / ERGAS
- Download SR GeoTIFF button

### Phase 3 — Upload & On-Demand SR (1–2 days)
- POST /api/upload — accept GeoTIFF ≤50MB
- GET  /api/jobs/{id} — poll status
- GET  /api/jobs/{id}/download — stream result
- ThreadPoolExecutor for background inference (no Celery/Redis)

### Phase 4 — Copernicus Live Fetch UI (1–2 days)
- webapp/static/fetch.html
- Fixed Bengaluru AOI, date picker, cloud cover slider
- Live step progress: Auth → Search → Download → Stack → SR → Done

### Phase 5 — Metrics Dashboard (1 day)
- webapp/static/metrics.html
- PSNR/SSIM/ERGAS/SAM score cards
- EDSR vs SwinIR bar chart (Chart.js)
- Training loss curve, NDWI improvement table

### Phase 6 — Backend Hardening (1 day)
- Rate limiting: 10 uploads/min per IP
- CORS: Vercel domain only
- Health check endpoint GET /health
- Disk cleanup every hour

### Phase 7 — Docker + Deployment (1 day)
- Dockerfile using pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
- vercel.json: proxy /api/* to HuggingFace Spaces
- GitHub Actions CI smoke test
- Deploy: HF Spaces (GPU backend) + Vercel (frontend)

### Phase 8 — Polish & SEO (1 day)
- Meta tags, OG image, PWA manifest
- Loading skeletons, 404 page, consistent navbar

---

## 5. Full Task Checklist

### Frontend (Vercel)
- [ ] landing.html — hero, features, CTA
- [ ] viewer.html — SR map, Bengaluru bbox, metrics overlay
- [ ] upload.html — drag-and-drop + job progress
- [ ] fetch.html — Copernicus fetch with live steps
- [ ] metrics.html — charts + model comparison
- [ ] 404 page, footer, PWA manifest, OG image

### Backend API (HuggingFace Spaces)
- [ ] POST /api/upload
- [ ] GET /api/jobs/{id}
- [ ] GET /api/jobs/{id}/download
- [ ] POST /api/fetch
- [ ] GET /api/metrics/{tile_id}
- [ ] GET /api/tiles
- [ ] GET /health
- [ ] Rate limiting, CORS, env vars, disk cleanup

### ML / Inference
- [ ] Pre-fetch 10–20 Bengaluru tiles via fetch_copernicus.py
- [ ] Cache SR GeoTIFFs for all tiles (demo-ready)
- [ ] ThreadPoolExecutor background jobs
- [ ] Store metrics per tile (PSNR/SSIM/ERGAS/SAM)

### Docker & Deployment
- [ ] Dockerfile (pytorch CUDA base image)
- [ ] vercel.json (static + API proxy)
- [ ] GitHub Actions CI
- [ ] HF Spaces GPU T4 deployment
- [ ] Vercel static deployment

### Monitoring
- [ ] Sentry free tier (error tracking)
- [ ] UptimeRobot free (ping /health every 5min)

### Future: Kubernetes
- [ ] k8s/deployment.yaml
- [ ] k8s/service.yaml
- [ ] k8s/hpa.yaml (min 1, max 5 GPU pods)

---

## 6. Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5 + Vanilla JS + CSS |
| Typography | Outfit + Inter (Google Fonts) |
| Map | Leaflet.js + leaflet-side-by-side |
| Charts | Chart.js |
| Backend | FastAPI (Python 3.13) |
| Inference | PyTorch 2.11 + CUDA 12.8 (EDSR) |
| GeoTIFF | rasterio + rio-tiler |
| Container | Docker (pytorch/pytorch:2.11.0-cuda12.8) |
| Frontend host | Vercel (free CDN) |
| Backend host | HuggingFace Spaces (free T4 GPU) |
| CI | GitHub Actions |
| Monitoring | Sentry + UptimeRobot |
| Future scaling | Kubernetes + HPA |

---

## 7. SIH Demo Strategy

Pre-run SR inference on all Bengaluru tiles before the presentation.
Cache results to disk. Demo will feel instant.

Demo flow (recommended ~4 minutes):
1. Landing page — branding and feature overview (30s)
2. Viewer — click pre-cached tiles, drag slider (60s)
3. Metrics — show PSNR/SSIM/ERGAS/SAM numbers (30s)
4. Fetch page — live Copernicus scene fetch (60s)
5. Upload — custom GeoTIFF with GPU inference progress (60s)

---

*Maintained by the TerraResolve team. Update as decisions change.*
