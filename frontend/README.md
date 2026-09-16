# TerraResolve Frontend Design System

A modern, high-tech Earth Observation (EO) and geospatial super-resolution HUD interface.

## 🎨 Design Tokens (`styles.css`)

### 1. Brand Palette
- **Brand Green Primary:** `#0e5c3c` (`--brand-green-primary`)
- **Brand Green Dark:** `#09402a` (`--brand-green-dark`)
- **Brand Green Light:** `#e8f5ec` (`--brand-green-light`)
- **Page Background:** `#f8fafc` (`--bg-page`)
- **Surface Elevation:** `#ffffff` (`--bg-surface`)

### 2. Glassmorphism Badges
- `--glass-dark-pill`: `rgba(9, 24, 18, 0.78)` with `backdrop-filter: blur(12px)`
- `--glass-dark-border`: `rgba(255, 255, 255, 0.14)`

### 3. Typography
- **Headings & Main Typography:** `Plus Jakarta Sans`
- **Telemetry & Band Badges:** `JetBrains Mono`

---

## 🧩 UI Components & Pages

| Component | Class / ID | Description |
| :--- | :--- | :--- |
| **Top Navbar** | `.navbar`, `.status-pill` | Clean white header with logo, navigation links, and live status pill |
| **Hero Title** | `.hero-title` | Large bold typography for "AI-Powered Satellite Super-Resolution" |
| **Primary CTA** | `.btn-start` | Forest green glowing button for "Start Image Processing" |
| **Upload CTA** | `.btn-upload` | Clean border card button for "Upload GeoTIFF" |
| **Hero Visual Card** | `.hero-image-card` | Curved card hosting Yelahanka orbital imagery with top-right and bottom-left chips |
| **Explorer Modal** | `#explorerModal`, `#modalMap` | Interactive full-screen Leaflet super-resolution comparison |
