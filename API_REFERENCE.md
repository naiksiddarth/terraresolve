# TerraResolve — API Reference

> **Intended audience**: AI agents and developers building Android or Web frontends against the TerraResolve FastAPI backend.
> **Backend entry point**: `python webapp/main.py`
> **Base URL**: User-provided at runtime — **never hardcode a URL** (see [Base URL Configuration](#base-url-configuration))
> **Framework**: FastAPI (Python) — auto-generates interactive docs at `/docs` and `/redoc`

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Base URL Configuration](#base-url-configuration) ← **read this first**
4. [Server Setup](#server-setup)
5. [Data Models](#data-models)
6. [API Endpoints](#api-endpoints)
   - [GET /api/tiles](#get-apitiles)
   - [GET /api/model](#get-apimodel)
   - [POST /api/model](#post-apimodel)
   - [GET /tiles/{source}/{tile_id}/{z}/{x}/{y}.png](#get-tilessourcetile_idzxypng)
7. [Static File Serving](#static-file-serving)
8. [Integration Guide — Web Client (JavaScript)](#integration-guide--web-client-javascript)
9. [Integration Guide — Android Client (Kotlin/Retrofit)](#integration-guide--android-client-kotlinretrofit)
10. [Error Reference](#error-reference)
11. [Key Constants & Conventions](#key-constants--conventions)

---

## Overview

TerraResolve is a deep-learning satellite super-resolution system. It takes low-resolution (LR)
Sentinel-2 L2A GeoTIFF imagery (10 m/px, 4 bands: Blue, Green, Red, NIR) and produces
super-resolved (SR) output at 4× resolution (~2.5 m/px) using a trained EDSR model. The backend:

- Loads a trained EDSR checkpoint automatically on startup
- Serves real LR / SR / HR imagery as **standard XYZ / slippy-map PNG tiles** — the same tile
  scheme used by Leaflet, Google Maps, and Mapbox
- Allows hot-swapping of checkpoints without restarting the server
- Caches SR inference results in memory and on disk to avoid re-running the model on every tile
  request

---

## Base URL Configuration

> **🚨 RULE FOR ALL FRONTEND AGENTS**: Never hardcode a base URL anywhere in the app.
> The server can run on any host and port (developer laptop, LAN server, cloud VM).
> The user must always supply the base URL at runtime.

### UX Flow

```
App launch
    │
    ├─ [No saved URL] ──► Show "Enter Server URL" prompt
    │                         User types e.g. http://192.168.1.42:8000
    │                         Validate by calling GET {url}/api/model
    │                         On success → save URL → proceed to app
    │                         On failure → show error, let user retry
    │
    └─ [Saved URL found] ──► Show dialog:
                              "Use saved server: http://192.168.1.42:8000?"
                              [Use Saved]  →  validate & proceed
                              [Enter New]  →  show URL prompt (same as above)
```

### Validation

After the user enters a URL, **always validate it** before proceeding:

1. `GET {baseUrl}/api/model`
2. If response is `200 OK` with `{ loaded, scale }` → URL is valid, save it and continue
3. If network error / non-200 → show error: `"Cannot reach server at {baseUrl}. Check the URL and that the server is running."`

### Web Client — `localStorage`

```javascript
// ─── Base URL management ─────────────────────────────────────────────────────
const STORAGE_KEY = 'terraresolve_base_url';

/** Returns a validated base URL, prompting the user as needed. */
async function resolveBaseUrl() {
  const saved = localStorage.getItem(STORAGE_KEY);

  if (saved) {
    // Offer to reuse or replace
    const useSaved = await showDialog({
      title: 'Server Connection',
      message: `Use saved server?\n${saved}`,
      confirmLabel: 'Use Saved',
      cancelLabel: 'Enter New',
    });
    if (useSaved) {
      if (await validateBaseUrl(saved)) return saved;
      showError(`Cannot reach ${saved}. Please enter a new URL.`);
    }
  }

  // Prompt for new URL
  return await promptAndSaveBaseUrl();
}

async function promptAndSaveBaseUrl() {
  while (true) {
    const url = await showInputDialog({
      title: 'Enter Server URL',
      placeholder: 'http://192.168.1.42:8000',
      hint: 'Include protocol and port. No trailing slash.',
    });
    if (!url) continue;                          // user dismissed without input
    const clean = url.replace(/\/$/, '');        // strip trailing slash
    if (await validateBaseUrl(clean)) {
      localStorage.setItem(STORAGE_KEY, clean);
      return clean;
    }
    showError(`Cannot reach server at ${clean}. Check the URL and try again.`);
  }
}

async function validateBaseUrl(url) {
  try {
    const res = await fetch(`${url}/api/model`, { signal: AbortSignal.timeout(5000) });
    return res.ok;
  } catch {
    return false;
  }
}

// ─── App bootstrap ────────────────────────────────────────────────────────────
(async () => {
  const BASE_URL = await resolveBaseUrl();   // ← use this everywhere, never a literal string
  await initApp(BASE_URL);
})();
```

### Android Client — `SharedPreferences`

```kotlin
// ─── Base URL management ─────────────────────────────────────────────────────
const val PREFS_NAME   = "terraresolve_prefs"
const val KEY_BASE_URL = "base_url"

object UrlPrefs {
    fun getSaved(context: Context): String? =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
               .getString(KEY_BASE_URL, null)

    fun save(context: Context, url: String) =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
               .edit().putString(KEY_BASE_URL, url.trimEnd('/')).apply()

    fun clear(context: Context) =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
               .edit().remove(KEY_BASE_URL).apply()
}

// ─── Validation ───────────────────────────────────────────────────────────────
/** Returns true if the server at [url] responds to GET /api/model with 200. */
suspend fun validateBaseUrl(url: String): Boolean = withContext(Dispatchers.IO) {
    try {
        val response = OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(5, TimeUnit.SECONDS)
            .build()
            .newCall(Request.Builder().url("${url.trimEnd('/')}/api/model").build())
            .execute()
        response.isSuccessful
    } catch (e: Exception) {
        false
    }
}

// ─── ViewModel ────────────────────────────────────────────────────────────────
class ServerSetupViewModel(application: Application) : AndroidViewModel(application) {

    private val _baseUrl = MutableStateFlow<String?>(UrlPrefs.getSaved(application))
    val baseUrl: StateFlow<String?> = _baseUrl

    val hasSavedUrl: Boolean get() = _baseUrl.value != null

    /** Call when user confirms they want to use the saved URL. */
    fun useSaved(onResult: (Boolean) -> Unit) {
        viewModelScope.launch {
            val valid = validateBaseUrl(_baseUrl.value!!)
            if (!valid) UrlPrefs.clear(getApplication())
            onResult(valid)
        }
    }

    /** Call with a URL the user typed. Saves it if valid. */
    fun submitNew(url: String, onResult: (Boolean) -> Unit) {
        viewModelScope.launch {
            val clean = url.trimEnd('/')
            val valid = validateBaseUrl(clean)
            if (valid) {
                UrlPrefs.save(getApplication(), clean)
                _baseUrl.value = clean
            }
            onResult(valid)
        }
    }
}

// ─── ApiClient — rebuilt whenever BASE_URL changes ────────────────────────────
object ApiClient {
    private var _baseUrl: String = ""
    lateinit var api: TerraResolveApi

    /** Call once after resolveBaseUrl() succeeds, before any API calls. */
    fun init(baseUrl: String) {
        if (baseUrl == _baseUrl) return
        _baseUrl = baseUrl
        api = Retrofit.Builder()
            .baseUrl("$baseUrl/")
            .client(
                OkHttpClient.Builder()
                    .connectTimeout(10, TimeUnit.SECONDS)
                    .readTimeout(90, TimeUnit.SECONDS)   // SR tiles can be slow
                    .addInterceptor(HttpLoggingInterceptor().apply {
                        level = HttpLoggingInterceptor.Level.BASIC
                    })
                    .build()
            )
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(TerraResolveApi::class.java)
    }
}
```

### Tile URLs after the base URL is known

The XYZ tile URL is always constructed at runtime from the resolved `baseUrl`:

```javascript
// Web
function tileUrl(source, tileId) {
  return `${BASE_URL}/tiles/${source}/${tileId}/{z}/{x}/{y}.png`;
}
```

```kotlin
// Android
fun tileUrl(source: String, tileId: String) =
    "${ApiClient.baseUrl}/tiles/$source/$tileId/{z}/{x}/{y}.png"
```

---

## Architecture & Data Flow

```
Client (Web / Android)
        │
        │  1. GET /api/tiles                            → list available scenes + bounding boxes
        │  2. GET /api/model                            → check which checkpoint is loaded
        │  3. GET /tiles/lr/{id}/{z}/{x}/{y}.png        → original Sentinel-2 tile
        │  4. GET /tiles/sr/{id}/{z}/{x}/{y}.png        → EDSR super-resolved tile  ← model output
        │  5. GET /tiles/hr/{id}/{z}/{x}/{y}.png        → high-res reference tile
        │
        ▼
   FastAPI Backend  (webapp/main.py)
        │
        ├── data/sen2naip/lr/     ← LR GeoTIFF inputs  (Sentinel-2 L2A, uint16 / 10 000)
        ├── data/sen2naip/hr/     ← HR GeoTIFF targets (NAIP-derived, uint8 or float32)
        ├── data/_sr_cache/       ← SR GeoTIFFs written to disk on first inference per tile_id
        └── checkpoints/          ← PyTorch model checkpoints (.pt)
```

**Tile URL pattern** — identical to OpenStreetMap / Google Maps XYZ slippy-map scheme:

```
/tiles/{source}/{tile_id}/{z}/{x}/{y}.png
```

| Segment | Meaning |
|---------|---------|
| `source` | `lr` \| `sr` \| `hr` |
| `tile_id` | File stem of the source GeoTIFF (e.g. `scene_001`) |
| `z` / `x` / `y` | Standard Web Mercator tile coordinates (EPSG:3857) |

---

## Server Setup

### Dependencies

```bash
# Core pipeline deps (already in requirements.txt):
#   torch, numpy, rasterio, pillow

# Viewer-only extra deps:
pip install fastapi uvicorn mercantile rio-tiler
```

### Run

```bash
python webapp/main.py
# Server listens on http://127.0.0.1:8000
```

### Startup Behaviour

On startup the server attempts to auto-load:

- **Config**: `configs/sen2naip.yaml`
- **Checkpoint**: `checkpoints/edsr_epoch50.pt`

If either file is absent the server starts normally, but `GET /tiles/sr/...` returns **503** until
a checkpoint is loaded via `POST /api/model`.

### CORS (for Web Clients on a Different Origin)

FastAPI does **not** add CORS headers by default. Add the middleware in `webapp/main.py` if your
web frontend runs on a different origin (e.g. `localhost:3000`):

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Data Models

### TileInfo — returned by `GET /api/tiles`

```jsonc
{
  "id": "scene_001",                        // string — unique scene ID (GeoTIFF file stem)
  "bbox": [77.55, 13.05, 77.65, 13.15],    // [west, south, east, north] — WGS84 degrees
  "center": [13.10, 77.60]                  // [latitude, longitude]  ← NOTE: lat first
}
```

> **⚠️ Coordinate order**:
> - `bbox` is `[west, south, east, north]` (GeoJSON / standard WGS84 order).
> - `center` is `[lat, lon]` (Leaflet convention). Keep this distinction when constructing map
>   bounds on the client.

### ModelInfo — returned by `GET /api/model`

```jsonc
{
  "loaded": "edsr::edsr_epoch50.pt",  // string | null  — "{model_name}::{checkpoint_filename}"
  "scale": 4                           // integer — upscaling factor (4 for EDSR default)
}
```

### Tile Image — returned by tile endpoints

| Property | Value |
|----------|-------|
| Content-Type | `image/png` |
| Dimensions | 256 × 256 pixels |
| Colour space | RGB, 8-bit per channel |
| Band mapping | GeoTIFF bands 3, 2, 1 → PNG R, G, B (true-colour: Red, Green, Blue from Sentinel-2) |
| Brightness stretch | Shared 1–99 percentile linear stretch from the HR reference — all three sources (LR / SR / HR) use the **same** stretch so brightness is directly comparable |

---

## API Endpoints

---

### `GET /api/tiles`

List all available real LR/HR scene pairs with geographic metadata for map display.

#### Request

```
GET /api/tiles
```

No query parameters.

#### Response — 200 OK

```json
[
  {
    "id": "scene_001",
    "bbox": [77.55, 13.05, 77.65, 13.15],
    "center": [13.10, 77.60]
  },
  {
    "id": "scene_002",
    "bbox": [77.20, 12.90, 77.30, 13.00],
    "center": [12.95, 77.25]
  }
]
```

Returns an **empty array `[]`** (not an error) when:
- `data/sen2naip/lr/` does not exist
- No `.tif` files are present
- A LR tile has no matching HR counterpart in `data/sen2naip/hr/`

#### Client Notes

- Fetch once on app launch; cache for the session.
- Use `bbox` to call `map.fitBounds()`.
- Use `id` in every subsequent tile URL.
- Tile IDs are **stable across model swaps** (they are GeoTIFF file stems, not model outputs).

---

### `GET /api/model`

Get information about the currently loaded model checkpoint.

#### Request

```
GET /api/model
```

No parameters.

#### Response — 200 OK (model loaded)

```json
{
  "loaded": "edsr::edsr_epoch50.pt",
  "scale": 4
}
```

#### Response — 200 OK (no model loaded)

```json
{
  "loaded": null,
  "scale": 4
}
```

#### Client Notes

- Check `loaded !== null` before enabling the SR layer in the UI.
- Display the model name as a status indicator (e.g. "EDSR ×4 | edsr_epoch50.pt").
- If `null`, show a banner: "Model not ready — SR tiles unavailable."

---

### `POST /api/model`

Hot-swap the model checkpoint without restarting the server. **Clears the SR tile cache.**

#### Request

```
POST /api/model?config={config_path}&checkpoint={checkpoint_path}
```

**Query Parameters** (both required):

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `config` | string | `configs/sen2naip.yaml` | Path to YAML config (server-side filesystem path) |
| `checkpoint` | string | `checkpoints/edsr_epoch50.pt` | Path to `.pt` checkpoint (server-side filesystem path) |

#### Response — 200 OK

```json
{
  "loaded": "edsr::edsr_epoch50.pt"
}
```

#### Response — 400 Bad Request

```json
{
  "detail": "Descriptive error message"
}
```

#### Client Notes

- Paths are **server-side filesystem paths** — the client cannot upload a file here, only trigger
  a swap between checkpoints already present on the server.
- After a successful swap, re-request SR tiles — the in-memory and on-disk cache is cleared
  automatically.
- Useful for A/B comparison of training epochs (e.g. `edsr_epoch10.pt` vs `edsr_epoch50.pt`).

---

### `GET /tiles/{source}/{tile_id}/{z}/{x}/{y}.png`

Fetch a single 256 × 256 PNG map tile for a given scene and image source. This is the **core
endpoint** consumed by Leaflet, Google Maps, Mapbox GL, or any XYZ-tile-compatible map SDK.

#### Request

```
GET /tiles/{source}/{tile_id}/{z}/{x}/{y}.png
```

**Path Parameters**:

| Parameter | Type | Allowed values | Description |
|-----------|------|----------------|-------------|
| `source` | string | `lr`, `sr`, `hr` | Which image source to render |
| `tile_id` | string | Any ID from `/api/tiles` | Which scene |
| `z` | integer | 0 – 22 | Zoom level (Web Mercator) |
| `x` | integer | — | Tile column |
| `y` | integer | — | Tile row |

**Source semantics**:

| Source | Full name | Resolution | Description |
|--------|-----------|------------|-------------|
| `lr` | Low Resolution (Original) | ~10 m/px | Raw Sentinel-2 L2A input. Normalised from uint16 ÷ 10 000. |
| `sr` | Super-Resolved | ~2.5 m/px | EDSR model output at 4× upscale. First request triggers inference + disk cache write. |
| `hr` | High Resolution (Reference) | ~2.5 m/px | NAIP-derived ground-truth reference for visual quality benchmarking. |

#### Response — 200 OK

```
Content-Type: image/png
Body: 256 × 256 8-bit RGB PNG tile bytes
```

#### Response — 400 Bad Request

```json
{ "detail": "source must be lr, sr, or hr" }
```

#### Response — 404 Not Found

```json
{ "detail": "Unknown tile id 'bad_id'" }
```

#### Response — 503 Service Unavailable

```json
{ "detail": "No model loaded yet -- POST /api/model first" }
```

#### Performance

| Source | First request | Subsequent requests |
|--------|--------------|---------------------|
| `lr` | Fast (< 1 s) | Fast |
| `hr` | Fast (< 1 s) | Fast |
| `sr` | **Slow (10 – 60 s)** — runs full model inference, writes SR GeoTIFF to `data/_sr_cache/` | Fast — served from cached file |

> **UI Implication**: Always show a loading indicator when the SR layer is first added. Listen to
> the tile layer's `loading` / `load` events and display a spinner or progress message like
> "Running super-resolution…" until tiles arrive.

---

## Static File Serving

```
GET /   →  webapp/static/index.html   (Leaflet side-by-side SR comparison viewer)
```

The backend also serves a fully working Leaflet viewer as a single-file HTML page
(`webapp/static/index.html`). This is the **canonical reference implementation** — it demonstrates
how to:

- Call `/api/tiles` to populate a scene dropdown
- Call `/api/model` to display model status
- Build `/tiles/{source}/{id}/{z}/{x}/{y}.png` XYZ tile URLs
- Wire two tile layers into `leaflet-side-by-side` for drag-to-compare

Study this file before building any new frontend.

---

## Integration Guide — Web Client (JavaScript)

> All examples below use `BASE_URL` as a variable resolved at runtime via `resolveBaseUrl()` (see [Base URL Configuration](#base-url-configuration)). **Never substitute a hardcoded string.**

### 1. Check model status on load

```javascript
async function checkModel() {
  const res = await fetch(`${BASE_URL}/api/model`);
  const data = await res.json();
  // data = { loaded: "edsr::edsr_epoch50.pt", scale: 4 }
  //      or { loaded: null, scale: 4 }
  if (!data.loaded) {
    showBanner('Model not loaded — SR tiles unavailable');
  }
  return data;
}
```

### 2. Fetch scene list and fit map

```javascript
async function loadScenes(map) {
  const res = await fetch(`${BASE_URL}/api/tiles`);
  const tiles = await res.json();
  // tiles = [{ id, bbox: [W,S,E,N], center: [lat, lon] }, ...]

  tiles.forEach(tile => {
    const [west, south, east, north] = tile.bbox;
    // fit map to first scene:
    map.fitBounds([[south, west], [north, east]], { maxZoom: 18, padding: [20, 20] });
  });

  return tiles;
}
```

### 3. Add XYZ tile layers to Leaflet

```javascript
function makeTileLayer(source, tileId) {
  return L.tileLayer(
    `${BASE_URL}/tiles/${source}/${tileId}/{z}/{x}/{y}.png`,
    {
      tileSize: 256,
      minZoom: 10,
      maxNativeZoom: 19,
      maxZoom: 22,
    }
  );
}

// Side-by-side split panel (SR left, LR right)
const leftLayer  = makeTileLayer('sr', tileId).addTo(map);
const rightLayer = makeTileLayer('lr', tileId).addTo(map);
const sbs = L.control.sideBySide(leftLayer, rightLayer).addTo(map);
```

### 4. Show loading indicator for slow SR tiles

```javascript
function makeSrLayerWithSpinner(tileId) {
  const layer = makeTileLayer('sr', tileId);
  layer.on('loading', () => showSpinner('Running super-resolution inference…'));
  layer.on('load',    () => hideSpinner());
  layer.on('tileerror', (e) => {
    hideSpinner();
    if (e.tile?.src?.includes('/sr/')) showError('SR tiles unavailable — check model status');
  });
  return layer;
}
```

### 5. Swap model checkpoint

```javascript
async function swapModel(configPath, checkpointPath) {
  const params = new URLSearchParams({ config: configPath, checkpoint: checkpointPath });
  const res = await fetch(`${BASE_URL}/api/model?${params}`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail);
  }
  return await res.json(); // { loaded: "edsr::new_checkpoint.pt" }
}
```

### 6. Full minimal bootstrap

```javascript
// BASE_URL is NEVER hardcoded — always resolved from user input + localStorage
(async () => {
  const BASE_URL = await resolveBaseUrl();   // shows prompt or reuse-dialog as needed

  const map = L.map('map', { minZoom: 4, maxZoom: 22 }).setView([20.5, 78.9], 5);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap © CARTO', maxZoom: 20, opacity: 0.35
  }).addTo(map);

  const model = await checkModel(BASE_URL);
  const tiles  = await loadScenes(BASE_URL, map);
  if (tiles.length && model.loaded) {
    const tileId = tiles[0].id;
    makeSrLayerWithSpinner(BASE_URL, tileId).addTo(map);
    makeTileLayer(BASE_URL, 'lr', tileId).addTo(map);
  }
})();
```

---

## Integration Guide — Android Client (Kotlin/Retrofit)

### 1. `build.gradle.kts` dependencies

```kotlin
// Networking
implementation("com.squareup.retrofit2:retrofit:2.9.0")
implementation("com.squareup.retrofit2:converter-gson:2.9.0")
implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

// Map SDK — MapLibre (open-source, recommended; no API key needed)
implementation("org.maplibre.gl:android-sdk:10.3.0")

// OR — Google Maps
// implementation("com.google.android.gms:play-services-maps:18.2.0")
```

### 2. Data classes

```kotlin
data class TileInfo(
    val id: String,
    val bbox: List<Double>,    // [west, south, east, north]
    val center: List<Double>   // [latitude, longitude]
)

data class ModelInfo(
    val loaded: String?,  // null if no checkpoint is loaded
    val scale: Int        // typically 4
)

data class ModelSwapResponse(
    val loaded: String
)
```

### 3. Retrofit API interface

```kotlin
import retrofit2.Response
import retrofit2.http.*

interface TerraResolveApi {

    @GET("api/tiles")
    suspend fun getTiles(): Response<List<TileInfo>>

    @GET("api/model")
    suspend fun getModel(): Response<ModelInfo>

    @POST("api/model")
    suspend fun setModel(
        @Query("config")     config: String,
        @Query("checkpoint") checkpoint: String
    ): Response<ModelSwapResponse>

    // Tile PNG images are fetched directly by the map SDK via URL — not through Retrofit.
}
```

### 4. Retrofit client

> The `ApiClient` is initialised with the URL resolved at runtime — see `ApiClient.init(baseUrl)`
> in [Base URL Configuration](#base-url-configuration). **Never set a hardcoded `BASE_URL`
> constant here.** Call `ApiClient.init(resolvedUrl)` once after the user confirms the server
> URL, then use `ApiClient.api` for all subsequent calls.

```kotlin
// ApiClient.init(baseUrl) is defined in the Base URL Configuration section above.
// It builds a Retrofit instance with the user-supplied URL and 90 s read timeout.
// Example call order in your Activity/Fragment:
//
//   viewModel.useSaved { valid ->
//       if (valid) {
//           ApiClient.init(UrlPrefs.getSaved(this)!!)
//           startMainActivity()
//       } else showErrorAndPromptNew()
//   }
```

### 5. ViewModel

```kotlin
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class MapViewModel : ViewModel() {
    private val _tiles     = MutableStateFlow<List<TileInfo>>(emptyList())
    val tiles: StateFlow<List<TileInfo>> = _tiles

    private val _modelInfo = MutableStateFlow<ModelInfo?>(null)
    val modelInfo: StateFlow<ModelInfo?> = _modelInfo

    fun init() {
        viewModelScope.launch {
            val tilesResult = ApiClient.api.getTiles()
            if (tilesResult.isSuccessful) {
                _tiles.value = tilesResult.body() ?: emptyList()
            }

            val modelResult = ApiClient.api.getModel()
            if (modelResult.isSuccessful) {
                _modelInfo.value = modelResult.body()
            }
        }
    }

    fun swapModel(config: String, checkpoint: String) {
        viewModelScope.launch {
            val result = ApiClient.api.setModel(config, checkpoint)
            if (result.isSuccessful) {
                // Refresh model info and clear any cached SR layers on the client
                _modelInfo.value = ModelInfo(result.body()?.loaded, _modelInfo.value?.scale ?: 4)
            }
        }
    }
}
```

### 6. Tile URL construction for map SDKs

The XYZ tile URL works directly with MapLibre, Google Maps, and OSMDroid.
Use the runtime-resolved `baseUrl` — **never a hardcoded string**:

```kotlin
// baseUrl comes from ApiClient after ApiClient.init(resolvedUrl) is called
fun tileUrl(source: String, tileId: String): String {
    val base = ApiClient.baseUrl   // property added to ApiClient, mirrors _baseUrl
    return "$base/tiles/$source/$tileId/{z}/{x}/{y}.png"
}
```

**MapLibre GL Android — add as a raster layer:**

```kotlin
fun addTileLayer(map: MapLibreMap, source: String, tileId: String, layerId: String) {
    val sourceId = "$layerId-source"
    map.style?.addSource(
        RasterSource(
            sourceId,
            TileSet("tileset", tileUrl(source, tileId)).apply {
                setMinZoom(10f)
                setMaxZoom(19f)
            },
            256
        )
    )
    map.style?.addLayer(RasterLayer(layerId, sourceId))
}

// Usage
addTileLayer(map, "sr", tileId, "sr-layer")
addTileLayer(map, "lr", tileId, "lr-layer")
```

### 7. Fit camera to a scene bounding box

```kotlin
fun fitToScene(tile: TileInfo, map: MapLibreMap) {
    val (west, south, east, north) = tile.bbox
    val bounds = LatLngBounds.from(north, east, south, west)
    map.animateCamera(
        CameraUpdateFactory.newLatLngBounds(bounds, 80 /* padding dp */)
    )
}
```

### 8. Handle slow SR tile loading

First SR tile for a `tile_id` triggers model inference (10–60 s). Show a spinner:

```kotlin
// Show spinner when SR layer is added
binding.progressBar.visibility = View.VISIBLE
binding.srStatusText.text = "Running super-resolution…"

// In MapLibre, listen for map idle (all tiles loaded):
map.addOnMapIdleListener {
    binding.progressBar.visibility = View.GONE
}

// For tile-level errors (e.g. 503 model not loaded):
map.addOnMapLoadingFinishedListener {
    // Check if SR layer has visible tiles; if not, show error banner
}
```

---

## Error Reference

| HTTP Status | Endpoint(s) | Cause | Recommended Client Action |
|-------------|-------------|-------|--------------------------|
| `200` | All | Success | Parse and use response |
| `400` | `GET /tiles/...` | `source` not in `{lr, sr, hr}` | Validate source string client-side before constructing URL |
| `400` | `POST /api/model` | Bad config/checkpoint path or corrupt file | Display `detail` message to user |
| `404` | `GET /tiles/...` | Unknown `tile_id` | Re-fetch `/api/tiles` to refresh scene list |
| `503` | `GET /tiles/sr/...` | No model checkpoint loaded | Show "Model not loaded" banner; poll `GET /api/model` |
| `500` | Any | Unexpected server error (rasterio failure, OOM, etc.) | Log the error; show a generic "Server error" message; retry once |

---

## Key Constants & Conventions

| Constant | Value | Notes |
|----------|-------|-------|
| Base URL | **User-provided at runtime** | Never hardcoded. Persisted in `localStorage` (web) / `SharedPreferences` (Android). |
| Default server port | `8000` | Set in `webapp/main.py` → `uvicorn.run(..., port=8000)` |
| Android emulator alias | `10.0.2.2` | Maps to host `127.0.0.1` — only valid in emulator, not on physical devices |
| localStorage key (web) | `terraresolve_base_url` | Key used to persist the base URL in the browser |
| SharedPreferences key (Android) | `base_url` in `terraresolve_prefs` | Key used to persist the base URL on device |
| Tile size | 256 px | Standard XYZ slippy-map tile |
| Default upscale factor | `4×` | LR 10 m/px → SR ~2.5 m/px |
| Input bands | 4 | Blue, Green, Red, NIR (Sentinel-2 L2A order) |
| LR normalisation | pixel ÷ 10 000 | Sentinel-2 L2A surface reflectance convention |
| Band order in PNG | R = band 3, G = band 2, B = band 1 | True-colour RGB from Sentinel-2 |
| SR disk cache path | `data/_sr_cache/{tile_id}.tif` | Survives server restarts; invalidated on `POST /api/model` |
| Auto-loaded checkpoint | `checkpoints/edsr_epoch50.pt` | Loaded at startup if file exists |
| Auto-loaded config | `configs/sen2naip.yaml` | Loaded at startup if file exists |
| Min zoom (tiles) | 10 | Scenes are small; below this the tile may render empty |
| Max native zoom | 19 | Tiles are upscaled by the client SDK above this |

---

## Available Models

Registered in `MODEL_REGISTRY` (see `terraresolve/registry.py`). Specify in YAML config as
`model.name`.

| Config key | Class | Description |
|------------|-------|-------------|
| `edsr` | `EDSR` | Enhanced Deep Super-Resolution. Default. 4×, 4 input channels, 64 features, 16 residual blocks. |
| `swinir` | `SwinIRLite` | Transformer-based SR. Use `configs/swinir.yaml`. |
| `sen2sr` | `Sen2SRModel` | Cross-sensor SR model specialised for Sentinel-2. |

---

## Available Evaluation Metrics

Computed internally during training / evaluation (`terraresolve/metrics/metrics.py`). Not exposed
as API endpoints but useful context for displaying quality scores in a frontend results panel.

| Metric key | Better | Range | Meaning |
|------------|--------|-------|---------|
| `psnr` | Higher | 0 – ∞ dB | Peak Signal-to-Noise Ratio — pixel-level fidelity |
| `ssim` | Higher | 0.0 – 1.0 | Structural Similarity Index — perceptual quality |
| `cc` | Higher | −1.0 – 1.0 | Pearson Correlation Coefficient — spatial correlation |
| `ergas` | Lower | 0 – ∞ | Relative Global Error in Synthesis — spectral accuracy |
| `sam` | Lower | 0 – 180° | Spectral Angle Mapper — spectral shape preservation |

---

## Reference Files in This Repo

| File | Purpose |
|------|---------|
| `webapp/main.py` | FastAPI backend — all endpoints defined here |
| `webapp/static/index.html` | Reference Leaflet web frontend wired to all endpoints |
| `configs/sen2naip.yaml` | Default model/training config |
| `configs/swinir.yaml` | SwinIR model config |
| `terraresolve/engine/inference.py` | Patch-based inference with overlap blending |
| `terraresolve/models/edsr.py` | EDSR model architecture |
| `terraresolve/models/swinir_lite.py` | SwinIR-Lite model architecture |
| `terraresolve/metrics/metrics.py` | PSNR, SSIM, CC, ERGAS, SAM implementations |
| `requirements.txt` | All Python dependencies |
