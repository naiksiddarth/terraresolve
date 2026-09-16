/**
 * TerraResolve Frontend Application Controller
 * Connects to the real backend running on http://localhost:7860
 * 
 * Endpoints:
 * - GET  /health/ready
 * - GET  /api/scenes
 * - GET  /api/scenes/{id}
 * - GET  /api/scenes/{id}/input  (Static PNG - L.imageOverlay)
 * - GET  /api/scenes/{id}/sr     (Static PNG - L.imageOverlay)
 * - POST /api/upload             (multipart/form-data, field "file")
 * - GET  /api/jobs/{job_id}      (Polling job status)
 * - GET  /api/jobs/{job_id}/result (Download output GeoTIFF)
 */

// Backend Base URL Configuration
const API_BASE = (window.location.port === '7860') ? '' : 'http://localhost:7860';

// Global state
let scenesList = [];
let currentSceneId = null;
let modalMapInstance = null;
let inputImageOverlay = null;
let srImageOverlay = null;
let downstreamMaskOverlay = null;
let currentLayerControl = null;
let activeJobPollingInterval = null;


// Replace all occurrences of "EDSR" with "SEN2SR" in DOM text
function correctModelNameInDOM() {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
  let node;
  while ((node = walker.nextNode())) {
    if (node.nodeValue && node.nodeValue.includes('EDSR')) {
      node.nodeValue = node.nodeValue.replace(/EDSR/g, 'SEN2SR');
    }
  }
}

/* =============================================================================
   1. HEALTH CHECK (/health/ready)
   ============================================================================= */
async function checkBackendHealth() {
  const statusText = document.getElementById('systemStatusText');
  const statusDot = document.querySelector('.status-dot');

  try {
    const res = await fetch(`${API_BASE}/health/ready`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();

    // Check response.model_loaded === true as specified
    if (data.model_loaded === true) {
      statusText.textContent = 'System Ready';
      if (statusDot) {
        statusDot.style.backgroundColor = '#10b981';
        statusDot.style.boxShadow = '0 0 6px rgba(16, 185, 129, 0.6)';
      }
    } else {
      statusText.textContent = 'Model Not Loaded';
      if (statusDot) {
        statusDot.style.backgroundColor = '#f59e0b';
        statusDot.style.boxShadow = '0 0 6px rgba(245, 158, 11, 0.6)';
      }
    }
  } catch (err) {
    statusText.textContent = 'Backend Offline';
    if (statusDot) {
      statusDot.style.backgroundColor = '#ef4444';
      statusDot.style.boxShadow = '0 0 6px rgba(239, 68, 68, 0.6)';
    }
  }
}

/* =============================================================================
   2. SCENES LIST & STATIC PNG OVERLAYS (L.imageOverlay)
   ============================================================================= */
async function loadAvailableScenes() {
  const select = document.getElementById('modalTileSelect');
  try {
    const res = await fetch(`${API_BASE}/api/scenes`);
    if (!res.ok) throw new Error(`Failed to fetch scenes: ${res.status}`);
    scenesList = await res.json();

    if (Array.isArray(scenesList) && scenesList.length > 0) {
      select.innerHTML = '';
      scenesList.forEach((scene) => {
        const id = scene.id || scene.scene_id || scene.name;
        const label = scene.name || scene.title || `Scene ${id}`;
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = `${label}`;
        select.appendChild(opt);
      });

      // Default to first scene
      currentSceneId = scenesList[0].id || scenesList[0].scene_id || scenesList[0].name;
    } else {
      select.innerHTML = '<option value="">No scenes indexed in backend</option>';
    }
  } catch (err) {
    console.warn('Could not load scenes from backend:', err);
    select.innerHTML = '<option value="yelahanka">Scene: Yelahanka, Bengaluru (Standby)</option>';
  }
}

/**
 * Normalizes bounding box array/object into Leaflet LatLngBounds
 */
function parseBoundingBox(rawBbox) {
  if (!rawBbox) return null;

  // Case 1: [south, west, north, east] or [west, south, east, north]
  if (Array.isArray(rawBbox) && rawBbox.length === 4) {
    let [a, b, c, d] = rawBbox.map(Number);
    // Determine lat vs lon
    let south, north, west, east;
    if (Math.abs(a) <= 90 && Math.abs(c) <= 90 && (Math.abs(b) > 40 || Math.abs(d) > 40)) {
      // a, c are latitudes; b, d are longitudes
      south = Math.min(a, c);
      north = Math.max(a, c);
      west = Math.min(b, d);
      east = Math.max(b, d);
    } else {
      // a, c might be longitudes (west, east) and b, d latitudes (south, north)
      west = Math.min(a, c);
      east = Math.max(a, c);
      south = Math.min(b, d);
      north = Math.max(b, d);
    }
    return L.latLngBounds([south, west], [north, east]);
  }

  // Case 2: { south, west, north, east } or { min_lat, min_lon, max_lat, max_lon }
  if (typeof rawBbox === 'object') {
    const south = rawBbox.south ?? rawBbox.min_lat ?? rawBbox.min_latitude;
    const north = rawBbox.north ?? rawBbox.max_lat ?? rawBbox.max_latitude;
    const west = rawBbox.west ?? rawBbox.min_lon ?? rawBbox.min_longitude;
    const east = rawBbox.east ?? rawBbox.max_lon ?? rawBbox.max_longitude;
    if (south !== undefined && north !== undefined && west !== undefined && east !== undefined) {
      return L.latLngBounds([south, west], [north, east]);
    }
  }

  return null;
}

/**
 * Loads a specific scene's detail and renders input & SR PNG overlays via L.imageOverlay
 */
async function displaySceneOnMap(sceneId) {
  if (!sceneId || !modalMapInstance) return;

  try {
    const cb = new Date().getTime();
    const res = await fetch(`${API_BASE}/api/scenes/${sceneId}?t=${cb}`);
    if (!res.ok) throw new Error(`Scene detail error ${res.status}`);
    const sceneDetail = await res.json();

    // Extract bounding box
    const bounds = parseBoundingBox(sceneDetail.bbox || sceneDetail.bounds || sceneDetail.bounding_box) 
                   || L.latLngBounds([13.0807, 77.5763], [13.1207, 77.6163]); // Fallback Yelahanka

    // Clean up previous image overlays and layer control
    if (inputImageOverlay) modalMapInstance.removeLayer(inputImageOverlay);
    if (srImageOverlay) modalMapInstance.removeLayer(srImageOverlay);
    if (downstreamMaskOverlay) modalMapInstance.removeLayer(downstreamMaskOverlay);
    if (currentLayerControl) modalMapInstance.removeControl(currentLayerControl);

    // Static PNG URLs as specified in contract (NOT L.tileLayer)
    const cacheBuster = new Date().getTime();
    const inputUrl = `${API_BASE}/api/scenes/${sceneId}/input?t=${cacheBuster}`;
    const srUrl = `${API_BASE}/api/scenes/${sceneId}/sr?t=${cacheBuster}`;
    const downstreamOverlayUrl = `${API_BASE}/api/scenes/${sceneId}/downstream/overlay?t=${cacheBuster}`;

    // Create Leaflet Image Overlays
    inputImageOverlay = L.imageOverlay(inputUrl, bounds, {
      opacity: 1.0,
      interactive: true,
      alt: 'Sentinel-2 LR Input (10m)'
    });

    srImageOverlay = L.imageOverlay(srUrl, bounds, {
      opacity: 1.0,
      interactive: true,
      alt: 'SEN2SR Super-Resolved (2.5m)'
    });

    downstreamMaskOverlay = L.imageOverlay(downstreamOverlayUrl, bounds, {
      opacity: 0.9,
      interactive: true,
      alt: 'NDWI Water & Shoreline (2.5m)'
    });

    // Add SR image overlay by default
    srImageOverlay.addTo(modalMapInstance);

    // Add layer control with base maps and downstream overlay toggle
    const baseMaps = {
      "SEN2SR Output (2.5m <4m)": srImageOverlay,
      "Original Sentinel-2 (10m)": inputImageOverlay
    };
    const overlayMaps = {
      "🌊 Water & Shoreline (NDWI)": downstreamMaskOverlay
    };
    currentLayerControl = L.control.layers(baseMaps, overlayMaps, { collapsed: false }).addTo(modalMapInstance);

    // Load downstream data
    loadDownstreamDataForScene(sceneId);

    // Force map to recalculate its container dimensions
    modalMapInstance.invalidateSize();
    modalMapInstance.fitBounds(bounds, { padding: [20, 20] });
    setTimeout(() => {
      modalMapInstance.invalidateSize();
      modalMapInstance.fitBounds(bounds, { padding: [20, 20] });
    }, 300);

  } catch (err) {
    console.warn(`Could not load scene imagery for ${sceneId}:`, err);
  }
}

/**
 * Loads downstream analysis metrics and graphic for the selected scene
 */
async function loadDownstreamDataForScene(sceneId) {
  if (!sceneId) return;
  try {
    const cb = new Date().getTime();
    const res = await fetch(`${API_BASE}/api/scenes/${sceneId}/downstream/metrics?t=${cb}`);
    if (!res.ok) {
      console.warn(`No downstream metrics available for ${sceneId}`);
      return;
    }
    const m = await res.json();

    const elAreaLr = document.getElementById('dsAreaLr');
    const elPxLr = document.getElementById('dsPxLr');
    const elAreaSr = document.getElementById('dsAreaSr');
    const elPxSr = document.getElementById('dsPxSr');
    const elShoreLr = document.getElementById('dsShoreLr');
    const elPerimLr = document.getElementById('dsPerimLr');
    const elShoreSr = document.getElementById('dsShoreSr');
    const elPerimSr = document.getElementById('dsPerimSr');
    const elSummary = document.getElementById('dsDensitySummary');
    const elImg = document.getElementById('dsGraphicImg');

    const lr = m['input_10m'] || {};
    const sr = m['sr_2.5m'] || {};
    const comp = m['comparison'] || {};

    if (elAreaLr && lr.area_ha !== undefined) elAreaLr.textContent = `${lr.area_ha.toFixed(2)} ha`;
    if (elPxLr && lr.water_pixels !== undefined) elPxLr.textContent = `${lr.water_pixels.toLocaleString()} pixels`;
    if (elAreaSr && sr.area_ha !== undefined) elAreaSr.textContent = `${sr.area_ha.toFixed(2)} ha`;
    if (elPxSr && sr.water_pixels !== undefined) elPxSr.textContent = `${sr.water_pixels.toLocaleString()} pixels`;

    if (elShoreLr && lr.shoreline_pixels !== undefined) elShoreLr.textContent = `${lr.shoreline_pixels.toLocaleString()} px`;
    if (elPerimLr && lr.estimated_perimeter_m !== undefined) elPerimLr.textContent = `Est. ${Math.round(lr.estimated_perimeter_m).toLocaleString()}m perimeter`;
    if (elShoreSr && sr.shoreline_pixels !== undefined) elShoreSr.textContent = `${sr.shoreline_pixels.toLocaleString()} px`;
    if (elPerimSr && sr.estimated_perimeter_m !== undefined) elPerimSr.textContent = `Est. ${Math.round(sr.estimated_perimeter_m).toLocaleString()}m perimeter`;

    if (elSummary && comp.shoreline_pixel_density_ratio !== undefined) {
      elSummary.innerHTML = `Shoreline boundary pixel density increases by <strong>${comp.shoreline_pixel_density_ratio}x</strong> in SR (+${comp.area_change_pct}% resolved water fringe).`;
    }
    if (elImg) {
      elImg.src = `${API_BASE}/api/scenes/${sceneId}/downstream?t=${cb}`;
    }

  } catch (err) {
    console.warn('Failed to load downstream data:', err);
  }
}


/* =============================================================================
   3. ASYNC UPLOAD & JOB POLLING FLOW
   ============================================================================= */

// Create UI Job Status Banner dynamically if it doesn't already exist
function getOrCreateJobStatusBanner() {
  let banner = document.getElementById('jobStatusBanner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'jobStatusBanner';
    banner.style.position = 'fixed';
    banner.style.bottom = '24px';
    banner.style.right = '24px';
    banner.style.zIndex = '2000';
    banner.style.background = '#ffffff';
    banner.style.border = '1.5px solid #0e5c3c';
    banner.style.borderRadius = '12px';
    banner.style.boxShadow = '0 10px 30px rgba(0,0,0,0.18)';
    banner.style.padding = '16px 20px';
    banner.style.maxWidth = '380px';
    banner.style.display = 'none';
    banner.style.flexDirection = 'column';
    banner.style.gap = '8px';
    banner.style.fontFamily = 'Plus Jakarta Sans, sans-serif';
    document.body.appendChild(banner);
  }
  return banner;
}

function showJobStatus(status, message, isError = false, downloadUrl = null) {
  const banner = getOrCreateJobStatusBanner();
  banner.style.display = 'flex';
  banner.style.borderColor = isError ? '#ef4444' : (status === 'COMPLETED' ? '#10b981' : '#0e5c3c');

  banner.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <div style="font-weight: 700; font-size: 14px; color: ${isError ? '#ef4444' : '#0f172a'}; display: flex; align-items: center; gap: 8px;">
        <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: ${isError ? '#ef4444' : (status === 'COMPLETED' ? '#10b981' : '#3b82f6')};"></span>
        Status: ${status}
      </div>
      <button onclick="document.getElementById('jobStatusBanner').style.display='none'" style="background: none; border: none; cursor: pointer; color: #94a3b8; font-size: 16px;">&times;</button>
    </div>
    <div style="font-size: 13px; color: #475569; line-height: 1.4;">${message}</div>
    ${downloadUrl ? `
      <a href="${downloadUrl}" download class="btn-start" style="margin-top: 6px; padding: 8px 16px; font-size: 13px; text-decoration: none; display: inline-flex; justify-content: center;">
        <i class="fa-solid fa-download"></i> Download SR GeoTIFF
      </a>
    ` : ''}
  `;
}

/**
 * Uploads a .tif / .tiff file via multipart/form-data to POST /api/upload
 */
async function uploadGeoTIFFFile(file) {
  if (!file) return;

  if (activeJobPollingInterval) {
    clearInterval(activeJobPollingInterval);
    activeJobPollingInterval = null;
  }

  showJobStatus('QUEUED', `Uploading "${file.name}" to SEN2SR enhancement pipeline...`);

  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || errData.error_message || `Upload failed with HTTP ${res.status}`);
    }

    const data = await res.json();
    const jobId = data.job_id;

    if (!jobId) {
      throw new Error('No job_id returned by upload endpoint');
    }

    showJobStatus(data.status || 'QUEUED', `Job #${jobId} registered. Initializing SEN2SR model...`);

    // Poll GET /api/jobs/{job_id} every 2 seconds
    pollJobStatus(jobId);

  } catch (err) {
    showJobStatus('FAILED', `Upload Error: ${err.message}`, true);
  }
}

/**
 * Polls GET /api/jobs/{job_id} every ~2000ms until COMPLETED or FAILED
 */
function pollJobStatus(jobId) {
  activeJobPollingInterval = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
      if (!res.ok) {
        throw new Error(`Polling error (HTTP ${res.status})`);
      }

      const jobData = await res.json();
      const status = jobData.status; // QUEUED, RUNNING, PREPROCESSING, INFERENCE, POSTPROCESSING, COMPLETED, FAILED
      const progress = jobData.progress_pct !== undefined ? ` (${jobData.progress_pct}%)` : '';

      if (status === 'COMPLETED') {
        clearInterval(activeJobPollingInterval);
        activeJobPollingInterval = null;
        const resultDownloadUrl = `${API_BASE}/api/jobs/${jobId}/result`;
        showJobStatus('COMPLETED', `Super-resolution completed successfully! Result is ready for download.`, false, resultDownloadUrl);
        // Refresh scene list if added
        loadAvailableScenes();
      } else if (status === 'FAILED') {
        clearInterval(activeJobPollingInterval);
        activeJobPollingInterval = null;
        // Show the real error_message field from the job response as mandated
        const errorMsg = jobData.error_message || 'Inference job failed unexpectedly.';
        showJobStatus('FAILED', `Error: ${errorMsg}`, true);
      } else {
        // QUEUED, RUNNING, PREPROCESSING, INFERENCE, POSTPROCESSING
        showJobStatus(status, `Processing stage: ${status}${progress}...`);
      }
    } catch (err) {
      console.warn('Polling error:', err);
    }
  }, 2000);
}

function triggerGeoTiffUpload() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.tif,.tiff,.geotiff';
  input.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
      uploadGeoTIFFFile(file);
    }
  };
  input.click();
}

/* =============================================================================
   4. UI NAVIGATION & MODAL CONTROLS
   ============================================================================= */
const navLinks = document.querySelectorAll('.nav-link');
navLinks.forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    navLinks.forEach(l => l.classList.remove('active'));
    link.classList.add('active');

    if (link.id === 'navExplorer') {
      openExplorerModal();
    } else if (link.id === 'navUpload') {
      triggerGeoTiffUpload();
    }
  });
});

const btnStartProcessing = document.getElementById('btnStartProcessing');
const btnUploadGeoTIFF = document.getElementById('btnUploadGeoTIFF');
const explorerModal = document.getElementById('explorerModal');
const closeModalBtn = document.getElementById('closeModalBtn');
const modalTileSelect = document.getElementById('modalTileSelect');

if (btnStartProcessing) {
  btnStartProcessing.addEventListener('click', () => {
    openExplorerModal();
  });
}

if (btnUploadGeoTIFF) {
  btnUploadGeoTIFF.addEventListener('click', () => {
    triggerGeoTiffUpload();
  });
}

if (closeModalBtn) {
  closeModalBtn.addEventListener('click', () => {
    explorerModal.classList.remove('active');
    document.getElementById('navHome').classList.add('active');
    document.getElementById('navExplorer').classList.remove('active');
  });
}

const btnToggleDownstream = document.getElementById('btnToggleDownstream');
const downstreamPanel = document.getElementById('downstreamPanel');
const btnCloseDownstreamPanel = document.getElementById('btnCloseDownstreamPanel');
const dsImgWrapper = document.getElementById('dsImgWrapper');

if (btnToggleDownstream && downstreamPanel) {
  btnToggleDownstream.addEventListener('click', () => {
    downstreamPanel.classList.toggle('active');
    btnToggleDownstream.classList.toggle('active');
  });
}

if (btnCloseDownstreamPanel && downstreamPanel) {
  btnCloseDownstreamPanel.addEventListener('click', () => {
    downstreamPanel.classList.remove('active');
    if (btnToggleDownstream) btnToggleDownstream.classList.remove('active');
  });
}

if (dsImgWrapper) {
  dsImgWrapper.addEventListener('click', () => {
    const img = document.getElementById('dsGraphicImg');
    if (img && img.src) {
      window.open(img.src, '_blank');
    }
  });
}

if (modalTileSelect) {
  modalTileSelect.addEventListener('change', (e) => {
    currentSceneId = e.target.value;
    displaySceneOnMap(currentSceneId);
  });
}


function openExplorerModal() {
  explorerModal.classList.add('active');
  document.getElementById('navExplorer').classList.add('active');
  document.getElementById('navHome').classList.remove('active');

  setTimeout(() => {
    if (!modalMapInstance) {
      // Centered at Yelahanka Bengaluru [13.1007, 77.5963]
      modalMapInstance = L.map('modalMap', {
        zoomControl: true,
        minZoom: 4,
        maxZoom: 22
      }).setView([13.1007, 77.5963], 13);

      // Keep existing Esri satellite basemap underneath as specified
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: '&copy; Esri &mdash; Earthstar Geographics',
        maxNativeZoom: 19,
        maxZoom: 22
      }).addTo(modalMapInstance);

      if (currentSceneId) {
        displaySceneOnMap(currentSceneId);
      }
    } else {
      modalMapInstance.invalidateSize();
      if (currentSceneId) {
        displaySceneOnMap(currentSceneId);
      }
    }
  }, 150);
}

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  correctModelNameInDOM();
  checkBackendHealth();
  loadAvailableScenes();

  // Periodic health check every 10 seconds
  setInterval(checkBackendHealth, 10000);
});

// Run once immediately
correctModelNameInDOM();
checkBackendHealth();
loadAvailableScenes();
