# TerraResolve — Jury Q&A

This document is a concise, jury-ready summary of the key questions a panel may ask about the project, along with clear answers grounded in the product and implementation plans.

---

## 1) What is TerraResolve, in one sentence?

**Answer:** TerraResolve is a deep-learning satellite super-resolution platform that converts medium-resolution Sentinel-2 imagery into sharper, georeferenced outputs suitable for downstream geospatial analysis and viewer-based decision support.

---

## 2) What problem is the project solving?

**Answer:** The problem is that Sentinel-2 imagery is available at 10 m/pixel resolution, which is often not detailed enough for urban, infrastructure, water-body, and vegetation analysis. The project aims to reconstruct sharper imagery to better than 4 m/pixel while preserving spectral fidelity, so the results remain scientifically useful instead of just visually appealing.

---

## 3) Why is the target resolution under 4 m important?

**Answer:** The competition target explicitly requires improving resolution from medium-resolution satellite imagery to under 4 m. This is meaningful because finer imagery supports land-use monitoring, urban structure detection, waterbody mapping, and vegetation analysis. The project focuses on this target while keeping the model calibrated for real remote-sensing use rather than only aesthetic enhancement.

---

## 4) Why use Sentinel-2 specifically?

**Answer:** Sentinel-2 is the mandated and real-world data source in the problem statement. It is publicly available, globally relevant, and already used in multiple geospatial workflows. TerraResolve taps into that ecosystem directly, and also includes a live Copernicus Data Space ingestion flow so the system can fetch a recent Sentinel-2 scene for direct processing.

---

## 5) What model are you using?

**Answer:** The core model is an EDSR-based super-resolution architecture, chosen because it is a proven and efficient CNN-based super-resolution model. The project also includes a lightweight SwinIR alternative for comparison, but the operational demo is intentionally anchored to EDSR because it is more straightforward to run, benchmark, and deploy reliably in a hackathon environment.

---

## 6) Is this really a working model or just a mockup?

**Answer:** It is a real working prototype, not a static mockup. The codebase contains a full training pipeline, model registry, dataset abstraction, evaluation metrics, inference pipeline, and GeoTIFF export flow. A 50-epoch EDSR training run was completed on real Sentinel-2/NAIP paired data, and the project includes a smoke test and a structured evaluation pipeline. The remaining work is around benchmarking, deployment polish, and broader dataset coverage.

---

## 7) What are the main training datasets used?

**Answer:** The main dataset used for the real training run is the SEN2NAIP cross-sensor subset, which pairs Sentinel-2 imagery with high-resolution NAIP imagery. This was selected because it provides a real LR-HR pair structure needed for supervised super-resolution training. The project also evaluates alternative real datasets such as SEN2VENuS and Maxar open-data collections to improve coverage and geographic relevance.

---

## 8) Is there a limitation in the current dataset?

**Answer:** Yes, openly. The current training data is geographically concentrated in the United States because NAIP is a US program. That means the model has mostly seen US-style landscapes, which is a legitimate generalization gap for Indian scenes. The project acknowledges this directly and treats it as a known limitation rather than hiding it.

---

## 9) How are you addressing the India-specific problem statement?

**Answer:** The product and roadmap target Bengaluru as the live demo region, and the architecture is designed to ingest live Copernicus Sentinel-2 scenes over a Bengaluru AOI. In parallel, the project is exploring Indian-specific real datasets such as SEN2VENuS/KUDALIAR and Maxar open-data scenes from India to improve geographic relevance and reduce the US-only training bias.

---

## 10) Why is Bengaluru the chosen demonstration area?

**Answer:** Bengaluru provides a focused, realistic, and manageable region for demo purposes. The product plan defines a bounded Bengaluru AOI, which allows the demo to preload a handful of scenes, render a clean viewer experience, and reduce operational complexity. This makes the demo feel fast and polished while still showing real geospatial performance.

---

## 11) How do you avoid a model that only looks good but is scientifically wrong?

**Answer:** The system explicitly measures spectral and structural quality, not just visual sharpness. It uses metrics such as PSNR, SSIM, CC, ERGAS, and SAM. Additionally, the project includes downstream tasks like NDWI and NDVI to test whether the sharpened output actually helps with real environmental analysis, such as water and vegetation mapping.

---

## 12) What makes the project more than just an image upscaler?

**Answer:** The product includes a full pipeline: data ingestion, model inference, georeferencing, user uploads, asynchronous jobs, downloadable outputs, and downstream geospatial usefulness. The system is designed not only to sharpen imagery but also to support a real workflow where a user can upload a GeoTIFF, trigger inference, and inspect the result through a geospatial viewer.

---

## 13) How does the live data pipeline work?

**Answer:** The project includes a Copernicus Data Space integration that authenticates against the Sentinel-2 catalog, searches for matching scenes within an AOI and date range, downloads the relevant data, stacks the L2A bands into the expected 4-band layout, and passes that into the model pipeline. This allows the system to work with the actual official Sentinel-2 source, not only with a static dataset folder.

---

## 14) What is the architecture of the product?

**Answer:** The architecture is split into a frontend and backend. The frontend is a static web experience hosted on Vercel, while the backend is a FastAPI service running in a Dockerized environment, with GPU-capable inference on HuggingFace Spaces or a similar deployment target. The platform handles tiles, jobs, metrics, uploads, and live fetch requests through a simple but production-aligned API structure.

---

## 15) Why not build a more complicated production stack immediately?

**Answer:** The hackathon and product architecture deliberately avoid unnecessary complexity. The first milestone is a reliable, working demo. That is why the project keeps the stack to a simple static frontend, a single FastAPI backend, SQLite for metadata, and one controlled inference worker. This keeps the system maintainable and debuggable while still looking production-like.

---

## 16) How do you handle user uploads and asynchronous processing?

**Answer:** The architecture includes a job-based workflow where users upload a GeoTIFF, the backend creates a job record in SQLite, the file is processed asynchronously, and the frontend polls status updates until completion. This is a standard pattern for inference-heavy systems and prevents the API from blocking while intensive model work runs.

---

## 17) What about georeferencing and GeoTIFF correctness?

**Answer:** This is a major technical strength. The inference pipeline is designed to preserve georeferencing metadata, rescale the transform correctly for the super-resolution factor, and write a valid GeoTIFF output. This is not just a visual upsample; it is a geospatial output meant to remain usable in GIS workflows and map viewers.

---

## 18) What does the viewer do for the demo?

**Answer:** The viewer is designed as a map-style comparison interface that lets the user compare low-resolution Sentinel-2 imagery with the enhanced output side by side. The product plan includes a slider-based comparison, a Bengaluru-locked viewport, metric overlays, and downloadable outputs. This gives judges a clear demonstration of before/after improvement.

---

## 19) What are the biggest technical risks or weaknesses?

**Answer:** The most important risks are dataset geographic coverage, model generalization to Indian terrain, and ensuring the inference remains stable under deployment conditions. The team also has to balance a scientifically valid model with the need for a polished demo. The project clearly documents these risks and treats them as engineering constraints to address next rather than hidden problems.

---

## 20) Why is TerraResolve a strong SIH-style project?

**Answer:** It combines a real research problem with a production-style user interface and deployment strategy. It uses actual satellite data, a real neural network, a scientific evaluation layer, geospatial correctness, and a practical demo experience. This gives it both technical credibility and presentation value, which is exactly what judges look for in a strong hackathon submission.

---

## 21) What are the next steps after this prototype?

**Answer:** The near-term roadmap is to improve data diversity, benchmark the model properly, run full-scene inference on real images, add stronger downstream evaluations, and improve deployment polish. Longer term, the system can incorporate more Indian datasets, broader model comparisons, and a more robust operational infrastructure.

---

## 22) If the jury asks, “What is the strongest claim you can defend right now?”

**Answer:** The strongest defendable claim is: TerraResolve is a working prototype of a real satellite super-resolution system that processes Sentinel-2 imagery with an EDSR pipeline, preserves geospatial metadata, supports live and uploaded workflows, and demonstrates a real geospatial improvement story in a polished web viewer. The strongest honest caveat is that the model’s data diversity and full India-specific validation are still open next steps.

---

## 23) If the jury asks, “What is your honest state of readiness?”

**Answer:** The system is well beyond a concept sketch: the codebase, training pipeline, inference logic, evaluation framework, and demo product architecture are already in place. The remaining readiness gaps are not foundational—they are calibration, benchmarking, dataset scope refinement, and deployment hardening. That is the difference between a working prototype and a fully production-hardened product.

---

## 24) Final message to the jury

**Answer:** TerraResolve is not just a model demo; it is a complete product concept for a satellite super-resolution workflow grounded in real geospatial data, scientific metrics, and a deployable user experience. The team has built a credible prototype, documented the technical constraints honestly, and laid out a clear roadmap from prototype to stronger real-world deployment.
