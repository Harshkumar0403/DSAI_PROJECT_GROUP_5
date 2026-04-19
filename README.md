# 🌍 Forest Loss Detection System

An end-to-end AI-powered web application for detecting forest cover loss using satellite imagery. This system leverages Sentinel-1 & Sentinel-2 data, Google Earth Engine (GEE), and a deep learning Siamese UNet model to analyze changes over time and provide interactive visual insights.

---

## 🚀 Overview

This project enables users to:

- Select an Area of Interest (AOI) on an interactive map  
- Choose two time periods (T1 and T2)  
- Visualize satellite imagery and forest data  
- Run a deep learning model to detect forest loss  
- Compare before/after imagery using an intuitive UI
- deployed link - https://huggingface.co/spaces/suranjan90/forest-cover-loss

The system is designed for environmental monitoring, research, and decision support in forestry and land-use analysis.

---

## 🧠 Key Features

- Interactive map-based UI (Leaflet.js)
- Satellite data from Sentinel-1 & Sentinel-2
- Integration with Google Earth Engine (GEE)
- Deep learning model for change detection
- FastAPI backend with streaming inference
- Real-time progress updates via SSE (Server-Sent Events)
- Adjustable prediction threshold (no re-inference required)
- Tile-based inference for large AOIs
- Optional GPU acceleration (CUDA)

---

## 🏗️ System Architecture
Frontend (HTML + JS + Leaflet)\
↓\
FastAPI Backend\
↓\
Inference Pipeline\
↓\
Google Earth Engine + DL Model


### Components

### 1. Frontend
- Built with HTML, CSS, and JavaScript  
- Uses Leaflet for map rendering and AOI selection  
- Communicates with backend via REST APIs and SSE  

### 2. Backend (FastAPI)
- Handles API requests and orchestration  
- Streams prediction progress to UI  
- Manages sessions and caching  
- Loads model from Hugging Face  

### 3. Inference Pipeline
- Fetches satellite data using GEE  
- Preprocesses and tiles data  
- Runs model inference  
- Merges predictions into final output  

---

## 🔬 Model Details

The system uses a Siamese UNet with ASPP (Atrous Spatial Pyramid Pooling):

- Takes bi-temporal inputs (T1 and T2)  
- Each input contains:
  - Sentinel-2 bands: B2, B3, B4, B8  
  - Sentinel-1 bands: VV, VH  
- Outputs a pixel-wise probability map of forest loss  

---

## 🛰️ Data Pipeline

### Steps

1. Input Validation
   - Ensures T1 < T2 (gap: 1–3 years)
   - Limits AOI size (default ≤ 200 km²)

2. Satellite Data Retrieval
   - Sentinel-2: Cloud-masked composites (Nov–Mar)
   - Sentinel-1: Annual median SAR data

3. Feature Engineering
   - NDVI computation
   - Band stacking for T1 and T2

4. Tile Generation
   - AOI split into 256×256 tiles
   - Aligned to satellite projection

5. Model Inference
   - Each tile normalized and passed through model
   - Predictions stitched into full map

6. Post-processing
   - Thresholding to generate binary mask
   - Overlay visualization

---

## ⚙️ Backend API

### Main Endpoints

| Endpoint | Description |
|--------|------------|
| `POST /api/layers` | Fetch satellite layers (T1, T2, Hansen) |
| `GET /api/predict/stream` | Run prediction with live progress |
| `POST /api/rethresh` | Adjust threshold without re-running model |
| `POST /api/prefetch` | Preload tiles for faster inference |
| `GET /api/pred_img/{sid}` | Get prediction image |
| `GET /health` | Check system status |

---

## 💻 Frontend

### Key Components

- Interactive map with AOI selection  
- Layer toggles (Sentinel, Hansen, Prediction)  
- Swipe comparison (T1 vs T2)  
- Real-time progress bar during inference  
- GPU/CPU selection  
- Opacity and visualization controls  

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd forest-loss-detection
pip install -r requirements.txt
```

🔐 Environment Variables
```env
Create a .env file:
GEE_SERVICE_ACCOUNT=your-service-account
GEE_KEY_JSON={...}
GEE_PROJECT=your-project-id

HF_MODEL_REPO=your-username/model-repo
HF_MODEL_FILENAME=best_model.pth
HF_TOKEN=your-hf-token
```
▶️ Running the Application
```bash
uvicorn main:app --reload
```
Open in browser:
```
http://localhost:7860
```
## ⚡ Performance Optimizations

- Model caching to avoid reloading
- Tile prefetching for faster inference
- SSE streaming for responsive UI
- GPU support (if available)
- Threshold adjustment without re-inference

## 📊 Use Cases
- Forest monitoring and conservation
- Deforestation detection
- Environmental research
- Policy and land-use planning
- Satellite data analysis

## ⚠️ Limitations
- AOI size is limited for performance reasons
- Depends on GEE availability and quotas
- Model accuracy depends on training data
- Hansen dataset available only up to 2024

## 🔮 Future Improvements
- Support for larger AOIs (batch processing)
- Time-series analysis (more than 2 timestamps)
- Improved model (Transformers / ConvNeXt / Swin)
- Downloadable reports and analytics
- Integration with additional datasets

## 📌 Summary

This project demonstrates a production-ready pipeline combining:

- Remote sensing (Sentinel data)
- Cloud geospatial processing (GEE)
- Deep learning (Siamese UNet)
- Modern web stack (FastAPI + JS UI)
