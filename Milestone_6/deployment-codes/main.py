"""
main.py  –  Forest Loss Detection  –  FastAPI backend

Endpoints:
  GET  /                   → index.html
  GET  /static/*           → assets
  POST /api/layers         → Sentinel + Hansen URLs
  GET  /api/predict/stream → SSE stream of prediction progress + result
  POST /api/rethresh       → re-threshold cached prob map
  GET  /api/pred_img/{sid} → PNG prediction image
  GET  /health
"""

import json
import logging
import os
import threading
import uuid
from pathlib import Path

import ee
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import inference as inf

from dotenv import load_dotenv # for local development; not used in production on Hugging Face Spaces
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("forest")

# ── Config ────────────────────────────────────────────────
GEE_SA   = os.environ.get("GEE_SERVICE_ACCOUNT", "")
GEE_KEY  = os.environ.get("GEE_KEY_JSON", "")
GEE_PROJ = os.environ.get("GEE_PROJECT", "")
HF_REPO  = os.environ.get("HF_MODEL_REPO",    "suranjan90/forest-cover-loss")
HF_FILE  = os.environ.get("HF_MODEL_FILENAME", "best_model.pth")
HF_TOK   = os.environ.get("HF_TOKEN", None)

import torch
CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else ""
DEFAULT_DEVICE = "cpu"
log.info(
    "Runtime availability: cpu=yes cuda=%s%s",
    "yes" if CUDA_AVAILABLE else "no",
    f" ({CUDA_NAME})" if CUDA_NAME else "",
)


def resolve_device(requested: str = "") -> str:
    requested = (requested or DEFAULT_DEVICE).strip().lower()
    if requested == "cuda":
        if CUDA_AVAILABLE:
            return "cuda"
        log.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return "cpu"

# ── GEE init ──────────────────────────────────────────────
def init_gee():
    if GEE_KEY:
        import tempfile
        key = json.loads(GEE_KEY)
        f   = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(key, f); f.close()
        creds = ee.ServiceAccountCredentials(GEE_SA, f.name)
        ee.Initialize(creds, project=GEE_PROJ)
        log.info("GEE: service-account OK")
    else:
        try:
            ee.Initialize(project=GEE_PROJ or None)
            log.info("GEE: cached-credentials OK")
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=GEE_PROJ or None)

init_gee()

# ── Model path ────────────────────────────────────────────
_model_path = None
_mp_lock    = threading.Lock()

def get_model_path():
    global _model_path
    with _mp_lock:
        if _model_path is None:
            from huggingface_hub import hf_hub_download
            log.info("Downloading model from %s ...", HF_REPO)
            _model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, token=HF_TOK)
            log.info("Model cached at %s", _model_path)
    return _model_path

# ── App ───────────────────────────────────────────────────
app = FastAPI(title="Forest Loss Detection")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

INDEX_HTML = STATIC_DIR / "index.html"

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))

# ── Request models ────────────────────────────────────────
class LayersReq(BaseModel):
    coords:   list
    t1_year:  int = Field(ge=2015, le=2025)
    t2_year:  int = Field(ge=2015, le=2025)
    sid:      str = ""
    max_km2:  float = Field(default=200.0, ge=1.0, le=1000.0)

class PredictReq(BaseModel):
    sid:       str
    threshold: float = Field(default=0.55, ge=0.0, le=1.0)

class RethreshReq(BaseModel):
    sid:       str
    threshold: float = Field(default=0.55, ge=0.0, le=1.0)

class PrefetchReq(BaseModel):
    sid: str

# ── /api/layers ───────────────────────────────────────────
@app.post("/api/layers")
async def api_layers(req: LayersReq):
    sid = req.sid or str(uuid.uuid4())
    try:
        result = inf.prepare_layers(req.coords, req.t1_year, req.t2_year,
                                    sid, max_km2=req.max_km2)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("layers error")
        raise HTTPException(500, "GEE error: " + str(e)[:200])
    result["sid"] = sid
    return JSONResponse(result)

# ── /api/prefetch (tiles only) ────────────────────────────
@app.post("/api/prefetch")
async def api_prefetch(req: PrefetchReq):
    if not req.sid:
        raise HTTPException(400, "sid required")
    ok, status = inf.start_prefetch_tiles(req.sid)
    if not ok:
        raise HTTPException(404, status)
    return JSONResponse({"status": status})

@app.post("/api/prefetch/cancel")
async def api_prefetch_cancel(req: PrefetchReq):
    if not req.sid:
        raise HTTPException(400, "sid required")
    canceled = inf.cancel_prefetch(req.sid)
    return JSONResponse({"canceled": bool(canceled)})

# ── /api/predict  (POST alias → redirects callers to SSE) ─
# Some browsers / cached JS may still POST here; return a clear error.
@app.post("/api/predict")
async def api_predict_post(req: PredictReq):
    raise HTTPException(
        405,
        "Use GET /api/predict/stream?sid=...&threshold=... (SSE endpoint)"
    )

# ── /api/predict/stream  (SSE) ────────────────────────────
@app.get("/api/predict/stream")
async def api_predict_stream(sid: str, threshold: float = 0.55, device: str = DEFAULT_DEVICE):
    """
    Server-Sent Events endpoint.
    Client connects with EventSource, receives progress events,
    and a final 'done' event with results.
    The pred_img is retrieved separately via /api/pred_img/{sid}.
    """
    if not sid:
        raise HTTPException(400, "sid required")
    run_device = resolve_device(device)
    log.info(
        "Prediction request sid=%s threshold=%.3f requested_device=%s actual_device=%s",
        sid, threshold, device, run_device,
    )

    def _generate():
        try:
            model_path = get_model_path()
        except Exception as e:
            yield "data: " + json.dumps({"type":"error","msg":str(e)[:200]}) + "\n\n"
            return

        for chunk in inf.run_predict_stream(sid, threshold, model_path, run_device):
            yield chunk

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering on HF
        },
    )

# ── /api/rethresh ─────────────────────────────────────────
@app.post("/api/rethresh")
async def api_rethresh(req: RethreshReq):
    try:
        result = inf.run_rethresh(req.sid, req.threshold)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("rethresh error")
        raise HTTPException(500, str(e)[:200])
    result["pred_url"] = "/api/pred_img/" + req.sid + "?t=" + str(req.threshold)
    return JSONResponse(result)

# ── /api/pred_img/{sid} ───────────────────────────────────
@app.get("/api/pred_img/{sid}")
async def api_pred_img(sid: str):
    sess = inf.get_session(sid)
    png  = sess.get("pred_png") if sess else None
    if png is None:
        raise HTTPException(404, "No prediction for this session.")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})

@app.get("/api/pred_tile/{sid}/{z}/{x}/{y}.png")
async def api_pred_tile(sid: str, z: int, x: int, y: int, mode: str = "fill"):
    try:
        png = inf.render_prediction_tile(sid, z, x, y, mode=mode)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        log.exception("pred tile error")
        raise HTTPException(500, str(e)[:200])
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})

# ── Health ────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "default_device": DEFAULT_DEVICE,
        "selected_device": DEFAULT_DEVICE,
        "available_devices": ["cpu", "cuda"] if CUDA_AVAILABLE else ["cpu"],
        "cuda_available": CUDA_AVAILABLE,
        "cuda_name": CUDA_NAME,
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
