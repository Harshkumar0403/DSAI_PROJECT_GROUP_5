"""
inference.py  –  satellite pipeline + model inference
"""

import io
import logging
import math
import threading
import time

import ee
import numpy as np
import requests
import torch
from PIL import Image
from pyproj import Transformer

from model import SiameseUNet_ASPP_DS

TILE_PX     = 256
SCALE       = 10.0
MAX_AOI_KM2 = 200.0

DATA_MEAN = torch.tensor([0.0187, 0.0389, 0.0231,  0.2954,
                           -8.4814, -14.8727]).view(6, 1, 1)
DATA_STD  = torch.tensor([0.0137, 0.0186, 0.0188,  0.0970,
                            2.2510,   2.2649]).view(6, 1, 1)

ALL_BANDS = ["B2_t1","B3_t1","B4_t1","B8_t1","VV_t1","VH_t1",
             "B2_t2","B3_t2","B4_t2","B8_t2","VV_t2","VH_t2"]

log = logging.getLogger("forest.inference")

def normalize(x):
    return (x - DATA_MEAN) / (DATA_STD + 1e-6)

# ── Model cache ───────────────────────────────────────────
_model_cache = {}
_model_lock  = threading.Lock()

def load_model(path, device):
    cache_key = (path, device)
    with _model_lock:
        if cache_key not in _model_cache:
            started = time.perf_counter()
            m = SiameseUNet_ASPP_DS(in_channels=6)
            sd = torch.load(path, map_location=device)
            if list(sd.keys())[0].startswith("module."):
                sd = {k[7:]: v for k, v in sd.items()}
            m.load_state_dict(sd)
            m.to(device).eval()
            _model_cache[cache_key] = m
            load_ms = (time.perf_counter() - started) * 1000
            log.info("Model loaded path=%s device=%s load_ms=%.1f", path, device, load_ms)
        else:
            log.info("Model cache hit path=%s device=%s", path, device)
    return _model_cache[cache_key]

# ── Session store ─────────────────────────────────────────
_session      = {}
_session_lock = threading.Lock()

# Limit concurrent Earth Engine tile prefetch jobs (protects against throttling).
# Keep this low on HF Spaces to avoid hammering EE.
_PREFETCH_MAX_CONCURRENT = 1
_prefetch_sem = threading.Semaphore(_PREFETCH_MAX_CONCURRENT)

_prefetch_threads = {}
_prefetch_threads_lock = threading.Lock()

def get_session(sid):
    with _session_lock:
        return _session.get(sid, {})

def set_session(sid, data):
    with _session_lock:
        _session[sid] = data

def _set_prefetch_state(sid, **kwargs):
    with _session_lock:
        sess = _session.get(sid)
        if not sess:
            return
        sess.setdefault("prefetch", {})
        sess["prefetch"].update(kwargs)

def cancel_prefetch(sid):
    with _session_lock:
        sess = _session.get(sid)
        if not sess:
            return False
        pf = sess.setdefault("prefetch", {})
        ev = pf.get("cancel_event")
        if ev is None:
            ev = threading.Event()
            pf["cancel_event"] = ev
        ev.set()
        pf["status"] = "canceled"
    log.info("Prefetch canceled sid=%s", sid)
    return True

def start_prefetch_tiles(sid):
    """
    Start background prefetch of tiles for a session (download only).
    Safe to call multiple times; reuses an in-flight job.
    """
    with _session_lock:
        sess = _session.get(sid)
        if not sess:
            return False, "Session not found"
        pf = sess.setdefault("prefetch", {})
        status = pf.get("status")
        if status in ("running", "done"):
            return True, status
        # Reset prefetch state
        pf["status"] = "queued"
        pf["error"] = ""
        pf["tiles_total"] = 0
        pf["tiles_done"] = 0
        pf["started_at"] = time.time()
        pf["finished_at"] = None
        pf["cancel_event"] = threading.Event()
        # Drop any previous prefetched payload
        sess.pop("prefetch_arrs", None)
        sess.pop("prefetch_order", None)
        sess.pop("prefetch_gi", None)

    def _runner():
        # Concurrency guard for EE download throttling
        acquired = _prefetch_sem.acquire(timeout=1)
        if not acquired:
            # Keep queued; predict can still download on-demand
            _set_prefetch_state(sid, status="queued")
            return
        try:
            _prefetch_tiles_worker(sid)
        finally:
            _prefetch_sem.release()

    t = threading.Thread(target=_runner, name=f"prefetch-{sid[:8]}", daemon=True)
    with _prefetch_threads_lock:
        _prefetch_threads[sid] = t
    t.start()
    return True, "running"

def _prefetch_tiles_worker(sid):
    with _session_lock:
        sess = _session.get(sid)
        if not sess:
            return
        pf = sess.setdefault("prefetch", {})
        cancel_event = pf.get("cancel_event") or threading.Event()
        pf["cancel_event"] = cancel_event
        pf["status"] = "running"

    try:
        stack = sess["stack"]; proj = sess["proj"]; aoi = sess["aoi"]
        tiles, gi = _tile_grid(aoi, proj)
        n = len(tiles)
        _set_prefetch_state(sid, tiles_total=n, tiles_done=0)
        log.info("Prefetch start sid=%s tiles=%d", sid, n)

        arrs, order = [], []
        for i, t in enumerate(tiles):
            if cancel_event.is_set():
                _set_prefetch_state(sid, status="canceled", finished_at=time.time())
                log.info("Prefetch canceled mid-run sid=%s done=%d/%d", sid, i, n)
                return
            arrs.append(_dl_tile(stack, t, proj))
            order.append((t["col"], t["row"]))
            if (i + 1) % 1 == 0:
                _set_prefetch_state(sid, tiles_done=i + 1)

        with _session_lock:
            sess = _session.get(sid)
            if not sess:
                return
            sess["prefetch_arrs"] = arrs
            sess["prefetch_order"] = order
            sess["prefetch_gi"] = gi
            sess.setdefault("prefetch", {})
            sess["prefetch"]["status"] = "done"
            sess["prefetch"]["finished_at"] = time.time()
        log.info("Prefetch done sid=%s tiles=%d", sid, n)
    except Exception as e:
        _set_prefetch_state(sid, status="error", error=str(e)[:300], finished_at=time.time())
        log.exception("Prefetch failed sid=%s", sid)

# ── Sentinel helpers ──────────────────────────────────────
def _mask_s2(image):
    qa   = image.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).divide(10000)

def _s2_mosaic(aoi, year):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(str(year-1)+"-11-01", str(year)+"-03-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .map(_mask_s2)
        .map(lambda i: i.addBands(i.normalizedDifference(["B8","B4"]).rename("NDVI")))
        .qualityMosaic("NDVI")
        .select(["B2","B3","B4","B8"])
        .clip(aoi)
    )

def _s1_median(aoi, year):
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(str(year)+"-01-01", str(year)+"-12-31")
        .filter(ee.Filter.eq("instrumentMode","IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH"))
        .median().select(["VV","VH"]).clip(aoi)
    )

def _ref_proj(aoi, scale=SCALE):
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi).first()
            .select("B2").projection().atScale(scale))

# ── Validation ────────────────────────────────────────────
def validate_years(t1, t2):
    gap = t2 - t1
    if gap < 1: raise ValueError("T2 must be at least 1 year after T1.")
    if gap > 3: raise ValueError("Maximum gap between T1 and T2 is 3 years.")

def validate_aoi(aoi, max_km2):
    km2 = aoi.area(maxError=100).getInfo() / 1e6
    if km2 > max_km2:
        raise ValueError("Area " + str(round(km2,1)) + " km2 exceeds limit of " + str(max_km2) + " km2.")
    return round(km2, 2)

# ── Thumbnail URLs ────────────────────────────────────────
def _map_tiles(image, vis_params):
    map_id = image.getMapId({**vis_params, "format": "png"})
    return map_id["tile_fetcher"].url_format

def _hansen_loss_mask(aoi, t1, t2):
    gfc  = ee.Image("UMD/hansen/global_forest_change_2024_v1_12").clip(aoi)
    return (gfc.select("lossyear").gte(t1 % 100)
              .And(gfc.select("lossyear").lte(t2 % 100))
              .And(gfc.select("treecover2000").gte(30))
              .selfMask())

def _hansen_forest30_mask(aoi):
    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12").clip(aoi)
    return gfc.select("treecover2000").gte(30).selfMask()

# ── AOI bbox (WGS-84) ─────────────────────────────────────
def _aoi_bounds_wgs84(aoi):
    bc   = aoi.bounds(1, "EPSG:4326").coordinates().getInfo()[0]
    lons = [c[0] for c in bc]
    lats = [c[1] for c in bc]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]

# ── Prepare layers (fast, no tile download) ───────────────
def prepare_layers(coords, t1, t2, sid, max_km2=MAX_AOI_KM2):
    validate_years(t1, t2)
    aoi = ee.Geometry.Polygon(coords)
    km2 = validate_aoi(aoi, max_km2)

    proj  = _ref_proj(aoi)
    s2_t1 = _s2_mosaic(aoi, t1).reproject(proj)
    s2_t2 = _s2_mosaic(aoi, t2).reproject(proj)
    s1_t1 = _s1_median(aoi, t1).reproject(proj)
    s1_t2 = _s1_median(aoi, t2).reproject(proj)

    before = s2_t1.addBands(s1_t1).rename(
        ["B2_t1","B3_t1","B4_t1","B8_t1","VV_t1","VH_t1"])
    after  = s2_t2.addBands(s1_t2).rename(
        ["B2_t2","B3_t2","B4_t2","B8_t2","VV_t2","VH_t2"])
    stack  = ee.Image.cat([before, after]).float().unmask(0).clip(aoi)

    hansen = _hansen_loss_mask(aoi, t1, t2) if t2 <= 2024 else None
    forest30 = _hansen_forest30_mask(aoi)
    t1_url = _map_tiles(
        s2_t1.clip(aoi),
        {"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.3, "gamma": 1.4},
    )
    t2_url = _map_tiles(
        s2_t2.clip(aoi),
        {"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.3, "gamma": 1.4},
    )
    han_url = (_map_tiles(
        hansen.clip(aoi),
        {"min": 0, "max": 1, "palette": ["00c2ff"]},
    ) if hansen is not None else None)
    forest30_url = _map_tiles(
        forest30.clip(aoi),
        {"min": 0, "max": 1, "palette": ["7ddc3a"]},
    )
    bounds  = _aoi_bounds_wgs84(aoi)

    set_session(sid, {
        "stack": stack, "proj": proj, "aoi": aoi,
        "bounds": bounds, "t1": t1, "t2": t2, "km2": km2,
    })

    return {
        "t1_url": t1_url,
        "t2_url": t2_url,
        "hansen_url": han_url,
        "forest30_url": forest30_url,
        "hansen_available": hansen is not None,
        "bounds": bounds,
        "area_km2": km2,
        "t1_year": t1,
        "t2_year": t2,
    }

# ── Tile grid (pixel-aligned to S2 grid) ──────────────────
def _bbox_native(aoi, proj_info):
    crs    = proj_info["crs"]
    coords = aoi.bounds(1,"EPSG:4326").coordinates().getInfo()[0]
    lons   = [c[0] for c in coords]
    lats   = [c[1] for c in coords]
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = tr.transform(lons, lats)
    except ImportError:
        sx,_,tx,_,sy,ty = proj_info["transform"]
        xs, ys = [], []
        for lon, lat in zip(lons, lats):
            pt = ee.Geometry.Point([lon, lat])
            px = (ee.Image.pixelCoordinates(ee.Projection(crs))
                  .sample(pt, 1).first().getInfo()["properties"])
            xs.append(tx + px["x"] * sx)
            ys.append(ty + px["y"] * sy)
    return min(xs), min(ys), max(xs), max(ys)

def _tile_grid(aoi, proj):
    info = proj.getInfo()
    sx,_,tx,_,sy,ty = info["transform"]
    crs = info["crs"]
    sx, sy = abs(sx), abs(sy)
    xn, yn, xx, yx = _bbox_native(aoi, info)

    # Snap origin outward to the pixel grid
    c0 = math.floor((xn - tx) / sx)
    r0 = math.floor((ty - yx) / sy)
    x0 = tx + c0 * sx
    y0 = ty - r0 * sy

    tw = TILE_PX * sx
    th = TILE_PX * sy
    nc = math.ceil((xx - x0) / tw)
    nr = math.ceil((y0 - yn) / th)

    tiles = []
    for r in range(nr):
        for c in range(nc):
            xL = x0 + c*tw; xR = xL + tw
            yT = y0 - r*th; yB = yT - th
            g  = ee.Geometry.Rectangle([xL,yB,xR,yT], proj=crs, evenOdd=False)
            tiles.append({"col":c,"row":r,"geom":g})

    # Exact geographic extent of the full stitched raster (native CRS corners)
    # Convert the four corners of the full grid back to WGS-84
    raster_x0 = x0
    raster_y0 = y0                  # top-left (north, because sy is positive upward)
    raster_x1 = x0 + nc * tw       # right
    raster_y1 = y0 - nr * th       # bottom

    try:
        from pyproj import Transformer
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        corners_x = [raster_x0, raster_x1, raster_x0, raster_x1]
        corners_y = [raster_y0, raster_y0, raster_y1, raster_y1]
        lons, lats = tr.transform(corners_x, corners_y)
        pred_bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    except Exception:
        # Fallback: use the AOI bounds (less accurate, but same as before)
        pred_bounds = None

    gi = {"nc": nc, "nr": nr, "x0": x0, "y0": y0,
          "sx": sx, "sy": sy, "crs": crs,
          "pred_bounds": pred_bounds}
    return tiles, gi

# ── Download tile ─────────────────────────────────────────
def _dl_tile(stack, tile, proj):
    scale = abs(proj.getInfo()["transform"][0])
    url   = (stack.select(ALL_BANDS).clip(tile["geom"]).reproject(proj)
             .getDownloadURL({"region":tile["geom"],"scale":scale,
                              "format":"NPY","bands":ALL_BANDS}))
    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
    except Exception:
        return np.zeros((12, TILE_PX, TILE_PX), dtype=np.float32)
    arr  = np.load(io.BytesIO(r.content))
    data = np.stack([arr[b].astype(np.float32) for b in ALL_BANDS])
    C, H, W = data.shape
    if H != TILE_PX or W != TILE_PX:
        pad = np.zeros((C, TILE_PX, TILE_PX), dtype=np.float32)
        pad[:, :H, :W] = data
        data = pad
    return data

# ── Batch inference ───────────────────────────────────────
def _batch_probs(arrs, model, device, bs=8):
    out = []
    total = len(arrs)
    num_batches = math.ceil(total / bs) if total else 0
    log.info(
        "Starting batched inference device=%s tiles=%d batch_size=%d num_batches=%d",
        device, total, bs, num_batches,
    )
    if device == "cuda":
        try:
            log.info("CUDA device=%s", torch.cuda.get_device_name(0))
        except Exception:
            pass
    for batch_idx, i in enumerate(range(0, total, bs), start=1):
        chunk = arrs[i:i+bs]
        batch_started = time.perf_counter()
        t1s = torch.stack([normalize(torch.tensor(a[:6], dtype=torch.float32))
                           for a in chunk]).to(device)
        t2s = torch.stack([normalize(torch.tensor(a[6:], dtype=torch.float32))
                           for a in chunk]).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        with torch.no_grad():
            p,_,_ = model(t1s, t2s)
            if device == "cuda":
                torch.cuda.synchronize()
            probs = torch.sigmoid(p).cpu().numpy()[:,0]
        batch_ms = (time.perf_counter() - batch_started) * 1000
        log.info(
            "Inference batch %d/%d device=%s batch_items=%d tensor_shape=%s elapsed_ms=%.1f",
            batch_idx, num_batches, device, len(chunk), tuple(t1s.shape), batch_ms,
        )
        out.extend(probs.astype(np.float32))
    return out

# ── Stitch ────────────────────────────────────────────────
def _stitch(preds, gi):
    canvas = np.zeros((gi["nr"]*TILE_PX, gi["nc"]*TILE_PX), dtype=np.float32)
    for (c,r), p in preds.items():
        canvas[r*TILE_PX:(r+1)*TILE_PX, c*TILE_PX:(c+1)*TILE_PX] = p
    return canvas

# ── PNG helper ────────────────────────────────────────────
def _prob_to_png_bytes(prob_map, threshold):
    mask = (prob_map > threshold).astype(np.uint8)
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:,:,0] = 255; rgba[:,:,1] = 122; rgba[:,:,2] = 0
    rgba[:,:,3] = (mask * 210).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()

def _tile_xyz_bounds_3857(x, y, z):
    world = 40075016.68557849
    tile_span = world / (2 ** z)
    xmin = -world / 2 + x * tile_span
    xmax = xmin + tile_span
    ymax = world / 2 - y * tile_span
    ymin = ymax - tile_span
    return xmin, ymin, xmax, ymax

def render_prediction_tile(sid, z, x, y, size=256, mode="fill"):
    sess = get_session(sid)
    if not sess or "prob_map" not in sess or "gi" not in sess:
        raise ValueError("No prediction cached. Run detection first.")

    threshold = sess.get("threshold", 0.55)
    prob_map = sess["prob_map"]
    gi = sess["gi"]
    mode = (mode or "fill").strip().lower()
    if mode not in ("fill", "outline"):
        mode = "fill"

    xmin, ymin, xmax, ymax = _tile_xyz_bounds_3857(x, y, z)
    px = np.linspace(xmin, xmax, size, endpoint=False) + (xmax - xmin) / (2 * size)
    py = np.linspace(ymax, ymin, size, endpoint=False) - (ymax - ymin) / (2 * size)
    xx, yy = np.meshgrid(px, py)

    tr = Transformer.from_crs("EPSG:3857", gi["crs"], always_xy=True)
    src_x, src_y = tr.transform(xx, yy)

    col = np.floor((src_x - gi["x0"]) / gi["sx"]).astype(np.int32)
    row = np.floor((gi["y0"] - src_y) / gi["sy"]).astype(np.int32)

    h, w = prob_map.shape
    valid = (row >= 0) & (row < h) & (col >= 0) & (col < w)

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    if np.any(valid):
        hit = np.zeros((size, size), dtype=bool)
        mask = prob_map[row[valid], col[valid]] > threshold
        if np.any(mask):
            valid_rows, valid_cols = np.where(valid)
            hit_rows = valid_rows[mask]
            hit_cols = valid_cols[mask]
            hit[hit_rows, hit_cols] = True

        if mode == "outline":
            # Outline in output pixel space: draw boundary pixels only.
            # A boundary pixel is a 'hit' pixel adjacent to at least one non-hit neighbor.
            if np.any(hit):
                up = np.pad(hit[:-1, :], ((1, 0), (0, 0)), constant_values=False)
                dn = np.pad(hit[1:, :], ((0, 1), (0, 0)), constant_values=False)
                lf = np.pad(hit[:, :-1], ((0, 0), (1, 0)), constant_values=False)
                rt = np.pad(hit[:, 1:], ((0, 0), (0, 1)), constant_values=False)
                edge = hit & (~(up & dn & lf & rt))
                rgba[edge, 0] = 255
                rgba[edge, 1] = 122
                rgba[edge, 2] = 0
                rgba[edge, 3] = 255
        else:
            # Filled blobs (existing behavior)
            if np.any(hit):
                rgba[hit, 0] = 255
                rgba[hit, 1] = 122
                rgba[hit, 2] = 0
                rgba[hit, 3] = 210

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()

# ── SSE progress helper ───────────────────────────────────
import json as _json

def _sse(data: dict) -> str:
    return "data: " + _json.dumps(data) + "\n\n"

# ── run_predict  (generator, yields SSE strings) ──────────
def run_predict_stream(sid, threshold, model_path, device):
    """
    Generator that yields SSE-formatted strings.
    Last event has type 'done' and contains the full result dict.
    On error yields type 'error'.
    """
    sess = get_session(sid)
    if not sess:
        yield _sse({"type":"error","msg":"Session not found. Re-select your AOI."})
        return

    stack = sess["stack"]; proj = sess["proj"]; aoi = sess["aoi"]

    try:
        started = time.perf_counter()
        log.info("Prediction start sid=%s device=%s threshold=%.3f", sid, device, threshold)
        batch_size = 8
        arrs = None
        order = None
        gi = None

        # If a prefetch job is running/queued, wait and stream progress.
        # If prefetch is done, reuse its cached tiles.
        pf = sess.get("prefetch", {}) or {}
        pf_status = pf.get("status")

        if pf_status in ("queued", "running"):
            yield _sse({"type":"progress","step":"prefetch","pct":5,"msg":"Waiting for tile prefetch..."})
            last_done = -1
            while True:
                with _session_lock:
                    cur = _session.get(sid, {})
                    pf2 = (cur.get("prefetch", {}) or {})
                    status2 = pf2.get("status")
                    done2 = int(pf2.get("tiles_done") or 0)
                    total2 = int(pf2.get("tiles_total") or 0)
                    err2 = pf2.get("error") or ""
                if status2 == "done":
                    break
                if status2 in ("canceled", "error"):
                    log.warning("Prefetch not available sid=%s status=%s error=%s", sid, status2, err2[:120])
                    break
                if total2 > 0 and done2 != last_done:
                    last_done = done2
                    pct = 5 + int(60 * done2 / max(1, total2))
                    yield _sse({
                        "type":"progress",
                        "step":"prefetch",
                        "pct": pct,
                        "msg": f"Prefetching tiles {done2}/{total2}",
                    })
                time.sleep(0.4)

        with _session_lock:
            sess2 = _session.get(sid, {})
            if sess2.get("prefetch", {}).get("status") == "done":
                arrs = sess2.get("prefetch_arrs")
                order = sess2.get("prefetch_order")
                gi = sess2.get("prefetch_gi")

        if arrs is None or order is None or gi is None:
            # Fallback: compute grid + download on-demand
            yield _sse({"type":"progress","step":"grid","pct":5,"msg":"Computing tile grid..."})
            tiles, gi = _tile_grid(aoi, proj)
            n = len(tiles)
            log.info(
                "Tile grid ready sid=%s tiles=%d rows=%d cols=%d pred_bounds=%s",
                sid, n, gi["nr"], gi["nc"], gi["pred_bounds"],
            )
            arrs, order = [], []
            download_started = time.perf_counter()
            for i, t in enumerate(tiles):
                pct = 10 + int(60 * i / n)
                yield _sse({"type":"progress","step":"tile","pct":pct,
                            "msg":"Downloading tile " + str(i+1) + "/" + str(n)})
                arrs.append(_dl_tile(stack, t, proj))
                order.append((t["col"], t["row"]))
            download_ms = (time.perf_counter() - download_started) * 1000
            log.info("Tile download complete sid=%s tiles=%d elapsed_ms=%.1f", sid, n, download_ms)
        else:
            n = len(order)
            log.info("Using prefetched tiles sid=%s tiles=%d", sid, n)

        yield _sse({"type":"progress","step":"model","pct":70,"msg":"Loading model..."})
        model = load_model(model_path, device)

        yield _sse({"type":"progress","step":"infer","pct":75,"msg":"Running model inference on " + device.upper() + "..."})
        infer_started = time.perf_counter()
        probs    = _batch_probs(arrs, model, device, bs=batch_size)
        infer_ms = (time.perf_counter() - infer_started) * 1000
        log.info(
            "Inference complete sid=%s device=%s batch_size=%d tiles=%d elapsed_ms=%.1f",
            sid, device, batch_size, n, infer_ms,
        )
        prob_map = _stitch({k:v for k,v in zip(order,probs)}, gi)

        with _session_lock:
            _session[sid]["prob_map"]    = prob_map
            _session[sid]["gi"]          = gi
            _session[sid]["pred_bounds"] = gi["pred_bounds"]
            _session[sid]["threshold"]   = threshold

        yield _sse({"type":"progress","step":"png","pct":92,"msg":"Generating prediction image..."})
        png      = _prob_to_png_bytes(prob_map, threshold)
        bin_map  = (prob_map > threshold).astype(np.float32)
        loss_pct = round(float(bin_map.mean()) * 100, 2)

        # Use exact raster bounds if available, fall back to AOI bounds
        pred_bounds = gi["pred_bounds"] if gi["pred_bounds"] else sess["bounds"]

        yield _sse({"type":"done","pct":100,
                    "loss_pct":  loss_pct,
                    "tiles":     n,
                    "area_km2":  sess["km2"],
                    "threshold": threshold,
                    "device":    device,
                    "batch_size": batch_size,
                    "pred_bounds": pred_bounds,
                    "aoi_bounds":  sess["bounds"]})
        # Store PNG separately (keyed by sid, retrieved via /api/pred_img)
        with _session_lock:
            _session[sid]["pred_png"] = png
        total_ms = (time.perf_counter() - started) * 1000
        log.info(
            "Prediction done sid=%s device=%s tiles=%d batch_size=%d loss_pct=%.2f total_ms=%.1f",
            sid, device, n, batch_size, loss_pct, total_ms,
        )

    except Exception as e:
        log.exception("Prediction failed sid=%s device=%s", sid, device)
        yield _sse({"type":"error","msg":str(e)[:300]})

def run_rethresh(sid, threshold):
    sess = get_session(sid)
    if not sess or "prob_map" not in sess:
        raise ValueError("No prediction cached. Run detection first.")
    prob_map    = sess["prob_map"]
    pred_bounds = sess.get("pred_bounds") or sess["bounds"]
    loss_pct    = round(float((prob_map > threshold).mean()) * 100, 2)
    png         = _prob_to_png_bytes(prob_map, threshold)
    with _session_lock:
        _session[sid]["pred_png"] = png
        _session[sid]["threshold"] = threshold
    return {
        "loss_pct":    loss_pct,
        "threshold":   threshold,
        "pred_bounds": pred_bounds,
        "aoi_bounds":  sess["bounds"],
    }
