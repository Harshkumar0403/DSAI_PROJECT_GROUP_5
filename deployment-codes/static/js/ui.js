/* ui.js  –  All UI controls, state, feedback
 * Depends on: MAP, API
 */

'use strict';

const UI = (() => {

  const STATE = {
    sid:        '',
    device:     'cpu',
    availableDevices: ['cpu'],
    cudaAvailable: false,
    cudaName:   '',
    predStyle:  'fill',  // 'fill' | 'outline'
    t1Year:     2021,
    t2Year:     2023,
    threshold:  0.55,
    aoi:        null,
    bounds:     null,
    predBounds: null,
    hasLayers:  false,
    hasPred:    false,
    hansenAvailable: true,
    maxKm2:     200,
    swipeEnabled: false,
    opacity: {
      hansen: 0.8,
      pred: 0.85,
      forest30: 0.35,
    },
    _es:        null,   // active EventSource for prediction
  };

  // ── Toast ──────────────────────────────────────
  let _toastTimer = null;
  function toast(msg, type, duration) {
    duration = duration || 3500;
    const wrap = document.getElementById('toast-wrap');
    wrap.innerHTML = '';
    const el = document.createElement('div');
    el.className = 'toast ' + (type || '');
    el.textContent = msg;
    wrap.appendChild(el);
    requestAnimationFrame(() => el.classList.add('show'));
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => {
      el.classList.remove('show');
      setTimeout(() => el.remove(), 300);
    }, duration);
  }

  // ── Progress bar ───────────────────────────────
  function _setProgress(pct) {
    // pct: 0-100 to show, null/undefined to hide
    const wrap = document.getElementById('progress-bar-wrap');
    const bar  = document.getElementById('progress-bar');
    const lbl  = document.getElementById('progress-label');
    if (pct == null) {
      bar.style.width = '100%';
      setTimeout(() => {
        wrap.style.display = 'none';
        bar.style.width = '0%';
        if (lbl) lbl.textContent = '';
      }, 350);
    } else {
      wrap.style.display = 'block';
      bar.style.width = pct + '%';
    }
  }

  function _setProgressMsg(msg) {
    const lbl = document.getElementById('progress-label');
    if (lbl) lbl.textContent = msg;
  }

  // ── Button loading state ────────────────────────
  function _setBtnLoading(btn, loading) {
    if (!btn) return;
    btn.classList.toggle('loading', loading);
    btn.disabled = loading;
  }

  function _renderDeviceControls() {
    const cpuBtn = document.getElementById('device-cpu');
    const cudaBtn = document.getElementById('device-cuda');
    const status = document.getElementById('device-status-text');
    const badge = document.getElementById('device-badge');
    if (!cpuBtn || !cudaBtn || !status || !badge) return;

    cpuBtn.classList.toggle('active', STATE.device === 'cpu');
    cudaBtn.classList.toggle('active', STATE.device === 'cuda');
    cudaBtn.disabled = !STATE.cudaAvailable;

    if (STATE.cudaAvailable) {
      status.textContent = STATE.device === 'cuda'
        ? 'Using ' + (STATE.cudaName || 'CUDA')
        : 'CUDA available';
    } else {
      status.textContent = 'CUDA unavailable';
    }

    badge.classList.remove('gpu', 'cpu');
    badge.classList.add(STATE.device === 'cuda' ? 'gpu' : 'cpu');
    badge.textContent = STATE.device.toUpperCase();
  }

  function _renderPredStyleControls() {
    const fillBtn = document.getElementById('pred-style-fill');
    const outlineBtn = document.getElementById('pred-style-outline');
    const text = document.getElementById('pred-style-text');
    if (!fillBtn || !outlineBtn || !text) return;
    fillBtn.classList.toggle('active', STATE.predStyle === 'fill');
    outlineBtn.classList.toggle('active', STATE.predStyle === 'outline');
    text.textContent = STATE.predStyle === 'outline' ? 'Outline' : 'Fill';
  }

  function _setPredStyle(style) {
    style = (style || 'fill').toLowerCase();
    if (style !== 'outline') style = 'fill';
    STATE.predStyle = style;
    _renderPredStyleControls();
    if (STATE.hasPred && STATE.sid) {
      const predUrl = '/api/pred_tile/' + encodeURIComponent(STATE.sid) + '/{z}/{x}/{y}.png?mode=' +
        encodeURIComponent(STATE.predStyle);
      MAP.showPrediction(predUrl);
      MAP.setLayerOpacity('pred', STATE.opacity.pred);
    }
  }

  async function _initRuntimeInfo() {
    try {
      const data = await API.getHealth();
      STATE.availableDevices = data.available_devices || ['cpu'];
      STATE.cudaAvailable = !!data.cuda_available;
      STATE.cudaName = data.cuda_name || '';
      STATE.device = 'cpu';
      _renderDeviceControls();
    } catch (_) {
      _renderDeviceControls();
    }
  }

  function _setDevice(device) {
    if (device === 'cuda' && !STATE.cudaAvailable) return;
    STATE.device = device === 'cuda' ? 'cuda' : 'cpu';
    _renderDeviceControls();
  }

  // ── Layer panel ─────────────────────────────────
  // Layer visibility state tracked separately so rebuilds don't lose it
  const _layerVis = { t1: true, t2: true, forest30: false, hansen: false, pred: true };
  let _preCompareLayerVis = null;

  function _buildLayerPanel() {
    const p = document.getElementById('layer-panel');
    p.innerHTML = '';

    _addLayerRow(p, 't1',     'Sentinel T1 \u00b7 ' + STATE.t1Year, '#22c55e', STATE.hasLayers && !STATE.swipeEnabled);
    _addLayerRow(p, 't2',     'Sentinel T2 \u00b7 ' + STATE.t2Year, '#86efac', STATE.hasLayers && !STATE.swipeEnabled);
    _addLayerRow(p, 'forest30', 'Hansen forest >30%',                '#7ddc3a', STATE.hasLayers);
    _addLayerRow(p, 'hansen', 'Hansen GFC loss',                     '#00c2ff', STATE.hasLayers && STATE.hansenAvailable);
    _addLayerRow(p, 'pred',   'Prediction mask',                     '#ff7a00', STATE.hasPred);

    // Swipe toggle — separate from layer rows
    _addSwipeRow(p);
    _buildOpacityPanel();
  }

  function _buildOpacityPanel() {
    const panel = document.getElementById('layer-opacity-panel');
    if (!panel) return;
    panel.innerHTML = '';
    _addOpacityRow(panel, 'forest30', 'Forest >30% opacity', STATE.hasLayers && _layerVis.forest30);
    _addOpacityRow(panel, 'hansen', 'Hansen opacity', STATE.hasLayers && _layerVis.hansen);
    _addOpacityRow(panel, 'pred', 'Prediction opacity', STATE.hasPred && _layerVis.pred);
  }

  function _addOpacityRow(container, id, label, enabled) {
    const row = document.createElement('div');
    row.className = 'opacity-row' + (enabled ? '' : ' disabled');

    const top = document.createElement('div');
    top.className = 'opacity-row-top';

    const name = document.createElement('div');
    name.className = 'opacity-label';
    name.textContent = label;

    const value = document.createElement('div');
    value.className = 'opacity-value';
    value.textContent = Math.round(STATE.opacity[id] * 100) + '%';

    const input = document.createElement('input');
    input.type = 'range';
    input.min = '0';
    input.max = '1';
    input.step = '0.05';
    input.value = String(STATE.opacity[id]);
    input.disabled = !enabled;
    input.addEventListener('input', e => {
      const opacity = parseFloat(e.target.value);
      STATE.opacity[id] = opacity;
      value.textContent = Math.round(opacity * 100) + '%';
      MAP.setLayerOpacity(id, opacity);
    });

    top.appendChild(name);
    top.appendChild(value);
    row.appendChild(top);
    row.appendChild(input);
    container.appendChild(row);
  }

  function _addLayerRow(container, id, label, color, enabled) {
    const checked = _layerVis[id];

    const div = document.createElement('div');
    div.className = 'layer-row' + (enabled ? '' : ' disabled');
    div.setAttribute('data-id', id);

    // Color dot
    const dot = document.createElement('div');
    dot.className = 'layer-dot';
    dot.style.background = color;

    // Label
    const name = document.createElement('div');
    name.className = 'layer-name';
    name.textContent = label;

    // Toggle switch built from raw elements (no innerHTML, no label wrapping)
    const switchEl = document.createElement('div');
    switchEl.className = 'switch';

    const inp = document.createElement('input');
    inp.type    = 'checkbox';
    inp.checked = checked;
    inp.id      = 'lyr-' + id;

    const track = document.createElement('div');
    track.className = 'switch-track';
    track.setAttribute('data-for', id);

    // Only the input fires the change; row click toggles the input
    inp.addEventListener('change', e => {
      _layerVis[id] = e.target.checked;
      MAP.setLayerVisible(id, e.target.checked);
      _buildOpacityPanel();
      e.stopPropagation();
    });

    switchEl.appendChild(inp);
    switchEl.appendChild(track);

    div.appendChild(dot);
    div.appendChild(name);
    div.appendChild(switchEl);

    // Row click (outside the switch) toggles the checkbox
    div.addEventListener('click', e => {
      if (e.target === inp || switchEl.contains(e.target)) return;
      inp.checked = !inp.checked;
      inp.dispatchEvent(new Event('change'));
    });

    container.appendChild(div);
  }

  function _addSwipeRow(container) {
    const div = document.createElement('div');
    div.className = 'layer-row' + (STATE.hasLayers ? '' : ' disabled');
    div.setAttribute('data-id', 'swipe');

    const dot = document.createElement('div');
    dot.className = 'layer-dot';
    dot.style.cssText = 'background:#a78bfa;border-radius:2px';

    const name = document.createElement('div');
    name.className = 'layer-name';
    name.textContent = 'Swipe comparison';

    const switchEl = document.createElement('div');
    switchEl.className = 'switch';

    const inp = document.createElement('input');
    inp.type    = 'checkbox';
    inp.id      = 'lyr-swipe';
    inp.checked = STATE.swipeEnabled;

    const track = document.createElement('div');
    track.className = 'switch-track';

    inp.addEventListener('change', e => {
      STATE.swipeEnabled = e.target.checked;
      if (STATE.swipeEnabled) {
        _preCompareLayerVis = { t1: _layerVis.t1, t2: _layerVis.t2 };
        _layerVis.t1 = true;
        _layerVis.t2 = true;
        MAP.setLayerVisible('t1', true);
        MAP.setLayerVisible('t2', true);
      } else if (_preCompareLayerVis) {
        _layerVis.t1 = _preCompareLayerVis.t1;
        _layerVis.t2 = _preCompareLayerVis.t2;
        MAP.setLayerVisible('t1', _layerVis.t1);
        MAP.setLayerVisible('t2', _layerVis.t2);
        _preCompareLayerVis = null;
      }
      MAP.setCompareEnabled(STATE.swipeEnabled && STATE.hasLayers);
      _buildLayerPanel();
      e.stopPropagation();
    });

    switchEl.appendChild(inp);
    switchEl.appendChild(track);

    div.appendChild(dot);
    div.appendChild(name);
    div.appendChild(switchEl);

    div.addEventListener('click', e => {
      if (e.target === inp || switchEl.contains(e.target)) return;
      inp.checked = !inp.checked;
      inp.dispatchEvent(new Event('change'));
    });

    container.appendChild(div);
  }

  // ── Stats card ─────────────────────────────────
  function _updateStats(data) {
    const card = document.getElementById('stats-card');
    card.innerHTML = '';
    card.classList.add('visible');

    const hint = document.getElementById('empty-hint');
    if (hint) hint.style.display = 'none';

    function addRow(label, val, cls) {
      const r = document.createElement('div');
      r.className = 'stat-row';
      const lEl = document.createElement('div'); lEl.className = 'stat-label'; lEl.textContent = label;
      const vEl = document.createElement('div'); vEl.className = 'stat-val' + (cls ? ' '+cls : ''); vEl.textContent = val;
      r.appendChild(lEl); r.appendChild(vEl);
      card.appendChild(r);
    }

    function addDivider() {
      const d = document.createElement('div'); d.className = 'stats-divider'; card.appendChild(d);
    }

    addRow('Area',         (data.area_km2 || '—') + ' km\u00b2');
    addRow('Tiles',        data.tiles || '—');
    addDivider();
    addRow('Forest loss',  (data.loss_pct || 0) + '%', data.loss_pct > 5 ? 'loss' : 'ok');
    addRow('Threshold',    data.threshold);
  }

  function _updateLossPct(loss_pct, threshold) {
    const vals = document.querySelectorAll('#stats-card .stat-val');
    if (vals[2]) {
      vals[2].textContent = loss_pct + '%';
      vals[2].className   = 'stat-val ' + (loss_pct > 5 ? 'loss' : 'ok');
    }
    if (vals[3]) vals[3].textContent = threshold;
  }

  // ── Slider bindings ─────────────────────────────
  function _bindSlider(id, valId, isFloat) {
    const sl  = document.getElementById(id);
    const val = document.getElementById(valId);
    if (!sl || !val) return;
    const fmt = v => isFloat ? parseFloat(v).toFixed(2) : v;
    val.textContent = fmt(sl.value);
    sl.addEventListener('input', () => { val.textContent = fmt(sl.value); });
  }

  // ── Settings panel ──────────────────────────────
  function _initSettings() {
    const maxEl = document.getElementById('max-km2-input');
    if (maxEl) {
      maxEl.value = STATE.maxKm2;
      maxEl.addEventListener('change', () => {
        const v = parseFloat(maxEl.value);
        if (!isNaN(v) && v >= 1) STATE.maxKm2 = v;
        else maxEl.value = STATE.maxKm2;
      });
    }
  }

  // ── AOI drawn (called from map.js) ─────────────
  async function onAoiDrawn(coords, areakm2, source) {
    if (areakm2 > STATE.maxKm2) {
      toast('Area ' + areakm2.toFixed(1) + ' km\u00b2 exceeds ' + STATE.maxKm2 + ' km\u00b2 limit.', 'error', 6000);
      MAP.clearSelection();
      MAP.unlockMap();
      return;
    }

    STATE.aoi    = coords;
    STATE.t1Year = parseInt(document.getElementById('t1-slider').value, 10);
    STATE.t2Year = parseInt(document.getElementById('t2-slider').value, 10);

    const gap = STATE.t2Year - STATE.t1Year;
    if (gap < 1 || gap > 3) {
      toast('Year gap must be 1\u20133 years.', 'error', 4000);
      return;
    }

    // Lock map panning immediately after valid draw
    MAP.lockAfterAoi();

    const label = source === 'view' ? 'Visible map area' : 'AOI';
    toast(label + ' \u00b7 ' + areakm2.toFixed(1) + ' km\u00b2 \u00b7 fetching Earth Engine layers\u2026', 'info', 10000);
    _setProgress(10);
    _setProgressMsg('Fetching Earth Engine map layers...');

    const prevSid = STATE.sid;
    try {
      const data = await API.fetchLayers(
        coords, STATE.t1Year, STATE.t2Year, STATE.sid, STATE.maxKm2
      );
      STATE.sid       = data.sid;
      STATE.bounds    = data.bounds;
      STATE.hasLayers = true;
      STATE.hasPred   = false;
      STATE.hansenAvailable = data.hansen_available !== false;
      STATE.predBounds = null;
      if (!STATE.hansenAvailable) _layerVis.hansen = false;

      MAP.showSentinelLayers(data);
      MAP.setLayerOpacity('forest30', STATE.opacity.forest30);
      MAP.setLayerOpacity('hansen', STATE.opacity.hansen);
      MAP.setCompareEnabled(STATE.swipeEnabled);
      _buildLayerPanel();
      _setProgress(null);
      toast('Layers ready \u00b7 T1 ' + data.t1_year + ' \u2192 T2 ' + data.t2_year, 'ok', 3500);

      // Cancel any older prefetch job and start tile prefetch for this sid.
      if (prevSid && prevSid !== STATE.sid) {
        API.cancelPrefetch(prevSid).catch(() => {});
      }
      API.prefetchTiles(STATE.sid).catch(() => {});
    } catch (err) {
      _setProgress(null);
      MAP.unlockMap();
      toast('Error: ' + ((err && err.message) ? err.message : JSON.stringify(err)), 'error', 7000);
    }
  }

  function onAoiCleared() {
    if (STATE.sid) API.cancelPrefetch(STATE.sid).catch(() => {});
    STATE.aoi = null; STATE.bounds = null;
    STATE.predBounds = null;
    STATE.hasLayers = false; STATE.hasPred = false;
    STATE.sid = '';
    MAP.unlockMap();
    _buildLayerPanel();
    document.getElementById('stats-card').classList.remove('visible');
    const hint = document.getElementById('empty-hint');
    if (hint) hint.style.display = '';
    toast('AOI cleared \u00b7 draw a new region', '', 2500);
  }

  // ── Run detection ──────────────────────────────
  function _runDetection() {
    if (!STATE.sid) { toast('Select an AOI on the map first.', 'warn', 4000); return; }

    // Abort any in-flight prediction
    if (STATE._es) { STATE._es.close(); STATE._es = null; }

    STATE.threshold = parseFloat(document.getElementById('thr-slider').value);
    const btn = document.getElementById('btn-detect');
    _setBtnLoading(btn, true);
    _setProgress(5);
    _setProgressMsg('Starting...');
    toast('Starting model inference\u2026', 'info', 60000);

    STATE._es = API.predictStream(
      STATE.sid,
      STATE.threshold,
      STATE.device,
      // onProgress
      ({ pct, msg }) => {
        _setProgress(pct);
        _setProgressMsg(msg);
      },
      // onDone
      (result) => {
        STATE._es = null;
        if (result.device) STATE.device = result.device;
        STATE.hasPred    = true;
        STATE.predBounds = result.pred_bounds;   // EXACT raster bounds

        const predUrl = '/api/pred_tile/' + encodeURIComponent(STATE.sid) + '/{z}/{x}/{y}.png?mode=' +
          encodeURIComponent(STATE.predStyle);
        MAP.showPrediction(predUrl);
        MAP.setLayerOpacity('pred', STATE.opacity.pred);
        _buildLayerPanel();
        _renderDeviceControls();
        _renderPredStyleControls();
        _updateStats({
          area_km2:  result.area_km2,
          tiles:     result.tiles,
          loss_pct:  result.loss_pct,
          threshold: result.threshold,
        });
        _setProgress(null);
        _setBtnLoading(btn, false);
        toast(
          'Detection complete \u00b7 ' + result.loss_pct + '% forest loss \u00b7 ' +
          (result.device || STATE.device).toUpperCase() + ' \u00b7 batch ' +
          (result.batch_size || 8),
          'ok',
          5000
        );
      },
      // onError
      (msg) => {
        STATE._es = null;
        _setProgress(null);
        _setBtnLoading(btn, false);
        toast('Error: ' + msg, 'error', 8000);
      }
    );
  }

  // ── Re-threshold ───────────────────────────────
  async function _rethresh() {
    if (!STATE.sid || !STATE.hasPred) { toast('Run detection first.', 'warn', 3000); return; }
    STATE.threshold = parseFloat(document.getElementById('thr-slider').value);
    const btn = document.getElementById('btn-rethresh');
    _setBtnLoading(btn, true);

    try {
      const data = await API.rethresh(STATE.sid, STATE.threshold);
      const predUrl = '/api/pred_tile/' + encodeURIComponent(STATE.sid) + '/{z}/{x}/{y}.png?mode=' +
        encodeURIComponent(STATE.predStyle);
      MAP.showPrediction(predUrl);
      MAP.setLayerOpacity('pred', STATE.opacity.pred);
      _updateLossPct(data.loss_pct, data.threshold);
      toast('Threshold \u2192 ' + data.threshold + ' \u00b7 ' + data.loss_pct + '% loss', 'ok', 2500);
    } catch (err) {
      toast('Error: ' + ((err && err.message) ? err.message : JSON.stringify(err)), 'error', 5000);
    } finally {
      _setBtnLoading(btn, false);
    }
  }

  // ── Init ───────────────────────────────────────
  function init() {
    _bindSlider('t1-slider', 't1-val', false);
    _bindSlider('t2-slider', 't2-val', false);
    _bindSlider('thr-slider', 'thr-val', true);
    _initSettings();
    _renderDeviceControls();
    _initRuntimeInfo();
    _renderPredStyleControls();
    _buildLayerPanel();

    const selectView = () => {
      MAP.clearSelection();
      const view = MAP.getCurrentViewAoi();
      onAoiDrawn(view.coords, view.area, 'view');
    };

    document.getElementById('btn-detect').addEventListener('click', _runDetection);
    document.getElementById('btn-rethresh').addEventListener('click', _rethresh);
    document.getElementById('btn-draw-rect').addEventListener('click', () => MAP.startRectangleDraw());
    document.getElementById('btn-use-view').addEventListener('click', selectView);
    document.getElementById('btn-map-draw').addEventListener('click', () => MAP.startRectangleDraw());
    document.getElementById('btn-map-view-aoi').addEventListener('click', selectView);
    document.getElementById('device-cpu').addEventListener('click', () => _setDevice('cpu'));
    document.getElementById('device-cuda').addEventListener('click', () => _setDevice('cuda'));
    document.getElementById('pred-style-fill').addEventListener('click', () => _setPredStyle('fill'));
    document.getElementById('pred-style-outline').addEventListener('click', () => _setPredStyle('outline'));

    // Re-draw button in results section
    const reDrawBtn = document.getElementById('btn-redraw');
    if (reDrawBtn) {
      reDrawBtn.addEventListener('click', () => {
        if (STATE.sid) API.cancelPrefetch(STATE.sid).catch(() => {});
        MAP.clearSelection();
        MAP.unlockMap();
        STATE.hasLayers = false; STATE.hasPred = false;
        STATE.aoi = null; STATE.bounds = null; STATE.predBounds = null;
        STATE.sid = '';
        _buildLayerPanel();
        document.getElementById('stats-card').classList.remove('visible');
        const hint = document.getElementById('empty-hint');
        if (hint) hint.style.display = '';
        toast('Draw a new region on the map', 'info', 3000);
      });
    }

    window.UI = { onAoiDrawn, onAoiCleared };
  }

  return { init, onAoiDrawn, onAoiCleared };

})();
