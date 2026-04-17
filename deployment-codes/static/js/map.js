/* map.js - Leaflet map, AOI tools, compare slider, GEE layers
 * Exposes MAP namespace.
 */

'use strict';

const MAP = (() => {

  let _map = null;
  let _drawn = null;
  let _swipeX = 50;
  let _swipeActive = false;
  let _aoiRect = null;
  let _compareEnabled = false;
  let _drawMode = false;
  let _drawStart = null;
  let _drawPreview = null;
  let _areaTooltip = null;

  const _layers = { t1: null, t2: null, forest30: null, hansen: null, pred: null };
  const _layerSources = { t1: null, t2: null, forest30: null, hansen: null, pred: null };
  const _layerVisible = { t1: true, t2: true, forest30: false, hansen: false, pred: true };
  const _layerOpacity = { t1: 1, t2: 1, forest30: 0.35, hansen: 0.8, pred: 0.85 };

  function init() {
    _map = L.map('map', {
      center: [26.2, 94.2],
      zoom: 9,
      zoomControl: false,
    });

    _ensurePane('gee-t1', 310);
    _ensurePane('gee-t2', 320);
    _ensurePane('gee-forest30', 328);
    _ensurePane('gee-labels', 325);
    _ensurePane('gee-hansen', 330);
    _ensurePane('gee-pred', 340);
    _ensurePane('gee-aoi', 350);

    L.control.zoom({ position: 'topleft' }).addTo(_map);

    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 19,
      }
    ).addTo(_map);

    L.tileLayer(
      'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
      {
        pane: 'gee-labels',
        attribution: 'Labels &copy; Esri',
        maxZoom: 19,
      }
    ).addTo(_map);

    _initDraw();
    _initSwipe();
  }

  function _ensurePane(name, zIndex) {
    if (_map.getPane(name)) return;
    _map.createPane(name);
    _map.getPane(name).style.zIndex = String(zIndex);
    _map.getPane(name).style.pointerEvents = 'none';
  }

  function _initDraw() {
    _drawn = new L.FeatureGroup().addTo(_map);

    const ctrl = new L.Control.Draw({
      position: 'topleft',
      draw: {
        rectangle: false,
        polygon: false,
        polyline: false,
        circle: false,
        circlemarker: false,
        marker: false,
      },
      edit: { featureGroup: _drawn, remove: true },
    });
    _map.addControl(ctrl);
    _map.on(L.Draw.Event.DELETED, _onDeleted);
    _map.on('mousedown', _onMapMouseDown);
    _map.on('mousemove', _onMapMouseMove);
    _map.on('mouseup', _onMapMouseUp);
    _map.on('mouseout', _onMapMouseUp);
    _areaTooltip = L.tooltip({
      permanent: false,
      direction: 'center',
      className: 'area-tooltip',
      offset: [0, 0],
    });
  }

  function _shapeOptions() {
    return {
      color: '#3d7cf5',
      weight: 2,
      dashArray: '7 4',
      fillColor: '#3d7cf5',
      fillOpacity: 0.07,
      pane: 'gee-aoi',
    };
  }

  function _onMapMouseDown(e) {
    if (!_drawMode || _compareEnabled) return;
    _drawStart = e.latlng;
    if (_drawPreview) _map.removeLayer(_drawPreview);
    _drawPreview = L.rectangle(L.latLngBounds(_drawStart, _drawStart), _shapeOptions()).addTo(_map);
    _map.dragging.disable();
  }

  function _onMapMouseMove(e) {
    if (!_drawMode || !_drawStart || !_drawPreview) return;
    const bounds = L.latLngBounds(_drawStart, e.latlng);
    _drawPreview.setBounds(bounds);
    const area = _haversineKm2(_latLngBoundsToCoords(bounds));
    _areaTooltip
      .setLatLng(bounds.getCenter())
      .setContent(area.toFixed(1) + ' km²');
    if (!_map.hasLayer(_areaTooltip)) _areaTooltip.addTo(_map);
  }

  function _onMapMouseUp(e) {
    if (!_drawMode || !_drawStart || !_drawPreview || !e.latlng) return;
    const bounds = L.latLngBounds(_drawStart, e.latlng);
    const southWest = bounds.getSouthWest();
    const northEast = bounds.getNorthEast();
    if (southWest.equals(northEast)) {
      _cancelDrawMode();
      return;
    }
    const layer = L.rectangle(bounds, _shapeOptions());
    _finalizeRectangle(layer);
    _cancelDrawMode();
  }

  function _cancelDrawMode() {
    _drawMode = false;
    _drawStart = null;
    if (_drawPreview) {
      _map.removeLayer(_drawPreview);
      _drawPreview = null;
    }
    if (_areaTooltip && _map.hasLayer(_areaTooltip)) _map.removeLayer(_areaTooltip);
    _map.dragging.enable();
    document.getElementById('map').style.cursor = '';
  }

  function _finalizeRectangle(layer) {
    _drawn.clearLayers();
    if (_aoiRect) {
      _map.removeLayer(_aoiRect);
      _aoiRect = null;
    }
    const coords = layer.toGeoJSON().geometry.coordinates[0];
    const area = _haversineKm2(coords);
    if (window.UI) UI.onAoiDrawn(coords, area, 'bbox');
  }

  function _onDeleted() {
    if (_drawPreview) _map.removeLayer(_drawPreview);
    _drawPreview = null;
    _unlockMap();
    clearDataLayers();
    if (window.UI) UI.onAoiCleared();
  }

  function startRectangleDraw() {
    clearSelection();
    _drawMode = true;
    _drawStart = null;
    document.getElementById('map').style.cursor = 'crosshair';
  }

  function _lockMap() {
    _map.doubleClickZoom.disable();
    _map.keyboard.disable();
    _map.boxZoom.disable();
    document.getElementById('map').style.cursor = 'default';
  }

  function _unlockMap() {
    _map.dragging.enable();
    _map.doubleClickZoom.enable();
    _map.keyboard.enable();
    _map.boxZoom.enable();
    document.getElementById('map').style.cursor = '';
  }

  function lockAfterAoi() { _lockMap(); }
  function unlockMap() { _unlockMap(); }

  function _haversineKm2(coords) {
    const R = 6371;
    let area = 0;
    const n = coords.length;
    for (let i = 0; i < n; i++) {
      const [lo1, la1] = coords[i];
      const [lo2, la2] = coords[(i + 1) % n];
      const x1 = lo1 * Math.PI / 180 * R * Math.cos(la1 * Math.PI / 180);
      const y1 = la1 * Math.PI / 180 * R;
      const x2 = lo2 * Math.PI / 180 * R * Math.cos(la2 * Math.PI / 180);
      const y2 = la2 * Math.PI / 180 * R;
      area += x1 * y2 - x2 * y1;
    }
    return Math.abs(area / 2);
  }

  function _latLngBoundsToCoords(bounds) {
    const sw = bounds.getSouthWest();
    const ne = bounds.getNorthEast();
    return [
      [sw.lng, sw.lat],
      [ne.lng, sw.lat],
      [ne.lng, ne.lat],
      [sw.lng, ne.lat],
      [sw.lng, sw.lat],
    ];
  }

  function clearDataLayers() {
    Object.keys(_layers).forEach(key => {
      if (_layers[key] && _map.hasLayer(_layers[key])) _map.removeLayer(_layers[key]);
      _layers[key] = null;
      _layerSources[key] = null;
    });
    if (_aoiRect) {
      _map.removeLayer(_aoiRect);
      _aoiRect = null;
    }
    if (_drawPreview) {
      _map.removeLayer(_drawPreview);
      _drawPreview = null;
    }
    document.getElementById('swipe-wrap').classList.remove('active');
    document.getElementById('compare-slider-wrap').classList.remove('active');
    _swipeActive = false;
    _compareEnabled = false;
    _resetCompareClip();
    _hideCompareView();
    document.getElementById('badge-t1').style.display = 'none';
    document.getElementById('badge-t2').style.display = 'none';
  }

  function clearSelection() {
    if (_drawn) _drawn.clearLayers();
    _cancelDrawMode();
    clearDataLayers();
  }

  function _tileLayer(url, pane, opacity) {
    return L.tileLayer(url, {
      pane,
      opacity: opacity || 1,
      tileSize: 256,
      updateWhenIdle: false,
      keepBuffer: 4,
      crossOrigin: true,
    });
  }

  function showSentinelLayers(data) {
    clearDataLayers();

    _layerVisible.t1 = true;
    _layerVisible.t2 = true;
    _layerVisible.forest30 = false;
    _layerVisible.hansen = false;

    const bounds = data.bounds;

    if (data.t2_url) {
      _layerSources.t2 = data.t2_url;
      _layers.t2 = _tileLayer(data.t2_url, 'gee-t2', _layerOpacity.t2);
      _layers.t2.addTo(_map);
    }
    if (data.t1_url) {
      _layerSources.t1 = data.t1_url;
      _layers.t1 = _tileLayer(data.t1_url, 'gee-t1', _layerOpacity.t1);
      _layers.t1.addTo(_map);
    }
    if (data.hansen_url) {
      _layerSources.hansen = data.hansen_url;
      _layers.hansen = _tileLayer(data.hansen_url, 'gee-hansen', _layerOpacity.hansen);
    }
    if (data.forest30_url) {
      _layerSources.forest30 = data.forest30_url;
      _layers.forest30 = _tileLayer(data.forest30_url, 'gee-forest30', _layerOpacity.forest30);
    }

    _aoiRect = L.rectangle(bounds, {
      color: '#3d7cf5',
      weight: 1.5,
      fill: false,
      dashArray: '5 4',
      opacity: 0.7,
      pane: 'gee-aoi',
    }).addTo(_map);

    _map.fitBounds(bounds, { padding: [24, 24] });

    const b1 = document.getElementById('badge-t1');
    const b2 = document.getElementById('badge-t2');
    b1.textContent = 'T1 · ' + data.t1_year;
    b2.textContent = 'T2 · ' + data.t2_year;
    b1.style.display = 'block';
    b2.style.display = 'block';

    document.getElementById('swipe-wrap').classList.add('active');
    _swipeActive = true;
    _applySwipe(50);
    setCompareEnabled(_compareEnabled);
  }

  function showPrediction(predUrl) {
    if (_layers.pred && _map.hasLayer(_layers.pred)) _map.removeLayer(_layers.pred);
    const url = predUrl + (predUrl.includes('?') ? '&' : '?') + '_ts=' + Date.now();
    _layerSources.pred = url;
    _layers.pred = _tileLayer(url, 'gee-pred', _layerOpacity.pred);
    _layers.pred.addTo(_map);
    _layerVisible.pred = true;
    _updateCompareLayerState('pred');
    if (_compareEnabled) _renderCompareView();
  }

  function setLayerVisible(id, visible) {
    _layerVisible[id] = !!visible;
    const l = _layers[id];
    if (l) {
      if (visible) {
        if (!_map.hasLayer(l)) _map.addLayer(l);
      } else if (_map.hasLayer(l)) {
        _map.removeLayer(l);
      }
    }
    _updateCompareLayerState(id);
    if (_compareEnabled) _renderCompareView();
  }

  function setLayerOpacity(id, opacity) {
    _layerOpacity[id] = opacity;
    const layer = _layers[id];
    if (layer && typeof layer.setOpacity === 'function') layer.setOpacity(opacity);
    _updateCompareLayerState(id);
  }

  function hasLayer(id) { return !!_layers[id]; }

  function getCurrentViewAoi() {
    const b = _map.getBounds();
    const south = b.getSouth();
    const west = b.getWest();
    const north = b.getNorth();
    const east = b.getEast();
    const coords = [
      [west, south],
      [east, south],
      [east, north],
      [west, north],
      [west, south],
    ];
    return {
      coords,
      area: _haversineKm2(coords),
      bounds: [[south, west], [north, east]],
    };
  }

  function _initSwipe() {
    const line = document.getElementById('swipe-line');
    const slider = document.getElementById('compare-slider');
    let dragging = false;

    line.addEventListener('pointerdown', e => {
      dragging = true;
      line.setPointerCapture(e.pointerId);
      e.preventDefault();
      e.stopPropagation();
    });

    window.addEventListener('pointermove', e => {
      if (!dragging) return;
      const mapEl = document.getElementById('map-wrap');
      const rect = mapEl.getBoundingClientRect();
      const pct = Math.max(2, Math.min(98, ((e.clientX - rect.left) / rect.width) * 100));
      _applySwipe(pct);
      e.preventDefault();
    }, { passive: false });

    window.addEventListener('pointerup', e => {
      if (!dragging) return;
      dragging = false;
      try { line.releasePointerCapture(e.pointerId); } catch (_) {}
    });

    window.addEventListener('pointercancel', () => { dragging = false; });

    if (slider) {
      slider.addEventListener('input', e => _applySwipe(parseFloat(e.target.value)));
    }
  }

  function _applySwipe(pct) {
    _swipeX = pct;
    document.getElementById('swipe-line').style.left = pct + '%';
    const slider = document.getElementById('compare-slider');
    if (slider && Math.abs(parseFloat(slider.value) - pct) > 0.5) slider.value = pct;

    const t2Pane = _map.getPane('gee-t2');
    const t1Pane = _map.getPane('gee-t1');
    const mapEl = document.getElementById('map-wrap');
    const width = mapEl.clientWidth;
    const divider = Math.max(0, Math.min(width, width * (pct / 100)));
    if (t2Pane) {
      if (_compareEnabled) {
        t2Pane.style.width = '100%';
        t2Pane.style.overflow = 'visible';
      } else {
        t2Pane.style.width = '100%';
        t2Pane.style.overflow = 'visible';
      }
    }
    if (t1Pane) {
      t1Pane.style.width = '100%';
      t1Pane.style.overflow = 'visible';
    }

    const t2ComparePane = document.querySelector('#compare-view [data-layer="t2"]');
    if (t2ComparePane) {
      t2ComparePane.style.width = divider + 'px';
    }
  }

  function setCompareEnabled(enabled) {
    _compareEnabled = !!enabled && _swipeActive && !!_layerSources.t1 && !!_layerSources.t2 && !!_aoiRect;
    const wrap = document.getElementById('swipe-wrap');
    const sliderWrap = document.getElementById('compare-slider-wrap');
    wrap.classList.toggle('active', _compareEnabled && _swipeActive);
    sliderWrap.classList.toggle('active', _compareEnabled && _swipeActive);
    if (_compareEnabled) {
      _showCompareView();
      _renderCompareView();
    } else {
      _hideCompareView();
    }
    _applySwipe(_swipeX);
  }

  function _resetCompareClip() {
    const t1Pane = _map ? _map.getPane('gee-t1') : null;
    const t2Pane = _map ? _map.getPane('gee-t2') : null;
    if (t1Pane) {
      t1Pane.style.width = '100%';
      t1Pane.style.overflow = 'visible';
    }
    if (t2Pane) {
      t2Pane.style.width = '100%';
      t2Pane.style.overflow = 'visible';
    }
  }

  function _showCompareView() {
    const wrap = document.getElementById('map-wrap');
    wrap.classList.add('compare-mode');
  }

  function _hideCompareView() {
    const wrap = document.getElementById('map-wrap');
    wrap.classList.remove('compare-mode');
    const panes = document.querySelectorAll('#compare-view .compare-pane');
    panes.forEach(pane => { pane.innerHTML = ''; });
  }

  function _renderCompareView() {
    if (!_compareEnabled || !_aoiRect) return;

    const compareView = document.getElementById('compare-view');
    const bounds = _aoiRect.getBounds();
    const width = compareView.clientWidth || compareView.offsetWidth;
    const height = compareView.clientHeight || compareView.offsetHeight;
    if (!width || !height) return;

    const zoom = Math.max(0, Math.min(19, _map.getBoundsZoom(bounds, false)));
    const sw = L.CRS.EPSG3857.latLngToPoint(bounds.getSouthWest(), zoom);
    const ne = L.CRS.EPSG3857.latLngToPoint(bounds.getNorthEast(), zoom);
    const pixelWidth = Math.max(1, ne.x - sw.x);
    const pixelHeight = Math.max(1, sw.y - ne.y);
    const scale = Math.min(width / pixelWidth, height / pixelHeight);
    const drawWidth = pixelWidth * scale;
    const drawHeight = pixelHeight * scale;
    const offsetX = (width - drawWidth) / 2;
    const offsetY = (height - drawHeight) / 2;

    ['t1', 't2', 'forest30', 'hansen', 'pred'].forEach(id => {
      const pane = document.querySelector('#compare-view [data-layer="' + id + '"]');
      if (!pane) return;
      pane.innerHTML = '';
      pane.style.opacity = String(_layerOpacity[id] == null ? 1 : _layerOpacity[id]);
      pane.style.display = _layerVisible[id] && _layerSources[id] ? 'block' : 'none';
      if (!_layerVisible[id] || !_layerSources[id]) return;

      const tileMinX = Math.floor(sw.x / 256);
      const tileMaxX = Math.floor((ne.x - 1) / 256);
      const tileMinY = Math.floor(ne.y / 256);
      const tileMaxY = Math.floor((sw.y - 1) / 256);
      const worldTiles = Math.pow(2, zoom);

      for (let x = tileMinX; x <= tileMaxX; x++) {
        for (let y = tileMinY; y <= tileMaxY; y++) {
          if (y < 0 || y >= worldTiles) continue;
          const wrappedX = ((x % worldTiles) + worldTiles) % worldTiles;
          const img = document.createElement('img');
          img.alt = id + ' tile';
          img.draggable = false;
          img.src = _tileUrl(_layerSources[id], zoom, wrappedX, y);
          img.style.left = offsetX + ((x * 256) - sw.x) * scale + 'px';
          img.style.top = offsetY + ((y * 256) - ne.y) * scale + 'px';
          img.style.width = (256 * scale) + 'px';
          img.style.height = (256 * scale) + 'px';
          pane.appendChild(img);
        }
      }
    });

    _updateCompareLayerState('forest30');
    _updateCompareLayerState('hansen');
    _updateCompareLayerState('pred');
    _applySwipe(_swipeX);
  }

  function _updateCompareLayerState(id) {
    const pane = document.querySelector('#compare-view [data-layer="' + id + '"]');
    if (!pane) return;
    pane.style.display = _compareEnabled && _layerVisible[id] && _layerSources[id] ? 'block' : 'none';
    pane.style.opacity = String(_layerOpacity[id] == null ? 1 : _layerOpacity[id]);
  }

  function _tileUrl(template, z, x, y) {
    return template
      .replace('{z}', String(z))
      .replace('{x}', String(x))
      .replace('{y}', String(y));
  }

  function refreshSwipe() {
    if (_compareEnabled) {
      _renderCompareView();
    } else if (_swipeActive) {
      _applySwipe(_swipeX);
    }
  }

  return {
    init,
    clearDataLayers,
    showSentinelLayers,
    showPrediction,
    setLayerVisible,
    setLayerOpacity,
    setCompareEnabled,
    hasLayer,
    startRectangleDraw,
    getCurrentViewAoi,
    clearSelection,
    lockAfterAoi,
    unlockMap,
    refreshSwipe,
  };

})();
