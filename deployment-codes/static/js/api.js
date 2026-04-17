/* api.js  –  Backend communication
 * All fetch/EventSource calls to FastAPI are here.
 */

'use strict';

const API = (() => {

  async function _post(url, body) {
    const resp = await fetch(url, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw { status: resp.status, message: data.detail || 'Unknown error' };
    return data;
  }

  /**
   * Fetch Sentinel + Hansen layer URLs.
   */
  async function fetchLayers(coords, t1Year, t2Year, sid, maxKm2) {
    return _post('/api/layers', {
      coords: coords, t1_year: t1Year, t2_year: t2Year,
      sid: sid || '', max_km2: maxKm2 || 200,
    });
  }

  async function prefetchTiles(sid) {
    return _post('/api/prefetch', { sid });
  }

  async function cancelPrefetch(sid) {
    return _post('/api/prefetch/cancel', { sid });
  }

  async function getHealth() {
    const resp = await fetch('/health');
    const data = await resp.json();
    if (!resp.ok) throw { status: resp.status, message: data.detail || 'Unknown error' };
    return data;
  }

  /**
   * Stream prediction progress via SSE.
   * @param {string}   sid
   * @param {number}   threshold
   * @param {Function} onProgress  called with {pct, msg} on each progress event
   * @param {Function} onDone      called with full result object on completion
   * @param {Function} onError     called with error message string
   * @returns {EventSource}  caller can call .close() to abort
   */
  function predictStream(sid, threshold, device, onProgress, onDone, onError) {
    const url = '/api/predict/stream?sid=' + encodeURIComponent(sid) +
                '&threshold=' + encodeURIComponent(threshold) +
                '&device=' + encodeURIComponent(device || 'cpu');
    const es  = new EventSource(url);

    es.onmessage = function(e) {
      let msg;
      try { msg = JSON.parse(e.data); }
      catch(_) { return; }

      if (msg.type === 'progress') {
        onProgress({ pct: msg.pct, msg: msg.msg });
      } else if (msg.type === 'done') {
        es.close();
        onDone(msg);
      } else if (msg.type === 'error') {
        es.close();
        onError(msg.msg || 'Unknown error');
      }
    };

    es.onerror = function() {
      es.close();
      onError('Connection to server lost during prediction.');
    };

    return es;
  }

  /**
   * Re-threshold cached probability map (fast, no model re-run).
   */
  async function rethresh(sid, threshold) {
    return _post('/api/rethresh', { sid, threshold });
  }

  return { fetchLayers, prefetchTiles, cancelPrefetch, getHealth, predictStream, rethresh };
})();
