"use strict";
// cog2tms web UI: vanilla JS + MapLibre GL only. All user-provided text goes
// through textContent, never innerHTML.
// The {z}/{x}/{y} grid of a raster source is the standard slippy grid and
// tileSize only declares the image resolution, so no zoomOffset trick here
// (verified in maplibre-gl src, raster_tile_source.ts).

const $ = (id) => document.getElementById(id);
const MAX_ZOOM = 24;  // same ceiling as the server's MAX_ZOOM setting
const FIELDS = ["format", "quality", "size", "blackmax", "bidx", "rescale"];
const DEFAULTS = {format: "png", quality: "80", size: "256", blackmax: "20", bidx: "", rescale: ""};
const BASE_STYLE = "https://tiles.openfreemap.org/styles/bright";  // free vector MVT basemap, no key
const BG_STYLE = {version: 8, sources: {},
                  layers: [{id: "bg", type: "background", paint: {"background-color": "#e8ebee"}}]};

// page directory as text: new URL() would percent-encode the {z}/{x}/{y} braces
const pageDir = () => location.href.replace(/[^/]*$/, "");

let map, info = null, fpData = null;  // info: c2t.tms.json payload

const cogUrl = () => $("url").value.trim();

function tileQuery() {
  const q = new URLSearchParams({u: cogUrl()});
  for (const f of FIELDS) {
    const el = $(f), v = el.value.trim();
    // only params differing from the SERVICE defaults (DEFAULTS) appear: the
    // UI default "auto" format therefore shows up in urls, png does not;
    // disabled controls (quality under png, blackmax under jpeg) never make it in
    if (el.disabled || v === DEFAULTS[f]) continue;
    q.set(f, v);
  }
  return q.toString();
}

const tileTemplate = () => pageDir() + "c2t/{z}/{x}/{y}?" + tileQuery();

async function getJSON(rel) {
  const r = await fetch(rel);
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try { detail = (await r.json()).detail || detail; } catch (e) { /* not JSON */ }
    throw new Error(detail);
  }
  return r.json();
}

async function load() {
  const u = cogUrl();
  if (!u) return;
  showError("");
  try {
    const qs = new URLSearchParams({u}).toString();
    const [tj, gj] = await Promise.all([
      getJSON("c2t.tms.json?" + qs),
      getJSON("c2t/0/0/0.geojson?" + qs),  // z/x/y unused: always the footprint
    ]);
    info = tj;
    fpData = gj;
    renderInfo(tj, gj);
    setUrls();
    ensureOverlays();
    if (!viewFromUrl) {
      // open at the TileJSON minzoom, centered on the bbox center
      const [w, s, e, n] = tj.bbox;
      map.jumpTo({center: [(w + e) / 2, (s + n) / 2],
                  zoom: Math.max(0, Math.min(tj.minzoom ?? 0, MAX_ZOOM))});
    }
    share();
  } catch (e) {
    showError(e.message);
  }
}

// (re)build the cog raster source + footprint line layer on the current style
function ensureOverlays() {
  if (!info || !map.isStyleLoaded()) return;  // style.load will call us again
  for (const id of ["cog", "fp"]) {
    if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(id)) map.removeSource(id);
  }
  map.addSource("cog", {
    type: "raster", tiles: [tileTemplate()], tileSize: +$("size").value,
    minzoom: info.minzoom ?? 0, maxzoom: Math.min(info.maxzoom ?? MAX_ZOOM, MAX_ZOOM),
  });
  map.addLayer({id: "cog", type: "raster", source: "cog",
                paint: {"raster-opacity": $("opacity").value / 100}});
  if (fpData) {
    map.addSource("fp", {type: "geojson", data: fpData});
    map.addLayer({id: "fp", type: "line", source: "fp",
                  paint: {"line-color": "#e11d48", "line-width": 2, "line-dasharray": [4, 3]}});
  }
}

function renderInfo(tj, gj) {
  const dl = $("infodl");
  dl.replaceChildren();
  const ring = gj.geometry.coordinates[0];
  const add = (k, v) => {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k;
    dd.textContent = v;
    dl.append(dt, dd);
  };
  add("CRS", gj.properties.crs);
  add("bbox", tj.bbox.map((v) => v.toFixed(4)).join(", "));
  add("zoom", `${tj.minzoom}\u2013${tj.maxzoom} (native)`);
  add("footprint", `${ring.length - 1} vertices`);
  $("info").hidden = false;
}

function setUrls() {
  const t = tileTemplate();
  $("u-xyz").textContent = t;
  $("u-josm").textContent = 'tms:'+t;
  $("urls").hidden = false;
}

// JOSM Remote Control (/imagery adds a TMS layer on port 8111, feature off by
// default in JOSM). url= must stay last: JOSM reads everything after it
// literally, our query's '&' included. bounds = minlat,minlon,maxlat,maxlon.
// no-cors fetch: only delivery matters, JOSM shows its own permission dialog.
function openInJosm() {
  if (!info) return;
  const [w, s, e, n] = info.bbox;
  fetch("http://127.0.0.1:8111/imagery?type=tms"
      + "&title=" + encodeURIComponent("cog2tms " + cogUrl())
      + `&bounds=${s},${w},${n},${e}`  // JOSM order: minlat,minlon,maxlat,maxlon
      + `&min_zoom=${info.minzoom ?? 0}&max_zoom=${info.maxzoom ?? MAX_ZOOM}`
      + "&url=" + tileTemplate(),
      {mode: "no-cors"})
    .catch(() => showError("JOSM not reachable on 127.0.0.1:8111 — enable "
        + "Remote Control in JOSM Preferences (Expert settings)"));
}

function showError(msg) {
  const e = $("err");
  e.textContent = msg;
  e.classList.toggle("show", !!msg);
}

// --- shareable state: page query mirrors u + options + view ---

function share() {
  const q = new URLSearchParams();
  if (cogUrl()) q.set("u", cogUrl());
  for (const f of FIELDS) {
    const el = $(f), v = el.value.trim();
    if (el.disabled || v === DEFAULTS[f]) continue;  // service defaults omitted
    q.set(f, v);
  }
  if (!$("base").checked) q.set("base", "0");
  if ($("opacity").value !== "100") q.set("opacity", $("opacity").value);
  if (info) q.set("c", `${map.getCenter().lat.toFixed(5)},${map.getCenter().lng.toFixed(5)},${Math.round(map.getZoom())}`);
  const s = q.toString();
  history.replaceState(null, "", s ? "?" + s : location.pathname);
}

let viewFromUrl = null;

function restore() {
  const q = new URLSearchParams(location.search);
  for (const f of FIELDS) {
    if (!q.has(f)) continue;
    const el = $(f), v = q.get(f);
    // selects must match an option (a hand-edited link could carry junk)
    if (el instanceof HTMLSelectElement && ![...el.options].some((o) => o.value === v)) continue;
    el.value = v;
  }
  if (q.get("base") === "0") $("base").checked = false;
  if (q.has("opacity")) $("opacity").value = q.get("opacity");
  if (q.has("u")) $("url").value = q.get("u");
  if (q.has("c")) {
    const [lat, lng, z] = q.get("c").split(",").map(Number);
    if ([lat, lng, z].every(Number.isFinite)) {
      viewFromUrl = true;
      return {center: [lng, lat], zoom: Math.min(z, MAX_ZOOM)};
    }
  }
  return null;
}

function tileAt(lng, lat, z) {  // standard slippy tile address at integer zoom z
  const n = 1 << z;
  const r = Math.max(Math.min(lat, 85.051129), -85.051129) * Math.PI / 180;
  const x = Math.floor((lng + 180) / 360 * n);
  const y = Math.floor((0.5 - Math.log(Math.tan(Math.PI / 4 + r / 2)) / (2 * Math.PI)) * n);
  return `${z}/${Math.min(Math.max(x, 0), n - 1)}/${Math.min(Math.max(y, 0), n - 1)}`;
}

// --- wiring ---

const savedView = restore();
map = new maplibregl.Map({
  container: "map",
  style: $("base").checked ? BASE_STYLE : BG_STYLE,
  maxZoom: MAX_ZOOM,
  // attribution always visible: the default control collapses to an ⓘ icon
  // once the canvas is <= 640 px wide (common with the side panel)
  attributionControl: {compact: false},
  ...(savedView ?? {center: [2.4, 46.6], zoom: 5}),
});
map.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
map.on("style.load", ensureOverlays);  // re-add overlays after every setStyle

// quality applies to jpeg and webp (auto resolves to webp or png depending on
// the browser's Accept header, so it stays useful under auto); blackmax
// creates transparency, which only jpeg cannot express
const syncFormatControls = () => {
  $("quality").disabled = $("format").value === "png";
  $("blackmax").disabled = $("format").value === "jpeg";
};
syncFormatControls();

$("loadform").addEventListener("submit", (e) => { e.preventDefault(); load(); });
$("josm-open").addEventListener("click", openInJosm);
for (const f of FIELDS) $(f).addEventListener("change", () => {
  if (f === "format") syncFormatControls();  // before rebuild: disabled controls drop out of the urls
  if (info) { ensureOverlays(); setUrls(); share(); }
});
$("base").addEventListener("change", () => { map.setStyle($("base").checked ? BASE_STYLE : BG_STYLE); share(); });
$("opacity").addEventListener("input", () => {
  if (map.getLayer("cog")) map.setPaintProperty("cog", "raster-opacity", $("opacity").value / 100);
  share();
});
map.on("moveend", share);
map.on("mousemove", (e) => { $("curtile").textContent = tileAt(e.lngLat.lng, e.lngLat.lat, Math.round(map.getZoom())); });
map.getCanvas().addEventListener("mouseout", () => { $("curtile").textContent = "\u2014"; });
for (const b of document.querySelectorAll("[data-copy]")) {
  b.addEventListener("click", async () => {
    const text = $(b.dataset.copy).textContent;
    try {
      await navigator.clipboard.writeText(text);
    } catch (e) {  // plain http: clipboard API unavailable, legacy fallback
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.append(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    b.textContent = "copied";
    setTimeout(() => { b.textContent = "copy"; }, 1200);
  });
}

if (cogUrl()) load();
