"""
app.py — Flask backend for the Air Quality Forecast Dashboard.

Endpoints:
  GET  /                        → main page (index.html)
  GET  /api/status              → data freshness info (date, downloaded_at)
  GET  /api/pollutants          → list of available pollutants with display names
  GET  /api/sites               → list of available sites
  GET  /api/map/<pollutant>/<time_index>  → PNG overlay image + bounds as JSON
  GET  /api/forecast/<site>/<pollutant>  → time-series JSON for Plotly
  GET  /api/current/<site>/<time_index>  → current conditions + AQI for one site
  POST /api/refresh             → clear cached dataset and reload from GitHub
"""

from flask import Flask, jsonify, request, render_template, send_file
import numpy as np
import pandas as pd
import json
import io
import base64
import os
import logging
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for server use
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image

from cams_read import get_cams_air_quality, get_latest_forecast_meta

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# Load site list once at startup
with open("sample_sites_poland.json", "r", encoding="utf-8") as f:
    SAMPLE_SITES = json.load(f)

# Module-level dataset cache (lives for the lifetime of the process)
_ds = None
_data_date = None


def get_dataset():
    """Return cached dataset, loading from GitHub Release on first call."""
    global _ds, _data_date
    if _ds is None:
        log.info("Loading CAMS dataset from GitHub Release...")
        _ds, _data_date = get_cams_air_quality()
    return _ds, _data_date


# ---------------------------------------------------------------------------
# AQI helpers (same logic as original app.py)
# ---------------------------------------------------------------------------

def compute_aqi(pm25, pm10):
    """Simplified AQI from PM2.5 and PM10."""
    aqi = 0

    if pm25 <= 12:
        aqi = max(aqi, 50 * (pm25 / 12))
    elif pm25 <= 35.4:
        aqi = max(aqi, 51 + 49 * ((pm25 - 12) / (35.4 - 12)))
    elif pm25 <= 55.4:
        aqi = max(aqi, 101 + 49 * ((pm25 - 35.4) / (55.4 - 35.4)))
    elif pm25 <= 150.4:
        aqi = max(aqi, 151 + 49 * ((pm25 - 55.4) / (150.4 - 55.4)))
    elif pm25 <= 250.4:
        aqi = max(aqi, 201 + 49 * ((pm25 - 150.4) / (250.4 - 150.4)))
    else:
        aqi = max(aqi, 251 + 49 * ((pm25 - 250.4) / (500.4 - 250.4)))

    if pm10 <= 54:
        aqi = max(aqi, 50 * (pm10 / 54))
    elif pm10 <= 154:
        aqi = max(aqi, 51 + 49 * ((pm10 - 54) / (154 - 54)))
    elif pm10 <= 254:
        aqi = max(aqi, 101 + 49 * ((pm10 - 154) / (254 - 154)))
    elif pm10 <= 354:
        aqi = max(aqi, 151 + 49 * ((pm10 - 254) / (354 - 254)))
    elif pm10 <= 424:
        aqi = max(aqi, 201 + 49 * ((pm10 - 354) / (424 - 354)))
    else:
        aqi = max(aqi, 251 + 49 * ((pm10 - 424) / (604 - 424)))

    return min(500, int(aqi))


def aqi_category(aqi):
    if aqi <= 50:   return "Good",                          "#22c55e"
    if aqi <= 100:  return "Moderate",                      "#eab308"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups", "#f97316"
    if aqi <= 200:  return "Unhealthy",                     "#ef4444"
    if aqi <= 300:  return "Very Unhealthy",                "#a855f7"
    return               ("Hazardous",                      "#7f1d1d")


# ---------------------------------------------------------------------------
# Map overlay helper
# ---------------------------------------------------------------------------

CMAP = plt.cm.get_cmap("YlOrRd")


def build_map_overlay(ds, pollutant, time_index):
    """
    Render pollutant grid as a base64 PNG and return bounds.
    Returns dict with keys: image (data URI), bounds [[S,W],[N,E]], vmin, vmax, units.
    """
    data = ds[pollutant].isel(time=time_index)
    lats = data.latitude.values
    lons = data.longitude.values
    values = data.values.astype(float)

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    rgba = CMAP(norm(values))
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")

    units = ds[pollutant].attrs.get("units", "µg/m³")

    return {
        "image": f"data:image/png;base64,{encoded}",
        "bounds": [[float(lats.min()), float(lons.min())],
                   [float(lats.max()), float(lons.max())]],
        "vmin": round(vmin, 4),
        "vmax": round(vmax, 4),
        "units": units,
    }


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    meta = get_latest_forecast_meta()
    _, data_date = get_dataset()
    return jsonify({
        "date": data_date,
        "downloaded_at_utc": meta.get("downloaded_at_utc") if meta else None,
        "ok": True,
    })


@app.route("/api/sites")
def api_sites():
    sites = [{"id": k, "name": k, "lat": v["lat"], "lon": v["lon"]}
             for k, v in SAMPLE_SITES.items()]
    return jsonify(sites)


@app.route("/api/pollutants")
def api_pollutants():
    ds, _ = get_dataset()
    if ds is None:
        return jsonify({"error": "Dataset not available"}), 503

    skip = {"latitude", "longitude", "time", "step"}
    result = []
    for var in ds.data_vars:
        if var in skip:
            continue
        attrs = ds[var].attrs
        species = attrs.get("species", var.replace("_", " ").title())
        if var == "particulate_matter_2.5um":
            species = "PM2.5"
        elif var == "particulate_matter_10um":
            species = "PM10"
        species = species.replace("Aerosol", "").replace("Grain", "").strip()
        units = attrs.get("units", "")
        label = f"{species} ({units})" if units else species
        result.append({"id": var, "label": label, "species": species, "units": units})

    return jsonify(result)


@app.route("/api/times")
def api_times():
    ds, _ = get_dataset()
    if ds is None:
        return jsonify({"error": "Dataset not available"}), 503

    times = pd.to_datetime(ds.time.values)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    diffs = [abs((t - now).total_seconds()) for t in times]
    default_index = int(np.argmin(diffs))

    return jsonify({
        "times": [t.strftime("%Y-%m-%d %H:%M UTC") for t in times],
        "default_index": default_index,
    })


@app.route("/api/map/<pollutant>/<int:time_index>")
def api_map(pollutant, time_index):
    ds, _ = get_dataset()
    if ds is None:
        return jsonify({"error": "Dataset not available"}), 503
    if pollutant not in ds.data_vars:
        return jsonify({"error": f"Unknown pollutant: {pollutant}"}), 400
    if time_index < 0 or time_index >= len(ds.time):
        return jsonify({"error": "time_index out of range"}), 400

    try:
        overlay = build_map_overlay(ds, pollutant, time_index)
        return jsonify(overlay)
    except Exception as e:
        log.exception("Error building map overlay")
        return jsonify({"error": str(e)}), 500


@app.route("/api/forecast/<path:site_id>/<pollutant>")
def api_forecast(site_id, pollutant):
    ds, _ = get_dataset()
    if ds is None:
        return jsonify({"error": "Dataset not available"}), 503
    if site_id not in SAMPLE_SITES:
        return jsonify({"error": f"Unknown site: {site_id}"}), 400
    if pollutant not in ds.data_vars:
        return jsonify({"error": f"Unknown pollutant: {pollutant}"}), 400

    site = SAMPLE_SITES[site_id]
    lat_idx = int(np.abs(ds.latitude.values - site["lat"]).argmin())
    lon_idx = int(np.abs(ds.longitude.values - site["lon"]).argmin())

    series = ds[pollutant].isel(latitude=lat_idx, longitude=lon_idx)
    times = pd.to_datetime(ds.time.values)
    values = series.values.tolist()
    units = ds[pollutant].attrs.get("units", "")

    return jsonify({
        "times": [t.strftime("%Y-%m-%d %H:%M") for t in times],
        "values": values,
        "units": units,
        "pollutant": pollutant,
        "site": site_id,
    })


@app.route("/api/current/<path:site_id>/<int:time_index>")
def api_current(site_id, time_index):
    ds, _ = get_dataset()
    if ds is None:
        return jsonify({"error": "Dataset not available"}), 503
    if site_id not in SAMPLE_SITES:
        return jsonify({"error": f"Unknown site: {site_id}"}), 400

    site = SAMPLE_SITES[site_id]
    lat_idx = int(np.abs(ds.latitude.values - site["lat"]).argmin())
    lon_idx = int(np.abs(ds.longitude.values - site["lon"]).argmin())

    key_pollutants = {
        "particulate_matter_2.5um": "PM2.5",
        "particulate_matter_10um":  "PM10",
        "nitrogen_dioxide":         "NO₂",
        "ozone":                    "O₃",
        "sulphur_dioxide":          "SO₂",
        "carbon_monoxide":          "CO",
    }

    current = {}
    for var, label in key_pollutants.items():
        if var in ds.data_vars:
            try:
                val = float(ds[var].isel(time=time_index, latitude=lat_idx, longitude=lon_idx).values)
                if not np.isnan(val):
                    units = ds[var].attrs.get("units", "")
                    current[label] = {"value": round(val, 3), "units": units}
            except Exception:
                pass

    # AQI
    aqi_data = None
    pm25 = current.get("PM2.5", {}).get("value")
    pm10 = current.get("PM10", {}).get("value")
    if pm25 is not None and pm10 is not None:
        aqi = compute_aqi(pm25, pm10)
        cat, color = aqi_category(aqi)
        aqi_data = {"value": aqi, "category": cat, "color": color}

    return jsonify({
        "site": site_id,
        "lat": site["lat"],
        "lon": site["lon"],
        "pollutants": current,
        "aqi": aqi_data,
    })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    global _ds, _data_date
    _ds = None
    _data_date = None
    log.info("Cache cleared — dataset will reload on next request.")
    return jsonify({"ok": True, "message": "Cache cleared. Data will reload on next request."})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
