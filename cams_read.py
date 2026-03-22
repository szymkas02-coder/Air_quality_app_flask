"""
cams_read.py

Loads the pre-downloaded CAMS Europe Air Quality forecast from the
GitHub Release asset published daily by GitHub Actions.
No API keys or cloud storage accounts required.
"""

import re
import json
import logging
import os
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import xarray as xr
import streamlit as st

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GitHub Release URLs
# The "latest-data" tag is recreated daily by the Actions workflow.
# These are stable, public HTTPS URLs — no auth needed for public repos.
# ---------------------------------------------------------------------------

def _base_url() -> str:
    """
    Build the GitHub Release asset base URL.
    Reads GITHUB_REPO from st.secrets or environment variable.
    Format: "owner/repo-name"  e.g. "jan-kowalski/air-quality-app"
    """
    try:
        repo = st.secrets["GITHUB_REPO"]
    except Exception:
        repo = os.environ["GITHUB_REPO"]
    return f"https://github.com/{repo}/releases/download/latest-data"


NC_FILENAME   = "cams_latest.nc"
META_FILENAME = "cams_meta.json"


# ---------------------------------------------------------------------------
# Post-processing (unchanged from original cams_read.py)
# ---------------------------------------------------------------------------

def _add_absolute_time(ds: xr.Dataset) -> xr.Dataset:
    if "FORECAST" in ds.attrs:
        match = re.search(r"(\d{8})\+", ds.attrs["FORECAST"])
        if match:
            base_time = pd.to_datetime(match.group(1), format="%Y%m%d")
            abs_time = base_time + pd.to_timedelta(ds.time.values, unit="h")
            ds = ds.assign_coords(time=abs_time)
    return ds


def _postprocess(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sel(level=0).squeeze()

    var_rename = {
        "apg_conc":   "alder_pollen",
        "nh3_conc":   "ammonia",
        "bpg_conc":   "birch_pollen",
        "co_conc":    "carbon_monoxide",
        "gpg_conc":   "grass_pollen",
        "mpg_conc":   "mugwort_pollen",
        "no2_conc":   "nitrogen_dioxide",
        "no_conc":    "nitrogen_monoxide",
        "opg_conc":   "olive_pollen",
        "o3_conc":    "ozone",
        "pm2p5_conc": "particulate_matter_2.5um",
        "pm10_conc":  "particulate_matter_10um",
        "rwpg_conc":  "ragweed_pollen",
        "so2_conc":   "sulphur_dioxide",
    }
    rename_present = {k: v for k, v in var_rename.items() if k in ds.data_vars}
    ds = ds.rename(rename_present)

    if "latitude" in ds.coords and "longitude" in ds.coords:
        lats = np.round(ds.latitude.values, 3)
        lons = np.round(ds.longitude.values, 3)
        lons = np.where(lons > 180, lons - 360, lons)
        ds = ds.assign_coords(latitude=lats, longitude=lons)

    ds = _add_absolute_time(ds)
    return ds


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_bytes(url: str, label: str) -> bytes:
    """Download a file from a URL with a simple progress message."""
    log.info(f"Fetching {label} from {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    log.info(f"{label}: {len(resp.content) / 1e6:.1f} MB received")
    return resp.content


def get_latest_forecast_meta() -> dict | None:
    """Return the metadata dict (date, downloaded_at_utc) or None on error."""
    try:
        url = f"{_base_url()}/{META_FILENAME}"
        return json.loads(_fetch_bytes(url, "metadata"))
    except Exception as exc:
        log.warning(f"Could not fetch metadata: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_cams_air_quality() -> tuple[xr.Dataset | None, str | None]:
    """
    Download the pre-built CAMS NetCDF from the GitHub Release and return
    a processed xarray Dataset.

    Returns (ds, date_str) or (None, None) on failure.
    Cached for the lifetime of the Streamlit server process.
    Call st.cache_resource.clear() to force a refresh.
    """
    st.info("☁️ Loading CAMS data from GitHub Release...")

    try:
        nc_url = f"{_base_url()}/{NC_FILENAME}"
        nc_bytes = _fetch_bytes(nc_url, "NetCDF")

        ds = xr.open_dataset(BytesIO(nc_bytes), engine="scipy")
        ds = _postprocess(ds)

        meta = get_latest_forecast_meta()
        date_str = meta["date"] if meta else datetime.now(timezone.utc).strftime("%Y-%m-%d")

        st.success(f"✅ CAMS data loaded — forecast date: {date_str}")
        return ds, date_str

    except Exception:
        import traceback
        st.error(f"❌ Failed to load CAMS data:\n{traceback.format_exc()}")
        return None, None
