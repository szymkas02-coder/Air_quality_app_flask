"""
scripts/download_cams.py

Downloads today's CAMS Europe Air Quality forecast and saves it as a local .nc file.
Intended to be called from GitHub Actions, which then uploads the file as a release asset.
"""

import cdsapi
import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from zipfile import ZipFile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CDSAPI_URL = os.environ["CDSAPI_URL"]
CDSAPI_KEY = os.environ["CDSAPI_KEY"]

OUTPUT_NC   = Path("cams_latest.nc")
OUTPUT_META = Path("cams_meta.json")

DATASET = "cams-europe-air-quality-forecasts"
VARIABLES = [
    "alder_pollen", "ammonia", "birch_pollen", "carbon_monoxide",
    "grass_pollen", "mugwort_pollen", "nitrogen_dioxide", "nitrogen_monoxide",
    "olive_pollen", "ozone", "particulate_matter_2.5um", "particulate_matter_10um",
    "ragweed_pollen", "sulphur_dioxide",
]
AREA = [56, 7, 47, 26]  # [N, W, S, E]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_request(date_str: str) -> dict:
    return {
        "variable": VARIABLES,
        "model": ["ensemble"],
        "level": ["0"],
        "date": [f"{date_str}/{date_str}"],
        "type": ["forecast"],
        "time": ["00:00"],
        "leadtime_hour": [str(h) for h in range(97)],
        "data_format": "netcdf_zip",
        "area": AREA,
    }


def download(date_str: str) -> None:
    """Download CAMS forecast for date_str, extract .nc, write to OUTPUT_NC."""
    log.info(f"Initialising CDS client (URL: {CDSAPI_URL})")
    client = cdsapi.Client(url=CDSAPI_URL, key=CDSAPI_KEY)

    zip_path = Path("/tmp/cams_download.zip")
    log.info(f"Submitting request for {date_str} ...")
    client.retrieve(DATASET, build_request(date_str)).download(str(zip_path))
    log.info(f"ZIP downloaded ({zip_path.stat().st_size / 1e6:.1f} MB)")

    with ZipFile(zip_path) as zf:
        nc_name = zf.namelist()[0]
        log.info(f"Extracting '{nc_name}'")
        nc_bytes = zf.read(nc_name)

    OUTPUT_NC.write_bytes(nc_bytes)
    log.info(f"Saved → {OUTPUT_NC} ({OUTPUT_NC.stat().st_size / 1e6:.1f} MB)")
    zip_path.unlink(missing_ok=True)


def write_meta(date_str: str) -> None:
    """Write a small JSON sidecar with forecast date and download timestamp."""
    meta = {
        "date": date_str,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_META.write_text(json.dumps(meta, indent=2))
    log.info(f"Metadata written → {OUTPUT_META}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    now_utc = datetime.now(timezone.utc)

    # CAMS forecast isn't published until ~10:00 UTC — use yesterday before that
    if now_utc.hour < 10:
        target = (now_utc - timedelta(days=1)).date()
        log.warning(f"Before 10 UTC — using yesterday: {target}")
    else:
        target = now_utc.date()

    date_str = target.strftime("%Y-%m-%d")

    try:
        download(date_str)
        write_meta(date_str)
    except Exception as exc:
        log.error(f"Failed: {exc}")
        sys.exit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
