# Air Quality Forecast App — Flask

A Flask-based web dashboard for visualising real-time and forecast air quality across Poland, powered by CAMS (Copernicus Atmosphere Monitoring Service) data.

## Features

- REST API backend serving pollutant map overlays and per-site forecast time-series
- Interactive frontend with Plotly charts and a Leaflet.js map
- In-process dataset cache for fast repeated requests
- `/api/refresh` endpoint to reload data without restarting the server
- Deployable to Heroku (Procfile + runtime.txt included)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET |  | Main dashboard page |
| GET |  | Data freshness info |
| GET |  | Available pollutants |
| GET |  | Monitoring site list |
| GET |  | PNG map overlay + bounds |
| GET |  | Time-series JSON |
| GET |  | Current conditions + AQI |
| POST |  | Clear cache and reload data |

## Tech Stack

Python · Flask · xarray · Pandas · Plotly · Folium · Matplotlib · Pillow · Gunicorn

## Running locally

```bash
pip install -r requirements.txt
flask run
```

> CAMS data is loaded from a GitHub Release asset via . CDS API credentials are required for live data.
