# api/main.py
"""
FastAPI backend serving 24-hour power demand forecasts.
Endpoints:
  GET /                 -> health check
  GET /forecast?start=  -> 144-block (24h) recursive forecast from a start datetime
  GET /historical?start=&end=  -> actual demand for a range (for actual-vs-forecast plots)
  GET /info             -> model metadata + available data range
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from forecaster import DemandForecaster

app = FastAPI(title="APU Load Forecasting API", version="1.0")

# Allow the frontend (served from a different port/file) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # fine for a local demo; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + data ONCE at startup (not per request)
forecaster = DemandForecaster()


@app.get("/health")
def health():
    return {"status": "ok", "message": "APU Load Forecasting API is running."}


@app.get("/info")
def info():
    return {
        "model": "LightGBM (recursive 24h)",
        "target": forecaster.target,
        "n_features": len(forecaster.feature_cols),
        "data_start": str(forecaster.data_start),
        "data_end": str(forecaster.data_end),
        "recursive_24h_mape_pct": forecaster.meta.get("recursive_24h_mape_pct"),
        "single_step_mape_pct": forecaster.meta.get("single_step_mape_pct"),
        "earliest_forecast_start": str(forecaster.data_start + 1008 * pd.Timedelta(minutes=10)),
    }


@app.get("/forecast")
def forecast(start: str = Query(..., description="Start datetime, e.g. 2017-12-10 or 2017-12-10 00:00:00"),
             horizon: int = Query(144, ge=1, le=144, description="Number of 10-min blocks (max 144 = 24h)")):
    try:
        series = forecaster.forecast(start, horizon=horizon)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "start": start,
        "horizon": horizon,
        "forecast": [
            {"timestamp": ts.isoformat(), "predicted_demand": round(val, 2)}
            for ts, val in series.items()
        ],
    }


@app.get("/historical")
def historical(start: str = Query(..., description="Range start datetime"),
               end: str = Query(..., description="Range end datetime")):
    try:
        series = forecaster.historical(start, end)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(series) == 0:
        raise HTTPException(status_code=404, detail="No data in that range.")

    return {
        "start": start,
        "end": end,
        "actual": [
            {"timestamp": ts.isoformat(), "demand": round(val, 2)}
            for ts, val in series.items()
        ],
    }
# --- Serve the frontend dashboard ---
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

@app.get("/")
def dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")