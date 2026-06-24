---
title: APU Load Forecasting
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---                                                                                                                               # APU Load Forecasting — 24-Hour Power Demand Prediction

A complete, end-to-end machine-learning system that forecasts electricity demand for a
132KV transmission network (utility "APU") 24 hours ahead, at 10-minute resolution, and
serves the forecast through a REST API and an interactive dashboard.

**Author:** Abhinav Nirapure · B.Tech Data Science & AI, IIIT Naya Raipur
**Stack:** Python · LightGBM · FastAPI · Chart.js · Docker

---

## What it does

Given a year of 10-minute interval load data for three feeders (F1, F2, F3), the system:

1. Cleans and validates the raw data (handling a non-obvious datetime-format corruption).
2. Enriches it with external weather (Open-Meteo API) and an India/Jharkhand holiday calendar.
3. Trains a LightGBM model to forecast **total system demand** (F1 + F2 + F3).
4. Produces a **recursive 24-hour forecast** (144 ten-minute blocks) via a FastAPI backend.
5. Visualises forecast-vs-actual load on an interactive web dashboard.

---

## Results

| Metric | Single-step (10-min ahead) | Recurcd sive 24-hour |
|---|---|---|
| **MAPE** | 0.58% | **4.6%** (mean over a held-out 21-day window) |
| Best day (24h) | — | 1.83% |

The 24-hour figure is the realistic operating metric: it is measured by recursively
forecasting all 144 blocks of each test day (feeding predictions back in as inputs),
which mirrors true deployment. The single-step figure is reported for reference only.

---


## Project structure

```
apu-load-forecasting/
├── api/
│   ├── main.py             # FastAPI app: endpoints + serves the dashboard
│   ├── forecaster.py       # Model loading + recursive forecasting logic
│   └── features.py         # Feature engineering (shared by training & serving)
├── data/
│   ├── raw/                # Original CSV + cached weather (not in container)
│   └── processed/          # Cleaned & enriched dataset (used at runtime)
├── frontend/
│   └── index.html          # Single-page Chart.js dashboard
├── models/
│   ├── lgbm_total_demand.pkl   # Trained model artifact
│   └── model_metadata.json     # Feature list + performance metadata
├── notebooks/
│   ├── 01_eda_cleaning.ipynb        # Milestone 1: EDA & cleaning
│   └── 02_feature_engineering.ipynb # Feature engineering + model training
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Running locally

**Prerequisites:** Python 3.11+ (developed on 3.14), and the dependencies in `requirements.txt`.

```bash
# 1. Clone
git clone https://github.com/<your-username>/apu-load-forecasting.git
cd apu-load-forecasting

# 2. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server (serves API + dashboard)
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open **http://127.0.0.1:8000/** for the dashboard, or **/docs** for the interactive API documentation.

---

## Running with Docker

The entire system (API + dashboard) runs in a single container.

```bash
docker build -t apu-forecast .
docker run -p 8000:8000 apu-forecast
```

Open **http://127.0.0.1:8000/**.

> The image uses `python:3.11-slim` for reliable pre-built ML library wheels. The application
> code is version-agnostic; 3.11 simply gives the most dependable container build.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/info` | Model metadata + available data range |
| GET | `/forecast?start=YYYY-MM-DD&horizon=144` | 24-hour recursive forecast from the given start |
| GET | `/historical?start=...&end=...` | Actual demand for a date range |

Example:
```
GET /forecast?start=2017-12-10&horizon=144
```

---

## Methodology highlights

**Data cleaning (Milestone 1).** The raw `Datetime` column silently mixed two formats — the
first ~12 days used `dd-mm-yyyy`, the remainder `m/d/yyyy`, and rows were out of chronological
order. A naive single-format parse corrupts ~60% of rows into invalid dates. This was detected
and corrected with format-aware parsing and re-sorting. After repair: a complete, gap-free,
strictly 10-minute timeline with no missing values. Extreme load values (e.g. the F3 summer
evening peak) were validated as **real demand events** — smooth, cross-feeder-correlated ramps —
and retained rather than removed.

**External features (Milestone 2).** Weather for Dhanbad, Jharkhand (2017) was sourced from the
Open-Meteo historical archive, adding **cloud cover** (absent from the provided data). A
reconciliation analysis showed the API weather and the dataset's own weather columns are weakly
correlated (temperature r ≈ 0.37) and describe different conditions; the dataset's internally
consistent weather columns were therefore used as primary model features, with API cloud cover
added as a supplementary signal. A 2017 India/Jharkhand public-holiday calendar (17 holidays,
including Chhath Puja and Jharkhand Formation Day) provides a holiday flag.

**Modeling (Milestone 3).** A single-step LightGBM regressor is applied recursively to produce
the 24-hour forecast. Features (29 total) include calendar fields, cyclical (sin/cos) time
encodings, weather, holiday flag, and strictly backward-looking lags (t−1, t−6, t−1 day,
t−1 week) and rolling statistics. **Leakage controls:** individual feeders (which sum to the
target) are excluded; all lag/rolling features look strictly backward; the train/test split is
chronological (never shuffled); hyperparameter tuning used `TimeSeriesSplit`. Two documented
improvement experiments (targeted features; time-aware hyperparameter search) established that
~4.6% is the natural error floor for single-model recursive forecasting on this data, driven by
error compounding during the steep morning demand ramp.

---

## Tech stack rationale

- **LightGBM** — fast, accurate gradient boosting well-suited to tabular time-series features.
- **FastAPI** — async REST framework with automatic interactive documentation.
- **Chart.js** — lightweight, dependency-free charting for the single-page dashboard.
- **Docker** — single-container packaging for reproducible deployment.

  ## Screenshots

### Interactive Dashboard — 24-Hour Forecast vs Actual
![Dashboard](docs/screenshots/Forecast1.png)

### Forecast vs Actual — Detail
![Forecast detail](docs/screenshots/forecast2.png)

### Analysis
![Analysis](docs/screenshots/forecast3.png) 


**🔗 Live Demo:** [abhinaviiitnr-apu-load-forecasting.hf.space](https://abhinaviiitnr-apu-load-forecasting.hf.space)

**📦 Repository:** [github.com/abhinaviiitnr/apu-load-forecasting](https://github.com/abhinaviiitnr/apu-load-forecasting)
