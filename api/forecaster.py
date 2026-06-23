# api/forecaster.py
"""
Core forecasting logic: load the trained model and produce a recursive
24-hour (144-block) demand forecast, seeding lags from historical data.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# --- Resolve paths relative to the project root (works regardless of CWD) ---
BASE_DIR = Path(__file__).resolve().parent.parent      # project root
MODEL_PATH    = BASE_DIR / "models" / "lgbm_total_demand.pkl"
META_PATH     = BASE_DIR / "models" / "model_metadata.json"
DATA_PATH     = BASE_DIR / "data" / "processed" / "utility_enriched.csv"

FREQ = pd.Timedelta(minutes=10)


class DemandForecaster:
    """Loads the model + data once, then serves forecasts and history."""

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            self.meta = json.load(f)
        self.feature_cols = self.meta["feature_cols"]
        self.target = self.meta["target"]
        self.best_iter = self.meta["best_iteration"]

        # Load the enriched dataset and build the full feature set
        # (build_features = the SAME logic used at training time)
        from features import build_features
        raw = pd.read_csv(DATA_PATH, index_col="Datetime", parse_dates=["Datetime"])
        self.df = build_features(raw, dropna=False)
        # Precompute the demand series for fast lag lookups
        self.demand = self.df[self.target]

        self.data_start = self.df.index.min()
        self.data_end   = self.df.index.max()

    # ---- helpers ----
    def _static_features_for(self, t):
        """Calendar/weather/holiday features for time t (known in advance)."""
        # These come from the stored dataframe row (the forecast day's known context)
        if t not in self.df.index:
            raise KeyError(f"Timestamp {t} not in available data range.")
        return self.df.loc[t, self.feature_cols].copy()

    def forecast(self, start_time, horizon=144):
        """Recursive 24h (default 144-block) forecast beginning at start_time."""
        start_time = pd.Timestamp(start_time)

        # Need at least 1 week (1008 blocks) of history before start_time to seed lags
        hist_needed = start_time - 1008 * FREQ
        if hist_needed < self.data_start:
            raise ValueError(
                f"Not enough history before {start_time}. "
                f"Earliest forecastable start is {self.data_start + 1008 * FREQ}."
            )

        # Running history: real values up to start, predictions appended after
        demand_history = self.demand.loc[:start_time - FREQ].copy()

        preds = {}
        for step in range(horizon):
            t = start_time + step * FREQ
            row = self._static_features_for(t)

            def hv(ts):
                return demand_history.get(ts, np.nan)

            # lags
            row["lag_1"]    = hv(t - 1 * FREQ)
            row["lag_6"]    = hv(t - 6 * FREQ)
            row["lag_144"]  = hv(t - 144 * FREQ)
            row["lag_1008"] = hv(t - 1008 * FREQ)
            # rolling
            last6   = [hv(t - i * FREQ) for i in range(1, 7)]
            last144 = [hv(t - i * FREQ) for i in range(1, 145)]
            row["roll_mean_6"]   = np.nanmean(last6)
            row["roll_std_6"]    = np.nanstd(last6, ddof=1)
            row["roll_mean_144"] = np.nanmean(last144)
            row["roll_std_144"]  = np.nanstd(last144, ddof=1)

            # ensure exact feature order the model expects
            x = row[self.feature_cols].values.reshape(1, -1)
            pred = float(self.model.predict(x, num_iteration=self.best_iter)[0])

            preds[t] = pred
            demand_history[t] = pred

        return pd.Series(preds, name="forecast")

    def historical(self, start_time, end_time):
        """Actual demand for a date range (for plotting actual-vs-forecast)."""
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        sl = self.demand.loc[start_time:end_time]
        return sl