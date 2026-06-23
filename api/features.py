# api/features.py
"""
Single source of truth for feature engineering.
Both model training and the serving API use this function, guaranteeing
the features at inference exactly match those at training time.
"""
import numpy as np
import pandas as pd

TARGET = "Total_Demand"

# The exact 29 features the model expects (order matters for prediction).
FEATURE_COLS = [
    'Temperature', 'Humidity', 'WindSpeed', 'CloudCover', 'IsHoliday',
    'hour', 'minute', 'dayofweek', 'day', 'month', 'dayofyear', 'weekofyear', 'is_weekend',
    'hour_sin', 'hour_cos', 'tod_sin', 'tod_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'lag_1', 'lag_6', 'lag_144', 'lag_1008',
    'roll_mean_6', 'roll_std_6', 'roll_mean_144', 'roll_std_144',
]


def add_time_features(df):
    """Calendar + cyclical encodings (period-correct sin/cos)."""
    df = df.copy()
    idx = df.index

    df['hour']       = idx.hour
    df['minute']     = idx.minute
    df['dayofweek']  = idx.dayofweek
    df['day']        = idx.day
    df['month']      = idx.month
    df['dayofyear']  = idx.dayofyear
    df['weekofyear'] = idx.isocalendar().week.astype(int)
    df['is_weekend'] = (idx.dayofweek >= 5).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    minute_of_day = df['hour'] * 60 + df['minute']
    df['tod_sin'] = np.sin(2 * np.pi * minute_of_day / 1440)
    df['tod_cos'] = np.cos(2 * np.pi * minute_of_day / 1440)

    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df


def add_lag_features(df, target=TARGET):
    """Backward-looking lags and rolling stats (no leakage)."""
    df = df.copy()

    df['lag_1']    = df[target].shift(1)
    df['lag_6']    = df[target].shift(6)
    df['lag_144']  = df[target].shift(144)
    df['lag_1008'] = df[target].shift(1008)

    shifted = df[target].shift(1)   # ensures rolling window ends at t-1
    df['roll_mean_6']   = shifted.rolling(window=6).mean()
    df['roll_std_6']    = shifted.rolling(window=6).std()
    df['roll_mean_144'] = shifted.rolling(window=144).mean()
    df['roll_std_144']  = shifted.rolling(window=144).std()

    return df


def build_features(df, target=TARGET, dropna=False):
    """
    Full feature pipeline: base enriched data -> all 29 model features.
    Set dropna=True for training (removes warm-up rows with incomplete lags).
    For serving we keep all rows (lags are filled in recursively).
    """
    df = add_time_features(df)
    df = add_lag_features(df, target=target)
    if dropna:
        df = df.dropna()
    return df