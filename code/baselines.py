"""Simple baseline forecasters.

Every baseline takes a `history` array of shape (L, T_hist) containing
observed outages up to the forecast issue time, plus a `horizon`, and
returns a prediction of shape (L, horizon). No peeking at the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zero_baseline(history: np.ndarray, horizon: int) -> np.ndarray:
    L = history.shape[0]
    return np.zeros((L, horizon), dtype=np.float32)


def persistence_baseline(history: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast = last observed value, repeated across the horizon."""
    last = history[:, -1]
    return np.tile(last[:, None], (1, horizon)).astype(np.float32)


def historical_mean_baseline(history: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast = per-county mean outage over training history."""
    mean = history.mean(axis=1)
    return np.tile(mean[:, None], (1, horizon)).astype(np.float32)


def recent_window_baseline(history: np.ndarray, horizon: int,
                           window: int = 24) -> np.ndarray:
    """Forecast = per-county mean over the last `window` hours.

    Captures ongoing storm/recovery state better than a long-run mean."""
    w = min(window, history.shape[1])
    mean = history[:, -w:].mean(axis=1)
    return np.tile(mean[:, None], (1, horizon)).astype(np.float32)


def exponential_decay_baseline(history: np.ndarray, horizon: int,
                               half_life_hours: float = 12.0) -> np.ndarray:
    """Exponentially decay the last observed value toward the long-run mean.

    Captures the recovery dynamics observed in EDA: outage counts after a
    peak fall back to background at rates consistent with a ~6–24h half-life.
    """
    last = history[:, -1]
    long_mean = history.mean(axis=1)
    lam = np.log(2) / half_life_hours
    t = np.arange(1, horizon + 1, dtype=np.float32)
    decay = np.exp(-lam * t)  # (horizon,)
    # broadcast: (L, 1) * (1, horizon) + (L, 1) * (1, horizon)
    pred = (last[:, None] - long_mean[:, None]) * decay[None, :] + long_mean[:, None]
    return np.clip(pred, 0, None).astype(np.float32)


def seasonal_naive_baseline(history: np.ndarray, horizon: int,
                             period: int = 24) -> np.ndarray:
    """Forecast hour-t = value from `period` hours ago."""
    T = history.shape[1]
    if T < period:
        return persistence_baseline(history, horizon)
    preds = np.empty((history.shape[0], horizon), dtype=np.float32)
    for h in range(horizon):
        src = T - period + (h % period)
        src = src if src < T else src - period
        preds[:, h] = history[:, src]
    return preds


BASELINES = {
    "zero": zero_baseline,
    "persistence": persistence_baseline,
    "historical_mean": historical_mean_baseline,
    "recent_24h_mean": lambda h, H: recent_window_baseline(h, H, window=24),
    "recent_72h_mean": lambda h, H: recent_window_baseline(h, H, window=72),
    "exp_decay_12h": lambda h, H: exponential_decay_baseline(h, H, half_life_hours=12.0),
    "exp_decay_24h": lambda h, H: exponential_decay_baseline(h, H, half_life_hours=24.0),
    "seasonal_24h": lambda h, H: seasonal_naive_baseline(h, H, period=24),
}
