"""Evaluation metrics for outage forecasting.

The grading rubric is RMSE averaged across counties, computed separately
for the 24-hour and 48-hour horizons. Per-county RMSE = sqrt(mean((y - yhat)^2))
over the horizon; the reported metric averages over counties.
"""
from __future__ import annotations

import numpy as np


def per_county_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute RMSE per county.

    Both arrays must have shape (location, horizon).
    Returns a (location,) array of RMSEs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))


def mean_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE averaged across counties — the headline submission metric."""
    return float(per_county_rmse(y_true, y_pred).mean())


def rmse_report(y_true: np.ndarray, y_pred: np.ndarray,
                locations: np.ndarray, k_worst: int = 10) -> dict:
    """Rich RMSE report: overall, median, and per-county ranking."""
    per = per_county_rmse(y_true, y_pred)
    order = np.argsort(-per)
    worst = [(locations[i], float(per[i])) for i in order[:k_worst]]
    best = [(locations[i], float(per[i])) for i in order[-k_worst:][::-1]]
    return {
        "mean_rmse": float(per.mean()),
        "median_rmse": float(np.median(per)),
        "max_rmse": float(per.max()),
        "min_rmse": float(per.min()),
        "worst_counties": worst,
        "best_counties": best,
        "per_county": per,
    }
