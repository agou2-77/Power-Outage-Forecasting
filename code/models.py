"""Forecasting models.

Three families, all of which produce (n_counties, horizon) predictions from
history available at the issue time:

  1. `LGBMDirectForecaster` — one LightGBM regressor trained on all
     (county, horizon) pairs with horizon_h as a feature. Target is log1p(y).
  2. `SarimaxPerCounty`    — fit SARIMAX(p,d,q) to each county's out series.
  3. `Seq2SeqForecaster`   — LSTM encoder + per-horizon head, shared across
     counties, features = past outage & weather.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

import features as ft


# -----------------------------------------------------------------------------
# LightGBM direct multi-horizon
# -----------------------------------------------------------------------------

@dataclass
class LGBMDirectForecaster:
    """Gradient-boosted direct multi-horizon forecaster.

    Train once across all (issue_time, county, horizon) samples; at inference,
    produce predictions for h = 1..horizon_max by replicating the issue-time
    feature row across horizons.
    """
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 600
    min_child_samples: int = 50
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambda_l2: float = 0.1
    random_state: int = 42
    feature_cols: list[str] = field(default_factory=list)
    model: object | None = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None,
            early_stopping_rounds: int = 50, verbose: int = 50):
        import lightgbm as lgb
        self.feature_cols = [c for c in train_df.columns
                             if c not in ("y_target", "y_target_log",
                                           "location_idx", "timestamp_idx")]
        X = train_df[self.feature_cols].values
        y = train_df["y_target_log"].values.astype(np.float32)

        train_set = lgb.Dataset(X, label=y, feature_name=self.feature_cols)
        callbacks = [lgb.log_evaluation(period=verbose)]
        valid_sets = [train_set]
        valid_names = ["train"]
        if val_df is not None and len(val_df) > 0:
            Xv = val_df[self.feature_cols].values
            yv = val_df["y_target_log"].values.astype(np.float32)
            valid_sets.append(lgb.Dataset(Xv, label=yv, reference=train_set))
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds,
                                                verbose=False))

        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "verbose": -1,
            "seed": self.random_state,
        }

        self.model = lgb.train(params, train_set,
                               num_boost_round=self.n_estimators,
                               valid_sets=valid_sets,
                               valid_names=valid_names,
                               callbacks=callbacks)
        return self

    def predict(self, panel: pd.DataFrame, issue_time_idx: int,
                horizons: Sequence[int]) -> np.ndarray:
        """Predict (L, len(horizons)) outages for the given issue time."""
        feat_row = panel[panel["timestamp_idx"] == issue_time_idx].copy()
        feat_row = feat_row.sort_values("location_idx").reset_index(drop=True)
        L = len(feat_row)
        out = np.zeros((L, len(horizons)), dtype=np.float32)
        base_cols = [c for c in self.feature_cols if c != "horizon_h"]
        X_base = feat_row[base_cols].values  # (L, F-1)
        has_horizon = "horizon_h" in self.feature_cols
        h_idx = self.feature_cols.index("horizon_h") if has_horizon else None

        for j, h in enumerate(horizons):
            if has_horizon:
                X_full = np.empty((L, len(self.feature_cols)), dtype=np.float32)
                # place horizon_h in the right column
                other_slots = [i for i in range(len(self.feature_cols)) if i != h_idx]
                X_full[:, other_slots] = X_base
                X_full[:, h_idx] = float(h)
            else:
                X_full = X_base
            yhat_log = self.model.predict(X_full)
            out[:, j] = np.clip(np.expm1(yhat_log), 0, None)
        return out


# -----------------------------------------------------------------------------
# Per-county SARIMAX
# -----------------------------------------------------------------------------

@dataclass
class SarimaxPerCounty:
    order: tuple = (2, 0, 2)
    log_target: bool = True
    models: dict = field(default_factory=dict)

    def fit(self, out_history: np.ndarray, locations: np.ndarray, verbose: bool = False):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self.models = {}
        for i, loc in enumerate(locations):
            y = out_history[i].astype(np.float64)
            if self.log_target:
                y = np.log1p(y)
            if len(y) < 20 or np.allclose(y, y[0]):
                self.models[loc] = None
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = SARIMAX(y, order=self.order,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
                    res = mod.fit(disp=False, maxiter=50)
                self.models[loc] = res
            except Exception as e:
                if verbose:
                    print(f"  SARIMAX failed for {loc}: {e!s:.80s}")
                self.models[loc] = None
        return self

    def predict(self, locations: np.ndarray, horizon: int) -> np.ndarray:
        L = len(locations)
        out = np.zeros((L, horizon), dtype=np.float32)
        for i, loc in enumerate(locations):
            m = self.models.get(loc)
            if m is None:
                continue
            try:
                fc = np.asarray(m.forecast(steps=horizon), dtype=np.float64)
                if self.log_target:
                    fc = np.expm1(fc)
                out[i] = np.clip(fc, 0, None)
            except Exception:
                pass
        return out


# -----------------------------------------------------------------------------
# Simple ensemble blender
# -----------------------------------------------------------------------------

def blend_predictions(preds: list[np.ndarray],
                      weights: list[float] | None = None) -> np.ndarray:
    """Linear blend of predictions (all shape (L, H))."""
    if weights is None:
        weights = [1.0 / len(preds)] * len(preds)
    stacked = np.stack(preds, axis=0)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 1, 1)
    return (stacked * w).sum(axis=0)
