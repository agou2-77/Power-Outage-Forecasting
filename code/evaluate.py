"""Cross-validation evaluation harness.

Runs rolling-origin backtests for baselines + trained models and
produces a CV-mean RMSE summary per model.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

import data_utils as du
import features as ft
import metrics as m
import splits as sp
import baselines as bl
from models import LGBMDirectForecaster, SarimaxPerCounty


def _train_lgbm(panel: pd.DataFrame, cols: list[str],
                t_train_end: int, issue_time_stride: int = 3,
                n_estimators: int = 400, num_leaves: int = 63,
                learning_rate: float = 0.05, verbose: int = 0,
                ) -> LGBMDirectForecaster:
    train_df = ft.build_direct_training_set(
        panel, cols, horizons=list(range(1, 49)),
        t_end_exclusive=t_train_end, t_start=200,
        issue_time_stride=issue_time_stride, keep_zero_rate=0.3,
    )
    # Use last ~10% by timestamp for inner val
    t_cut = train_df["timestamp_idx"].quantile(0.9)
    tr = train_df[train_df["timestamp_idx"] <= t_cut]
    va = train_df[train_df["timestamp_idx"] > t_cut]
    model = LGBMDirectForecaster(n_estimators=n_estimators,
                                  num_leaves=num_leaves,
                                  learning_rate=learning_rate)
    model.fit(tr, val_df=va, verbose=verbose or 1_000_000)
    return model


def backtest_baselines(data, horizon: int = 48, n_folds: int = 5,
                        stride: int = 48) -> pd.DataFrame:
    T = data.out.shape[1]
    folds = sp.rolling_origin_splits(T, horizon, n_folds=n_folds, stride=stride)
    rows = []
    for split in folds:
        truth = data.out[:, split.target_slice()]
        history = data.out[:, :split.train_end]
        issue_time = data.timestamps[split.train_end - 1]
        for name, fn in bl.BASELINES.items():
            pred = fn(history, horizon)
            rows.append({
                "model": name, "issue_time": issue_time,
                "rmse_24": m.mean_rmse(truth[:, :24], pred[:, :24]),
                "rmse_48": m.mean_rmse(truth, pred),
            })
    return pd.DataFrame(rows)


def backtest_lgbm(data, n_folds: int = 5, stride: int = 48,
                   issue_time_stride: int = 3, verbose: int = 0) -> pd.DataFrame:
    T = data.out.shape[1]
    horizon = 48
    folds = sp.rolling_origin_splits(T, horizon, n_folds=n_folds, stride=stride)

    # One panel shared across folds
    panel, cols = ft.build_panel_features(
        out=data.out, tracked=data.tracked,
        weather=data.weather, feature_names=data.features,
        timestamps=data.timestamps,
    )

    rows = []
    for split in folds:
        t0 = time.time()
        model = _train_lgbm(panel, cols,
                            t_train_end=split.train_end,
                            issue_time_stride=issue_time_stride)
        fit_time = time.time() - t0
        issue_t = split.train_end - 1
        pred = model.predict(panel, issue_time_idx=issue_t,
                             horizons=list(range(1, 49)))
        truth = data.out[:, split.target_slice()]
        rows.append({
            "model": "lgbm",
            "issue_time": data.timestamps[issue_t],
            "rmse_24": m.mean_rmse(truth[:, :24], pred[:, :24]),
            "rmse_48": m.mean_rmse(truth, pred),
            "fit_time_s": fit_time,
            "best_iter": model.model.best_iteration,
        })
        if verbose:
            print(f"  fold @ {rows[-1]['issue_time']}: "
                  f"24={rows[-1]['rmse_24']:.2f}, 48={rows[-1]['rmse_48']:.2f}, "
                  f"fit={fit_time:.1f}s")
    return pd.DataFrame(rows)


def backtest_sarimax(data, n_folds: int = 5, stride: int = 48,
                      order: tuple = (2, 0, 2), verbose: int = 0) -> pd.DataFrame:
    T = data.out.shape[1]
    horizon = 48
    folds = sp.rolling_origin_splits(T, horizon, n_folds=n_folds, stride=stride)
    rows = []
    for split in folds:
        t0 = time.time()
        history = data.out[:, :split.train_end]
        model = SarimaxPerCounty(order=order).fit(history, data.locations, verbose=False)
        fit_time = time.time() - t0
        pred = model.predict(data.locations, horizon)
        truth = data.out[:, split.target_slice()]
        rows.append({
            "model": f"sarimax{order}",
            "issue_time": data.timestamps[split.train_end - 1],
            "rmse_24": m.mean_rmse(truth[:, :24], pred[:, :24]),
            "rmse_48": m.mean_rmse(truth, pred),
            "fit_time_s": fit_time,
        })
        if verbose:
            print(f"  fold @ {rows[-1]['issue_time']}: "
                  f"24={rows[-1]['rmse_24']:.2f}, 48={rows[-1]['rmse_48']:.2f}, "
                  f"fit={fit_time:.1f}s")
    return pd.DataFrame(rows)


def summarize(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(df_list, ignore_index=True)
    summary = (df.groupby("model")[["rmse_24", "rmse_48"]]
               .agg(["mean", "median", "max"]).round(2))
    return summary
