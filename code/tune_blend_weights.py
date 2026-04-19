"""Scan blend weights on the internal hold-out.

The 24h and 48h predictions are graded separately, so we search for the best
(w_lgbm, w_s2s, w_decay) combination for each horizon independently.

We re-use the cached hold-out predictions on disk rather than re-fitting.
Uses a 2-stage grid search.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

import data_utils as du
import metrics as m
import features as ft
import baselines as bl
from models import LGBMDirectForecaster


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _make_cached_lgbm(data: du.OutageData, panel, cols,
                      t_end_exclusive: int, issue_t: int) -> np.ndarray:
    """Fit LGBM with the canonical final config and predict from issue_t."""
    train_df = ft.build_direct_training_set(
        panel, cols, horizons=list(range(1, 49)),
        t_end_exclusive=t_end_exclusive, t_start=200,
        issue_time_stride=2, keep_zero_rate=0.3,
    )
    t_cut = train_df["timestamp_idx"].quantile(0.92)
    tr = train_df[train_df["timestamp_idx"] <= t_cut]
    va = train_df[train_df["timestamp_idx"] > t_cut]
    mdl = LGBMDirectForecaster(n_estimators=600, num_leaves=63, learning_rate=0.05,
                                min_child_samples=50, feature_fraction=0.8,
                                bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.1)
    mdl.fit(tr, val_df=va, verbose=0)
    return mdl.predict(panel, issue_time_idx=issue_t,
                        horizons=list(range(1, 49)))


def main() -> None:
    data = du.load_train()
    T = data.out.shape[1]
    hold_out = 48
    truth = data.out[:, T - hold_out:]

    # LGBM hold-out prediction (same config used everywhere)
    panel, cols = ft.build_panel_features(
        out=data.out, tracked=data.tracked,
        weather=data.weather, feature_names=data.features,
        timestamps=data.timestamps,
    )
    pred_lgbm = _make_cached_lgbm(data, panel, cols,
                                    t_end_exclusive=T - hold_out,
                                    issue_t=T - hold_out - 1)
    pred_s2s = np.load(os.path.join(RESULTS_DIR, "pred_s2s_holdout.npy"))
    pred_decay = bl.exponential_decay_baseline(
        data.out[:, :T - hold_out], horizon=48, half_life_hours=18.0,
    )

    # Component RMSEs
    print("Component RMSEs on hold-out:")
    for name, pred in [("lgbm", pred_lgbm), ("seq2seq", pred_s2s), ("decay", pred_decay)]:
        print(f"  {name:8s} 24h={m.mean_rmse(truth[:, :24], pred[:, :24]):.2f}  "
              f"48h={m.mean_rmse(truth, pred):.2f}")

    # Grid search per horizon
    ws = np.arange(0.0, 1.01, 0.05)
    best = {"24": (float("inf"), None), "48": (float("inf"), None)}
    for w_lgbm in ws:
        for w_s2s in ws:
            w_decay = 1.0 - w_lgbm - w_s2s
            if w_decay < 0 or w_decay > 1.0:
                continue
            pred_blend = w_lgbm * pred_lgbm + w_s2s * pred_s2s + w_decay * pred_decay
            r24 = m.mean_rmse(truth[:, :24], pred_blend[:, :24])
            r48 = m.mean_rmse(truth, pred_blend)
            if r24 < best["24"][0]:
                best["24"] = (r24, (w_lgbm, w_s2s, w_decay))
            if r48 < best["48"][0]:
                best["48"] = (r48, (w_lgbm, w_s2s, w_decay))

    print("\nBest weights found:")
    print(f"  24h: RMSE={best['24'][0]:.3f}  weights(lgbm,s2s,decay)={best['24'][1]}")
    print(f"  48h: RMSE={best['48'][0]:.3f}  weights(lgbm,s2s,decay)={best['48'][1]}")

    # Also report combined score (24+48)/2 blend with one set of weights
    best_combined = (float("inf"), None)
    for w_lgbm in ws:
        for w_s2s in ws:
            w_decay = 1.0 - w_lgbm - w_s2s
            if w_decay < 0 or w_decay > 1.0:
                continue
            pred_blend = w_lgbm * pred_lgbm + w_s2s * pred_s2s + w_decay * pred_decay
            r24 = m.mean_rmse(truth[:, :24], pred_blend[:, :24])
            r48 = m.mean_rmse(truth, pred_blend)
            avg = 0.5 * (r24 + r48)
            if avg < best_combined[0]:
                best_combined = (avg, (w_lgbm, w_s2s, w_decay, r24, r48))
    print(f"  avg 24h/48h: RMSE={best_combined[0]:.3f}  "
          f"weights={best_combined[1][:3]}  (24h={best_combined[1][3]:.2f}, 48h={best_combined[1][4]:.2f})")


if __name__ == "__main__":
    main()
