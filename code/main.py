"""End-to-end pipeline (LightGBM + exponential-decay blend).

Trains the final model on the full training data, writes predictions for the
24h and 48h horizons, runs the greedy generator allocation, and emits the
three submission artifacts required by the grader:

  results/predictions_24h.csv
  results/predictions_48h.csv
  results/recommended_counties.txt

The pipeline is deliberately a small number of idempotent steps so it can
also be driven from the final Jupyter notebook.

The Seq2Seq (LSTM) experiment is trained in a separate process
(``make_s2s_predictions.py``) to avoid an OpenMP runtime conflict between
PyTorch and LightGBM on macOS.  If the resulting ``results/pred_s2s.npy``
file exists it is loaded and included in the blend with a small weight; if
not, the blend falls back to LightGBM + exponential decay only.
"""
from __future__ import annotations

import os

# Avoid OpenMP runtime conflict if torch happens to be imported downstream.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time

import numpy as np
import pandas as pd

import data_utils as du
import features as ft
import metrics as m
import splits as sp
import baselines as bl
import policy as pol
import submission as sub
from models import LGBMDirectForecaster, blend_predictions


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TEMPLATE_24H = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "dataset", "submission_template_24h.csv",
)
TEMPLATE_48H = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "dataset", "submission_template_48h.csv",
)


# ---------------------------------------------------------------------------
# LGBM training helper
# ---------------------------------------------------------------------------

def train_lgbm_final(data: du.OutageData, panel: pd.DataFrame,
                      cols: list[str],
                      n_estimators: int = 600,
                      num_leaves: int = 63,
                      learning_rate: float = 0.05,
                      verbose: int = 200) -> LGBMDirectForecaster:
    """Fit LightGBM on the full training data with a recent-slice inner val set
    (used only for early stopping).  Chosen hyperparameters come from the
    5-config sweep logged in ``docs/hyperparam_sweep.txt``."""
    T = data.out.shape[1]
    train_df = ft.build_direct_training_set(
        panel, cols, horizons=list(range(1, 49)),
        t_end_exclusive=T, t_start=200,
        issue_time_stride=2, keep_zero_rate=0.3,
    )
    t_cut = train_df["timestamp_idx"].quantile(0.92)
    tr = train_df[train_df["timestamp_idx"] <= t_cut]
    va = train_df[train_df["timestamp_idx"] > t_cut]
    model = LGBMDirectForecaster(
        n_estimators=n_estimators, num_leaves=num_leaves,
        learning_rate=learning_rate, min_child_samples=50,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=0.1,
    )
    model.fit(tr, val_df=va, verbose=verbose)
    return model


def _exp_decay_prediction(history: np.ndarray, horizon: int,
                           half_life: float = 18.0) -> np.ndarray:
    return bl.exponential_decay_baseline(history, horizon, half_life_hours=half_life)


# ---------------------------------------------------------------------------
# Internal hold-out eval for the report (LGBM + exp_decay only)
# ---------------------------------------------------------------------------

def internal_holdout_eval(data: du.OutageData) -> dict:
    """Train LGBM on data[:-48], evaluate on the last 48 hours.

    Reports: LGBM RMSE, exp-decay baseline RMSE, and their blend.
    Seq2Seq numbers come from the smoke-test script; they are NOT produced
    here because torch + libgbm in the same process segfault on macOS.
    """
    T = data.out.shape[1]
    hold_out = 48
    truth = data.out[:, T - hold_out:]

    panel, cols = ft.build_panel_features(
        out=data.out, tracked=data.tracked,
        weather=data.weather, feature_names=data.features,
        timestamps=data.timestamps,
    )

    train_df = ft.build_direct_training_set(
        panel, cols, horizons=list(range(1, 49)),
        t_end_exclusive=T - hold_out, t_start=200,
        issue_time_stride=2, keep_zero_rate=0.3,
    )
    t_cut = train_df["timestamp_idx"].quantile(0.92)
    tr = train_df[train_df["timestamp_idx"] <= t_cut]
    va = train_df[train_df["timestamp_idx"] > t_cut]
    lgbm = LGBMDirectForecaster(n_estimators=600, num_leaves=63, learning_rate=0.05,
                                 min_child_samples=50, feature_fraction=0.8,
                                 bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.1)
    lgbm.fit(tr, val_df=va, verbose=200)
    issue_t = T - hold_out - 1
    pred_lgbm = lgbm.predict(panel, issue_time_idx=issue_t,
                              horizons=list(range(1, 49)))

    pred_decay = _exp_decay_prediction(data.out[:, :T - hold_out],
                                         horizon=48, half_life=18.0)

    # Cached Seq2Seq prediction (reported for comparison, NOT blended in —
    # the weight sweep in tune_blend_weights.py found optimal w_s2s = 0 on
    # this hold-out because its 48h RMSE is ~1.7x LGBM's).
    pred_s2s = _load_cached_s2s_holdout()

    # Per-horizon blend weights, selected by grid search on this hold-out.
    # 24h uses more exp-decay because early forecast is dominated by the
    # residual-decay signal; 48h weights LGBM more since it captures the
    # longer-range recovery curvature.
    pred_blend_24 = blend_predictions(
        [pred_lgbm[:, :24], pred_decay[:, :24]], weights=[0.70, 0.30],
    )
    pred_blend_48 = blend_predictions(
        [pred_lgbm, pred_decay], weights=[0.80, 0.20],
    )

    def _eval(name, pred):
        return {
            "model": name,
            "rmse_24": m.mean_rmse(truth[:, :24], pred[:, :24]),
            "rmse_48": m.mean_rmse(truth, pred),
        }

    rows = [
        _eval("lgbm", pred_lgbm),
        _eval("exp_decay", pred_decay),
    ]
    if pred_s2s is not None:
        rows.append({"model": "seq2seq",
                     "rmse_24": m.mean_rmse(truth[:, :24], pred_s2s[:, :24]),
                     "rmse_48": m.mean_rmse(truth, pred_s2s)})
    rows.append({
        "model": "blend",
        "rmse_24": m.mean_rmse(truth[:, :24], pred_blend_24),
        "rmse_48": m.mean_rmse(truth, pred_blend_48),
    })

    return {"models": rows, "panel_cols": cols, "has_s2s": pred_s2s is not None}


def _load_cached_s2s_holdout() -> np.ndarray | None:
    """Return cached Seq2Seq predictions for the hold-out, if any."""
    path = os.path.join(RESULTS_DIR, "pred_s2s_holdout.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def _load_cached_s2s_test() -> np.ndarray | None:
    path = os.path.join(RESULTS_DIR, "pred_s2s_test.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


# ---------------------------------------------------------------------------
# Final predictions on the actual test issue time
# ---------------------------------------------------------------------------

def make_final_predictions(data: du.OutageData) -> dict:
    """Train final LightGBM on FULL training data and blend with decay (+ Seq2Seq
    if a cached prediction array exists)."""
    print("\n>>> Training final LGBM model on all training data...")
    panel, cols = ft.build_panel_features(
        out=data.out, tracked=data.tracked,
        weather=data.weather, feature_names=data.features,
        timestamps=data.timestamps,
    )

    t0 = time.time()
    lgbm = train_lgbm_final(data, panel, cols,
                             n_estimators=600, num_leaves=63, learning_rate=0.05,
                             verbose=300)
    print(f"[lgbm] fit done in {time.time()-t0:.1f}s "
          f"(best_iter={lgbm.model.best_iteration})")

    T = data.out.shape[1]
    issue_t = T - 1
    pred_lgbm = lgbm.predict(panel, issue_time_idx=issue_t,
                              horizons=list(range(1, 49)))
    pred_decay = _exp_decay_prediction(data.out, horizon=48, half_life=18.0)

    pred_s2s = _load_cached_s2s_test()

    # Per-horizon blend — weights chosen by grid search on the 48h hold-out
    # (see tune_blend_weights.py).  Seq2Seq is cached and reported but NOT
    # used in the submission blend because the sweep assigned it weight 0.
    pred_blend_24 = blend_predictions(
        [pred_lgbm[:, :24], pred_decay[:, :24]], weights=[0.70, 0.30],
    )
    pred_blend_48 = blend_predictions(
        [pred_lgbm, pred_decay], weights=[0.80, 0.20],
    )
    print("[blend] 24h weights: lgbm=0.70  exp_decay=0.30")
    print("[blend] 48h weights: lgbm=0.80  exp_decay=0.20")

    return {
        "pred_lgbm": pred_lgbm,
        "pred_seq2seq": pred_s2s,
        "pred_decay": pred_decay,
        "pred_blend_24": pred_blend_24,
        "pred_blend_48": pred_blend_48,
        "locations": data.locations,
    }


def write_final_artifacts(preds: dict) -> None:
    pred_24 = preds["pred_blend_24"]
    pred_48 = preds["pred_blend_48"]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sub.write_predictions_csv(pred_24, preds["locations"], TEMPLATE_24H,
                               os.path.join(RESULTS_DIR, "predictions_24h.csv"))
    sub.write_predictions_csv(pred_48, preds["locations"], TEMPLATE_48H,
                               os.path.join(RESULTS_DIR, "predictions_48h.csv"))
    print(f"[submission] wrote predictions_24h.csv and predictions_48h.csv")

    assignment = pol.greedy_allocation(pred_48, preds["locations"])
    sub.write_counties_txt(assignment.fips_list,
                            os.path.join(RESULTS_DIR, "recommended_counties.txt"))
    print(f"[policy] assignment: {assignment.counts}")
    print(f"[policy] expected restored customer-hours: "
          f"{assignment.expected_customer_hours:,.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MLPS Final Project — Power Outage Forecasting")
    print("=" * 70)
    data = du.load_train()
    print(f"Train: {data.out.shape}, features={len(data.features)}")

    eval_results = internal_holdout_eval(data)
    print("\nInternal 48h hold-out RMSE:")
    for r in eval_results["models"]:
        print(f"  {r['model']:10s} 24h={r['rmse_24']:6.2f}  48h={r['rmse_48']:6.2f}")
    if not eval_results["has_s2s"]:
        print("  (Seq2Seq not included — run make_s2s_predictions.py to add it.)")

    preds = make_final_predictions(data)
    write_final_artifacts(preds)


if __name__ == "__main__":
    main()
