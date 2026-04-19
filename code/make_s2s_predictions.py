"""Train the Seq2Seq LSTM and save predictions for the blend.

Runs in its own Python process so it does NOT conflict with LightGBM's
OpenMP runtime on macOS (running both in the same process segfaults).

Outputs (written to ``results/``):
  pred_s2s_holdout.npy  — last-48h hold-out prediction  (L × 48)
  pred_s2s_test.npy     — final 48h prediction issued from T-1 (L × 48)
"""
from __future__ import annotations

import os
import time

# Stabilise libomp across PyTorch / NumPy on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

import torch  # noqa: F401 — imported early to anchor libomp

import data_utils as du
import metrics as m
from seq2seq import Seq2SeqForecaster


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = du.load_train()
    T = data.out.shape[1]
    print(f"Train shape: {data.out.shape}   T={T}")

    # ------------------------------------------------------------------
    # 1) Last-48h hold-out prediction (for reporting RMSE)
    # ------------------------------------------------------------------
    t_end = T - 48
    t0 = time.time()
    mdl = Seq2SeqForecaster(
        seq_len=48, horizon=48, hidden_dim=96, num_layers=2,
        epochs=6, batch_size=128, learning_rate=1e-3,
        use_weather=True, dropout=0.15,
    )
    mdl.fit(data.out[:, :t_end], data.weather[:, :t_end],
            feature_names=data.features, verbose=True)
    pred_holdout = mdl.predict(data.out, data.weather, issue_time_idx=t_end - 1)
    truth = data.out[:, t_end:t_end + 48]
    print(f"[s2s holdout] 24h={m.mean_rmse(truth[:, :24], pred_holdout[:, :24]):.2f}  "
          f"48h={m.mean_rmse(truth, pred_holdout):.2f}  "
          f"fit={time.time()-t0:.1f}s")
    np.save(os.path.join(RESULTS_DIR, "pred_s2s_holdout.npy"), pred_holdout)

    # ------------------------------------------------------------------
    # 2) Final test prediction — train on FULL data, issue from T-1
    # ------------------------------------------------------------------
    print("\n>>> Training Seq2Seq on FULL training data...")
    t0 = time.time()
    mdl_final = Seq2SeqForecaster(
        seq_len=48, horizon=48, hidden_dim=96, num_layers=2,
        epochs=6, batch_size=128, learning_rate=1e-3,
        use_weather=True, dropout=0.15,
    )
    mdl_final.fit(data.out, data.weather, feature_names=data.features, verbose=True)
    pred_test = mdl_final.predict(data.out, data.weather, issue_time_idx=T - 1)
    print(f"[s2s test] pred shape={pred_test.shape}  fit={time.time()-t0:.1f}s")
    np.save(os.path.join(RESULTS_DIR, "pred_s2s_test.npy"), pred_test)

    print("\nWrote:")
    print(f"  {RESULTS_DIR}/pred_s2s_holdout.npy")
    print(f"  {RESULTS_DIR}/pred_s2s_test.npy")


if __name__ == "__main__":
    main()
