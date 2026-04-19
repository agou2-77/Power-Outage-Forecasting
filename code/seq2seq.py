"""LSTM encoder-decoder for outage forecasting.

Improvements over the demo notebook baseline:
  * log1p(out) target with per-county z-score normalization of features.
  * Dead weather features dropped upstream.
  * 48h encoder lookback (vs 24 in the demo) — captures storm phase.
  * Two-layer LSTM with dropout.
  * Forecast horizon emitted from a per-horizon linear head (no teacher forcing).
  * At inference, no future weather required — the head projects directly from
    the encoder's final hidden state.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _build_sliding_windows(
    y: np.ndarray, w: np.ndarray, seq_len: int, horizon: int,
    issue_time_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) arrays for training one seq2seq model.

    y: (T, L) scaled outages
    w: (T, L, F) scaled weather (F may be 0 for weather-free variant)
    X: (N, seq_len, 1 + F) — past outages + past weather
    Y: (N, horizon)          — future outages (log-scaled)
    """
    T, L = y.shape
    F = w.shape[-1] if (w is not None and w.ndim == 3) else 0
    X_list, Y_list = [], []
    # Choose valid issue times: need seq_len past + horizon future.
    valid_t0 = list(range(seq_len, T - horizon + 1, issue_time_stride))
    for t0 in valid_t0:
        past_y = y[t0 - seq_len:t0]     # (seq_len, L)
        future_y = y[t0:t0 + horizon]   # (horizon, L)
        if F > 0:
            past_w = w[t0 - seq_len:t0]  # (seq_len, L, F)
            xi = np.concatenate([past_y[..., None], past_w], axis=-1)  # (seq_len, L, 1+F)
        else:
            xi = past_y[..., None]
        # Flatten across L → one sample per location
        xi = xi.transpose(1, 0, 2)  # (L, seq_len, 1+F)
        yi = future_y.T             # (L, horizon)
        X_list.append(xi)
        Y_list.append(yi)
    if not X_list:
        return (np.empty((0, seq_len, 1 + F), dtype=np.float32),
                np.empty((0, horizon), dtype=np.float32))
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    Y = np.concatenate(Y_list, axis=0).astype(np.float32)
    return X, Y


class _LSTMSeq2Seq(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 horizon: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        # h: (num_layers, batch, hidden)
        last = h[-1]            # (batch, hidden)
        return self.head(last)  # (batch, horizon)


@dataclass
class Seq2SeqForecaster:
    seq_len: int = 48
    horizon: int = 48
    hidden_dim: int = 96
    num_layers: int = 2
    dropout: float = 0.15
    epochs: int = 6
    batch_size: int = 128
    learning_rate: float = 1e-3
    use_weather: bool = True
    random_seed: int = 42
    device: str = "cpu"

    model: nn.Module | None = None
    y_mu: float = 0.0
    y_sd: float = 1.0
    w_mu: np.ndarray | None = None
    w_sd: np.ndarray | None = None
    feature_names: list = field(default_factory=list)

    def fit(self, out: np.ndarray, weather: np.ndarray | None,
            feature_names: np.ndarray | None = None,
            verbose: bool = True):
        """Train on full history. Arrays are (L, T) for out and (L, T, F) for weather."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        L, T = out.shape
        y = np.log1p(out.T.astype(np.float32))    # (T, L) in log space
        # Standardize globally (simple & works for log-scaled outages)
        self.y_mu = float(np.nanmean(y))
        self.y_sd = float(np.nanstd(y) + 1e-6)
        y_scaled = (y - self.y_mu) / self.y_sd

        if self.use_weather and weather is not None:
            w = weather.transpose(1, 0, 2).astype(np.float32)  # (T, L, F)
            self.w_mu = np.nanmean(w.reshape(-1, w.shape[-1]), axis=0)
            self.w_sd = np.nanstd(w.reshape(-1, w.shape[-1]), axis=0) + 1e-6
            w_scaled = (w - self.w_mu) / self.w_sd
            w_scaled = np.nan_to_num(w_scaled)
            F = w.shape[-1]
            self.feature_names = list(feature_names) if feature_names is not None else []
        else:
            w_scaled = None
            F = 0
            self.feature_names = []

        X, Y = _build_sliding_windows(
            y_scaled, w_scaled, self.seq_len, self.horizon, issue_time_stride=1,
        )
        if verbose:
            print(f"[seq2seq] train windows: X={X.shape}, Y={Y.shape}")

        input_dim = 1 + F
        self.model = _LSTMSeq2Seq(
            input_dim=input_dim, hidden_dim=self.hidden_dim,
            num_layers=self.num_layers, horizon=self.horizon, dropout=self.dropout,
        ).to(self.device)

        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(X_t, Y_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.model.train()
        for epoch in range(self.epochs):
            tot, n = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                tot += loss.item() * xb.size(0)
                n += xb.size(0)
            if verbose:
                print(f"[seq2seq] epoch {epoch + 1}/{self.epochs}  loss={tot/n:.4f}")
        return self

    def predict(self, out: np.ndarray, weather: np.ndarray | None,
                issue_time_idx: int) -> np.ndarray:
        """Predict (L, horizon) outages for the given issue time.

        issue_time_idx is the index of the last KNOWN hour.
        The prediction covers times (issue_time_idx+1) ... (issue_time_idx+horizon).
        """
        assert self.model is not None, "fit() first"
        L, T = out.shape
        # Build input: past seq_len hours ending at issue_time_idx
        start = issue_time_idx - self.seq_len + 1
        end = issue_time_idx + 1
        y = np.log1p(out[:, start:end].astype(np.float32))  # (L, seq_len)
        y_scaled = (y - self.y_mu) / self.y_sd
        if self.use_weather and weather is not None and self.w_mu is not None:
            w = weather[:, start:end, :].astype(np.float32)  # (L, seq_len, F)
            w_scaled = (w - self.w_mu) / self.w_sd
            w_scaled = np.nan_to_num(w_scaled)
            xin = np.concatenate([y_scaled[..., None], w_scaled], axis=-1)
        else:
            xin = y_scaled[..., None]

        x_t = torch.tensor(xin, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred_log_scaled = self.model(x_t).cpu().numpy()  # (L, horizon)
        # Invert standardization + log1p
        pred_log = pred_log_scaled * self.y_sd + self.y_mu
        pred = np.expm1(pred_log)
        return np.clip(pred, 0, None).astype(np.float32)
