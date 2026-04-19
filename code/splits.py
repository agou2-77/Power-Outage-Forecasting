"""Time-aware CV splits for hourly outage forecasting.

The test window is a single 24/48h block immediately after training.
For hyperparameter tuning we use rolling-origin backtesting:
multiple non-overlapping (issue_time, horizon) windows drawn from the tail
of the training series so every split respects causality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Split:
    """A single backtest fold.

    train_end is an *exclusive* index — the forecaster can use data
    at indices [0, train_end).
    The forecast target is out[:, train_end : train_end + horizon].
    """
    train_end: int
    horizon: int

    def target_slice(self) -> slice:
        return slice(self.train_end, self.train_end + self.horizon)


def rolling_origin_splits(
    n_timestamps: int,
    horizon: int,
    n_folds: int = 5,
    stride: int = 48,
    min_train: int | None = None,
) -> list[Split]:
    """Generate rolling-origin splits anchored to the tail of the series.

    The most recent fold ends at `n_timestamps` — mimicking the real test setup.
    Earlier folds are strided back by `stride` hours.
    """
    if min_train is None:
        min_train = max(horizon * 10, 24 * 14)  # need enough history

    splits = []
    # Most recent fold: the last `horizon` hours are the validation slice.
    last_end = n_timestamps - horizon
    for k in range(n_folds):
        end = last_end - k * stride
        if end < min_train:
            break
        splits.append(Split(train_end=end, horizon=horizon))
    return list(reversed(splits))  # oldest first


def iter_splits(n_timestamps: int, horizon: int, n_folds: int = 5,
                stride: int = 48) -> Iterator[Split]:
    yield from rolling_origin_splits(n_timestamps, horizon, n_folds, stride)
