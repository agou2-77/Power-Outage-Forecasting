"""Data loading, cleaning, and pre-processing for the MLPS outage project.

Load xarray NetCDF files, drop constant/dead weather features,
and expose train/test tensors in a consistent (location, timestamp) layout.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "dataset",
    "data",
)

TRAIN_PATH = os.path.join(DATA_DIR, "train.nc")
TEST_24H_PATH = os.path.join(DATA_DIR, "test_24h_demo.nc")
TEST_48H_PATH = os.path.join(DATA_DIR, "test_48h_demo.nc")

# Features confirmed constant in EDA (std == 0 across all (county, hour)).
DEAD_FEATURES = (
    "aod", "bgrun", "cfrzr", "cicep", "crain", "csnow",
    "hail_2", "ltng", "prate", "sdwe_1", "siconc", "ssrun",
    "tcoli", "tcolw", "tp", "unknown_7", "unknown_9",
)


@dataclass(frozen=True)
class OutageData:
    """Immutable container for train / validation / test arrays.

    Shape convention: (location L, timestamp T[, feature F]).
    """
    out: np.ndarray            # (L, T) outage counts
    tracked: np.ndarray        # (L, T) customers tracked (proxy for population)
    weather: np.ndarray | None # (L, T, F_live) or None for test slices with no weather
    locations: np.ndarray      # (L,) FIPS codes as strings
    timestamps: pd.DatetimeIndex  # (T,) hourly timestamps
    features: np.ndarray       # (F_live,) live feature names


def _drop_dead_features(weather: np.ndarray, feature_names: Sequence[str]
                        ) -> tuple[np.ndarray, np.ndarray]:
    keep_mask = np.array([f not in DEAD_FEATURES for f in feature_names])
    return weather[..., keep_mask], np.asarray(feature_names)[keep_mask]


def load_train(path: str = TRAIN_PATH) -> OutageData:
    ds = xr.open_dataset(path)
    out = ds["out"].transpose("location", "timestamp").values.astype(np.float32)
    tracked = ds["tracked"].transpose("location", "timestamp").values.astype(np.float32)
    weather = ds["weather"].transpose("location", "timestamp", "feature").values.astype(np.float32)
    features = ds["feature"].values.astype(str)
    weather, features = _drop_dead_features(weather, features)

    return OutageData(
        out=out,
        tracked=tracked,
        weather=weather,
        locations=ds["location"].values.astype(str),
        timestamps=pd.to_datetime(ds["timestamp"].values),
        features=features,
    )


def load_test(path: str) -> OutageData:
    """Load a test file. Note: demo test files have *synthetic* `out` values."""
    ds = xr.open_dataset(path)
    out = ds["out"].transpose("location", "timestamp").values.astype(np.float32)
    tracked = ds["tracked"].transpose("location", "timestamp").values.astype(np.float32)

    # Test slices should not contain weather, but handle gracefully if they do.
    if "weather" in ds.data_vars:
        weather = ds["weather"].transpose("location", "timestamp", "feature").values.astype(np.float32)
        features = ds["feature"].values.astype(str)
        weather, features = _drop_dead_features(weather, features)
    else:
        weather = None
        features = np.array([], dtype=object)

    return OutageData(
        out=out,
        tracked=tracked,
        weather=weather,
        locations=ds["location"].values.astype(str),
        timestamps=pd.to_datetime(ds["timestamp"].values),
        features=features,
    )


def temporal_split(data: OutageData, val_hours: int = 48
                   ) -> tuple[OutageData, OutageData]:
    """Split a training dataset chronologically.

    Keep the final `val_hours` as a held-out validation slice —
    48 by default to mirror the 48h test horizon.
    """
    T = len(data.timestamps)
    split = T - val_hours
    train = OutageData(
        out=data.out[:, :split],
        tracked=data.tracked[:, :split],
        weather=data.weather[:, :split] if data.weather is not None else None,
        locations=data.locations,
        timestamps=data.timestamps[:split],
        features=data.features,
    )
    val = OutageData(
        out=data.out[:, split:],
        tracked=data.tracked[:, split:],
        weather=data.weather[:, split:] if data.weather is not None else None,
        locations=data.locations,
        timestamps=data.timestamps[split:],
        features=data.features,
    )
    return train, val


def clip_negative(x: np.ndarray) -> np.ndarray:
    """Outage counts are non-negative; clamp predictions before scoring."""
    return np.clip(x, 0.0, None)
