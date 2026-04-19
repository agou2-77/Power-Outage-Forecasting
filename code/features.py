"""Feature engineering for direct multi-horizon outage forecasting.

At forecast issue time t0, build per-(county, horizon) feature vectors using
only information available at or before t0. Each horizon h in 1..H gets its
own training target (out[county, t0 + h]), so the same feature matrix can be
reused across horizons with an added `horizon_h` column.

Design principles:
  * No future leakage — all features are strictly backward-looking.
  * Horizon-as-feature lets one model serve all h = 1..48.
  * Include county static features (log_tracked, one-hot county id, lat/long proxy).
  * Weather features are sampled at issue time and at past lags (1, 3, 6, 12, 24h).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Feature lag schedules -------------------------------------------------------

OUTAGE_LAGS_HOURS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
OUTAGE_ROLLING_WINDOWS = (3, 12, 24, 72, 168)
WEATHER_LAGS_HOURS = (0, 3, 12, 24)

# Weather features prioritized by EDA correlation with outages
PRIMARY_WX = (
    "cape", "cape_1", "pwat", "mstav", "sh2", "gust",
    "t2m", "u10", "v10", "refc", "gh_4", "lcc", "wz",
)


@dataclass(frozen=True)
class FeatureConfig:
    outage_lags: tuple[int, ...] = OUTAGE_LAGS_HOURS
    outage_rolls: tuple[int, ...] = OUTAGE_ROLLING_WINDOWS
    weather_lags: tuple[int, ...] = WEATHER_LAGS_HOURS
    primary_wx: tuple[str, ...] = PRIMARY_WX
    include_county_onehot: bool = False  # 83 dummies blows up feature width
    include_calendar: bool = True
    log_target: bool = True
    horizon_max: int = 48


def _safe_lag(arr: np.ndarray, lag: int) -> np.ndarray:
    """Shift a (L, T) array by `lag` positions along the time axis.

    Past lag (positive) means arr[:, t-lag]. Negative lag = future (disallowed).
    Out-of-bounds positions are filled with NaN.
    """
    if lag < 0:
        raise ValueError("negative lag disallowed (would leak future)")
    L, T = arr.shape
    out = np.full_like(arr, np.nan, dtype=np.float32)
    if lag < T:
        out[:, lag:] = arr[:, : T - lag]
    return out


def _rolling_stat(arr: np.ndarray, window: int, fn) -> np.ndarray:
    """Compute a backward-looking rolling stat per row (no future leakage).

    Uses pandas which is faster than hand-rolled for moderate sizes."""
    df = pd.DataFrame(arr.T)  # (T, L) for pandas
    rolled = df.rolling(window=window, min_periods=1).apply(fn, raw=True)
    return rolled.values.T.astype(np.float32)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    df = pd.DataFrame(arr.T)
    return df.rolling(window=window, min_periods=1).mean().values.T.astype(np.float32)


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    df = pd.DataFrame(arr.T)
    return df.rolling(window=window, min_periods=1).max().values.T.astype(np.float32)


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    df = pd.DataFrame(arr.T)
    return df.rolling(window=window, min_periods=1).std().values.T.astype(np.float32)


def hours_since_peak(arr: np.ndarray, threshold: float = 100.0) -> np.ndarray:
    """For each (county, t), compute hours since last value >= threshold.

    Used as a "storm-phase" feature. Infinities filled with a large cap.
    """
    L, T = arr.shape
    result = np.full((L, T), 10000.0, dtype=np.float32)
    counter = np.full(L, 10000.0, dtype=np.float32)
    for t in range(T):
        counter = counter + 1.0
        peak_now = arr[:, t] >= threshold
        counter[peak_now] = 0.0
        result[:, t] = counter
    return result


def build_panel_features(
    out: np.ndarray,
    tracked: np.ndarray,
    weather: np.ndarray | None,
    feature_names: np.ndarray,
    timestamps: pd.DatetimeIndex,
    config: FeatureConfig = FeatureConfig(),
) -> tuple[pd.DataFrame, list[str]]:
    """Build a panel feature frame with one row per (county, timestamp).

    Returns:
      panel — DataFrame indexed by (location_idx, timestamp_idx) with columns:
              {lag features, rolling features, weather features, calendar},
              plus a `y_t` column holding the outage at that timestamp.
      feature_cols — list of feature column names (excludes y_t).

    The caller then constructs the training set by joining this panel on the
    issue time (t0) and the target y_{t0 + h} for each horizon h in 1..H.
    """
    L, T = out.shape
    panel_parts: list[np.ndarray] = []
    col_names: list[str] = []

    # Target at each timestamp — used both for lag feature construction
    # and as the prediction target after appropriate shifting.
    y = out.astype(np.float32)
    y_log = np.log1p(y) if config.log_target else y

    # --- 1. Outage lag features (strict past) ---
    for lag in config.outage_lags:
        panel_parts.append(_safe_lag(y_log, lag))
        col_names.append(f"y_lag{lag}")

    # --- 2. Outage rolling stats ---
    for win in config.outage_rolls:
        # Rolling computed on current-and-past; to avoid leakage at the
        # target time we lag by 1 (i.e., stats of past `win` hours ending at t-1).
        panel_parts.append(_safe_lag(_rolling_mean(y_log, win), 1))
        col_names.append(f"y_roll{win}_mean")
        panel_parts.append(_safe_lag(_rolling_max(y_log, win), 1))
        col_names.append(f"y_roll{win}_max")

    # --- 3. Hours since last peak (storm-phase feature) ---
    panel_parts.append(_safe_lag(hours_since_peak(y, threshold=100.0), 1))
    col_names.append("hrs_since_peak_100")
    panel_parts.append(_safe_lag(hours_since_peak(y, threshold=1000.0), 1))
    col_names.append("hrs_since_peak_1000")

    # --- 4. Weather features (sampled at the forecast issue time) ---
    if weather is not None:
        feat_idx = {name: i for i, name in enumerate(feature_names)}
        # Only keep primary weather features that exist
        chosen = [f for f in config.primary_wx if f in feat_idx]
        for f in chosen:
            w = weather[:, :, feat_idx[f]].astype(np.float32)  # (L, T)
            for lag in config.weather_lags:
                panel_parts.append(_safe_lag(w, lag))
                col_names.append(f"wx_{f}_lag{lag}")

    # --- 5. Calendar features (hour/day of week) ---
    if config.include_calendar:
        hod = timestamps.hour.values.astype(np.float32)
        dow = timestamps.dayofweek.values.astype(np.float32)
        hod_sin = np.sin(2 * np.pi * hod / 24.0)
        hod_cos = np.cos(2 * np.pi * hod / 24.0)
        dow_sin = np.sin(2 * np.pi * dow / 7.0)
        dow_cos = np.cos(2 * np.pi * dow / 7.0)
        for name, vec in [("hod_sin", hod_sin), ("hod_cos", hod_cos),
                           ("dow_sin", dow_sin), ("dow_cos", dow_cos)]:
            # Broadcast to (L, T)
            panel_parts.append(np.broadcast_to(vec[None, :], (L, T)).copy())
            col_names.append(name)

    # --- 6. County static features (log tracked — population proxy) ---
    log_tracked = np.log1p(np.nan_to_num(tracked, nan=0.0))
    panel_parts.append(log_tracked)
    col_names.append("log_tracked")

    # Stack into (n_features, L, T) then reshape to (L*T, n_features)
    stacked = np.stack(panel_parts, axis=0)  # (F, L, T)
    F = stacked.shape[0]
    panel = stacked.reshape(F, L * T).T  # (L*T, F)

    df = pd.DataFrame(panel, columns=col_names, dtype=np.float32)
    df["y_t"] = y.reshape(-1)
    loc_idx = np.repeat(np.arange(L, dtype=np.int32), T)
    time_idx = np.tile(np.arange(T, dtype=np.int32), L)
    df["location_idx"] = loc_idx
    df["timestamp_idx"] = time_idx

    return df, col_names


def build_direct_training_set(
    panel: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int],
    t_end_exclusive: int,
    t_start: int | None = None,
    log_target: bool = True,
    issue_time_stride: int = 1,
    keep_zero_rate: float = 1.0,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Join features at issue time t0 with target y_{t0 + h}.

    Returns a long DataFrame with columns:
      feature_cols + [horizon_h, y_target, y_target_log, location_idx, timestamp_idx]

    Only rows with t0 + h < t_end_exclusive are retained to avoid leaking
    post-cutoff data into training.

    Args:
      issue_time_stride: sample every k-th hour as an issue time (default 1).
        Using 3 drops the panel to ~1/3 the size with little info loss, since
        issue times at consecutive hours carry near-identical features.
      keep_zero_rate: probability of retaining a row whose target is 0.
        Since ~70% of targets are zero, setting this to 0.3 balances classes.
    """
    if t_start is None:
        # Drop the first chunk where long lags are NaN
        t_start = max(max(OUTAGE_LAGS_HOURS) + 1, max(OUTAGE_ROLLING_WINDOWS))

    rng = np.random.default_rng(rng_seed)
    # Precompute y lookup once: (loc, t) -> y
    y_vals = panel["y_t"].values
    loc_idx_all = panel["location_idx"].values
    time_idx_all = panel["timestamp_idx"].values
    T_max = int(time_idx_all.max()) + 1
    L_max = int(loc_idx_all.max()) + 1
    y_lut = np.full((L_max, T_max), np.nan, dtype=np.float32)
    y_lut[loc_idx_all, time_idx_all] = y_vals

    # Stride-filtered issue times
    valid_t0 = np.arange(t_start, t_end_exclusive, issue_time_stride)
    issue_mask = np.isin(time_idx_all, valid_t0)
    base_rows = panel[issue_mask][feature_cols + ["location_idx", "timestamp_idx"]]

    out_frames = []
    for h in horizons:
        sub = base_rows[base_rows["timestamp_idx"] + h < t_end_exclusive].copy()
        tgt = y_lut[sub["location_idx"].values, sub["timestamp_idx"].values + h]
        sub["y_target"] = tgt
        sub["horizon_h"] = h
        sub = sub.dropna(subset=["y_target"])
        if keep_zero_rate < 1.0:
            is_zero = sub["y_target"].values == 0
            keep = np.ones(len(sub), dtype=bool)
            keep[is_zero] = rng.random(is_zero.sum()) < keep_zero_rate
            sub = sub.iloc[keep]
        out_frames.append(sub)

    df = pd.concat(out_frames, ignore_index=True)
    if log_target:
        df["y_target_log"] = np.log1p(df["y_target"].values)
    return df


def extract_issue_time_features(
    panel: pd.DataFrame,
    feature_cols: list[str],
    issue_time_idx: int,
    horizons: list[int],
) -> pd.DataFrame:
    """Get features for predicting at a single issue time across all counties & horizons."""
    row = panel[panel["timestamp_idx"] == issue_time_idx][
        feature_cols + ["location_idx"]
    ].copy()
    # Replicate across horizons
    rep = row.loc[row.index.repeat(len(horizons))].reset_index(drop=True)
    rep["horizon_h"] = np.tile(horizons, len(row))
    return rep
