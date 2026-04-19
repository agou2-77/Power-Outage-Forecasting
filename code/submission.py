"""Submission file generation.

Ensures prediction CSVs exactly match the template format required by the
grader, and writes the Part II county selection as a plain text file.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd


def _load_template(path: str) -> pd.DataFrame:
    """Load a submission template CSV, preserving row order."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["location"] = df["location"].astype(str)
    return df


def write_predictions_csv(
    pred: np.ndarray,
    locations: Sequence[str],
    template_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Write predictions in the expected CSV format.

    pred: (L, H) array of forecasted outage counts.
    locations: list of FIPS codes (strings), length L, matching the axis-0
      of `pred` and the ordering produced by the dataset loader.
    template_path: path to the provided submission_template_24h.csv or _48h.csv.
    output_path: where to write the completed CSV.
    """
    template = _load_template(template_path)
    L, H = pred.shape
    if len(locations) != L:
        raise ValueError(f"#locations {len(locations)} != pred rows {L}")

    # Map (loc, timestamp) -> predicted value
    ts_unique = np.sort(template["timestamp"].unique())
    if len(ts_unique) != H:
        raise ValueError(f"template has {len(ts_unique)} timestamps, pred has H={H}")

    pred_frame = pd.DataFrame({
        "location": np.repeat(np.asarray(locations), H),
        "timestamp": np.tile(ts_unique, L),
        "pred": pred.reshape(-1),
    })
    pred_frame["timestamp"] = pd.to_datetime(pred_frame["timestamp"])

    # Merge onto template order
    merged = template.merge(
        pred_frame, on=["timestamp", "location"], how="left",
        suffixes=("_tmpl", ""),
    )
    if merged["pred"].isna().any():
        missing = merged[merged["pred"].isna()].head()
        raise ValueError(f"Unfilled predictions for some (timestamp, location) rows:\n{missing}")

    out = merged[["timestamp", "location", "pred"]].copy()
    # Match template's display format (e.g. "6/30/23 1:00")
    out["timestamp"] = out["timestamp"].dt.strftime("%-m/%-d/%y %-H:%M")
    out.to_csv(output_path, index=False)
    return out


def write_counties_txt(counties: Sequence[str], output_path: str) -> None:
    """Write the 5 generator-placement FIPS codes as a plain text list.

    Matches the required format: [26001, 26001, 26001, 26001, 26001]
    (brackets around a comma-separated list, one-line).
    """
    if len(counties) != 5:
        raise ValueError(f"must select exactly 5 counties (got {len(counties)})")
    payload = "[" + ", ".join(str(c) for c in counties) + "]"
    with open(output_path, "w") as f:
        f.write(payload + "\n")
