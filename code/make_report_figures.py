"""Generate figures used in the PDF report."""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

import data_utils as du


REPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report",
)


def fig_system_outage(data: du.OutageData) -> None:
    """System-wide total active customer outages vs time."""
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    total = data.out.sum(axis=0)
    ax.plot(data.timestamps, total, lw=0.9, color="tab:red")
    # Vertical dashed line at the last training hour.
    last_ts = data.timestamps[-1]
    ax.axvline(last_ts, color="black", lw=0.8, linestyle="--",
               label=f"issue time\n{last_ts}")
    # Annotate the storm peak.
    peak_idx = int(total.argmax())
    ax.annotate(f"peak: {int(total[peak_idx]):,}",
                xy=(data.timestamps[peak_idx], total[peak_idx]),
                xytext=(8, -14), textcoords="offset points",
                fontsize=8, color="tab:red")
    ax.set_ylabel("# customers without power")
    ax.set_xlabel("time (hourly)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "fig_system_outage.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_fold_predictions(data: du.OutageData, pred_blend: np.ndarray) -> None:
    """Overlay blended forecast on the system-wide recovery tail + test
    window for sanity checking.  Only generated if pred_blend is provided."""
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    tail_hours = 240
    start = data.out.shape[1] - tail_hours
    ts = data.timestamps[start:]
    total = data.out.sum(axis=0)[start:]
    ax.plot(ts, total, lw=0.9, color="tab:red", label="observed")
    # Prediction: 48h after the last observed hour.
    pred_ts = [data.timestamps[-1] + np.timedelta64(h + 1, "h") for h in range(48)]
    ax.plot(pred_ts, pred_blend.sum(axis=0), lw=1.2,
            color="tab:blue", label="blended forecast")
    ax.axvline(data.timestamps[-1], color="black", lw=0.8, linestyle="--")
    ax.set_ylabel("# customers (system-wide)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "fig_forecast_tail.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    data = du.load_train()
    fig_system_outage(data)


if __name__ == "__main__":
    main()
