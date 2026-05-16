"""Composite area-level risk score combining intensity, trend, violence, forecast."""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

log = get_logger(__name__)


def _zscale(s: pd.Series) -> pd.Series:
    """Min-max scale to [0, 1]; safe for constant series."""
    rng = s.max() - s.min()
    if rng == 0 or pd.isna(rng):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / rng


def compute_area_risk(
    daily_panel: pd.DataFrame,
    forecast_df: pd.DataFrame | None = None,
    recent_days: int = 28,
    trend_long: int = 90,
) -> pd.DataFrame:
    """Compute a normalized risk score per LAPD area.

    Components (each scaled to [0, 1]):
        intensity : mean daily crimes over the last `recent_days`
        trend     : (recent mean - long-term mean) / long-term std
        violence  : violent_share over recent window
        forecast  : mean of forecasted next-period crimes (optional)
    """
    last_day = daily_panel["date"].max()
    recent_start = last_day - pd.Timedelta(days=recent_days)
    long_start = last_day - pd.Timedelta(days=trend_long)

    recent = daily_panel[daily_panel["date"] > recent_start]
    long = daily_panel[daily_panel["date"] > long_start]

    g_recent = (recent.groupby(["area_id", "area_name"], observed=True)
                       .agg(intensity=("crimes", "mean"),
                            violence=("violent_ratio", "mean"))
                       .reset_index())
    g_long = (long.groupby(["area_id", "area_name"], observed=True)
                  .agg(long_mean=("crimes", "mean"),
                       long_std=("crimes", "std"))
                  .reset_index())

    out = g_recent.merge(g_long, on=["area_id", "area_name"], how="left")
    out["trend"] = ((out["intensity"] - out["long_mean"]) /
                    (out["long_std"].replace(0, np.nan))).fillna(0)

    if forecast_df is not None and not forecast_df.empty:
        f = (forecast_df.groupby(["area_id", "area_name"], observed=True)["y_pred"]
                          .mean().reset_index().rename(columns={"y_pred": "forecast"}))
        out = out.merge(f, on=["area_id", "area_name"], how="left")
    else:
        out["forecast"] = out["intensity"]

    # Scale components
    out["intensity_n"] = _zscale(out["intensity"])
    out["trend_n"] = _zscale(out["trend"])
    out["violence_n"] = _zscale(out["violence"])
    out["forecast_n"] = _zscale(out["forecast"])

    w = config.RISK_WEIGHTS
    out["risk_score"] = (
        w["intensity"] * out["intensity_n"] +
        w["trend"] * out["trend_n"] +
        w["violence"] * out["violence_n"] +
        w["forecast"] * out["forecast_n"]
    )
    out["risk_rank"] = out["risk_score"].rank(ascending=False, method="min").astype(int)
    out["risk_tier"] = pd.qcut(out["risk_score"], q=4,
                               labels=["Low", "Medium", "High", "Critical"])
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)
