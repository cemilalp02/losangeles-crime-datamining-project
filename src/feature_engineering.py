"""Feature engineering: incident-level features and area/time aggregations."""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config
from .utils import get_logger, timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Incident-level features (used for classification & explainability)
# ---------------------------------------------------------------------------
def build_incident_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a tidy feature matrix for incident-level modelling."""
    cols = [
        "area_id", "area_name", "hour", "dow", "month", "year",
        "is_weekend", "part_of_day",
        "victim_age", "victim_sex", "victim_descent",
        "premise_code", "weapon_used",
        "lat", "lon", "report_lag_days",
        "crime_code", "crime_desc", "part_class",
        config.TARGET_VIOLENT, config.TARGET_ARREST,
    ]
    feats = df[cols].copy()

    # Imputations
    feats["victim_age"] = feats["victim_age"].astype("float32").fillna(feats["victim_age"].median())
    feats["report_lag_days"] = feats["report_lag_days"].fillna(0).clip(lower=0, upper=365)
    feats["lat"] = feats["lat"].fillna(feats["lat"].median())
    feats["lon"] = feats["lon"].fillna(feats["lon"].median())

    # Cyclical encodings
    feats["hour_sin"] = np.sin(2 * np.pi * feats["hour"] / 24).astype("float32")
    feats["hour_cos"] = np.cos(2 * np.pi * feats["hour"] / 24).astype("float32")
    feats["dow_sin"] = np.sin(2 * np.pi * feats["dow"].astype(float) / 7).astype("float32")
    feats["dow_cos"] = np.cos(2 * np.pi * feats["dow"].astype(float) / 7).astype("float32")
    feats["month_sin"] = np.sin(2 * np.pi * feats["month"].astype(float) / 12).astype("float32")
    feats["month_cos"] = np.cos(2 * np.pi * feats["month"].astype(float) / 12).astype("float32")

    return feats


# ---------------------------------------------------------------------------
# Spatio-temporal aggregations (used for forecasting & risk scoring)
# ---------------------------------------------------------------------------
def build_daily_area_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Daily panel: rows = (date, area) with crime counts and violent ratio."""
    with timer("build_daily_area_panel", log):
        d = df.assign(date=df["date_occurred"].dt.normalize())
        agg = (
            d.groupby(["date", "area_id", "area_name"], observed=True)
             .agg(
                 crimes=("dr_no", "count"),
                 violent=(config.TARGET_VIOLENT, "sum"),
                 arrests=(config.TARGET_ARREST, "sum"),
             )
             .reset_index()
        )
        agg["violent_ratio"] = (agg["violent"] / agg["crimes"]).astype("float32")
        agg["arrest_ratio"] = (agg["arrests"] / agg["crimes"]).astype("float32")

        # Build full date x area grid so missing days = 0 crimes.
        full_dates = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
        areas = agg[["area_id", "area_name"]].drop_duplicates()
        grid = (pd.MultiIndex.from_product([full_dates, areas["area_id"]],
                                           names=["date", "area_id"]).to_frame(index=False)
                .merge(areas, on="area_id", how="left"))
        agg = grid.merge(agg, on=["date", "area_id", "area_name"], how="left")
        agg[["crimes", "violent", "arrests"]] = agg[["crimes", "violent", "arrests"]].fillna(0)
        agg[["violent_ratio", "arrest_ratio"]] = agg[["violent_ratio", "arrest_ratio"]].fillna(0)
    return agg


def build_weekly_area_panel(daily: pd.DataFrame) -> pd.DataFrame:
    with timer("build_weekly_area_panel", log):
        w = daily.copy()
        w["week_start"] = w["date"] - pd.to_timedelta(w["date"].dt.dayofweek, unit="D")
        agg = (
            w.groupby(["week_start", "area_id", "area_name"], observed=True)
             .agg(crimes=("crimes", "sum"),
                  violent=("violent", "sum"),
                  arrests=("arrests", "sum"))
             .reset_index()
             .rename(columns={"week_start": "date"})
        )
        agg["violent_ratio"] = (agg["violent"] / agg["crimes"].replace(0, np.nan)).fillna(0).astype("float32")
        agg["arrest_ratio"] = (agg["arrests"] / agg["crimes"].replace(0, np.nan)).fillna(0).astype("float32")
    return agg


def trim_incomplete_tail(daily: pd.DataFrame, threshold: float = 0.70,
                         min_days: int = 14) -> pd.DataFrame:
    """Drop the most recent days where city-wide reporting looks incomplete.

    Heuristic: compare the 7-day rolling sum of city-wide crimes to the
    same window 6 weeks earlier; if it has dropped below `threshold`
    (e.g. 70%) of that earlier value for at least `min_days` consecutive days,
    treat the tail as incomplete and trim it.
    """
    total = daily.groupby("date", observed=True)["crimes"].sum().sort_index()
    roll7 = total.rolling(7, min_periods=1).sum()
    baseline = roll7.shift(42)            # ~6 weeks earlier
    ratio = roll7 / baseline
    bad = ratio < threshold
    if bad.tail(min_days).all():
        # find the first date in the contiguous trailing bad window
        rev = bad.iloc[::-1]
        cut = rev.idxmin() if (~rev).any() else rev.index[-1]
        # cut is the latest "good" date; trim everything strictly after it
        keep = total.index <= cut
        cutoff_date = total.index[keep].max()
        log.info(f"trim_incomplete_tail: cutting at {cutoff_date.date()} "
                 f"(was {total.index.max().date()})")
        return daily[daily["date"] <= cutoff_date].copy()
    return daily.copy()


def add_lag_features(panel: pd.DataFrame, lags=(1, 7, 14, 28), rolls=(7, 28, 90)) -> pd.DataFrame:
    """Add lag, rolling, and area-baseline features for the `crimes` series.

    All target-derived features use `shift(1)` to avoid leakage from the
    same-day target into its own lag.
    """
    p = panel.sort_values(["area_id", "date"]).copy()
    g = p.groupby("area_id", observed=True)["crimes"]
    for L in lags:
        p[f"lag_{L}"] = g.shift(L).astype("float32")
    for R in rolls:
        p[f"roll_mean_{R}"] = (
            g.shift(1).rolling(R).mean().reset_index(level=0, drop=True).astype("float32")
        )
        p[f"roll_std_{R}"] = (
            g.shift(1).rolling(R).std().reset_index(level=0, drop=True).astype("float32")
        )

    # Calendar features
    p["dow"] = p["date"].dt.dayofweek.astype("int8")
    p["month"] = p["date"].dt.month.astype("int8")
    p["year"] = p["date"].dt.year.astype("int16")
    p["doy"] = p["date"].dt.dayofyear.astype("int16")
    p["weekofyear"] = p["date"].dt.isocalendar().week.astype("int16")
    p["is_weekend"] = (p["dow"] >= 5).astype("int8")

    # Expanding mean per (area, day-of-week) -> very strong feature.
    # Use shift(1) within group to keep it leak-free.
    p["adow_mean"] = (
        p.groupby(["area_id", "dow"], observed=True)["crimes"]
         .transform(lambda s: s.shift(1).expanding(min_periods=4).mean())
         .astype("float32")
    )
    return p
