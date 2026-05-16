"""Cleaning, normalisation, and target derivation for LAPD crime data."""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config
from .utils import get_logger, timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_time_occ(series: pd.Series) -> pd.Series:
    """`TIME OCC` is HHMM as int (e.g. 2130). Convert to hour 0-23."""
    s = series.fillna(0).astype("int32")
    return (s // 100).clip(lower=0, upper=23).astype("int8")


def _within_la(lat: pd.Series, lon: pd.Series) -> pd.Series:
    bb = config.LA_BBOX
    return (
        lat.between(bb["lat_min"], bb["lat_max"]) &
        lon.between(bb["lon_min"], bb["lon_max"])
    )


def _victim_age_clean(age: pd.Series) -> pd.Series:
    """Replace 0/negative/>110 ages with NaN and convert to nullable Int."""
    a = pd.to_numeric(age, errors="coerce")
    a = a.where((a > 0) & (a <= 110))
    return a.astype("Int16")


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply schema renames, type coercion, target derivation and basic QC."""
    with timer("preprocessing.clean", log):
        df = df.rename(columns=config.COLUMN_RENAME).copy()

        # ---- temporal ------------------------------------------------------
        df["date_occurred"] = pd.to_datetime(df["date_occurred"], errors="coerce")
        df["date_reported"] = pd.to_datetime(df["date_reported"], errors="coerce")
        df["hour"] = _parse_time_occ(df["time_occurred"])
        df["dow"] = df["date_occurred"].dt.dayofweek.astype("Int8")
        df["month"] = df["date_occurred"].dt.month.astype("Int8")
        df["year"] = df["date_occurred"].dt.year.astype("Int16")
        df["week"] = df["date_occurred"].dt.isocalendar().week.astype("Int16")
        df["is_weekend"] = df["dow"].isin([5, 6])
        df["report_lag_days"] = (df["date_reported"] - df["date_occurred"]).dt.days
        df["part_of_day"] = pd.cut(
            df["hour"],
            bins=[-1, 5, 11, 17, 21, 23],
            labels=["Late Night", "Morning", "Afternoon", "Evening", "Night"],
            ordered=False,
        )

        # ---- victim --------------------------------------------------------
        df["victim_age"] = _victim_age_clean(df["victim_age"])
        df["victim_sex"] = (
            df["victim_sex"].astype("string").str.upper().where(
                df["victim_sex"].astype("string").str.upper().isin(["M", "F"])
            ).fillna("Unknown").astype("category")
        )
        df["victim_descent"] = df["victim_descent"].astype("string").fillna("X").astype("category")

        # ---- spatial -------------------------------------------------------
        df["has_coords"] = _within_la(df["lat"], df["lon"])
        df.loc[~df["has_coords"], ["lat", "lon"]] = np.nan

        # ---- targets -------------------------------------------------------
        df[config.TARGET_VIOLENT] = (
            df["crime_code"].isin(config.VIOLENT_CRIME_CODES)
        ).astype("int8")
        df[config.TARGET_ARREST] = df["status_code"].astype("string").isin(config.ARREST_STATUSES).astype("int8")
        df["status_group"] = df["status_code"].astype("string").map(config.STATUS_MAP).fillna("Unknown").astype("category")

        # ---- weapon flag ---------------------------------------------------
        df["weapon_used"] = df["weapon_code"].notna().astype("int8")

        # ---- drop rows with hopelessly bad timestamps ----------------------
        before = len(df)
        df = df[df["date_occurred"].notna()]
        log.info(f"dropped {before - len(df):,} rows with missing date_occurred")

        # ---- final ordering -----------------------------------------------
        df = df.sort_values("date_occurred").reset_index(drop=True)
    return df


def save_clean(df: pd.DataFrame) -> None:
    out = config.CLEAN_PARQUET
    with timer(f"save clean parquet -> {out.name}", log):
        df.to_parquet(out, index=False)
    log.info(f"clean parquet saved: {out} ({out.stat().st_size/1024**2:.1f} MB)")
