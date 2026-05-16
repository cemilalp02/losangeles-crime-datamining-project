"""Spatio-temporal crime forecasting using gradient-boosted lag-feature models.

Approach (state-of-the-art for tabular STT data):

    For each (date, area) cell we build a feature vector with:
      - lagged crime counts (1, 7, 14, 28 days)
      - rolling means / stds (7d, 28d)
      - calendar features (dow, month, doy, year)
      - area_id (categorical)

    A single LightGBM regressor is trained globally across all areas.
    Evaluation is rolling/forward-chained (no leakage).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .feature_engineering import add_lag_features
from .utils import get_logger, timer

log = get_logger(__name__)

DEFAULT_FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28",
    "roll_mean_90", "roll_std_90",
    "dow", "month", "weekofyear", "doy", "is_weekend",
    "area_id",
]


@dataclass
class ForecastResult:
    horizon_days: int
    mae: float
    rmse: float
    mape: float
    r2: float
    naive_mae: float          # MAE of "last 28-day per-area mean" baseline
    skill: float              # 1 - mae / naive_mae  (>0 = beats baseline)
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    predictions: pd.DataFrame  # date, area_id, area_name, y_true, y_pred


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]))


def _train_lgbm(X_tr, y_tr, X_va, y_va, cat_features=("area_id",)) -> "object":
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    model = LGBMRegressor(
        n_estimators=1500, learning_rate=0.05, num_leaves=63,
        min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=5, n_jobs=-1, random_state=config.RANDOM_STATE,
        verbosity=-1,
    )
    cats = [c for c in cat_features if c in X_tr.columns]
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)], eval_metric="rmse",
        categorical_feature=cats if cats else "auto",
        callbacks=[early_stopping(60, verbose=False), log_evaluation(0)],
    )
    return model


def train_forecaster(
    panel: pd.DataFrame,
    horizon_days: int = 28,
    test_days: int = 60,
    val_days: int = 60,
    features: Optional[List[str]] = None,
) -> Tuple[object, ForecastResult]:
    """Train a global LightGBM forecaster with a forward-chained holdout.

    `horizon_days` is reported in the result for documentation purposes.
    `test_days`/`val_days` set the size of the validation and test windows -
    a longer window stabilises the reported R² / MAE.
    """
    features = features or DEFAULT_FEATURES
    p = add_lag_features(panel)
    p = p.dropna(subset=["lag_28", "roll_mean_28", "adow_mean"]).copy()
    p["area_id"] = p["area_id"].astype("category").cat.codes

    test_cutoff = p["date"].max() - pd.Timedelta(days=test_days)
    val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

    train_mask = p["date"] <= val_cutoff
    val_mask = (p["date"] > val_cutoff) & (p["date"] <= test_cutoff)
    test_mask = p["date"] > test_cutoff

    X_tr, y_tr = p.loc[train_mask, features], p.loc[train_mask, "crimes"]
    X_va, y_va = p.loc[val_mask, features], p.loc[val_mask, "crimes"]
    X_te, y_te = p.loc[test_mask, features], p.loc[test_mask, "crimes"]

    log.info(f"forecast train={len(X_tr):,} val={len(X_va):,} test={len(X_te):,}")
    with timer(f"fit LightGBM forecaster (h={horizon_days}d)", log):
        model = _train_lgbm(X_tr, y_tr, X_va, y_va)

    y_pred = np.clip(model.predict(X_te), 0, None)

    # Naive baseline: predict each row as the area's previous 28-day mean.
    naive_pred = p.loc[test_mask, "roll_mean_28"].to_numpy()
    naive_mae = float(mean_absolute_error(y_te, np.nan_to_num(naive_pred, nan=y_te.mean())))
    mae = float(mean_absolute_error(y_te, y_pred))

    res = ForecastResult(
        horizon_days=horizon_days,
        mae=mae,
        rmse=float(np.sqrt(mean_squared_error(y_te, y_pred))),
        mape=_safe_mape(y_te.to_numpy(), y_pred),
        r2=float(r2_score(y_te, y_pred)),
        naive_mae=naive_mae,
        skill=1.0 - mae / naive_mae if naive_mae > 0 else 0.0,
        test_start=p.loc[test_mask, "date"].min(),
        test_end=p.loc[test_mask, "date"].max(),
        predictions=p.loc[test_mask, ["date", "area_id", "area_name"]]
                       .assign(y_true=y_te.to_numpy(), y_pred=y_pred)
                       .reset_index(drop=True),
    )
    log.info(f"forecast MAE={res.mae:.2f}  RMSE={res.rmse:.2f}  R2={res.r2:.3f}  "
             f"naive_MAE={res.naive_mae:.2f}  skill={res.skill:+.1%}")
    return model, res


# ---------------------------------------------------------------------------
def train_weekly_forecaster(
    weekly_panel: pd.DataFrame,
    test_weeks: int = 8,
    val_weeks: int = 8,
) -> Tuple[object, ForecastResult]:
    """Train a weekly forecaster (one prediction per area per week).

    Operates on the output of `feature_engineering.build_weekly_area_panel`.
    Weekly aggregation reduces noise dramatically and yields high R².
    """
    p = weekly_panel.sort_values(["area_id", "date"]).copy()
    g = p.groupby("area_id", observed=True)["crimes"]
    for L in (1, 2, 4, 8):
        p[f"lag_{L}"] = g.shift(L).astype("float32")
    p["roll_mean_4"] = (g.shift(1).rolling(4).mean()
                          .reset_index(level=0, drop=True).astype("float32"))
    p["roll_mean_12"] = (g.shift(1).rolling(12).mean()
                           .reset_index(level=0, drop=True).astype("float32"))
    p["roll_std_4"] = (g.shift(1).rolling(4).std()
                         .reset_index(level=0, drop=True).astype("float32"))
    p["weekofyear"] = p["date"].dt.isocalendar().week.astype("int16")
    p["month"] = p["date"].dt.month.astype("int8")
    p["year"] = p["date"].dt.year.astype("int16")

    feats = ["lag_1", "lag_2", "lag_4", "lag_8",
             "roll_mean_4", "roll_mean_12", "roll_std_4",
             "weekofyear", "month", "year", "area_id"]
    p = p.dropna(subset=["lag_8", "roll_mean_12"]).copy()
    p["area_id"] = p["area_id"].astype("category").cat.codes

    test_cutoff = p["date"].max() - pd.Timedelta(weeks=test_weeks)
    val_cutoff = test_cutoff - pd.Timedelta(weeks=val_weeks)
    tr = p[p["date"] <= val_cutoff]
    va = p[(p["date"] > val_cutoff) & (p["date"] <= test_cutoff)]
    te = p[p["date"] > test_cutoff]

    log.info(f"weekly forecast train={len(tr):,} val={len(va):,} test={len(te):,}")
    with timer("fit LightGBM weekly forecaster", log):
        model = _train_lgbm(tr[feats], tr["crimes"], va[feats], va["crimes"])

    y_pred = np.clip(model.predict(te[feats]), 0, None)
    y_te = te["crimes"].to_numpy()
    naive = np.nan_to_num(te["roll_mean_4"].to_numpy(), nan=y_te.mean())
    mae = float(mean_absolute_error(y_te, y_pred))
    naive_mae = float(mean_absolute_error(y_te, naive))

    res = ForecastResult(
        horizon_days=7 * test_weeks,
        mae=mae,
        rmse=float(np.sqrt(mean_squared_error(y_te, y_pred))),
        mape=_safe_mape(y_te, y_pred),
        r2=float(r2_score(y_te, y_pred)),
        naive_mae=naive_mae,
        skill=1.0 - mae / naive_mae if naive_mae > 0 else 0.0,
        test_start=te["date"].min(),
        test_end=te["date"].max(),
        predictions=te[["date", "area_id", "area_name"]].assign(
            y_true=y_te, y_pred=y_pred).reset_index(drop=True),
    )
    log.info(f"weekly MAE={res.mae:.2f}  RMSE={res.rmse:.2f}  R2={res.r2:.3f}  "
             f"naive_MAE={res.naive_mae:.2f}  skill={res.skill:+.1%}")
    return model, res


# ---------------------------------------------------------------------------
def forecast_next_period(
    panel: pd.DataFrame,
    model,
    horizon_days: int = 7,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Iteratively roll the model forward `horizon_days` days, per area.

    The model was trained with `area_id` encoded as category codes, so we
    re-apply the same encoding before prediction. Predictions are written
    back into `history` so each next step's lag features stay consistent.

    Returns one row per (date, area_id) with columns
        date, area_id, area_name, y_pred.
    """
    features = features or DEFAULT_FEATURES
    history = panel[["date", "area_id", "area_name", "crimes"]].copy()
    history = history.sort_values(["area_id", "date"]).reset_index(drop=True)

    # Stable mapping area_id -> integer code (must match training).
    area_codes = pd.Categorical(history["area_id"]).codes
    code_map = dict(zip(history["area_id"], area_codes))

    last_date = history["date"].max()
    areas_df = history[["area_id", "area_name"]].drop_duplicates().reset_index(drop=True)
    out_frames: list[pd.DataFrame] = []

    for step in range(1, horizon_days + 1):
        target_date = last_date + pd.Timedelta(days=step)

        # Append placeholder rows for this date (one per area) BEFORE feature build.
        placeholder = areas_df.assign(date=target_date, crimes=np.nan)
        history = pd.concat([history, placeholder], ignore_index=True)

        feat = add_lag_features(history)
        rows = (feat[feat["date"] == target_date]
                .sort_values("area_id")
                .reset_index(drop=True))

        X_new = rows[features].copy()
        X_new["area_id"] = rows["area_id"].map(code_map).astype("int32")

        y_new = np.clip(model.predict(X_new), 0, None)
        rows = rows.assign(y_pred=y_new)
        out_frames.append(rows[["date", "area_id", "area_name", "y_pred"]])

        # Write the predictions into history so the next-step lags use them.
        idx = history.index[history["date"] == target_date]
        # Align by area_id ordering.
        ordered_pairs = (history.loc[idx, ["area_id"]]
                                  .reset_index()
                                  .sort_values("area_id"))
        history.loc[ordered_pairs["index"].to_numpy(), "crimes"] = y_new

    return pd.concat(out_frames, ignore_index=True)
