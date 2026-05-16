"""End-to-end pipeline runner.

Usage (PowerShell):
    py pipeline.py --step all
    py pipeline.py --step preprocess
    py pipeline.py --step features
    py pipeline.py --step models
    py pipeline.py --step forecast
    py pipeline.py --step risk
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src import (
    config,
    data_loader,
    preprocessing,
    feature_engineering,
    eda,
    hotspot,
    classification,
    forecasting,
    risk_scoring,
    explainability,
)
from src.utils import get_logger, timer

log = get_logger("pipeline")


def step_preprocess() -> pd.DataFrame:
    raw = data_loader.load_raw()
    clean = preprocessing.clean(raw)
    preprocessing.save_clean(clean)
    return clean


def step_features(clean: pd.DataFrame | None = None):
    clean = clean if clean is not None else data_loader.load_clean()
    inc = feature_engineering.build_incident_features(clean)
    inc.to_parquet(config.FEATURES_DIR / "incident_features.parquet", index=False)

    daily = feature_engineering.build_daily_area_panel(clean)
    daily.to_parquet(config.DAILY_AREA_PARQUET, index=False)

    weekly = feature_engineering.build_weekly_area_panel(daily)
    weekly.to_parquet(config.WEEKLY_AREA_PARQUET, index=False)
    log.info(f"daily panel rows={len(daily):,}  weekly panel rows={len(weekly):,}")
    return inc, daily, weekly


def step_eda(clean: pd.DataFrame | None = None) -> dict:
    clean = clean if clean is not None else data_loader.load_clean()
    eda.plot_crimes_over_time(clean)
    eda.plot_hour_dow_heatmap(clean)
    eda.plot_top_areas(clean)
    eda.plot_top_crime_types(clean)
    eda.plot_victim_demographics(clean)
    eda.plot_arrest_rate_by_crime(clean)
    eda.plot_violent_share_by_area(clean)
    overview = eda.overview(clean).iloc[0].to_dict()
    overview = {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in overview.items()}
    (config.REPORTS_DIR / "eda_overview.json").write_text(
        json.dumps(overview, indent=2, default=str)
    )
    return overview


def step_models() -> dict:
    inc = pd.read_parquet(config.FEATURES_DIR / "incident_features.parquet")
    metrics: dict = {}

    for target in (config.TARGET_VIOLENT, config.TARGET_ARREST):
        log.info(f"=== training classifiers for target={target} ===")
        results = classification.train_classifier(inc, target=target,
                                                  models=("logreg", "lgbm"),
                                                  sample=200_000)
        for name, r in results.items():
            metrics[f"{target}__{name}"] = {
                "accuracy": r.accuracy, "f1_pos": r.f1_pos,
                "roc_auc": r.roc_auc, "pr_auc": r.pr_auc,
            }

        # Persist best model + run explainability on it
        best_name = max(results, key=lambda k: results[k].pr_auc)
        best_pipe = results[best_name].pipeline
        with open(config.MODELS_DIR / f"{target}_{best_name}.pkl", "wb") as f:
            pickle.dump(best_pipe, f)
        log.info(f"saved best model for {target}: {best_name}")

        # SHAP summary on the LightGBM model when available
        if "lgbm" in results:
            X = inc.dropna(subset=[target])[classification.NUMERIC + classification.CATEGORICAL]
            explainability.shap_summary(results["lgbm"].pipeline, X,
                                        fname=f"shap_{target}",
                                        sample=3000)

    (config.REPORTS_DIR / "classification_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str))
    return metrics


def step_forecast() -> dict:
    daily = pd.read_parquet(config.DAILY_AREA_PARQUET)
    daily = feature_engineering.trim_incomplete_tail(daily)

    # ----- Daily area forecaster ---------------------------------------
    model, result = forecasting.train_forecaster(daily, horizon_days=28)
    with open(config.MODELS_DIR / "forecaster_lgbm.pkl", "wb") as f:
        pickle.dump(model, f)
    result.predictions.to_parquet(config.FEATURES_DIR / "forecast_predictions.parquet",
                                  index=False)

    next7 = forecasting.forecast_next_period(daily, model, horizon_days=7)
    next7.to_parquet(config.FEATURES_DIR / "forecast_next7.parquet", index=False)

    # ----- Weekly area forecaster (lower noise; primary for risk) ------
    weekly = pd.read_parquet(config.WEEKLY_AREA_PARQUET)
    # Trim incomplete weekly tail too (last 4 weeks).
    last_complete = daily["date"].max()
    weekly = weekly[weekly["date"] <= last_complete - pd.Timedelta(days=14)]
    wmodel, wresult = forecasting.train_weekly_forecaster(weekly)
    with open(config.MODELS_DIR / "forecaster_weekly_lgbm.pkl", "wb") as f:
        pickle.dump(wmodel, f)
    wresult.predictions.to_parquet(config.FEATURES_DIR / "forecast_weekly_predictions.parquet",
                                   index=False)

    metrics = {
        "daily": {
            "horizon_days": result.horizon_days, "MAE": result.mae,
            "RMSE": result.rmse, "MAPE": result.mape, "R2": result.r2,
            "naive_MAE": result.naive_mae, "skill": result.skill,
            "test_start": str(result.test_start), "test_end": str(result.test_end),
        },
        "weekly": {
            "horizon_weeks": 8,
            "MAE": wresult.mae, "RMSE": wresult.rmse, "MAPE": wresult.mape,
            "R2": wresult.r2, "naive_MAE": wresult.naive_mae, "skill": wresult.skill,
            "test_start": str(wresult.test_start), "test_end": str(wresult.test_end),
        },
    }
    (config.REPORTS_DIR / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"forecast metrics: {json.dumps(metrics, indent=2)}")
    return metrics


def step_hotspot() -> dict:
    clean = data_loader.load_clean()
    grid = hotspot.grid_hotspots(clean)
    grid.head(500).to_parquet(config.FEATURES_DIR / "grid_hotspots_top500.parquet",
                              index=False)
    db = hotspot.dbscan_hotspots(clean, sample_size=60_000, eps_km=0.4, min_samples=80)
    summary = hotspot.cluster_summary(db)
    summary.to_parquet(config.FEATURES_DIR / "dbscan_clusters.parquet", index=False)
    hotspot.folium_heatmap(clean, sample=120_000)
    hotspot.folium_heatmap(clean, sample=80_000, only_violent=True,
                           out_html=config.FIGURES_DIR / "hotspot_heatmap_violent.html")
    return {"dbscan_clusters": int(len(summary))}


def step_risk() -> pd.DataFrame:
    daily = pd.read_parquet(config.DAILY_AREA_PARQUET)
    daily = feature_engineering.trim_incomplete_tail(daily)
    fpath = config.FEATURES_DIR / "forecast_next7.parquet"
    forecast = pd.read_parquet(fpath) if fpath.exists() else None
    risk = risk_scoring.compute_area_risk(daily, forecast)
    risk.to_parquet(config.FEATURES_DIR / "area_risk.parquet", index=False)
    log.info("Top 5 highest-risk LAPD areas:\n" +
             risk.head(5)[["area_name", "risk_score", "risk_tier"]].to_string(index=False))
    return risk


def main() -> None:
    parser = argparse.ArgumentParser(description="Crime Forecast LA pipeline runner")
    parser.add_argument("--step", default="all",
                        choices=["all", "preprocess", "features", "eda",
                                 "models", "forecast", "hotspot", "risk"])
    args = parser.parse_args()

    if args.step in ("all", "preprocess"):
        with timer("STEP preprocess"):
            step_preprocess()
    if args.step in ("all", "features"):
        with timer("STEP features"):
            step_features()
    if args.step in ("all", "eda"):
        with timer("STEP eda"):
            step_eda()
    if args.step in ("all", "hotspot"):
        with timer("STEP hotspot"):
            step_hotspot()
    if args.step in ("all", "models"):
        with timer("STEP models"):
            step_models()
    if args.step in ("all", "forecast"):
        with timer("STEP forecast"):
            step_forecast()
    if args.step in ("all", "risk"):
        with timer("STEP risk"):
            step_risk()


if __name__ == "__main__":
    main()
