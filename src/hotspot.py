"""Spatial hotspot discovery: KDE heatmap, DBSCAN clustering, KMeans grids."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MiniBatchKMeans

from . import config
from .utils import get_logger, timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# DBSCAN density-based hotspots
# ---------------------------------------------------------------------------
def dbscan_hotspots(
    df: pd.DataFrame,
    sample_size: int = 50_000,
    eps_km: float = 0.4,
    min_samples: int = 80,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Run DBSCAN over (lat, lon) using haversine distance.

    Returns the sampled rows with an added `cluster` column. Cluster `-1` = noise.
    """
    pts = df.dropna(subset=["lat", "lon"]).copy()
    if len(pts) > sample_size:
        pts = pts.sample(sample_size, random_state=random_state)

    coords_rad = np.radians(pts[["lat", "lon"]].to_numpy())
    earth_radius_km = 6371.0088
    eps_rad = eps_km / earth_radius_km

    with timer(f"DBSCAN n={len(pts):,} eps={eps_km}km", log):
        db = DBSCAN(eps=eps_rad, min_samples=min_samples,
                    metric="haversine", algorithm="ball_tree", n_jobs=-1)
        labels = db.fit_predict(coords_rad)
    pts["cluster"] = labels
    n_clusters = (np.unique(labels) >= 0).sum()
    log.info(f"DBSCAN found {n_clusters} hotspot clusters, "
             f"{(labels == -1).mean():.1%} noise")
    return pts


def cluster_summary(clustered: pd.DataFrame) -> pd.DataFrame:
    """Summarise each DBSCAN cluster: size, centroid, top crime type."""
    valid = clustered[clustered["cluster"] >= 0]
    if valid.empty:
        return pd.DataFrame()
    summary = (valid.groupby("cluster")
                    .agg(size=("dr_no", "count"),
                         lat=("lat", "mean"),
                         lon=("lon", "mean"),
                         violent_share=(config.TARGET_VIOLENT, "mean"))
                    .reset_index())
    top_type = (valid.groupby(["cluster", "crime_desc"], observed=True)
                     .size().reset_index(name="n")
                     .sort_values(["cluster", "n"], ascending=[True, False])
                     .drop_duplicates("cluster")
                     [["cluster", "crime_desc"]]
                     .rename(columns={"crime_desc": "top_crime"}))
    return summary.merge(top_type, on="cluster").sort_values("size", ascending=False)


# ---------------------------------------------------------------------------
# Grid-based hotspots (rasterise LA into ~250m cells and rank)
# ---------------------------------------------------------------------------
def grid_hotspots(df: pd.DataFrame, cell_deg: float = 0.0025) -> pd.DataFrame:
    """Bucket incidents into a lat/lon grid; return cells ranked by intensity."""
    pts = df.dropna(subset=["lat", "lon"]).copy()
    pts["lat_bin"] = (pts["lat"] / cell_deg).round() * cell_deg
    pts["lon_bin"] = (pts["lon"] / cell_deg).round() * cell_deg
    grid = (pts.groupby(["lat_bin", "lon_bin"])
                .agg(crimes=("dr_no", "count"),
                     violent=(config.TARGET_VIOLENT, "sum"))
                .reset_index()
                .rename(columns={"lat_bin": "lat", "lon_bin": "lon"}))
    grid["violent_share"] = grid["violent"] / grid["crimes"]
    return grid.sort_values("crimes", ascending=False)


# ---------------------------------------------------------------------------
# KMeans area-archetype clustering
# ---------------------------------------------------------------------------
def kmeans_area_archetypes(daily_panel: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Cluster LAPD areas by their daily crime profile (volume, volatility, mix)."""
    g = (daily_panel.groupby(["area_id", "area_name"], observed=True)
                    .agg(mean_crimes=("crimes", "mean"),
                         std_crimes=("crimes", "std"),
                         violent=("violent", "mean"),
                         arrest=("arrests", "mean"))
                    .reset_index())
    X = g[["mean_crimes", "std_crimes", "violent", "arrest"]].fillna(0).to_numpy()
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
    km = MiniBatchKMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
    g["archetype"] = km.fit_predict(Xn)
    return g


# ---------------------------------------------------------------------------
# Folium map (interactive HTML output)
# ---------------------------------------------------------------------------
def folium_heatmap(df: pd.DataFrame, sample: int = 100_000,
                   out_html: Optional[Path] = None,
                   only_violent: bool = False) -> Path:
    import folium
    from folium.plugins import HeatMap

    out_html = out_html or (config.FIGURES_DIR / "hotspot_heatmap.html")
    pts = df.dropna(subset=["lat", "lon"])
    if only_violent:
        pts = pts[pts[config.TARGET_VIOLENT] == 1]
    if len(pts) > sample:
        pts = pts.sample(sample, random_state=config.RANDOM_STATE)

    m = folium.Map(location=[34.05, -118.25], zoom_start=11, tiles="cartodbpositron")
    HeatMap(pts[["lat", "lon"]].to_numpy(), radius=8, blur=12,
            max_zoom=13).add_to(m)
    m.save(str(out_html))
    log.info(f"folium heatmap -> {out_html}")
    return out_html
