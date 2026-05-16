"""Reusable EDA plotting / summarisation helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import config
from .utils import apply_plot_style, save_fig, get_logger

log = get_logger(__name__)


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row summary of dataset size, date range, missingness."""
    return pd.DataFrame({
        "rows": [len(df)],
        "columns": [df.shape[1]],
        "date_min": [df["date_occurred"].min()],
        "date_max": [df["date_occurred"].max()],
        "areas": [df["area_name"].nunique()],
        "crime_types": [df["crime_desc"].nunique()],
        "violent_share": [df[config.TARGET_VIOLENT].mean()],
        "arrest_share": [df[config.TARGET_ARREST].mean()],
        "missing_lat_pct": [df["lat"].isna().mean()],
    })


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    out = miss[miss > 0].rename("missing_pct").to_frame()
    out["missing_count"] = (df.isna().sum()).loc[out.index]
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_crimes_over_time(df: pd.DataFrame, freq: str = "ME") -> plt.Figure:
    apply_plot_style()
    s = df.set_index("date_occurred").sort_index()
    monthly = s["dr_no"].resample(freq).count()
    violent = s[s[config.TARGET_VIOLENT] == 1]["dr_no"].resample(freq).count()
    fig, ax = plt.subplots(figsize=(13, 5))
    monthly.plot(ax=ax, label="All crimes", color="#4361EE", lw=2)
    violent.plot(ax=ax, label="Violent crimes", color="#E63946", lw=2)
    ax.set_title("Reported Crimes in Los Angeles (2020 - present)")
    ax.set_ylabel("Count per period")
    ax.set_xlabel("")
    ax.legend()
    save_fig(fig, "01_crimes_over_time")
    return fig


def plot_hour_dow_heatmap(df: pd.DataFrame) -> plt.Figure:
    apply_plot_style()
    pivot = (df.groupby(["dow", "hour"]).size()
               .reset_index(name="n")
               .pivot(index="dow", columns="hour", values="n").fillna(0))
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap="rocket_r", ax=ax, cbar_kws={"label": "Crime count"})
    ax.set_title("When do crimes happen? (Day-of-week x Hour-of-day)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    save_fig(fig, "02_hour_dow_heatmap")
    return fig


def plot_top_areas(df: pd.DataFrame, top: int = 15) -> plt.Figure:
    apply_plot_style()
    counts = (df.groupby("area_name", observed=True).size()
                .sort_values(ascending=False).head(top))
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index,
                palette="rocket_r", ax=ax, legend=False)
    ax.set_title(f"Top {top} LAPD Areas by Crime Volume")
    ax.set_xlabel("Crime count")
    ax.set_ylabel("")
    save_fig(fig, "03_top_areas")
    return fig


def plot_top_crime_types(df: pd.DataFrame, top: int = 15) -> plt.Figure:
    apply_plot_style()
    counts = (df.groupby("crime_desc", observed=True).size()
                .sort_values(ascending=False).head(top))
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index,
                palette="mako_r", ax=ax, legend=False)
    ax.set_title(f"Top {top} Crime Types")
    ax.set_xlabel("Crime count")
    ax.set_ylabel("")
    save_fig(fig, "04_top_crime_types")
    return fig


def plot_victim_demographics(df: pd.DataFrame) -> plt.Figure:
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    age = df["victim_age"].dropna()
    age = age[(age >= 5) & (age <= 90)]
    sns.histplot(age, bins=40, color="#4361EE", ax=axes[0])
    axes[0].set_title("Victim Age Distribution")
    axes[0].set_xlabel("Age")

    sex = df["victim_sex"].value_counts()
    sns.barplot(x=sex.index.astype(str), y=sex.values,
                hue=sex.index.astype(str),
                palette="rocket", ax=axes[1], legend=False)
    axes[1].set_title("Victim Sex")
    axes[1].set_ylabel("Count")
    save_fig(fig, "05_victim_demographics")
    return fig


def plot_arrest_rate_by_crime(df: pd.DataFrame, top: int = 15) -> plt.Figure:
    apply_plot_style()
    g = (df.groupby("crime_desc", observed=True)
           .agg(n=("dr_no", "count"), arrests=(config.TARGET_ARREST, "sum"))
           .query("n > 1000"))
    g["arrest_rate"] = g["arrests"] / g["n"]
    g = g.sort_values("arrest_rate", ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=g["arrest_rate"], y=g.index, hue=g.index,
                palette="viridis", ax=ax, legend=False)
    ax.set_title(f"Top {top} Crime Types by Arrest Rate (n>1000)")
    ax.set_xlabel("Arrest rate")
    ax.set_ylabel("")
    save_fig(fig, "06_arrest_rate_by_crime")
    return fig


def plot_violent_share_by_area(df: pd.DataFrame) -> plt.Figure:
    apply_plot_style()
    g = (df.groupby("area_name", observed=True)[config.TARGET_VIOLENT]
           .mean().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=g.values, y=g.index, hue=g.index, palette="rocket_r",
                ax=ax, legend=False)
    ax.set_title("Violent-Crime Share by LAPD Area")
    ax.set_xlabel("Violent share")
    ax.set_ylabel("")
    save_fig(fig, "07_violent_share_by_area")
    return fig
