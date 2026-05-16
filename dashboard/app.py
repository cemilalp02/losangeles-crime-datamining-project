"""Crime Forecast LA - Streamlit Dashboard

Interactive dashboard for exploring spatio-temporal crime risk in Los Angeles.

Run from the project root:
    py -m streamlit run dashboard/app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Make the `src` package importable regardless of where streamlit is launched.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config  # noqa: E402

st.set_page_config(
    page_title="Crime Forecast LA",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clean() -> pd.DataFrame:
    if not config.CLEAN_PARQUET.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.CLEAN_PARQUET)


@st.cache_data(show_spinner=False)
def load_daily() -> pd.DataFrame:
    if not config.DAILY_AREA_PARQUET.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.DAILY_AREA_PARQUET)


@st.cache_data(show_spinner=False)
def load_risk() -> pd.DataFrame:
    p = config.FEATURES_DIR / "area_risk.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame:
    p = config.FEATURES_DIR / "forecast_next7.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_dbscan() -> pd.DataFrame:
    p = config.FEATURES_DIR / "dbscan_clusters.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ Crime Forecast LA")
st.sidebar.caption("Spatio-Temporal Crime Mining for Los Angeles")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🗺️ Hotspots", "📈 Time Trends", "🧠 Forecast & Risk", "🎯 Area Drilldown"],
)

clean = load_clean()
if clean.empty:
    st.error(
        "Processed data not found. Please run `py pipeline.py --step all` first to "
        "generate the parquet artifacts under `data/processed/` and `data/features/`."
    )
    st.stop()

areas = sorted(clean["area_name"].dropna().unique().tolist())
date_min = clean["date_occurred"].min().date()
date_max = clean["date_occurred"].max().date()

st.sidebar.markdown("### Filters")
date_range = st.sidebar.date_input("Date range", value=(date_min, date_max),
                                   min_value=date_min, max_value=date_max)
selected_areas = st.sidebar.multiselect("Areas", areas, default=[])
violent_only = st.sidebar.checkbox("Violent crimes only", value=False)


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        out = out[(out["date_occurred"] >= d0) & (out["date_occurred"] < d1)]
    if selected_areas:
        out = out[out["area_name"].isin(selected_areas)]
    if violent_only:
        out = out[out["is_violent"] == 1]
    return out


# ===========================================================================
# OVERVIEW
# ===========================================================================
if page == "📊 Overview":
    st.title("Crime Forecast LA - Overview")
    st.caption(f"Dataset: {date_min} → {date_max}, {len(clean):,} crime records, {len(areas)} LAPD areas.")

    sub = filter_df(clean)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Crimes (filter)", f"{len(sub):,}")
    c2.metric("Violent share", f"{sub['is_violent'].mean()*100:.1f}%" if len(sub) else "0.0%")
    c3.metric("Arrest share", f"{sub['is_arrest'].mean()*100:.1f}%" if len(sub) else "0.0%")
    c4.metric("Areas", f"{sub['area_name'].nunique()}")
    c5.metric("Crime types", f"{sub['crime_desc'].nunique()}")

    st.divider()
    cA, cB = st.columns(2)
    with cA:
        st.subheader("Top 15 Crime Types")
        ct = (sub.groupby("crime_desc", observed=True).size()
                  .sort_values(ascending=False).head(15))
        st.plotly_chart(
            px.bar(ct.iloc[::-1], orientation="h",
                   color=ct.iloc[::-1].values, color_continuous_scale="Reds")
              .update_layout(xaxis_title="count", yaxis_title="", showlegend=False,
                             coloraxis_showscale=False, height=480),
            use_container_width=True,
        )
    with cB:
        st.subheader("Top 15 Areas")
        ca = (sub.groupby("area_name", observed=True).size()
                  .sort_values(ascending=False).head(15))
        st.plotly_chart(
            px.bar(ca.iloc[::-1], orientation="h",
                   color=ca.iloc[::-1].values, color_continuous_scale="Blues")
              .update_layout(xaxis_title="count", yaxis_title="", showlegend=False,
                             coloraxis_showscale=False, height=480),
            use_container_width=True,
        )

    st.subheader("Hour × Day-of-week heatmap")
    pivot = (sub.groupby(["dow", "hour"]).size().reset_index(name="n")
                 .pivot(index="dow", columns="hour", values="n").fillna(0))
    if not pivot.empty:
        pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(pivot)]
        st.plotly_chart(
            px.imshow(pivot, color_continuous_scale="Reds", aspect="auto",
                      labels=dict(x="Hour of day", y="", color="Crimes")),
            use_container_width=True,
        )


# ===========================================================================
# HOTSPOTS
# ===========================================================================
elif page == "🗺️ Hotspots":
    st.title("Crime Hotspot Discovery")
    st.caption("Spatial concentrations of crime via point density and DBSCAN clustering.")
    sub = filter_df(clean).dropna(subset=["lat", "lon"])

    sample_n = st.slider("Sample size for map", 5_000, 100_000, 30_000, step=5_000)
    if len(sub) > sample_n:
        sub = sub.sample(sample_n, random_state=42)

    cA, cB = st.columns([3, 2])
    with cA:
        st.subheader("Crime density map")
        fig = px.density_map(
            sub, lat="lat", lon="lon", radius=6, zoom=10,
            center=dict(lat=34.05, lon=-118.25),
            map_style="carto-positron", height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    with cB:
        st.subheader("Top DBSCAN clusters")
        clusters = load_dbscan()
        if not clusters.empty:
            st.dataframe(
                clusters.head(15).style.format({
                    "lat": "{:.4f}", "lon": "{:.4f}",
                    "violent_share": "{:.1%}",
                }),
                use_container_width=True, height=560,
            )
        else:
            st.info("DBSCAN cluster artefact not yet generated.")


# ===========================================================================
# TIME TRENDS
# ===========================================================================
elif page == "📈 Time Trends":
    st.title("Temporal Patterns of Crime")
    sub = filter_df(clean)

    freq = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"], index=2)
    rule = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}[freq]
    ts_all = sub.set_index("date_occurred").sort_index()["dr_no"].resample(rule).count()
    ts_v = sub[sub["is_violent"] == 1].set_index("date_occurred").sort_index()["dr_no"].resample(rule).count()

    df_ts = pd.DataFrame({"All crimes": ts_all, "Violent": ts_v}).reset_index()
    fig = px.line(df_ts, x="date_occurred", y=["All crimes", "Violent"],
                  labels={"value": "count", "date_occurred": "", "variable": ""},
                  height=420)
    fig.update_traces(line=dict(width=2))
    st.plotly_chart(fig, use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        st.subheader("By month")
        m = sub.groupby("month").size().reset_index(name="n")
        st.plotly_chart(px.bar(m, x="month", y="n", color="n",
                                 color_continuous_scale="Reds")
                          .update_layout(coloraxis_showscale=False, height=320),
                          use_container_width=True)
    with cB:
        st.subheader("By hour")
        h = sub.groupby("hour").size().reset_index(name="n")
        st.plotly_chart(px.bar(h, x="hour", y="n", color="n",
                                 color_continuous_scale="Blues")
                          .update_layout(coloraxis_showscale=False, height=320),
                          use_container_width=True)


# ===========================================================================
# FORECAST & RISK
# ===========================================================================
elif page == "🧠 Forecast & Risk":
    st.title("7-Day Forecast and Area Risk Score")
    risk = load_risk()
    fc = load_forecast()

    if risk.empty:
        st.warning("Risk artefact not yet generated. Run `py pipeline.py --step risk` first.")
    else:
        st.subheader("Risk-ranked LAPD areas")
        st.caption(
            "Composite score = "
            f"{config.RISK_WEIGHTS['intensity']:.2f}·intensity + "
            f"{config.RISK_WEIGHTS['trend']:.2f}·trend + "
            f"{config.RISK_WEIGHTS['violence']:.2f}·violence + "
            f"{config.RISK_WEIGHTS['forecast']:.2f}·forecast"
        )
        cols = ["area_name", "risk_rank", "risk_score", "risk_tier",
                "intensity", "trend", "violence", "forecast"]
        st.dataframe(
            risk[cols].style.format({
                "risk_score": "{:.3f}", "intensity": "{:.2f}",
                "trend": "{:.2f}", "violence": "{:.1%}", "forecast": "{:.2f}",
            }),
            use_container_width=True, height=520,
        )

        fig = px.bar(risk, x="area_name", y="risk_score", color="risk_tier",
                     color_discrete_map={"Low": "#2A9D8F", "Medium": "#E9C46A",
                                         "High": "#F4A261", "Critical": "#E63946"},
                     height=420)
        fig.update_layout(xaxis_title="", yaxis_title="Risk score")
        st.plotly_chart(fig, use_container_width=True)

    if not fc.empty:
        st.subheader("Forecast: next 7 days, total daily crimes (all areas)")
        ts = fc.groupby("date")["y_pred"].sum().reset_index()
        st.plotly_chart(px.line(ts, x="date", y="y_pred", markers=True,
                                  labels={"y_pred": "predicted crimes"})
                          .update_layout(height=320),
                          use_container_width=True)
        st.subheader("Forecast by area (sum over 7 days)")
        by_a = fc.groupby("area_name")["y_pred"].sum().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(by_a, x="area_name", y="y_pred",
                                 color="y_pred", color_continuous_scale="Reds")
                          .update_layout(coloraxis_showscale=False,
                                         xaxis_title="", yaxis_title="forecast crimes",
                                         height=380),
                          use_container_width=True)


# ===========================================================================
# AREA DRILLDOWN
# ===========================================================================
elif page == "🎯 Area Drilldown":
    st.title("Area Drilldown")
    daily = load_daily()
    risk = load_risk()
    if daily.empty:
        st.warning("Daily panel not found. Run `py pipeline.py --step features`.")
    else:
        area = st.selectbox("Choose an LAPD area", sorted(daily["area_name"].unique()))
        a = daily[daily["area_name"] == area].sort_values("date")
        a["roll7"] = a["crimes"].rolling(7, min_periods=1).mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg crimes / day", f"{a['crimes'].mean():.1f}")
        c2.metric("Last-28-day avg", f"{a.tail(28)['crimes'].mean():.1f}")
        if not risk.empty:
            r = risk.set_index("area_name").loc[area]
            c3.metric("Risk tier", str(r["risk_tier"]))

        fig = px.area(a, x="date", y="roll7", labels={"roll7": "7-day moving avg"},
                      height=380)
        st.plotly_chart(fig, use_container_width=True)

        # Crime mix in this area
        sub = clean[clean["area_name"] == area]
        st.subheader("Top 12 crime types in this area")
        ct = (sub.groupby("crime_desc", observed=True).size()
                  .sort_values(ascending=False).head(12)).iloc[::-1]
        st.plotly_chart(px.bar(ct, orientation="h",
                                 color=ct.values, color_continuous_scale="Reds")
                          .update_layout(coloraxis_showscale=False, height=420,
                                         xaxis_title="", yaxis_title=""),
                          use_container_width=True)
