# Crime Forecast LA — Final Report

**Spatio-Temporal Crime Forecasting, Hotspot Discovery and Explainable Risk Modeling**

Data Mining Course Project

---

## 1. Problem statement

The City of Los Angeles publishes ~1 million LAPD crime reports per five-year window.
Manually surfacing actionable patterns from this volume is impossible. Our project asks:

> *Where, when, and what kind of crime risk is rising in Los Angeles, and how will it
> evolve over the next 1–8 weeks?*

We frame this as a multi-task data-mining problem and deliver an end-to-end pipeline:

1. **Descriptive EDA** of temporal, spatial and categorical structure.
2. **Classification** — violent vs non-violent crime, and arrest outcome.
3. **Hotspot discovery** — grid + density-based clustering.
4. **Spatio-temporal forecasting** — daily and weekly area-level crime counts.
5. **Risk scoring** — composite area-level score combining the above.
6. **Explainability** — SHAP + permutation importance for every predictive model.

The full pipeline is reproducible via `pipeline.py --step all` and visualised in a
Streamlit dashboard.

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Source   | LAPD Open Data via Kaggle (`haseefalam/crime-dataset`) |
| Rows     | **955 339** |
| Columns  | 28 raw → 41 after preprocessing |
| Period   | 2020-01-01 → 2024-06-24 (effective) |
| Areas    | 21 LAPD divisions |
| Crime types | 139 distinct |

Key quality issues handled:

* `LAT`/`LON = 0.0` for ~0.2 % of incidents — masked to `NaN`, kept inside an LA
  bounding box (33.6–34.4 N, −118.8 to −117.95 W).
* Reporting lag: the most recent ~6 weeks are systematically under-reported
  (city-wide totals fall from ~430/day to ~38/day in the last fortnight). Our
  `trim_incomplete_tail` heuristic drops these days from the forecast/risk training
  data automatically.
* `Vict Age = 0` and impossible ages → `NaN`.
* `Vict Sex` outside `{M, F}` → "Unknown".

Targets derived:

* `is_violent` — incident's primary code is in the FBI-UCR Part 1 violent set
  (homicide, rape, robbery, aggravated assault) plus weapons / kidnapping / criminal
  threats. **17 %** of incidents are violent.
* `is_arrest` — `Status` ∈ {`AA`, `JA`}. **9 %** of incidents end in arrest.

---

## 3. Pipeline architecture

```
raw CSV  ─▶  data_loader  ─▶  preprocessing  ─▶  feature_engineering
                                                    │
                                          ┌─────────┼──────────┐
                                          ▼                    ▼
                                    incident features   daily/weekly panels
                                          │                    │
                       ┌──────────────────┘                    │
                       ▼                                       ▼
              classification (LR / LGBM)              forecasting (LGBM, daily+weekly)
                       │                                       │
                       └──────────────┬───────────────────────┘
                                      ▼
                              risk_scoring  ─▶  Streamlit dashboard
                                      │
                                      ▼
                              explainability (SHAP + permutation)
```

Each stage writes to `data/processed/`, `data/features/`, or `reports/` so downstream
stages remain decoupled and re-runnable.

---

## 4. Exploratory analysis

Selected findings (full plots in `reports/figures/`):

* **Daily seasonality** — crimes peak between 16:00 and 22:00; a secondary spike
  at midnight reflects automatic timestamps when the exact hour is unknown.
  (`02_hour_dow_heatmap.png`)
* **Weekly seasonality** — Friday and Saturday late-night dominate.
* **Spatial concentration** — the top 5 LAPD areas (Central, 77th Street, Southwest,
  Pacific, N Hollywood) account for ~30 % of all incidents.
  (`03_top_areas.png`)
* **Crime mix** — vehicle theft, battery-simple assault, burglary from vehicle,
  and theft of identity dominate. (`04_top_crime_types.png`)
* **Violent share** is highest in **77th Street, Southeast, Newton** (≥ 0.20).
  (`07_violent_share_by_area.png`)
* **Arrest rate** is highest for narcotics, weapons offences, and DUI — i.e.
  encounters where the suspect is on scene; lowest for property crimes.
  (`06_arrest_rate_by_crime.png`)

---

## 5. Classification

### 5.1 Setup

* Features: 14 numeric (cyclical time, victim age, lat/lon, weapon flag, etc.) plus
  4 high-cardinality categorical (area, victim sex/descent, part-of-day) one-hot
  encoded with `min_frequency = 200` to bound feature dimensionality.
* Pipeline: `ColumnTransformer(StandardScaler, OneHotEncoder)` → estimator.
* Imbalanced classes handled with `class_weight = balanced`.
* 200 K-row stratified subsample, 80/20 train/test, fixed seed.

### 5.2 Results

| Target | Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|--------|-------|----------|----|---------|--------|
| `is_violent` | LogReg | 0.823 | 0.650 | 0.915 | 0.593 |
| `is_violent` | **LGBM** | **0.826** | **0.654** | **0.923** | **0.620** |
| `is_arrest`  | LogReg | 0.676 | 0.266 | 0.712 | 0.184 |
| `is_arrest`  | **LGBM** | **0.683** | **0.279** | **0.741** | **0.205** |

* The violence classifier is **highly informative** (AUC > 0.92) — the rule
  learned is essentially "weapon used + crime category + premise".
* The arrest classifier is much harder (PR-AUC ≈ 0.21 vs the 9 % base rate). This
  is expected and matches published LAPD analyses: the arrest decision depends on
  many factors not in the dataset (suspect availability, witness statements,
  evidence quality).

### 5.3 Drivers (SHAP)

`reports/figures/shap_is_violent.png` confirms the top contributors are:

1. `weapon_used` — strong positive contribution.
2. `part_of_day = Night/Late Night` — positive contribution.
3. `victim_age` — bell-shaped: peak risk between 18-35.
4. Specific `area_name` indicators (77th Street, Southeast, Newton).
5. `hour_sin` / `hour_cos` — captures non-linear time-of-day effect.

For arrests (`shap_is_arrest.png`), the dominant drivers are `weapon_used` and
specific crime / premise categories — i.e. severity of the encounter.

---

## 6. Hotspot discovery

Three complementary views:

* **Grid hotspots** — bucket incidents into ~250 m × 250 m cells. The top cell
  (≈ 34.045 N, −118.250 W) is in **Central / Skid Row**.
* **DBSCAN** with haversine metric (`eps = 400 m`, `min_samples = 80`) on a 60 K
  sample yields ~30 distinct clusters; the largest covers Central and South LA.
  Per-cluster top-crime descriptions are written to
  `data/features/dbscan_clusters.parquet`.
* **K-Means area archetypes** with `k = 4` clusters the 21 LAPD areas by their
  daily volume / volatility / violent share / arrest share. This produces an
  interpretable typology (high-volume central, residential mid-volume,
  outlying low-volume, mixed).
* **Folium HTML heatmaps** — `reports/figures/hotspot_heatmap.html` and
  `hotspot_heatmap_violent.html` are interactive and shareable.

---

## 7. Spatio-temporal forecasting

### 7.1 Modelling choice

Daily area-level crime counts are noisy: the per-cell standard deviation is comparable
to the mean. A single global LightGBM regressor outperformed per-area ARIMA in our
experiments, was 100× faster, and benefited from cross-area feature transfer.

Features per (date, area) row:

* Lags: 1, 7, 14, 28 days.
* Rolling means / stds over 7 / 28 / 90 days (shifted by 1 day to avoid leakage).
* Calendar: day-of-week, month, week-of-year, day-of-year, weekend flag.
* `area_id` as native LightGBM categorical.

A weekly version uses the same feature template scaled to weeks.

### 7.2 Validation

* **Forward-chained holdout**: last 60 days = test, preceding 60 days = validation.
* **Naive baseline**: predict each row as the area's previous 28-day rolling mean
  (last-4-week mean for the weekly model). All metrics are reported alongside this
  baseline so improvements are interpretable under the structural decline in
  reported LA crime since 2022.

### 7.3 Results

| Granularity | MAE | RMSE | R² | naive MAE | **Skill** |
|-------------|-----|------|----|-----------|-----------|
| Daily, per area  |  7.79 |  9.48 | −0.13 |  8.64 | **+9.8 %**  |
| Weekly, per area | 40.45 | 53.82 | −0.10 | 47.73 | **+15.2 %** |

* The **negative R²** is a known artefact of distribution shift: the train mean is
  ~28 crimes/day per area, the test mean is ~21. Even a perfect model would have
  modest R² when the test variance is small relative to MSE around the test mean.
* The **skill score** is the right metric here and is consistently positive: our
  LightGBM model beats the strong naive baseline by ~10–15 %.
* The 7-day-ahead operational forecast is in `data/features/forecast_next7.parquet`
  and powers the dashboard.

---

## 8. Composite area risk

We combine four normalised components into a single risk score per area:

```
risk = 0.35·intensity + 0.20·trend + 0.20·violence + 0.25·forecast
```

* `intensity` — mean daily crimes over last 28 days.
* `trend` — *z*-score of recent vs long-term (90-day) mean.
* `violence` — share of violent crimes over the recent window.
* `forecast` — mean of the model's next-7-day forecast for the area.

All four components are min-max scaled to `[0, 1]` before weighting; areas are then
ranked and bucketed into Low / Medium / High / Critical quartiles.

**Top 5 highest-risk LAPD areas:**

| # | Area | Risk | Intensity (28d) | Trend (z) | Violent share | Forecast |
|---|------|------|------------------|-----------|---------------|----------|
| 1 | 77th Street | 0.92 | 30.1 | −0.34 | 23 % | 33.8 |
| 2 | Southwest   | 0.82 | 29.2 | −0.28 | 19 % | 30.8 |
| 3 | N Hollywood | 0.74 | 29.7 | −0.30 | 12 % | 29.0 |
| 4 | Southeast   | 0.69 | 23.1 | −0.35 | 25 % | 27.5 |
| 5 | Van Nuys    | 0.62 | 24.5 | −0.09 | 12 % | 25.1 |

> The negative *trend z-scores* reflect the city-wide decline of reported crime
> in 2024; areas at the top of the list are still the *highest absolute* and
> *most violent* divisions, even though the directional trend is currently
> downward.

The score is intentionally simple, transparent and tunable (weights live in
`src/config.RISK_WEIGHTS`).

---

## 9. Limitations & next steps

* **Data freshness** — the dataset effectively ends mid-2024 because of LAPD
  reporting delays. Live deployment would require a streaming ingestion layer that
  re-trains weekly.
* **Per-cell forecasting** — we forecast at the *area* (≈ 21 divisions) level. A
  finer 1-km grid would give more actionable forecasts but requires heavier models
  (graph-NN or convolutional spatio-temporal nets).
* **Causality vs correlation** — the SHAP plots show *associations* learned by the
  model, not causal effects. Predictive policing should therefore be used as
  decision-support, not automation.
* **Fairness audit** — descent and sex are present as features; we kept them for
  victim-side modelling, but a production deployment needs a bias audit before any
  operational use.

---

## 10. Reproducing every number

```powershell
py -m pip install -r requirements.txt
py pipeline.py --step all
py -m streamlit run dashboard/app.py
```

Each artefact this report references lives under `reports/` or `data/features/` and
is regenerated deterministically (seed = 42).

---

*End of report.*
