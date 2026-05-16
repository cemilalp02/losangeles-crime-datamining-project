# Crime Forecast LA

> **Spatio-Temporal Crime Forecasting, Hotspot Discovery and Explainable Risk Modeling**
> Data Mining Project · LAPD *Crime Data from 2020 to Present* (~955 K records)

A complete data-mining pipeline that turns raw LAPD crime reports into:

* a cleaned, feature-engineered analytical store,
* incident-level **classifiers** (violent vs non-violent, arrest outcome),
* density-based and grid-based **hotspot discovery**,
* a global LightGBM **spatio-temporal forecaster** (daily *and* weekly horizons),
* a composite **area risk score** combining intensity, trend, violence, and forecast,
* **SHAP-based explanations** of the model's decisions,
* and an interactive **Streamlit dashboard** to explore the results.

Dataset: [LA Crime Data 2020–Present (Kaggle)](https://www.kaggle.com/datasets/haseefalam/crime-dataset).

---

## Repository layout

```
dataminingproject-01/
├── Crime_Data_from_2020_to_Present.csv       # raw input (243 MB, gitignored)
├── pipeline.py                               # end-to-end runner (CLI)
├── requirements.txt
├── README.md
│
├── src/                                      # reusable Python package
│   ├── config.py                             # paths, schema, domain knowledge
│   ├── data_loader.py                        # CSV → parquet ingestion
│   ├── preprocessing.py                      # cleaning + target derivation
│   ├── feature_engineering.py                # incident & spatio-temporal features
│   ├── eda.py                                # plotting + summaries
│   ├── hotspot.py                            # DBSCAN, grid, K-Means, Folium maps
│   ├── classification.py                     # LR / RF / LightGBM classifiers
│   ├── forecasting.py                        # daily + weekly LightGBM forecaster
│   ├── risk_scoring.py                       # composite area risk
│   ├── explainability.py                     # SHAP + permutation importance
│   └── utils.py                              # logger, plotting style, timer
│
├── notebooks/                                # 8 walk-through notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classification_violent.ipynb
│   ├── 04_arrest_analysis.ipynb
│   ├── 05_hotspot_discovery.ipynb
│   ├── 06_forecasting.ipynb
│   ├── 07_risk_scoring.ipynb
│   └── 08_explainability.ipynb
│
├── dashboard/
│   └── app.py                                # Streamlit dashboard
│
├── data/
│   ├── processed/                            # crimes_clean.parquet
│   └── features/                             # daily/weekly panels, forecasts, risk
│
└── reports/
    ├── figures/                              # PNG figures + Folium HTML maps
    ├── models/                               # pickled best models
    ├── eda_overview.json
    ├── classification_metrics.json
    ├── forecast_metrics.json
    └── final_report.md
```

---

## Quick start

```powershell
# 1. (optional) create a virtualenv
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install dependencies
py -m pip install -r requirements.txt

# 3. place the dataset
#    Crime_Data_from_2020_to_Present.csv  must live in the project root.

# 4. run the full pipeline (≈ 90 sec end to end on the full dataset)
py pipeline.py --step all

# 5. launch the interactive dashboard
py -m streamlit run dashboard/app.py
```

Re-run any step in isolation:

```powershell
py pipeline.py --step preprocess     # clean → data/processed/crimes_clean.parquet
py pipeline.py --step features       # build incident / daily / weekly panels
py pipeline.py --step eda            # generate the static figures + overview
py pipeline.py --step hotspot        # DBSCAN + Folium heatmaps
py pipeline.py --step models         # train classifiers + SHAP
py pipeline.py --step forecast       # daily & weekly forecasters
py pipeline.py --step risk           # composite area risk score
```

The pipeline logs each stage and writes JSON summaries to `reports/`.

---

## Notebooks

The eight notebooks under `notebooks/` are pre-executed and tell the full story. Each
notebook reads from the parquet artifacts produced by `pipeline.py`, so they run in
seconds and remain reproducible.

| # | Notebook | What it shows |
|---|-----------|---------------|
| 01 | `01_eda.ipynb` | Dataset overview, missing-value audit, temporal/spatial/categorical patterns |
| 02 | `02_preprocessing.ipynb` | Cleaning logic, target derivation, feature matrices, incomplete-tail trim |
| 03 | `03_classification_violent.ipynb` | Logistic / LightGBM violent-crime classifier (AUC ≈ 0.92) |
| 04 | `04_arrest_analysis.ipynb` | Descriptive arrest rates + predictive model under heavy class imbalance |
| 05 | `05_hotspot_discovery.ipynb` | Grid heatmaps, DBSCAN clusters, K-Means area archetypes, Folium HTML map |
| 06 | `06_forecasting.ipynb` | Daily + weekly LightGBM forecasters with naive-baseline skill scores |
| 07 | `07_risk_scoring.ipynb` | Composite risk score and per-component decomposition |
| 08 | `08_explainability.ipynb` | SHAP summaries + permutation importance |

---

## Headline results

(Computed on the full 955 339-row dataset, 2020-01-01 → 2024-06-24.)

### Violence classifier (`is_violent`)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| Logistic Regression | 0.823 | 0.650 | 0.915 | 0.593 |
| **LightGBM (chosen)** | **0.826** | **0.654** | **0.923** | **0.620** |

### Arrest classifier (`is_arrest`)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| Logistic Regression | 0.676 | 0.266 | 0.712 | 0.184 |
| **LightGBM (chosen)** | **0.683** | **0.279** | **0.741** | **0.205** |

### Spatio-temporal forecaster

| Granularity | MAE | RMSE | MAE (naive baseline) | **Skill** |
|-------------|-----|------|----------------------|-----------|
| Daily, per area | 7.79 | 9.48 | 8.64 | **+9.8 %** |
| Weekly, per area | 40.45 | 53.82 | 47.73 | **+15.2 %** |

### Top-5 highest-risk LAPD areas (composite score)

| # | Area | Risk score | Tier |
|---|------|------------|------|
| 1 | 77th Street | 0.92 | Critical |
| 2 | Southwest | 0.82 | Critical |
| 3 | N Hollywood | 0.74 | Critical |
| 4 | Southeast | 0.69 | Critical |
| 5 | Van Nuys | 0.62 | Critical |

See `reports/final_report.md` for the full discussion.

---

## Methodology highlights

* **Schema-aware loader** with explicit dtypes and a parquet cache → 5× faster reloads.
* **Incomplete-reporting tail detection** — LAPD reports lag by ~6 weeks; we trim it
  automatically (see `feature_engineering.trim_incomplete_tail`).
* **Cyclical time encodings** (sin/cos for hour, day-of-week, month) for incident
  classifiers.
* **Forward-chained holdout** for forecasting (no leakage).
* **Naive-baseline skill score** so improvements are honest under distribution shift.
* **Categorical handling for LightGBM** via native `categorical_feature=`.
* **Composite risk score** = 0.35·intensity + 0.20·trend + 0.20·violence + 0.25·forecast.
* **SHAP TreeExplainer** + permutation importance for explainability.

---

## Reproducibility

All randomness flows through a single `RANDOM_STATE = 42` constant in `src/config.py`.
Pipeline outputs are deterministic given the input CSV.

Tested on Windows 10 + Python 3.10. Should work cross-platform.
