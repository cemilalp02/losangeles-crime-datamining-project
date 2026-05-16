"""Incident-level classifiers:
    - Violent vs non-violent crime
    - Arrest outcome prediction
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support, roc_auc_score, average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from . import config
from .utils import get_logger, timer

log = get_logger(__name__)

NUMERIC = ["hour", "dow", "month", "victim_age", "report_lag_days",
           "weapon_used", "lat", "lon",
           "hour_sin", "hour_cos", "dow_sin", "dow_cos",
           "month_sin", "month_cos"]
CATEGORICAL = ["area_name", "victim_sex", "victim_descent", "part_of_day"]


# ---------------------------------------------------------------------------
@dataclass
class ClfResult:
    name: str
    accuracy: float
    f1_macro: float
    f1_pos: float
    roc_auc: float
    pr_auc: float
    report: str
    confusion: List[List[int]]
    feature_names: List[str] = field(default_factory=list)
    importances: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200,
                              sparse_output=True), CATEGORICAL),
    ])


def _build_pipeline(model_name: str = "logreg") -> Pipeline:
    pre = _build_preprocessor()
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=400, n_jobs=-1, class_weight="balanced",
                                 solver="liblinear")
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=180, max_depth=18, min_samples_leaf=8,
            n_jobs=-1, random_state=config.RANDOM_STATE,
            class_weight="balanced_subsample",
        )
    elif model_name == "lgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_data_in_leaf=80, n_jobs=-1, class_weight="balanced",
            random_state=config.RANDOM_STATE, verbosity=-1,
        )
    else:
        raise ValueError(f"unknown model {model_name!r}")
    return Pipeline([("pre", pre), ("clf", clf)])


def _evaluate(name: str, pipe: Pipeline, X_test, y_test) -> ClfResult:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary",
                                                  zero_division=0)
    res = ClfResult(
        name=name,
        accuracy=accuracy_score(y_test, pred),
        f1_macro=f1_score(y_test, pred, average="macro"),
        f1_pos=f1,
        roc_auc=roc_auc_score(y_test, proba),
        pr_auc=average_precision_score(y_test, proba),
        report=classification_report(y_test, pred, zero_division=0),
        confusion=confusion_matrix(y_test, pred).tolist(),
    )
    log.info(f"[{name}] acc={res.accuracy:.3f}  f1={res.f1_pos:.3f}  "
             f"AUC={res.roc_auc:.3f}  PR-AUC={res.pr_auc:.3f}")
    return res


# ---------------------------------------------------------------------------
def train_classifier(features: pd.DataFrame, target: str,
                     models=("logreg", "rf", "lgbm"),
                     test_size: float = 0.2,
                     sample: int | None = 250_000) -> Dict[str, ClfResult]:
    """Train a battery of classifiers; return per-model results."""
    df = features.dropna(subset=[target]).copy()
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=config.RANDOM_STATE)

    X = df[NUMERIC + CATEGORICAL]
    y = df[target].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE)

    out: Dict[str, ClfResult] = {}
    for name in models:
        with timer(f"train {name} -> {target}", log):
            pipe = _build_pipeline(name)
            pipe.fit(X_tr, y_tr)
        out[name] = _evaluate(name, pipe, X_te, y_te)
        out[name].pipeline = pipe  # type: ignore[attr-defined]
    return out


# ---------------------------------------------------------------------------
def feature_importances(pipe: Pipeline, top: int = 20) -> pd.DataFrame:
    """Extract feature importances or coefficients from a fitted pipeline."""
    pre: ColumnTransformer = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names: List[str] = []
    names.extend(NUMERIC)
    try:
        ohe: OneHotEncoder = pre.named_transformers_["cat"]
        names.extend(ohe.get_feature_names_out(CATEGORICAL).tolist())
    except Exception:
        pass

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).ravel()
    else:
        return pd.DataFrame()

    if len(imp) != len(names):
        names = [f"f{i}" for i in range(len(imp))]
    out = pd.DataFrame({"feature": names, "importance": imp})
    return out.sort_values("importance", ascending=False).head(top).reset_index(drop=True)
