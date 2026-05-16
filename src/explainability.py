"""Model explainability via SHAP and permutation importance."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config
from .utils import apply_plot_style, save_fig, get_logger

log = get_logger(__name__)


def _get_transformed(pipe, X: pd.DataFrame, sample: int = 5_000):
    """Run the preprocessor and return (X_transformed_dense, feature_names)."""
    if len(X) > sample:
        X = X.sample(sample, random_state=config.RANDOM_STATE)
    pre = pipe.named_steps["pre"]
    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]
    return Xt, list(names), X


def shap_summary(pipe, X: pd.DataFrame, max_display: int = 18,
                 fname: str = "shap_summary",
                 sample: int = 4000) -> Optional[Path]:
    """Generate a SHAP summary plot for a fitted classification pipeline.

    Falls back gracefully if shap is not available or model unsupported.
    """
    try:
        import shap
    except ImportError:
        log.warning("shap not installed; skipping SHAP summary")
        return None

    apply_plot_style()
    Xt, names, _ = _get_transformed(pipe, X, sample=sample)
    clf = pipe.named_steps["clf"]
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
    except Exception as e:
        log.warning(f"TreeExplainer failed ({e}); using LinearExplainer fallback")
        try:
            explainer = shap.LinearExplainer(clf, Xt)
            sv = explainer.shap_values(Xt)
        except Exception as e2:
            log.warning(f"SHAP unavailable for this model: {e2}")
            return None

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, features=Xt, feature_names=names,
                      max_display=max_display, show=False)
    fig = plt.gcf()
    out = save_fig(fig, fname)
    plt.close(fig)
    log.info(f"SHAP summary -> {out}")
    return out


def permutation_top(pipe, X: pd.DataFrame, y: pd.Series,
                    n_repeats: int = 5, top: int = 15) -> pd.DataFrame:
    """Permutation feature importance on the original (untransformed) features."""
    from sklearn.inspection import permutation_importance
    if len(X) > 30_000:
        idx = X.sample(30_000, random_state=config.RANDOM_STATE).index
        X = X.loc[idx]; y = y.loc[idx]
    r = permutation_importance(pipe, X, y, n_repeats=n_repeats,
                               random_state=config.RANDOM_STATE, n_jobs=-1,
                               scoring="roc_auc")
    out = (pd.DataFrame({"feature": X.columns,
                         "importance_mean": r.importances_mean,
                         "importance_std": r.importances_std})
             .sort_values("importance_mean", ascending=False)
             .head(top).reset_index(drop=True))
    return out
