"""Utility helpers (logging, plotting style, IO, timing)."""
from __future__ import annotations
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import seaborn as sns

from . import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "crime_forecast_la") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(h)
    logger.propagate = False
    return logger


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    log = logger or get_logger()
    t0 = time.perf_counter()
    log.info(f"[start] {label}")
    yield
    log.info(f"[done ] {label} in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
def apply_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.frameon": False,
    })


def save_fig(fig: plt.Figure, name: str) -> Path:
    out = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(out)
    return out


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def memory_mb(obj) -> float:
    """Estimate memory of a pandas DataFrame in MB."""
    try:
        return obj.memory_usage(deep=True).sum() / 1024 ** 2
    except Exception:
        return float("nan")
