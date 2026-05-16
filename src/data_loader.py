"""CSV ingestion with memory-efficient dtypes and parquet caching."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config
from .utils import get_logger, timer

log = get_logger(__name__)


def load_raw(
    csv_path: Optional[Path] = None,
    nrows: Optional[int] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load the raw LAPD CSV.

    Parameters
    ----------
    csv_path : optional override (default config.RAW_CSV).
    nrows    : sample size for quick experiments.
    use_cache: if True and a parquet cache exists, load it instead.
    """
    csv_path = csv_path or config.RAW_CSV
    cache_path = config.PROCESSED_DIR / "crimes_raw.parquet"

    if use_cache and cache_path.exists() and nrows is None:
        with timer(f"load parquet cache {cache_path.name}", log):
            df = pd.read_parquet(cache_path)
        return df

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found at {csv_path}")

    with timer(f"read_csv {csv_path.name} (nrows={nrows})", log):
        df = pd.read_csv(
            csv_path,
            dtype=config.RAW_DTYPES,
            parse_dates=config.DATE_COLUMNS,
            date_format="%m/%d/%Y %I:%M:%S %p",
            nrows=nrows,
            low_memory=False,
        )

    log.info(f"loaded shape={df.shape}")
    if nrows is None and use_cache:
        with timer(f"write parquet cache {cache_path.name}", log):
            df.to_parquet(cache_path, index=False)
    return df


def load_clean(parquet_path: Optional[Path] = None) -> pd.DataFrame:
    p = parquet_path or config.CLEAN_PARQUET
    if not p.exists():
        raise FileNotFoundError(
            f"Clean parquet not found at {p}. "
            "Run preprocessing first (notebooks/02_preprocessing.ipynb or pipeline.py)."
        )
    with timer(f"load clean {p.name}", log):
        df = pd.read_parquet(p)
    return df
