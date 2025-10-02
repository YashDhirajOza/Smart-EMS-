from __future__ import annotations

import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Callable, Dict


def load_time_series(
    path: Path,
    timestamp_column: str = "timestamp",
    transformers: Dict[str, Callable[[pd.Series], pd.Series]] | None = None,
) -> DataFrame:
    """Load and prepare a time-series dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=[timestamp_column])
    df = df.set_index(timestamp_column).sort_index()

    if transformers:
        for column, func in transformers.items():
            if column in df.columns:
                df[column] = func(df[column])

    df = df.interpolate(method="time").ffill().bfill()
    return df
