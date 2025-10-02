from __future__ import annotations

import numpy as np
import pandas as pd


def clamp(series: pd.Series, lower: float | None = None, upper: float | None = None) -> pd.Series:
    result = series.copy()
    if lower is not None:
        result = result.clip(lower=lower)
    if upper is not None:
        result = result.clip(upper=upper)
    return result


def add_noise(series: pd.Series, amplitude: float = 0.01) -> pd.Series:
    noise = pd.Series(np.random.uniform(-amplitude, amplitude, size=len(series)), index=series.index)
    return series + noise
