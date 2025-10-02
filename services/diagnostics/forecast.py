from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def project_time_to_threshold(
    series: pd.Series,
    target_value: float,
    horizon_minutes: int = 60,
) -> Tuple[datetime | None, float | None]:
    if len(series) < 3:
        return None, None

    timestamps = (series.index - series.index[0]).total_seconds().values.reshape(-1, 1)
    values = series.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(timestamps, values)

    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    if slope == 0:
        return None, intercept

    seconds_to_target = (target_value - intercept) / slope
    if seconds_to_target < 0:
        return None, intercept

    forecast_time = series.index[0] + timedelta(seconds=float(seconds_to_target))
    if forecast_time > series.index[-1] + timedelta(minutes=horizon_minutes):
        return None, intercept

    return forecast_time.to_pydatetime(), intercept


def forecast_metrics(metrics: Dict[str, pd.Series]) -> Dict[str, Dict[str, List[Dict[str, float | str]]]]:
    forecasts: Dict[str, Dict[str, List[Dict[str, float | str]]]] = {}
    for asset_id, series_map in metrics.items():
        forecasts[asset_id] = {}
        for metric, series in series_map.items():
            if metric == "state_of_charge":
                time_to_empty, intercept = project_time_to_threshold(series, target_value=0.1)
                forecasts[asset_id][metric] = [
                    {
                        "forecast_type": "time_to_empty",
                        "value": time_to_empty.isoformat() if time_to_empty else None,
                        "baseline": float(intercept) if intercept is not None else None,
                    }
                ]
    return forecasts
