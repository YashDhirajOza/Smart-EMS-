from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
from influxdb_client import InfluxDBClient
from sqlalchemy import text
from sqlalchemy.engine import create_engine

from config import INFLUX_SETTINGS, POSTGRES_SETTINGS
from forecast import forecast_metrics
from resolvers import RESOLVER_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("urjanet.diagnostics")


@dataclass(frozen=True)
class AssetThreshold:
    metric_name: str
    warn_min: Optional[float]
    warn_max: Optional[float]
    crit_min: Optional[float]
    crit_max: Optional[float]


class DiagnosticsEngine:
    def __init__(self, cycle_seconds: int = 30) -> None:
        self.cycle_seconds = cycle_seconds
        self.influx_client = InfluxDBClient(
            url=INFLUX_SETTINGS.url,
            token=INFLUX_SETTINGS.token,
            org=INFLUX_SETTINGS.org,
        )
        self.pg_engine = create_engine(POSTGRES_SETTINGS.sqlalchemy_url, future=True)

    def run_forever(self) -> None:
        logger.info("Diagnostics engine started with %s second cycle", self.cycle_seconds)
        while True:
            try:
                self.run_cycle()
            except Exception:  # noqa: BLE001
                logger.exception("Diagnostics cycle failed")
            time.sleep(self.cycle_seconds)

    def run_cycle(self) -> None:
        assets = self._fetch_assets()
        if not assets:
            logger.info("No assets found to evaluate")
            return

        for asset in assets:
            telemetry = self._fetch_recent_metrics(asset_id=asset["asset_id"])
            if telemetry is None:
                continue
            thresholds = self._fetch_thresholds(asset["asset_id"])
            health = self._calculate_health_score(telemetry, thresholds)
            self._store_health(asset["asset_id"], health)
            alerts = self._generate_alerts(asset["asset_id"], telemetry, thresholds)
            for alert in alerts:
                self._store_alert(alert)

        forecasts = self._forecast_assets(assets)
        if forecasts:
            logger.debug("Generated forecasts: %s", forecasts)

    def _fetch_assets(self) -> List[Dict[str, str]]:
        query = text("SELECT asset_id, asset_type FROM assets")
        with self.pg_engine.connect() as conn:
            result = conn.execute(query)
            return [dict(row._mapping) for row in result]

    def _fetch_thresholds(self, asset_id: str) -> Dict[str, AssetThreshold]:
        query = text(
            "SELECT metric_name, warn_min, warn_max, crit_min, crit_max FROM asset_thresholds WHERE asset_id = :asset_id"
        )
        with self.pg_engine.connect() as conn:
            result = conn.execute(query, {"asset_id": asset_id})
            return {
                row.metric_name: AssetThreshold(
                    metric_name=row.metric_name,
                    warn_min=row.warn_min,
                    warn_max=row.warn_max,
                    crit_min=row.crit_min,
                    crit_max=row.crit_max,
                )
                for row in result
            }

    def _fetch_recent_metrics(self, asset_id: str) -> Optional[pd.DataFrame]:
        query_api = self.influx_client.query_api()
        query = f"""
        from(bucket: "{INFLUX_SETTINGS.bucket}")
          |> range(start: -30m)
          |> filter(fn: (r) => r["_measurement"] == "asset_metrics")
          |> filter(fn: (r) => r["asset_id"] == "{asset_id}")
          |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        """
        df_list = query_api.query_data_frame(org=INFLUX_SETTINGS.org, query=query)
        if not df_list:
            return None
        df = df_list[0]
        if df.empty:
            return None
        df = df.set_index("_time")
        df.index = pd.to_datetime(df.index)
        return df

    @staticmethod
    def _calculate_health_score(data: pd.DataFrame, thresholds: Dict[str, AssetThreshold]) -> float:
        score = 100.0
        for column in data.columns:
            latest = data[column].iloc[-1]
            threshold = thresholds.get(column)
            if not threshold:
                continue
            penalty = 0.0
            if threshold.warn_min is not None and latest < threshold.warn_min:
                penalty += 10
            if threshold.warn_max is not None and latest > threshold.warn_max:
                penalty += 10
            if threshold.crit_min is not None and latest < threshold.crit_min:
                penalty += 25
            if threshold.crit_max is not None and latest > threshold.crit_max:
                penalty += 25
            score -= penalty
        return max(score, 0.0)

    def _store_health(self, asset_id: str, health_score: float) -> None:
        logger.debug("Updating health for %s to %.2f", asset_id, health_score)
        with self.pg_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO asset_health (asset_id, health_score, updated_at) "
                    "VALUES (:asset_id, :health, NOW()) "
                    "ON CONFLICT (asset_id) DO UPDATE SET health_score = EXCLUDED.health_score, updated_at = EXCLUDED.updated_at"
                ),
                {"asset_id": asset_id, "health": health_score},
            )

    def _generate_alerts(
        self,
        asset_id: str,
        data: pd.DataFrame,
        thresholds: Dict[str, AssetThreshold],
    ) -> Iterable[Dict[str, object]]:
        latest = data.iloc[-1]
        alerts = []
        for metric, threshold in thresholds.items():
            value = latest.get(metric)
            if value is None:
                continue
            severity = None
            if threshold.crit_max is not None and value > threshold.crit_max:
                severity = "critical"
            elif threshold.crit_min is not None and value < threshold.crit_min:
                severity = "critical"
            elif threshold.warn_max is not None and value > threshold.warn_max:
                severity = "warning"
            elif threshold.warn_min is not None and value < threshold.warn_min:
                severity = "warning"

            if severity:
                resolver = RESOLVER_REGISTRY.get(metric)
                resolver_action = resolver(float(value)) if resolver else {"action": "Inspect asset."}
                alerts.append(
                    {
                        "id": str(uuid.uuid4()),
                        "asset_id": asset_id,
                        "metric_name": metric,
                        "severity": severity,
                        "message": f"{metric} reading {value:.2f} breaching {severity} threshold.",
                        "resolver_action": json.dumps(resolver_action),
                    }
                )
        return alerts

    def _store_alert(self, alert: Dict[str, object]) -> None:
        with self.pg_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO alerts (id, asset_id, metric_name, severity, message, resolver_action, raised_at) "
                    "VALUES (:id, :asset_id, :metric_name, :severity, :message, :resolver_action::jsonb, NOW())"
                ),
                alert,
            )

    def _forecast_assets(self, assets: List[Dict[str, str]]) -> Dict[str, Dict[str, List[Dict[str, float | str]]]]:
        series_map: Dict[str, Dict[str, pd.Series]] = {}
        for asset in assets:
            data = self._fetch_recent_metrics(asset["asset_id"])
            if data is None:
                continue
            series_map[asset["asset_id"]] = {col: data[col] for col in data.columns}
        return forecast_metrics(series_map)


def main() -> None:
    engine = DiagnosticsEngine(cycle_seconds=60)
    engine.run_forever()


if __name__ == "__main__":
    main()
