from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class InfluxSettings:
    url: str = os.getenv("INFLUXDB_URL", "http://influxdb:8086")
    token: str = os.getenv("INFLUXDB_TOKEN", "local-dev-token")
    org: str = os.getenv("INFLUXDB_ORG", "urjanet")
    bucket: str = os.getenv("INFLUXDB_BUCKET", "urjanet_telemetry")


@dataclass(frozen=True)
class PostgresSettings:
    host: str = os.getenv("POSTGRES_HOST", "postgres")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "urjanet")
    password: str = os.getenv("POSTGRES_PASSWORD", "urjanet-secret")
    db: str = os.getenv("POSTGRES_DB", "urjanet")

    @property
    def sqlalchemy_url(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


INFLUX_SETTINGS = InfluxSettings()
POSTGRES_SETTINGS = PostgresSettings()
