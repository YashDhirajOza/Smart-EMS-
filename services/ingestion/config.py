from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MQTTSettings:
    host: str = os.getenv("MQTT_BROKER_HOST", "mosquitto")
    port: int = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    topic: str = os.getenv("MQTT_TOPIC", "urjanet/telemetry/#")
    username: str | None = os.getenv("MQTT_USERNAME")
    password: str | None = os.getenv("MQTT_PASSWORD")


@dataclass(frozen=True)
class InfluxSettings:
    url: str = os.getenv("INFLUXDB_URL", "http://influxdb:8086")
    token: str = os.getenv("INFLUXDB_TOKEN", "local-dev-token")
    org: str = os.getenv("INFLUXDB_ORG", "urjanet")
    bucket: str = os.getenv("INFLUXDB_BUCKET", "urjanet_telemetry")


MQTT_SETTINGS = MQTTSettings()
INFLUX_SETTINGS = InfluxSettings()
