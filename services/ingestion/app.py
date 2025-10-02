from __future__ import annotations

import json
import logging
import signal
import sys
from typing import Any

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision

from config import INFLUX_SETTINGS, MQTT_SETTINGS
from models import TelemetryPayload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("urjanet.ingestion")


class IngestionService:
    def __init__(self) -> None:
        self.mqtt_client = mqtt.Client(client_id="urjanet-ingestion")
        if MQTT_SETTINGS.username and MQTT_SETTINGS.password:
            self.mqtt_client.username_pw_set(MQTT_SETTINGS.username, MQTT_SETTINGS.password)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect

        self.influx_client = InfluxDBClient(
            url=INFLUX_SETTINGS.url,
            token=INFLUX_SETTINGS.token,
            org=INFLUX_SETTINGS.org,
        )
        self.writer = self.influx_client.write_api()

    def start(self) -> None:
        logger.info("Connecting to MQTT broker at %s:%s", MQTT_SETTINGS.host, MQTT_SETTINGS.port)
        self.mqtt_client.connect(MQTT_SETTINGS.host, MQTT_SETTINGS.port)
        self.mqtt_client.loop_start()

    def stop(self) -> None:
        logger.info("Shutting down ingestion service")
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        self.writer.close()
        self.influx_client.close()

    def on_connect(self, client: mqtt.Client, userdata: Any, flags: dict, rc: int) -> None:
        if rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error("Failed to connect to MQTT broker: %s", mqtt.error_string(rc))
            return
        logger.info("Connected to MQTT broker. Subscribing to %s", MQTT_SETTINGS.topic)
        client.subscribe(MQTT_SETTINGS.topic, qos=1)

    def on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        logger.warning("Disconnected from MQTT broker: %s", mqtt.error_string(rc))

    def on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        payload = msg.payload.decode("utf-8")
        try:
            data = json.loads(payload)
            telemetry = TelemetryPayload.parse_obj(data)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Invalid payload: %s - error: %s", payload, exc)
            return

        point = Point("asset_metrics") \
            .tag("asset_id", telemetry.asset_id) \
            .tag("asset_type", telemetry.asset_type)

        for metric, value in telemetry.metrics.items():
            point = point.field(metric, float(value))

        point = point.time(telemetry.timestamp, WritePrecision.S)

        try:
            self.writer.write(bucket=INFLUX_SETTINGS.bucket, record=point)
            logger.debug("Stored telemetry for %s", telemetry.asset_id)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to write point to InfluxDB")


def main() -> None:
    service = IngestionService()

    def handle_signal(signum: int, frame: Any) -> None:  # noqa: ANN401
        logger.info("Received signal %s", signum)
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    service.start()
    signal.pause()


if __name__ == "__main__":
    main()
