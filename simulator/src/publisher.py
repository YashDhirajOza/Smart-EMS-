from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict

import paho.mqtt.client as mqtt


def build_payload(
    *,
    timestamp: datetime,
    asset_id: str,
    asset_type: str,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "asset_id": asset_id,
        "asset_type": asset_type,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }


def publish_message(client: mqtt.Client, topic: str, payload: Dict[str, Any]) -> None:
    serialized = json.dumps(payload)
    result = client.publish(topic, serialized, qos=1)
    result.wait_for_publish()
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(f"Failed to publish to {topic}: {mqtt.error_string(result.rc)}")


def create_mqtt_client() -> mqtt.Client:
    client_id = os.getenv("SIMULATOR_CLIENT_ID", "urjanet-simulator")
    broker_host = os.getenv("MQTT_BROKER_HOST", "localhost")
    broker_port = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    username = os.getenv("MQTT_USERNAME")
    password = os.getenv("MQTT_PASSWORD")

    client = mqtt.Client(client_id=client_id)
    if username and password:
        client.username_pw_set(username, password)

    client.connect(broker_host, broker_port)
    return client
