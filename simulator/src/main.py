from __future__ import annotations

import asyncio
import itertools
import logging
import os
from datetime import datetime
from typing import Dict

import pandas as pd

from asset_profiles import ASSET_PROFILES, AssetProfile
from dataset_loader import load_time_series
from publisher import build_payload, create_mqtt_client, publish_message
from utils.scheduler import jittered_interval_runner
from utils.transformers import add_noise

logging.basicConfig(
    level=os.getenv("SIMULATOR_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("urjanet.simulator")


class AssetSimulator:
    def __init__(self, profile: AssetProfile):
        self.profile = profile
        self.dataset = load_time_series(profile.dataset_path)
        self.iterator = itertools.cycle(self.dataset.iterrows())
        self.client = None

    def start(self) -> None:
        if self.client is None:
            self.client = create_mqtt_client()

    def stop(self) -> None:
        if self.client:
            self.client.disconnect()
            self.client = None

    def build_metrics(self, row: pd.Series) -> Dict[str, float]:
        metrics = {}
        for metric_name, dataset_column in self.profile.metric_mappings.items():
            value = row.get(dataset_column)
            if value is None:
                continue
            metrics[metric_name] = float(value)
        if metrics:
            # Subtle randomization to avoid identical streams
            metrics = {k: v for k, v in add_noise(pd.Series(metrics)).items()}
        return metrics

    async def publish_once(self) -> None:
        if self.client is None:
            self.start()

        timestamp, row = next(self.iterator)
        metrics = self.build_metrics(row)
        payload = build_payload(
            timestamp=datetime.utcnow(),
            asset_id=self.profile.asset_id,
            asset_type=self.profile.asset_type,
            metrics=metrics,
        )
        publish_topic = f"{self.profile.base_topic}/{self.profile.asset_id}"
        publish_message(self.client, publish_topic, payload)
        logger.info("Published %s", payload)


async def run_simulation() -> None:
    simulators = [AssetSimulator(profile) for profile in ASSET_PROFILES]

    async def runner(sim: AssetSimulator) -> None:
        await jittered_interval_runner(sim.publish_once, sim.profile.publish_interval_seconds)

    await asyncio.gather(*(runner(sim) for sim in simulators))


def main() -> None:
    try:
        asyncio.run(run_simulation())
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")


if __name__ == "__main__":
    main()
