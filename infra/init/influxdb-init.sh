#!/bin/bash
set -euo pipefail

influx bucket create \
  --name "$DOCKER_INFLUXDB_INIT_BUCKET" \
  --org "$DOCKER_INFLUXDB_INIT_ORG" \
  --retention 720h || true

influx bucket create \
  --name "${DOCKER_INFLUXDB_INIT_BUCKET}_15m" \
  --org "$DOCKER_INFLUXDB_INIT_ORG" \
  --retention 43800h || true

influx task create \
  --org "$DOCKER_INFLUXDB_INIT_ORG" \
  --flux 'option task = {name: "downsample_asset_metrics", every: 15m}
from(bucket: "'"$DOCKER_INFLUXDB_INIT_BUCKET"'")
  |> range(start: -task.every)
  |> filter(fn: (r) => r["_measurement"] == "asset_metrics")
  |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
  |> to(bucket: "'"$DOCKER_INFLUXDB_INIT_BUCKET"'_15m", org: "'"$DOCKER_INFLUXDB_INIT_ORG"'")' || true
