#!/usr/bin/env bash
set -euo pipefail

python simulator/src/main.py &
SIM_PID=$!

sleep 5

echo "Stopping simulator"
kill ${SIM_PID}
