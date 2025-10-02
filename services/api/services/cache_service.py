from __future__ import annotations

from cachetools import TTLCache

energy_flow_cache = TTLCache(maxsize=32, ttl=5)
forecast_cache = TTLCache(maxsize=32, ttl=30)
