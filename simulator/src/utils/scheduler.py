from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Iterable


async def jittered_interval_runner(
    coro_factory: Callable[[], Awaitable[None]],
    interval_range: Iterable[int],
) -> None:
    intervals = list(interval_range)
    if not intervals:
        raise ValueError("interval_range must not be empty")

    while True:
        await coro_factory()
        await asyncio.sleep(random.choice(intervals))
