"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Metrics Types"""

import asyncio
import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Stats:
    num_running_reqs: int = 0
    num_used_tokens: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    num_queue_reqs: int = 0
    cache_hit_rate: float = 0.0


enable_metrics = False


def set_enable_metrics(value: bool):
    # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
    from prometheus_client import Histogram

    global enable_metrics, FUNC_LATENCY
    enable_metrics = value

    FUNC_LATENCY = Histogram(
        "sglang:func_latency_seconds",
        "Function latency in seconds",
        # captures latency in range [50ms - ~50s]
        buckets=exponential_buckets(start=0.05, width=1.5, length=18),
        labelnames=["name"],
    )


FUNC_LATENCY = None


def exponential_buckets(start: float, width: float, length: int) -> List[float]:
    buckets = []
    for i in range(length):
        buckets.append(start * (width**i))
    return buckets


def time_func_latency(name: Optional[str] = None) -> Callable[..., Any]:
    """
    A decorator to observe the latency of a function's execution. Supports both sync and async functions.

    NOTE: We use our own implementation of a timer decorator since prometheus_client does not support async
    context manager yet.

    Overhead: The overhead introduced here in case of an async function could likely be because of `await` introduced
    which will return in another coroutine object creation and under heavy load could see longer wall time
    (scheduling delays due to introduction of another awaitable).

    @param name: The name of this function

    @return: A function that wraps the given function and measures its execution time in prometheus metric.
    """

    def measure(func: Callable[..., Any]) -> Callable[..., Any]:
        nonlocal name

        name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not enable_metrics:
                return await func(*args, **kwargs)

            metric = FUNC_LATENCY
            start = time.monotonic()
            ret = func(*args, **kwargs)
            if isinstance(ret, asyncio.Future) or asyncio.iscoroutine(ret):
                try:
                    ret = await ret
                finally:
                    metric.labels(name=name).observe(time.monotonic() - start)
            return ret

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not enable_metrics:
                return func(*args, **kwargs)

            metric = FUNC_LATENCY
            start = time.monotonic()
            try:
                ret = func(*args, **kwargs)
            finally:
                metric.labels(name=name).observe(time.monotonic() - start)
            return ret

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return measure
