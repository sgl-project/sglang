# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Records the latency of some functions
"""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Optional

from sglang.srt.metrics.utils import exponential_buckets


def enable_func_timer():
    # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
    from prometheus_client import Histogram

    global FUNC_LATENCY

    FUNC_LATENCY = Histogram(
        "sglang:func_latency_seconds",
        "Function latency in seconds",
        # captures latency in range [50ms - ~50s]
        buckets=exponential_buckets(start=0.05, width=1.5, length=18),
        labelnames=["name"],
    )


FUNC_LATENCY = None


def time_func_latency(
    func: Callable = None, name: Optional[str] = None
) -> Callable[..., Any]:
    """
    A decorator to observe the latency of a function's execution. Supports both sync and async functions.

    NOTE: We use our own implementation of a timer decorator since prometheus_client does not support async
    context manager yet.

    Overhead: The overhead introduced here in case of an async function could likely be because of `await` introduced
    which will return in another coroutine object creation and under heavy load could see longer wall time
    (scheduling delays due to introduction of another awaitable).
    """

    def measure(func: Callable[..., Any]) -> Callable[..., Any]:
        nonlocal name

        name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
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

    if func:
        return measure(func)
    else:
        return measure
