# SPDX-License-Identifier: Apache-2.0

import time


class RealtimeStageTimer:
    __slots__ = ("_last", "_start")

    def __init__(self):
        now = time.perf_counter()
        self._start = now
        self._last = now

    def mark_ms(self) -> float:
        now = time.perf_counter()
        elapsed_ms = (now - self._last) * 1000.0
        self._last = now
        return elapsed_ms

    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
