# SPDX-License-Identifier: Apache-2.0

import time


class RealtimeStageTimer:
    __slots__ = ("_last",)

    def __init__(self):
        self._last = time.perf_counter()

    def mark_ms(self) -> float:
        now = time.perf_counter()
        elapsed_ms = (now - self._last) * 1000.0
        self._last = now
        return elapsed_ms
