"""Busy time of a disaggregated prefill engine, as reported to the autoscaler."""

from __future__ import annotations

import time
from typing import Optional


class PrefillBusyClock:
    """Wall time from an active mark to the next idle mark, excluding pauses."""

    def __init__(self) -> None:
        self._total_ns = 0
        self._busy_since_ns: Optional[int] = None
        self._last_close_ns = 0
        self._paused = False

    def mark_busy(self, event_ns: int) -> None:
        """Open a busy span at event_ns unless one is open or the engine is paused."""
        if self._busy_since_ns is None and not self._paused:
            # A backdated receive start never reopens time already closed.
            self._busy_since_ns = max(event_ns, self._last_close_ns)

    def mark_idle(self) -> int:
        """Close the open busy span now and return that time."""
        now_ns = time.monotonic_ns()
        self._mark_idle(now_ns)
        return now_ns

    def _mark_idle(self, now_ns: int) -> None:
        """Close the open busy span at now_ns."""
        if self._busy_since_ns is not None:
            self._total_ns += max(0, now_ns - self._busy_since_ns)
            self._busy_since_ns = None
        # Follow the clock, also when a CRIU restore moved it back.
        self._last_close_ns = now_ns

    def pause(self) -> None:
        """Close the open span now and ignore busy marks until resume."""
        self._pause(time.monotonic_ns())

    def _pause(self, now_ns: int) -> None:
        """Close the open span and ignore busy marks until resume."""
        self._mark_idle(now_ns)
        self._paused = True

    def resume(self) -> None:
        """Accept busy marks again and count from now until the next idle mark."""
        self._resume(time.monotonic_ns())

    def _resume(self, now_ns: int) -> None:
        """Accept busy marks again and count from now_ns until the next idle mark."""
        self._paused = False
        self.mark_busy(now_ns)

    def restart(self, total_us: int) -> None:
        """Drop the counted time and continue from total_us now, keeping any pause."""
        self._restart(total_us, time.monotonic_ns())

    def _restart(self, total_us: int, now_ns: int) -> None:
        """Drop the counted time and continue from total_us, keeping any pause."""
        self._total_ns = total_us * 1000
        self._busy_since_ns = None
        self._last_close_ns = now_ns

    def read_us(self) -> int:
        """Busy microseconds so far, including the open span up to now."""
        return self._read_us(time.monotonic_ns())

    def _read_us(self, now_ns: int) -> int:
        """Busy microseconds so far, including the open span up to now_ns."""
        total_ns = self._total_ns
        if self._busy_since_ns is not None:
            total_ns += max(0, now_ns - self._busy_since_ns)
        return total_ns // 1000
