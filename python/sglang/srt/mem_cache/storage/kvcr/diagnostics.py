# SPDX-License-Identifier: Apache-2.0
"""Bounded interval collector for KVCR's optional native telemetry seam.

No per-operation samples or page keys are retained. KVCR calls this collector
from its public main-side API, serialized by KVCRStore's IO lock. These are
diagnostic measurements, not HiCache's logical-page or bandwidth metrics.
"""

from __future__ import annotations

import bisect
import gc
import json
import threading
import time
from collections import deque


class DiagnosticGCObserver:
    """Observe process-wide GC pauses without changing the collector's policy.

    The owning storage instance closes this lease. Callbacks only record bounded
    primitive observations; logging and native API calls happen outside GC.
    """

    def __init__(self, max_events=256, *, clock=None, wall_clock=None):
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self._limit = max_events
        self._clock = clock or time.perf_counter
        self._wall_clock = wall_clock or time.time
        self._events = deque()
        self._started = {}
        self._dropped = 0
        self._closed = False
        self._callback = self._observe
        gc.callbacks.append(self._callback)

    def _observe(self, phase, info):
        if self._closed:
            return
        generation = info.get("generation")
        if type(generation) is not int or generation not in (0, 1, 2):
            return
        if phase == "start":
            self._started[generation] = (
                self._clock(),
                self._wall_clock(),
                threading.get_ident(),
            )
        elif phase == "stop":
            started = self._started.pop(generation, None)
            if started is None:
                return
            elapsed = (self._clock() - started[0]) * 1000
            if len(self._events) >= self._limit:
                self._dropped += 1
                return
            self._events.append(
                {
                    "generation": generation,
                    "start_epoch": started[1],
                    "epoch": self._wall_clock(),
                    "duration_ms": elapsed,
                    "thread_id": started[2],
                    "collected": info.get("collected", 0),
                    "uncollectable": info.get("uncollectable", 0),
                }
            )

    def drain(self):
        # GC callbacks and this drain run under CPython's GIL. Only this owner
        # removes entries; callbacks can append between individual deque pops.
        events = []
        while self._events:
            events.append(self._events.popleft())
        dropped, self._dropped = self._dropped, 0
        return {"collections": events, "dropped_observations": dropped}

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            gc.callbacks.remove(self._callback)
        except ValueError:
            pass
        self._started.clear()


class DiagnosticStats:
    # Native durations are seconds. Keep a fixed histogram, including overflow,
    # rather than accumulating an unbounded list while the server is idle.
    BOUNDS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 30)

    def __init__(self, max_series=128):
        self._max_series = max_series
        self._series = {}
        self._dropped = 0

    def _entry(self, kind, name, labels):
        key = (kind, name, tuple(labels))
        if key not in self._series:
            if len(self._series) >= self._max_series:
                self._dropped += 1
                return None
            self._series[key] = {}
        return self._series[key]

    def increase_counter(self, name, value, labelvalues=()):
        entry = self._entry("counter", name, labelvalues)
        if entry is not None:
            entry["value"] = entry.get("value", 0) + value

    def set_gauge(self, name, value, labelvalues=()):
        entry = self._entry("gauge", name, labelvalues)
        if entry is not None:
            entry["value"] = value

    def observe_histogram(self, name, value, labelvalues=()):
        entry = self._entry("histogram", name, labelvalues)
        if entry is None:
            return
        entry["count"] = entry.get("count", 0) + 1
        entry["sum"] = entry.get("sum", 0) + value
        entry["min"] = min(entry.get("min", value), value)
        entry["max"] = max(entry.get("max", value), value)
        bucket = bisect.bisect_left(self.BOUNDS, value)
        field = f"bucket_{bucket}"
        entry[field] = entry.get(field, 0) + 1

    def reduce(self):
        """Drain one interval; JSON tuple keys preserve metric/label identity.

        Histogram buckets are non-cumulative and indexed into BOUNDS; index
        len(BOUNDS) is overflow. Consumers can report bounds, not exact p90s.
        """
        result = {
            json.dumps((kind, name, labels, field), separators=(",", ":")): value
            for (kind, name, labels), entry in self._series.items()
            for field, value in entry.items()
        }
        if self._dropped:
            result["dropped_observations"] = self._dropped
        self._series.clear()
        self._dropped = 0
        return result

    def is_empty(self):
        return not self._series and not self._dropped
