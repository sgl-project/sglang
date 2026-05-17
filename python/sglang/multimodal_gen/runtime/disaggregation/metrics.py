# SPDX-License-Identifier: Apache-2.0
"""Observability metrics for disaggregated diffusion pipelines."""

import threading
import time
from dataclasses import dataclass


@dataclass
class _RequestTiming:
    start_time: float
    stage_start: float = 0.0


@dataclass
class RoleStats:
    role: str
    requests_completed: int = 0
    requests_failed: int = 0
    requests_in_flight: int = 0
    requests_timed_out: int = 0
    queue_depth: int = 0
    last_latency_s: float = 0.0
    avg_latency_s: float = 0.0
    max_latency_s: float = 0.0
    throughput_rps: float = 0.0
    uptime_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "requests_completed": self.requests_completed,
            "requests_failed": self.requests_failed,
            "requests_in_flight": self.requests_in_flight,
            "requests_timed_out": self.requests_timed_out,
            "queue_depth": self.queue_depth,
            "last_latency_s": round(self.last_latency_s, 4),
            "avg_latency_s": round(self.avg_latency_s, 4),
            "max_latency_s": round(self.max_latency_s, 4),
            "throughput_rps": round(self.throughput_rps, 4),
            "uptime_s": round(self.uptime_s, 1),
        }


class DisaggMetrics:
    """Thread-safe metrics collector for a single disagg role."""

    def __init__(self, role: str):
        self._role = role
        self._lock = threading.Lock()
        self._start_time = time.monotonic()

        self._completed = 0
        self._failed = 0
        self._timed_out = 0

        self._in_flight: dict[str, _RequestTiming] = {}

        self._last_latency = 0.0
        self._max_latency = 0.0
        self._total_latency = 0.0

        self._completion_times: list[float] = []
        self._throughput_window_s = 60.0

        self._queue_depth = 0

    @property
    def role(self) -> str:
        return self._role

    def record_request_start(self, request_id: str) -> None:
        with self._lock:
            self._in_flight[request_id] = _RequestTiming(start_time=time.monotonic())

    def record_request_complete(self, request_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            timing = self._in_flight.pop(request_id, None)
            if timing is not None:
                latency = now - timing.start_time
                self._last_latency = latency
                self._max_latency = max(self._max_latency, latency)
                self._total_latency += latency

            self._completed += 1
            self._completion_times.append(now)
            self._prune_completion_times(now)

    def record_request_failed(self, request_id: str) -> None:
        with self._lock:
            self._in_flight.pop(request_id, None)
            self._failed += 1

    def record_request_timeout(self, request_id: str) -> None:
        with self._lock:
            self._in_flight.pop(request_id, None)
            self._timed_out += 1

    def update_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depth = depth

    def snapshot(self) -> RoleStats:
        now = time.monotonic()
        with self._lock:
            self._prune_completion_times(now)
            total = self._completed + self._failed
            avg_latency = self._total_latency / total if total > 0 else 0.0
            rps = (
                len(self._completion_times) / self._throughput_window_s
                if self._completion_times
                else 0.0
            )

            return RoleStats(
                role=self._role,
                requests_completed=self._completed,
                requests_failed=self._failed,
                requests_in_flight=len(self._in_flight),
                requests_timed_out=self._timed_out,
                queue_depth=self._queue_depth,
                last_latency_s=self._last_latency,
                avg_latency_s=avg_latency,
                max_latency_s=self._max_latency,
                throughput_rps=rps,
                uptime_s=now - self._start_time,
            )

    def _prune_completion_times(self, now: float) -> None:
        cutoff = now - self._throughput_window_s
        while self._completion_times and self._completion_times[0] < cutoff:
            self._completion_times.pop(0)
