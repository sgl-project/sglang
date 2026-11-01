"""Request-local wall-clock spans for frontend timing.

Intervals are merged within each stage so concurrent calls do not count the
same wall-clock instant twice. Different stages may overlap and must not be
added to infer the overall elapsed time.
"""

import contextvars
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock

current_frontend_timing = contextvars.ContextVar(
    "frontend_request_timing", default=None
)


@contextmanager
def bind_frontend_timing(timing):
    token = current_frontend_timing.set(timing)
    try:
        yield
    finally:
        current_frontend_timing.reset(token)


@contextmanager
def measure_frontend_stage(timing, stage):
    if timing is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timing.add_span(stage, start, time.perf_counter())


@dataclass
class FrontendRequestTiming:
    started_at: float
    _spans: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)
    _finished_at: float | None = field(default=None, init=False, repr=False)

    def finish(self, finished_at):
        """Claim the logical request's first submission exactly once."""
        with self._lock:
            if self._finished_at is not None:
                return None
            if finished_at < self.started_at:
                raise ValueError("request end precedes request start")
            self._finished_at = finished_at
        return self.snapshot(finished_at)

    def fork(self):
        """Copy work shared before a batch was split into logical requests."""
        with self._lock:
            return FrontendRequestTiming(
                self.started_at,
                {stage: list(items) for stage, items in self._spans.items()},
            )

    def add_span(self, stage: str, started_at: float, finished_at: float) -> None:
        if finished_at < started_at:
            raise ValueError("span end precedes span start")
        with self._lock:
            self._spans.setdefault(stage, []).append((started_at, finished_at))

    def snapshot(self, finished_at: float) -> dict[str, float]:
        """Return elapsed time and the union of each stage's clipped spans."""
        if finished_at < self.started_at:
            raise ValueError("request end precedes request start")
        with self._lock:
            spans = {stage: list(items) for stage, items in self._spans.items()}
        durations = {"total": finished_at - self.started_at}
        for stage, items in spans.items():
            clipped = sorted(
                (max(start, self.started_at), min(end, finished_at))
                for start, end in items
                if end > self.started_at and start < finished_at
            )
            total = 0.0
            previous_end = self.started_at
            for start, end in clipped:
                total += max(0.0, end - max(start, previous_end))
                previous_end = max(previous_end, end)
            durations[stage] = total
        return durations


class FrontendTimingGroup:
    """Attribute shared batch execution to every participating request."""

    def __init__(self, timings):
        self.timings = [timing for timing in timings if timing is not None]

    def add_span(self, stage, start, end):
        for timing in self.timings:
            timing.add_span(stage, start, end)
