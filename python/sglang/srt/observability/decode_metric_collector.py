"""Low-overhead scheduler-local metrics for disaggregated decode."""

from __future__ import annotations

import bisect
import time
from typing import Callable, Iterable, List, Optional, Tuple


# Keep this list aligned with the default buckets used by
# ``generation_tokens_histogram`` in ``observability/metrics_collector.py``.
# This collector is scheduler-local, so the list is intentionally duplicated
# here instead of changing the existing metrics implementation.
DEFAULT_OUTPUT_LEN_BUCKETS = tuple(
    [
        100,
        300,
        500,
        700,
        1000,
        1500,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
        12500,
        15000,
        17500,
        20000,
        22500,
        25000,
        27500,
        30000,
        35000,
        40000,
        60000,
        80000,
        100000,
        200000,
        300000,
        400000,
        600000,
        800000,
        1000000,
        1100000,
    ]
)


class DecodeMetricCollector:
    """Track output-length quantiles over a rolling logical 15-second window.

    Counts are cumulative inside the collector, while each quantile update uses
    the difference from the previous snapshot. This keeps memory constant and
    makes the hot path a bucket increment plus a cached flag read.
    """

    WINDOW_SECONDS = 15.0

    def __init__(
        self,
        bucket_bounds: Iterable[int] = DEFAULT_OUTPUT_LEN_BUCKETS,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        bounds = tuple(int(value) for value in bucket_bounds)
        if not bounds or any(left >= right for left, right in zip(bounds, bounds[1:])):
            raise ValueError("bucket_bounds must be a strictly increasing sequence")

        self.bucket_bounds = bounds
        self.window_seconds = self.WINDOW_SECONDS
        self._clock = clock or time.monotonic
        self._bucket_counts: List[int] = [0] * (len(bounds) + 1)
        self._previous_bucket_counts: List[int] = [0] * (len(bounds) + 1)
        self.last_stat_time = self._clock()
        self.p50_output_len: Optional[int] = None
        self.p95_output_len: Optional[int] = None

    def observe_output_len(self, output_len: int) -> None:
        """Record one request's completed output length for the current forward."""
        bucket_index = bisect.bisect_left(self.bucket_bounds, max(0, int(output_len)))
        self._bucket_counts[bucket_index] += 1

    def observe_batch_output_lengths(self, reqs: Iterable[object]) -> None:
        """Record the current output length of each request in a decode forward."""
        for req in reqs:
            if getattr(req, "is_retracted", False):
                continue
            finished_len = getattr(req, "finished_len", None)
            self.observe_output_len(
                len(req.output_ids) if finished_len is None else finished_len
            )

    def maybe_update(
        self, now: Optional[float] = None
    ) -> Optional[Tuple[Optional[int], Optional[int]]]:
        """Update cached quantiles once per window and return them when updated."""
        now = self._clock() if now is None else now
        if now - self.last_stat_time < self.window_seconds:
            return None

        delta = [
            current - previous
            for current, previous in zip(
                self._bucket_counts, self._previous_bucket_counts
            )
        ]
        self._previous_bucket_counts = self._bucket_counts.copy()
        self.last_stat_time = now

        total = sum(delta)
        if total <= 0:
            self.p50_output_len = None
            self.p95_output_len = None
            return self.p50_output_len, self.p95_output_len

        self.p50_output_len = self._quantile(delta, 0.50, total)
        self.p95_output_len = self._quantile(delta, 0.95, total)
        return self.p50_output_len, self.p95_output_len

    def _quantile(self, counts: List[int], quantile: float, total: int) -> int:
        target = max(1, int(total * quantile + 0.999999))
        cumulative = 0
        for index, count in enumerate(counts):
            cumulative += count
            if cumulative >= target:
                if index < len(self.bucket_bounds):
                    return self.bucket_bounds[index]
                return self.bucket_bounds[-1]
        return self.bucket_bounds[-1]
