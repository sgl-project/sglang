"""Gauge with gt/le bucket labels for Grafana heatmap visualization.

Unlike Prometheus Histogram which uses cumulative buckets, this uses
non-cumulative buckets (gt < value <= le) suitable for heatmap display.

Note: Keep in sync with Rust implementation in
sgl-model-gateway/src/observability/gauge_histogram.rs
"""

import bisect
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union


class BucketLabels:
    """Bucket label pairs and count computation for a GaugeHistogram."""

    def __init__(self, upper_bounds: Sequence[Union[int, float]]):
        bounds = list(upper_bounds)
        self._upper_bounds: List[Union[int, float]] = bounds
        self._labels: List[Tuple[str, str]] = []
        for i, upper in enumerate(bounds):
            lower = bounds[i - 1] if i > 0 else 0
            if upper <= lower:
                raise ValueError("bucket bounds must be strictly ascending")
            self._labels.append((str(lower), str(upper)))
        last_lower = bounds[-1] if bounds else 0
        self._labels.append((str(last_lower), "+Inf"))

    def __len__(self) -> int:
        return len(self._labels)

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return iter(self._labels)

    def compute_bucket_counts(
        self, observations: Sequence[Union[int, float]]
    ) -> List[int]:
        """Compute how many observations fall into each bucket. O(n log b) complexity."""
        counts = [0] * len(self)
        for v in observations:
            # bisect_left finds insertion point; values at boundary go to current bucket
            idx = bisect.bisect_left(self._upper_bounds, v)
            counts[idx] += 1
        return counts


class GaugeHistogram:
    """Gauge with gt/le bucket labels for Grafana heatmap visualization."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str],
        bucket_bounds: Sequence[Union[int, float]],
        multiprocess_mode: str = "mostrecent",
    ):
        from prometheus_client import Gauge

        self._buckets = BucketLabels(bucket_bounds)
        self._labelnames = tuple(labelnames)
        self._handle_cache: Dict[Tuple[str, ...], Tuple[Any, ...]] = {}

        self._gauge = Gauge(
            name=name,
            documentation=documentation,
            labelnames=list(self._labelnames) + ["gt", "le"],
            multiprocess_mode=multiprocess_mode,
        )

    def _get_bucket_gauges(self, labels: Dict[str, str]) -> Tuple[Any, ...]:
        if len(labels) == len(self._labelnames):
            try:
                cache_key = tuple(str(labels[name]) for name in self._labelnames)
            except KeyError:
                cache_key = None
        else:
            cache_key = None

        if cache_key is not None:
            handles = self._handle_cache.get(cache_key)
            if handles is not None:
                return handles

        handles = tuple(
            self._gauge.labels(**labels, gt=gt, le=le) for gt, le in self._buckets
        )
        if cache_key is not None:
            self._handle_cache[cache_key] = handles
        return handles

    def set_raw(self, labels: Dict[str, str], values: Sequence[Union[int, float]]):
        """Set bucket counts directly."""
        if len(values) != len(self._buckets):
            raise ValueError(
                f"counts length ({len(values)}) must match bucket count ({len(self._buckets)})"
            )
        for gauge, count in zip(self._get_bucket_gauges(labels), values):
            gauge.set(count)

    def set_by_current_observations(
        self, labels: Dict[str, str], observations: Sequence[Union[int, float]]
    ):
        """Compute bucket counts from observations and set them."""
        counts = self._buckets.compute_bucket_counts(observations)
        self.set_raw(labels, counts)

    def buckets(self) -> BucketLabels:
        return self._buckets
