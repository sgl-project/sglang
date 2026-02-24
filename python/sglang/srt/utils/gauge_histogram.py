"""Gauge with gt/le bucket labels for Grafana heatmap visualization.

Unlike Prometheus Histogram which uses cumulative buckets, this uses
non-cumulative buckets (gt < value <= le) suitable for heatmap display.

Note: Keep in sync with Rust implementation in
sgl-model-gateway/src/observability/gauge_histogram.rs
"""

import bisect
from typing import Dict, Iterator, List, Tuple, Union


class BucketLabels:
    """Bucket label pairs and count computation for a GaugeHistogram."""

    def __init__(self, upper_bounds: List[Union[int, float]]):
        self._upper_bounds = upper_bounds
        self._labels: List[Tuple[str, str]] = []
        for i, upper in enumerate(upper_bounds):
            lower = upper_bounds[i - 1] if i > 0 else 0
            self._labels.append((str(lower), str(upper)))
        self._labels.append((str(upper_bounds[-1]), "+Inf"))

    def __len__(self) -> int:
        return len(self._labels)

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return iter(self._labels)

    def compute_bucket_counts(self, observations: List[Union[int, float]]) -> List[int]:
        """Compute how many observations fall into each bucket. O(n) complexity."""
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
        labelnames: List[str],
        bucket_bounds: List[Union[int, float]],
        multiprocess_mode: str = "mostrecent",
    ):
        from prometheus_client import Gauge

        self._buckets = BucketLabels(bucket_bounds)

        self._gauge = Gauge(
            name=name,
            documentation=documentation,
            labelnames=list(labelnames) + ["gt", "le"],
            multiprocess_mode=multiprocess_mode,
        )

    def set_raw(self, labels: Dict[str, str], values: List[int]):
        """Set bucket counts directly."""
        for (gt, le), count in zip(self._buckets, values):
            self._gauge.labels(**labels, gt=gt, le=le).set(count)

    def set_by_current_observations(
        self, labels: Dict[str, str], observations: List[Union[int, float]]
    ):
        """Compute bucket counts from observations and set them."""
        counts = self._buckets.compute_bucket_counts(observations)
        self.set_raw(labels, counts)

    def buckets(self) -> BucketLabels:
        return self._buckets
