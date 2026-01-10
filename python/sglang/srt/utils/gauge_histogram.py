"""Gauge with gt/le bucket labels for Grafana heatmap visualization.

Unlike Prometheus Histogram which uses cumulative buckets, this uses
non-cumulative buckets (gt < value <= le) suitable for heatmap display.

Note: Keep in sync with Rust implementation in
sgl-model-gateway/src/observability/gauge_histogram.rs
"""

from typing import Dict, List, Union


class GaugeHistogram:

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: List[str],
        bucket_bounds: List[Union[int, float]],
        multiprocess_mode: str = "mostrecent",
    ):
        from prometheus_client import Gauge

        self._bucket_bounds = bucket_bounds
        self._bucket_labels: List[tuple] = []
        for i, upper in enumerate(bucket_bounds):
            lower = bucket_bounds[i - 1] if i > 0 else 0
            self._bucket_labels.append((str(lower), str(upper)))
        self._bucket_labels.append((str(bucket_bounds[-1]), "+Inf"))

        self._gauge = Gauge(
            name=name,
            documentation=documentation,
            labelnames=list(labelnames) + ["gt", "le"],
            multiprocess_mode=multiprocess_mode,
        )

    def set_raw(self, labels: Dict[str, str], values: List[int]):
        """Set bucket counts directly."""
        for (gt, le), count in zip(self._bucket_labels, values):
            self._gauge.labels(**labels, gt=gt, le=le).set(count)

    def set_by_current_observations(
        self, labels: Dict[str, str], observations: List[Union[int, float]]
    ):
        """Compute bucket counts from observations and set them."""
        bucket_counts = self._compute_bucket_counts(observations)
        self.set_raw(labels, bucket_counts)

    def _compute_bucket_counts(
        self, observations: List[Union[int, float]]
    ) -> List[int]:
        """Compute how many observations fall into each bucket."""
        bounds = self._bucket_bounds
        counts = []
        for i, upper in enumerate(bounds):
            lower = bounds[i - 1] if i > 0 else 0
            count = sum(1 for v in observations if lower < v <= upper)
            counts.append(count)
        counts.append(sum(1 for v in observations if v > bounds[-1]))
        return counts

    @property
    def num_buckets(self) -> int:
        return len(self._bucket_labels)
