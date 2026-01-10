"""Gauge with gt/le bucket labels for Grafana heatmap visualization.

Unlike Prometheus Histogram which uses cumulative buckets, this uses
non-cumulative buckets (gt < value <= le) suitable for heatmap display.

Note: Keep in sync with Rust implementation in
sgl-model-gateway/src/observability/gauge_histogram.rs
"""

from typing import Dict, List


class GaugeHistogram:

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: List[str],
        bucket_bounds: List[int],
        multiprocess_mode: str = "mostrecent",
    ):
        from prometheus_client import Gauge

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

    def set(self, labels: Dict[str, str], values: List[int]):
        for (gt, le), count in zip(self._bucket_labels, values):
            self._gauge.labels(**labels, gt=gt, le=le).set(count)

    @property
    def num_buckets(self) -> int:
        return len(self._bucket_labels)
