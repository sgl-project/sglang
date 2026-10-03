# SPDX-License-Identifier: Apache-2.0
"""Prometheus exporter for a weight cache daemon.

The daemon is a standalone process -- in ``client`` mode it outlives the
engine -- so it cannot share the engine's multiprocess ``/metrics``. Instead
each daemon serves its own scrape endpoint on ``--metrics-port``.

Metrics are derived from the same snapshot the ``status`` request returns
(``WeightCacheDaemon._status_snapshot``), computed lazily at scrape time by a
custom collector. There is one source of truth: whatever
``python -m sglang.srt.weight_cache.status`` prints is exactly what Prometheus
sees, and adding a field to the snapshot is the only step needed to expose it.

Every series carries ``gpu_id``, ``tp_rank`` and ``pp_rank`` so the per-GPU
daemons of one node can be told apart after a Prometheus job scrapes them all.
"""

from typing import Any, Callable, Dict, Iterable, List

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    InfoMetricFamily,
    Metric,
)
from prometheus_client.registry import Collector

_PREFIX = "sglang:weight_cache_daemon_"
_LABELS = ["gpu_id", "tp_rank", "pp_rank"]

# Snapshot config keys copied verbatim onto the ``_info`` metric. Everything
# here is fixed for the daemon's lifetime, so the label set never churns.
_INFO_CONFIG_KEYS = (
    "model_path",
    "model_arch",
    "quantization",
    "dtype",
    "tp_size",
    "pp_size",
    "dp_size",
    "ep_size",
)


def _labels_of(snapshot: Dict[str, Any]) -> List[str]:
    config = snapshot.get("config") or {}
    return [
        str(snapshot.get("gpu_id", "")),
        str(config.get("tp_rank", "")),
        str(config.get("pp_rank", "")),
    ]


def _gauge(name: str, doc: str, labels: List[str], value: Any) -> Metric:
    family = GaugeMetricFamily(_PREFIX + name, doc, labels=_LABELS)
    family.add_metric(labels, value)
    return family


def snapshot_to_metrics(snapshot: Dict[str, Any]) -> Iterable[Metric]:
    """Map one ``status`` snapshot to Prometheus metric families."""
    labels = _labels_of(snapshot)
    config = snapshot.get("config") or {}

    # InfoMetricFamily appends ``_info`` itself.
    info = InfoMetricFamily(
        _PREFIX.rstrip("_"),
        "Static identity of this weight cache daemon.",
        labels=_LABELS,
    )
    info.add_metric(
        labels,
        {
            **{k: str(config.get(k, "")) for k in _INFO_CONFIG_KEYS},
            "transport_backend": str(snapshot.get("transport_backend") or ""),
            "socket_path": str(snapshot.get("socket_path") or ""),
            "pid": str(snapshot.get("pid") or ""),
        },
    )
    yield info

    # Serving pipeline: loaded -> serving. ``loaded`` flips once load() returns.
    yield _gauge(
        "loaded",
        "1 once the daemon has finished loading weights and is ready to serve.",
        labels,
        1 if snapshot.get("loaded_at") is not None else 0,
    )
    for name, doc, key in (
        (
            "preloaded_weights_bytes",
            "Bytes of model weights this daemon holds resident in GPU memory.",
            "preloaded_weights_bytes",
        ),
        ("num_tensors", "Number of tensors exported over IPC.", "num_tensors"),
        (
            "load_seconds",
            "Wall-clock seconds the daemon spent loading weights from disk.",
            "load_seconds",
        ),
        (
            "uptime_seconds",
            "Seconds since the daemon process started.",
            "uptime_seconds",
        ),
        (
            "live_clients",
            "Engine processes currently alive that fetched weights from this daemon.",
            "live_client_count",
        ),
        (
            "last_served_timestamp_seconds",
            "Unix time of the most recent successful fetch_state.",
            "last_served_at",
        ),
    ):
        value = snapshot.get(key)
        # A None field (still loading, never served) yields no sample rather
        # than a fake zero, so dashboards can tell "not yet" from "zero".
        if value is not None:
            yield _gauge(name, doc, labels, value)

    fetch = CounterMetricFamily(
        _PREFIX + "fetch_state",
        "fetch_state requests, by outcome: hit (served) or mismatch (rejected on CacheConfig).",
        labels=_LABELS + ["result"],
    )
    fetch.add_metric(labels + ["hit"], snapshot.get("serve_count") or 0)
    fetch.add_metric(labels + ["mismatch"], snapshot.get("mismatch_count") or 0)
    yield fetch


class WeightCacheDaemonCollector(Collector):
    """Collector that re-reads the daemon's status snapshot on every scrape."""

    def __init__(self, snapshot_fn: Callable[[], Dict[str, Any]]):
        self._snapshot_fn = snapshot_fn

    def collect(self) -> Iterable[Metric]:
        return snapshot_to_metrics(self._snapshot_fn())


def build_registry(snapshot_fn: Callable[[], Dict[str, Any]]) -> CollectorRegistry:
    registry = CollectorRegistry()
    registry.register(WeightCacheDaemonCollector(snapshot_fn))
    return registry


def start_metrics_server(
    snapshot_fn: Callable[[], Dict[str, Any]], port: int, addr: str = "0.0.0.0"
) -> CollectorRegistry:
    """Serve ``/metrics`` on a background thread; returns the registry."""
    registry = build_registry(snapshot_fn)
    start_http_server(port, addr=addr, registry=registry)
    return registry
