"""Merge native registries and Python mmap collectors without double counting.

The private snapshot format retains gauge modes and timestamps and keeps
histogram buckets noncumulative until the final Prometheus accumulation.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_SCRAPE_BYTES = 64 * 1024 * 1024
SOURCE_TIMEOUT_SECONDS = 4.0


@dataclass(frozen=True)
class MetricSource:
    source_id: str
    url: str
    kind: str = "python"

    def as_dict(self):
        return asdict(self)


def read_python_metrics(directory: str, source_id: str) -> dict:
    from prometheus_client import multiprocess

    files = list(Path(directory).glob("*.db"))
    size = 0
    for path in files:
        try:
            size += path.stat().st_size
        except FileNotFoundError:
            if not path.name.startswith("gauge_live"):
                raise
    if size > MAX_SCRAPE_BYTES:
        raise ValueError("Python metric sources exceed the 64 MiB scrape budget")
    # collect()/merge() discard mostrecent timestamps. Preserve the raw samples
    # here and use the same library's accumulation once, after source merging.
    metrics = multiprocess.MultiProcessCollector._read_metrics(files)
    families = []
    for metric in metrics.values():
        families.append(
            {
                "name": metric.name,
                "documentation": metric.documentation,
                "type": metric.type,
                "unit": metric.unit,
                "gauge_mode": (
                    metric._multiprocess_mode if metric.type == "gauge" else None
                ),
                "samples": [
                    {
                        "name": sample.name,
                        "labels": dict(sample.labels),
                        # Prometheus permits NaN and infinities; JSON numbers do not.
                        "value": str(sample.value),
                        "timestamp": (
                            str(sample.timestamp)
                            if sample.timestamp is not None
                            else None
                        ),
                    }
                    for sample in metric.samples
                ],
            }
        )
    return {"version": 1, "source_id": source_id, "metrics": families}


def _python_families(snapshot: dict, source_id: str):
    from prometheus_client.metrics_core import Metric

    if snapshot["version"] != 1 or snapshot["source_id"] != source_id:
        raise ValueError(f"Unexpected metrics source identity or version: {source_id}")
    for family in snapshot["metrics"]:
        metric = Metric(
            family["name"], family["documentation"], family["type"], family["unit"]
        )
        if metric.type == "gauge":
            metric._multiprocess_mode = family["gauge_mode"]
            if metric._multiprocess_mode not in (
                "all",
                "liveall",
                "sum",
                "livesum",
                "min",
                "livemin",
                "max",
                "livemax",
                "mostrecent",
                "livemostrecent",
            ):
                raise ValueError(f"Unknown gauge mode for {metric.name}")
        for sample in family["samples"]:
            labels = dict(sample["labels"])
            if "pid" in labels:
                # Process ids are node-local. all/liveall must keep two sources
                # distinct even when their operating systems reuse the same pid.
                labels["pid"] = f"{source_id}/{labels['pid']}"
            timestamp = sample["timestamp"]
            metric.add_sample(
                sample["name"],
                tuple(sorted(labels.items())),
                float(sample["value"]),
                float(timestamp) if timestamp is not None else None,
            )
        yield metric


def _native_families(text: str):
    from prometheus_client.parser import text_string_to_metric_families

    for metric in text_string_to_metric_families(text):
        if metric.type == "gauge":
            # Only the public listener contributes HTTP observations. Idle DP
            # worker registries can still contain the initial zero gauge.
            if metric.name not in (
                "sglang:http_requests_active",
                "sglang:routing_keys_active",
            ):
                raise ValueError(
                    f"Native gauge has no declared merge mode: {metric.name}"
                )
            metric._multiprocess_mode = "livesum"
        samples = []
        buckets = {}
        for sample in metric.samples:
            labels = tuple(sorted(sample.labels.items()))
            if metric.type == "histogram" and sample.name.endswith("_bucket"):
                key = tuple(label for label in labels if label[0] != "le")
                buckets.setdefault(key, []).append(sample)
            else:
                samples.append(sample._replace(labels=labels))
        for group in buckets.values():
            previous = 0.0
            for sample in sorted(group, key=lambda sample: float(sample.labels["le"])):
                if sample.value < previous:
                    raise ValueError(f"Nonmonotonic native histogram: {metric.name}")
                samples.append(
                    sample._replace(
                        labels=tuple(sorted(sample.labels.items())),
                        value=sample.value - previous,
                    )
                )
                previous = sample.value
        metric.samples = samples
        yield metric


def merge_sources(sources: list[tuple[MetricSource, dict | str]]):
    from prometheus_client import multiprocess

    merged = {}
    seen = set()
    histogram_bounds = {}
    for source, payload in sources:
        if source.source_id in seen:
            continue
        seen.add(source.source_id)
        families = (
            _python_families(payload, source.source_id)
            if source.kind == "python"
            else _native_families(payload)
        )
        for metric in families:
            if metric.type == "summary" and any(
                "quantile" in dict(sample.labels) for sample in metric.samples
            ):
                raise ValueError(f"Cannot aggregate summary quantiles: {metric.name}")
            if metric.type == "histogram":
                bounds = {}
                for sample in metric.samples:
                    if not sample.name.endswith("_bucket"):
                        continue
                    labels = dict(sample.labels)
                    bound = float(labels.pop("le"))
                    key = (metric.name, tuple(sorted(labels.items())))
                    bounds.setdefault(key, set()).add(bound)
                for key, value in bounds.items():
                    if histogram_bounds.setdefault(key, value) != value:
                        raise ValueError(
                            f"Conflicting histogram buckets: {metric.name}"
                        )
            existing = merged.get(metric.name)
            if existing is None:
                merged[metric.name] = metric
                continue
            if (metric.type, metric.documentation, metric.unit) != (
                existing.type,
                existing.documentation,
                existing.unit,
            ) or (
                metric.type == "gauge"
                and metric._multiprocess_mode != existing._multiprocess_mode
            ):
                raise ValueError(f"Conflicting metric family schema: {metric.name}")
            existing.samples.extend(metric.samples)
    return list(multiprocess.MultiProcessCollector._accumulate_metrics(merged, True))


class CollectedMetrics:
    def __init__(self, families):
        self.families = families

    def collect(self):
        return iter(self.families)


class MetricsCollection:
    """Event-loop-owned source configuration; collection is outside GPU processes."""

    def __init__(self, directory: str, source_id: str):
        self.directory = directory
        self.source_id = source_id
        self.sources: tuple[MetricSource, ...] = ()
        self.startup: list[dict] = []

    def set_startup_time(self, startup_time: dict, labels: dict[str, str]):
        timestamp = str(time.time())
        self.startup = []
        for name, help_text, values in (
            (
                "sglang:startup_time_seconds",
                "Engine startup duration by phase in seconds.",
                {
                    phase: startup_time[phase]
                    for phase in (
                        "load_weight",
                        "kv_cache_allocation",
                        "scheduler_e2e",
                        "tokenizer_e2e",
                    )
                },
            ),
            (
                "sglang:startup_cuda_graph_time_seconds",
                "CUDA graph capture duration by phase in seconds.",
                startup_time["cuda_graph"],
            ),
        ):
            self.startup.append(
                {
                    "name": name,
                    "documentation": help_text,
                    "type": "gauge",
                    "unit": "",
                    "gauge_mode": "mostrecent",
                    "samples": [
                        {
                            "name": name,
                            "labels": {**labels, "phase": phase},
                            "value": str(float(value)),
                            "timestamp": timestamp,
                        }
                        for phase, value in values.items()
                    ],
                }
            )

    async def snapshot(self):
        startup = self.startup
        snapshot = await asyncio.to_thread(
            read_python_metrics, self.directory, self.source_id
        )
        snapshot["metrics"].extend(startup)
        return snapshot

    def configure(self, sources: list[MetricSource]):
        unique = {}
        for source in sources:
            if source.kind not in ("python", "native"):
                raise ValueError(f"Unknown metric source kind: {source.kind}")
            if source.source_id in unique and unique[source.source_id] != source:
                raise ValueError(f"Conflicting metric source: {source.source_id}")
            unique[source.source_id] = source
        self.sources = tuple(unique.values())

    async def collect(self, session):
        total_bytes = 0

        async def fetch(source):
            nonlocal total_bytes
            async with session.get(
                source.url, timeout=SOURCE_TIMEOUT_SECONDS
            ) as response:
                response.raise_for_status()
                body = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_SCRAPE_BYTES:
                        raise ValueError(
                            "Metric sources exceed the 64 MiB scrape budget"
                        )
                    body.extend(chunk)
            payload = (
                json.loads(body) if source.kind == "python" else body.decode("utf-8")
            )
            return source, payload

        async def local():
            snapshot = await self.snapshot()
            return MetricSource(self.source_id, ""), snapshot

        sources = await asyncio.gather(
            local(),
            *(
                fetch(source)
                for source in self.sources
                if source.source_id != self.source_id
            ),
        )
        families = await asyncio.to_thread(merge_sources, sources)
        return CollectedMetrics(families)
