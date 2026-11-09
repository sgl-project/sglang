# SPDX-License-Identifier: Apache-2.0
"""Opt-in, process-local diffusion metrics; only replica leaders publish."""

from __future__ import annotations

import os
import tempfile
import time
from typing import TYPE_CHECKING, Hashable

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

    from sglang.multimodal_gen.runtime.server_args import ServerArgs

_multiproc_dir: tempfile.TemporaryDirectory | None = None
_metrics: DiffusionMetrics | None = None


def configure_metrics() -> None:
    """Set the multiprocess directory before importing Prometheus or spawning."""
    global _multiproc_dir
    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        _multiproc_dir = tempfile.TemporaryDirectory(prefix="sglang-diffusion-metrics-")
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = _multiproc_dir.name


def init_metrics(
    server_args: ServerArgs, rank: int = 0, *, role: str | None = None
) -> DiffusionMetrics | None:
    global _metrics
    group_size = max(1, server_args.num_gpus // (server_args.dp_size or 1))
    if not server_args.enable_metrics or rank % group_size:
        _metrics = None
        return None
    replica = rank // group_size
    _metrics = DiffusionMetrics(
        role=role or server_args.disagg_role.value,
        replica=server_args.scheduler_endpoint_for(replica),
    )
    return _metrics


def get_metrics() -> DiffusionMetrics | None:
    return _metrics


def start_role_metrics_server(server_args: ServerArgs) -> None:
    # prometheus selects its storage backend at import time, after configure_metrics
    from prometheus_client import CollectorRegistry, multiprocess, start_http_server

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    start_http_server(server_args.port, addr=server_args.host, registry=registry)


class DiffusionMetrics:
    """Callbacks are serialized by the scheduler or RequestTracker lock.

    Keys identify original requests, not generated outputs or GPU batches.
    Request IDs, prompts and adapter paths are never exported as labels.
    """

    def __init__(
        self, *, role: str, replica: str, registry: CollectorRegistry | None = None
    ):
        # defer this import until the launcher has configured multiprocess storage
        from prometheus_client import Counter, Gauge, Histogram

        labels = ("role", "replica")
        self._labels = (role, replica)
        self._requests: dict[Hashable, tuple[float, bool, bool]] = {}
        self._queued = 0
        self._running = 0
        self._observed_modules: set[str] = set()
        self.queue = Gauge(
            "sglang:diffusion_num_queue_reqs",
            "Requests waiting for first dispatch.",
            labels,
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        self.running = Gauge(
            "sglang:diffusion_num_running_reqs",
            "Dispatched requests not yet completed.",
            labels,
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        self.requests = Counter(
            "sglang:diffusion_requests_total",
            "Completed original client requests.",
            labels + ("status", "is_warmup"),
            registry=registry,
        )
        buckets = (0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200)
        self.latency = Histogram(
            "sglang:diffusion_request_latency_seconds",
            "Time from scheduler acceptance to completion, excluding HTTP postprocessing.",
            labels + ("status", "is_warmup"),
            buckets=buckets,
            registry=registry,
        )
        self.queue_time = Histogram(
            "sglang:diffusion_queue_time_seconds",
            "Time until first dispatch.",
            labels + ("is_warmup",),
            buckets=buckets,
            registry=registry,
        )
        self.batch_size = Histogram(
            "sglang:diffusion_generation_batch_size",
            "Original requests per dispatched batch.",
            labels + ("stop_reason",),
            buckets=(1, 2, 4, 8, 16, 32, 64),
            registry=registry,
        )
        self.stage_latency = Histogram(
            "sglang:diffusion_stage_host_latency_seconds",
            "Host wall time around a pipeline stage; not synchronized GPU execution time.",
            labels + ("stage",),
            buckets=(
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1,
                2,
                5,
                10,
                30,
                60,
                120,
                300,
                1200,
            ),
            registry=registry,
        )
        self.lora_loaded = Gauge(
            "sglang:diffusion_lora_loaded_adapters",
            "Loaded LoRA adapters.",
            labels,
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        self.lora_modules = Gauge(
            "sglang:diffusion_lora_active_modules",
            "Modules with active LoRA adapters.",
            labels,
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        self.lora_adapters = Gauge(
            "sglang:diffusion_lora_active_adapters",
            "Unique active LoRA adapters.",
            labels,
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        self.lora_module = Gauge(
            "sglang:diffusion_lora_module_active",
            "Whether a module has an active adapter.",
            labels + ("module",),
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        self._publish_depths()

    def _publish_depths(self):
        self.queue.labels(*self._labels).set(self._queued)
        self.running.labels(*self._labels).set(self._running)

    def enqueue(self, key: Hashable, *, is_warmup: bool, now: float | None = None):
        self._requests[key] = (
            time.monotonic() if now is None else now,
            is_warmup,
            False,
        )
        self._queued += 1
        self._publish_depths()

    def dispatch(self, key: Hashable):
        state = self._requests.get(key)
        if state is None or state[2]:
            return
        start, is_warmup, _ = state
        self._requests[key] = (start, is_warmup, True)
        self._queued -= 1
        self._running += 1
        self.queue_time.labels(*self._labels, str(is_warmup).lower()).observe(
            max(0.0, time.monotonic() - start)
        )
        self._publish_depths()

    def finish(self, key: Hashable, *, error: bool):
        state = self._requests.pop(key, None)
        if state is None:
            return
        start, is_warmup, dispatched = state
        if dispatched:
            self._running -= 1
        else:
            self._queued -= 1
        labels = (
            *self._labels,
            "error" if error else "success",
            str(is_warmup).lower(),
        )
        self.requests.labels(*labels).inc()
        self.latency.labels(*labels).observe(max(0.0, time.monotonic() - start))
        self._publish_depths()

    def observe_batch(self, size: int, stop_reason: str | None):
        reason = (stop_reason or "unspecified").partition(":")[0]
        self.batch_size.labels(*self._labels, reason).observe(size)

    def observe_stage(self, name: str, seconds: float):
        if name.startswith("denoising_step_"):
            name = "DenoisingStep"
        self.stage_latency.labels(*self._labels, name).observe(seconds)

    def update_lora(self, status: dict):
        active = status["active"]
        adapters = {
            nickname
            for entries in active.values()
            for entry in entries
            for nickname in entry["nicknames"]
            if nickname
        }
        self.lora_loaded.labels(*self._labels).set(len(status["loaded_adapters"]))
        self.lora_modules.labels(*self._labels).set(len(active))
        self.lora_adapters.labels(*self._labels).set(len(adapters))
        self._observed_modules.update(active)
        for module in self._observed_modules:
            self.lora_module.labels(*self._labels, module).set(int(module in active))
