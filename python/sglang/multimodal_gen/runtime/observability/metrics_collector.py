"""Prometheus metric primitives for the diffusion runtime."""

import os
from typing import Any

_PROMETHEUS_MULTIPROC_CONFIGURED = "SGLANG_PROMETHEUS_MULTIPROC_CONFIGURED"
DIFFUSION_LATENCY_BUCKETS = (
    0.01,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    3,
    4,
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
)
DIFFUSION_STAGE_LATENCY_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    20,
    60,
    120,
    300,
    800,
)
DIFFUSION_BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


def _stage_label(stage_name: str) -> str:
    if stage_name.startswith("denoising_step_"):
        return "DenoisingStep"
    return stage_name


def _stop_reason_label(stop_reason: str | None) -> str:
    if not stop_reason:
        return "unspecified"
    stop_reason = str(stop_reason)
    for prefix in ("config_cap", "cost_budget_next", "cost_budget"):
        if stop_reason.startswith(prefix):
            return prefix
    if ":" in stop_reason:
        return stop_reason.split(":", 1)[0]
    return stop_reason


def configure_prometheus_multiproc_dir() -> None:
    from sglang.srt.utils import set_prometheus_multiproc_dir

    if os.environ.get(_PROMETHEUS_MULTIPROC_CONFIGURED) == "1":
        return
    set_prometheus_multiproc_dir()
    os.environ[_PROMETHEUS_MULTIPROC_CONFIGURED] = "1"


class DiffusionMetricsCollector:
    """Prometheus metrics for diffusion runtime."""

    def __init__(self, registry: Any | None = None):
        from prometheus_client import Counter, Gauge, Histogram

        self.num_queue_reqs = Gauge(
            name="sglang:diffusion_num_queue_reqs",
            documentation="Number of requests in the diffusion scheduler waiting queue.",
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        self.num_running_reqs = Gauge(
            name="sglang:diffusion_num_running_reqs",
            documentation="Number of diffusion generation requests dispatched by the scheduler and not yet finished.",
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        self.requests_total = Counter(
            name="sglang:diffusion_requests_total",
            documentation="Total number of diffusion requests by status.",
            labelnames=["status", "is_warmup"],
            registry=registry,
        )
        self.request_latency_seconds = Histogram(
            name="sglang:diffusion_request_latency_seconds",
            documentation="Diffusion scheduler request latency in seconds, from enqueue to output handling.",
            labelnames=["status", "is_warmup"],
            buckets=DIFFUSION_LATENCY_BUCKETS,
            registry=registry,
        )
        self.queue_time_seconds = Histogram(
            name="sglang:diffusion_queue_time_seconds",
            documentation="Histogram of queueing time in seconds for diffusion generation requests.",
            labelnames=["is_warmup"],
            buckets=DIFFUSION_LATENCY_BUCKETS,
            registry=registry,
        )
        self.generation_batch_size = Histogram(
            name="sglang:diffusion_generation_batch_size",
            documentation="Histogram of diffusion generation batch sizes at scheduler dispatch.",
            labelnames=["stop_reason"],
            buckets=DIFFUSION_BATCH_SIZE_BUCKETS,
            registry=registry,
        )
        self.stage_latency_seconds = Histogram(
            name="sglang:diffusion_stage_latency_seconds",
            documentation="Histogram of diffusion pipeline stage latency in seconds.",
            labelnames=["stage"],
            buckets=DIFFUSION_STAGE_LATENCY_BUCKETS,
            registry=registry,
        )
        self.lora_loaded_adapters = Gauge(
            name="sglang:diffusion_lora_loaded_adapters",
            documentation="Number of loaded diffusion LoRA adapters.",
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        self.lora_active_modules = Gauge(
            name="sglang:diffusion_lora_active_modules",
            documentation="Number of diffusion modules with active LoRA adapters.",
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        self.lora_active_adapters = Gauge(
            name="sglang:diffusion_lora_active_adapters",
            documentation="Number of unique active diffusion LoRA adapters.",
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        self.lora_module_active = Gauge(
            name="sglang:diffusion_lora_module_active",
            documentation="Whether LoRA is active for a diffusion module (1 active, 0 inactive).",
            labelnames=["module"],
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        self._observed_modules: set[str] = set()

    def set_queue_depth(self, queue_depth: int) -> None:
        self.num_queue_reqs.set(max(queue_depth, 0))

    def set_running_reqs(self, running_reqs: int) -> None:
        self.num_running_reqs.set(max(running_reqs, 0))

    def observe_request(self, status: str, is_warmup: bool, latency_s: float) -> None:
        status_label = status if status in ("success", "error") else "unknown"
        warmup_label = _bool_label(is_warmup)
        labels = {"status": status_label, "is_warmup": warmup_label}
        self.requests_total.labels(**labels).inc()
        self.request_latency_seconds.labels(**labels).observe(max(latency_s, 0.0))

    def observe_queue_time(self, wait_s: float, is_warmup: bool) -> None:
        warmup_label = _bool_label(is_warmup)
        self.queue_time_seconds.labels(is_warmup=warmup_label).observe(max(wait_s, 0.0))

    def observe_generation_batch_size(
        self, batch_size: int, stop_reason: str | None
    ) -> None:
        self.generation_batch_size.labels(
            stop_reason=_stop_reason_label(stop_reason)
        ).observe(max(batch_size, 0))

    def observe_stage_latency(self, stage_name: str, latency_s: float) -> None:
        self.stage_latency_seconds.labels(
            stage=_stage_label(stage_name),
        ).observe(max(latency_s, 0.0))

    def clear_lora_status(self) -> None:
        self.lora_loaded_adapters.set(0)
        self.lora_active_modules.set(0)
        self.lora_active_adapters.set(0)
        for module_name in self._observed_modules:
            self.lora_module_active.labels(module=module_name).set(0)

    def update_lora_status(self, status: dict[str, Any]) -> None:
        loaded_adapters = status.get("loaded_adapters", [])
        loaded_count = len(loaded_adapters) if isinstance(loaded_adapters, list) else 0

        active = status.get("active", {})
        active_map = active if isinstance(active, dict) else {}
        reported_modules = {str(module_name) for module_name in active_map}

        active_adapters: set[str] = set()
        active_modules: set[str] = set()
        for module_name, module_entries in active_map.items():
            if not isinstance(module_entries, list):
                continue
            module_has_active_adapter = False
            for entry in module_entries:
                if not isinstance(entry, dict):
                    continue
                nicknames = entry.get("nicknames")
                if isinstance(nicknames, list):
                    adapters = [str(adapter).strip() for adapter in nicknames]
                else:
                    nickname = entry.get("nickname")
                    adapters = [str(nickname).strip()] if nickname else []
                valid_adapters = [adapter for adapter in adapters if adapter]
                if valid_adapters:
                    module_has_active_adapter = True
                    active_adapters.update(valid_adapters)
            if module_has_active_adapter:
                active_modules.add(str(module_name))

        self.lora_loaded_adapters.set(loaded_count)
        self.lora_active_modules.set(len(active_modules))
        self.lora_active_adapters.set(len(active_adapters))

        # Keep prior module labels so modules that become inactive are exported
        # as 0. The module namespace is bounded by pipeline component names.
        all_modules = self._observed_modules | reported_modules
        for module_name in all_modules:
            self.lora_module_active.labels(module=module_name).set(
                1 if module_name in active_modules else 0
            )
        self._observed_modules = all_modules
