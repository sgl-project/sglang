import threading
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


class DiffusionMetricsCollector:
    """Prometheus metrics for diffusion runtime."""

    def __init__(self):
        from prometheus_client import Counter, Gauge, Histogram

        self.num_queue_reqs = Gauge(
            name="sglang:diffusion_num_queue_reqs",
            documentation="Number of requests in the diffusion scheduler waiting queue.",
            multiprocess_mode="mostrecent",
        )
        self.num_running_reqs = Gauge(
            name="sglang:diffusion_num_running_reqs",
            documentation="Number of currently running diffusion requests.",
            multiprocess_mode="mostrecent",
        )
        self.requests_total = Counter(
            name="sglang:diffusion_requests_total",
            documentation="Total number of diffusion requests by status.",
            labelnames=["status", "is_warmup"],
        )
        self.request_latency_seconds = Histogram(
            name="sglang:diffusion_request_latency_seconds",
            documentation="End-to-end diffusion request latency in seconds.",
            labelnames=["status", "is_warmup"],
            buckets=(
                0.01,
                0.05,
                0.1,
                0.2,
                0.5,
                1,
                2,
                5,
                10,
                20,
                30,
                60,
                120,
                300,
            ),
        )
        self.queue_time_seconds = Histogram(
            name="sglang:diffusion_queue_time_seconds",
            documentation="Histogram of queueing time in seconds for diffusion generation requests.",
            buckets=(
                0.0,
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
            ),
        )
        self.lora_loaded_adapters = Gauge(
            name="sglang:diffusion_lora_loaded_adapters",
            documentation="Number of loaded diffusion LoRA adapters.",
            multiprocess_mode="mostrecent",
        )
        self.lora_active_modules = Gauge(
            name="sglang:diffusion_lora_active_modules",
            documentation="Number of diffusion modules with active LoRA adapters.",
            multiprocess_mode="mostrecent",
        )
        self.lora_active_adapters = Gauge(
            name="sglang:diffusion_lora_active_adapters",
            documentation="Number of unique active diffusion LoRA adapters.",
            multiprocess_mode="mostrecent",
        )
        self.lora_module_active = Gauge(
            name="sglang:diffusion_lora_module_active",
            documentation="Whether LoRA is active for a diffusion module (1 active, 0 inactive).",
            labelnames=["module"],
            multiprocess_mode="mostrecent",
        )
        self._observed_modules: set[str] = set()

    def set_queue_depth(self, queue_depth: int) -> None:
        self.num_queue_reqs.set(max(queue_depth, 0))

    def set_running_reqs(self, running_reqs: int) -> None:
        self.num_running_reqs.set(max(running_reqs, 0))

    def observe_request(self, status: str, is_warmup: bool, latency_s: float) -> None:
        status_label = status if status in ("success", "error") else "unknown"
        warmup_label = "true" if is_warmup else "false"
        labels = {"status": status_label, "is_warmup": warmup_label}
        self.requests_total.labels(**labels).inc()
        self.request_latency_seconds.labels(**labels).observe(max(latency_s, 0.0))

    def observe_queue_time(self, wait_s: float) -> None:
        self.queue_time_seconds.observe(max(wait_s, 0.0))

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
        active_modules = set(active_map.keys())

        active_adapters: set[str] = set()
        for module_entries in active_map.values():
            if not isinstance(module_entries, list):
                continue
            for entry in module_entries:
                if not isinstance(entry, dict):
                    continue
                nickname = entry.get("nickname")
                if not nickname:
                    continue
                for adapter in str(nickname).split(","):
                    adapter = adapter.strip()
                    if adapter:
                        active_adapters.add(adapter)

        self.lora_loaded_adapters.set(loaded_count)
        self.lora_active_modules.set(len(active_modules))
        self.lora_active_adapters.set(len(active_adapters))

        all_modules = self._observed_modules | active_modules
        for module_name in all_modules:
            self.lora_module_active.labels(module=module_name).set(
                1 if module_name in active_modules else 0
            )
        self._observed_modules = all_modules


_diffusion_metrics_collector_lock = threading.Lock()
_diffusion_metrics_collector: Optional[DiffusionMetricsCollector] = None


def get_diffusion_metrics_collector(
    server_args: Optional["ServerArgs"] = None,
) -> Optional[DiffusionMetricsCollector]:
    global _diffusion_metrics_collector

    if _diffusion_metrics_collector is not None:
        return _diffusion_metrics_collector

    if server_args is None or not server_args.enable_metrics:
        return None

    with _diffusion_metrics_collector_lock:
        if _diffusion_metrics_collector is None:
            _diffusion_metrics_collector = DiffusionMetricsCollector()

    return _diffusion_metrics_collector
