"""Runtime observer interface for diffusion serving."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.runtime.observability.metrics_collector import (
    DiffusionMetricsCollector,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


class DiffusionObserver:
    """Observer interface used by diffusion runtime code."""

    enabled = False

    def set_queue_depth(self, queue_depth: int) -> None:
        pass

    def reset_running_requests(self) -> None:
        pass

    def mark_requests_dispatched(self, reqs: list[Any]) -> None:
        pass

    def mark_requests_finished(self, reqs: list[Any]) -> None:
        pass

    def observe_queue_time(self, req: Any, enqueue_time: float) -> None:
        pass

    def observe_request_finished(
        self, req: Any, output_batch: Any, enqueue_time: float
    ) -> None:
        pass

    def observe_batch_dispatch(self, batch_size: int, stop_reason: str | None) -> None:
        pass

    def observe_stage_latency(self, stage_name: str, latency_s: float) -> None:
        pass

    def clear_lora_status(self) -> None:
        pass

    def update_lora_status(self, status: dict[str, Any]) -> None:
        pass


class NoopDiffusionObserver(DiffusionObserver):
    pass


class PrometheusDiffusionObserver(DiffusionObserver):
    """DiffusionObserver backed by Prometheus metrics."""

    enabled = True

    def __init__(self, collector: DiffusionMetricsCollector | None = None):
        self.collector = collector or DiffusionMetricsCollector()
        self._running_reqs = 0
        self._running_reqs_lock = threading.Lock()
        self.collector.set_running_reqs(0)

    def set_queue_depth(self, queue_depth: int) -> None:
        self.collector.set_queue_depth(queue_depth)

    def reset_running_requests(self) -> None:
        with self._running_reqs_lock:
            self._running_reqs = 0
            self.collector.set_running_reqs(0)

    def mark_requests_dispatched(self, reqs: list[Any]) -> None:
        """Track original scheduler requests that have entered execution."""
        count = len(reqs)
        if count == 0:
            return
        with self._running_reqs_lock:
            self._running_reqs += count
            self.collector.set_running_reqs(self._running_reqs)

    def mark_requests_finished(self, reqs: list[Any]) -> None:
        """Remove original scheduler requests from the running count."""
        count = len(reqs)
        if count == 0:
            return
        with self._running_reqs_lock:
            self._running_reqs = max(0, self._running_reqs - count)
            self.collector.set_running_reqs(self._running_reqs)

    def observe_queue_time(self, req: Any, enqueue_time: float) -> None:
        self.collector.observe_queue_time(
            time.monotonic() - enqueue_time,
            is_warmup=req.is_warmup,
        )

    def observe_request_finished(
        self, req: Any, output_batch: Any, enqueue_time: float
    ) -> None:
        status = "error" if output_batch.error is not None else "success"
        self.collector.observe_request(
            status=status,
            is_warmup=req.is_warmup,
            latency_s=time.monotonic() - enqueue_time,
        )

    def observe_batch_dispatch(self, batch_size: int, stop_reason: str | None) -> None:
        self.collector.observe_generation_batch_size(
            batch_size=batch_size,
            stop_reason=stop_reason,
        )

    def observe_stage_latency(self, stage_name: str, latency_s: float) -> None:
        self.collector.observe_stage_latency(stage_name, latency_s=latency_s)

    def clear_lora_status(self) -> None:
        self.collector.clear_lora_status()

    def update_lora_status(self, status: dict[str, Any]) -> None:
        self.collector.update_lora_status(status)


_noop_observer = NoopDiffusionObserver()
_diffusion_observer_lock = threading.Lock()
_diffusion_observer: DiffusionObserver = _noop_observer


def get_diffusion_observer(
    server_args: "ServerArgs" | None = None,
    *,
    enabled: bool = True,
) -> DiffusionObserver:
    """Return the process-wide observer, creating Prometheus metrics from ServerArgs."""
    global _diffusion_observer

    if not enabled or (server_args is not None and not server_args.enable_metrics):
        return _noop_observer

    if _diffusion_observer.enabled:
        return _diffusion_observer

    if server_args is None:
        return _noop_observer

    with _diffusion_observer_lock:
        if not _diffusion_observer.enabled:
            _diffusion_observer = PrometheusDiffusionObserver()

    return _diffusion_observer


def get_diffusion_metrics_collector(
    server_args: "ServerArgs" | None = None,
) -> DiffusionMetricsCollector | None:
    observer = get_diffusion_observer(server_args)
    if isinstance(observer, PrometheusDiffusionObserver):
        return observer.collector
    return None
