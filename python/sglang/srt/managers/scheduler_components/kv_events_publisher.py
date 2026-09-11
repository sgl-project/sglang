from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)

import msgspec
import zmq
from sglang.srt.disaggregation.kv_events import (
    EventPublisherFactory,
    KVEventBatch,
    is_kv_publisher_rank,
    select_kv_publisher_dp_rank,
)
from sglang.srt.managers.io_struct import hook_custom_types, sock_send

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class SchedulerStats: ...  # type: ignore[no-redef]


class KvMetrics(msgspec.Struct, tag=True, kw_only=True, array_like=True):
    request_active_slots: int = 0
    request_total_slots: int = 0
    kv_active_blocks: int = 0
    kv_total_blocks: int = 0
    num_requests_waiting: int = 0
    gpu_cache_usage_perc: float = 0.0
    gpu_prefix_cache_hit_rate: float = 0.0
    data_parallel_rank: int = 0


hook_custom_types(KvMetrics)


@dataclass(kw_only=True, slots=True)
class SchedulerKvEventsPublisher:
    kv_events_config: Optional[str]
    ps: ParallelState
    attn_tp_rank: int
    attn_cp_rank: int
    attn_dp_rank: int
    dp_rank: Optional[int]
    tree_cache: BasePrefixCache
    send_metrics_from_scheduler: Optional[zmq.Socket]
    max_running_requests: int
    max_total_num_tokens: int
    get_stats: Callable
    enable_kv_cache_events: bool = False
    kv_event_publisher: Any = None
    model: str = ""

    def __post_init__(self) -> None:
        self.init_kv_events(self.kv_events_config)

    def init_kv_events(self, kv_events_config: Optional[str]):
        self.enable_kv_cache_events = is_kv_publisher_rank(kv_events_config, self.ps)

        if self.enable_kv_cache_events:
            config = json.loads(kv_events_config)
            components = getattr(self.tree_cache, "tree_components", ())
            if (
                config.get("publisher") == "zmq"
                and config.get("snapshot_endpoint")
                and config.get("worker_id")
            ):
                core = getattr(self.tree_cache, "tree_core", self.tree_cache)
                recorder = getattr(core, "kv_events", None)
                if recorder is not None:
                    config["page_size"] = recorder.page_size
                config["model"] = config.get("model") or self.model
                config["is_bigram"] = bool(getattr(self.tree_cache, "is_eagle", False))
                kv_events_config = json.dumps(config)
            if (
                config.get("publisher") == "zmq"
                and config.get("snapshot_endpoint")
                and config.get("worker_id")
                and len(components) > 1
            ):
                names = {str(component) for component in components}
                supported = names <= {"full", "swa", "mamba"}
                # Python UnifiedTreeCore emits the component residency updates.
                # Alternative cores must provide the same contract before they
                # can advertise a queryable v1 component spec.
                python_core = (
                    type(getattr(self.tree_cache, "tree_core", None)).__name__
                    == "UnifiedTreeCore"
                )
                if supported and python_core:
                    self.tree_cache.tree_core.kv_events.component_aware = True
                config["cache_spec"] = {
                    "version": 1 if supported and python_core else 2,
                    "components": sum(
                        {"full": 1, "swa": 2, "mamba": 4}.get(name, 0) for name in names
                    ),
                    "swa_window_tokens": getattr(
                        self.tree_cache, "_sliding_window_size", 0
                    )
                    or 0,
                    "full_tier_mask": 6,
                    "swa_tier_mask": 6,
                    "mamba_tier_mask": 6,
                }
                kv_events_config = json.dumps(config)
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config,
                select_kv_publisher_dp_rank(
                    self.ps.attn_dp_size, self.ps.attn_dp_rank, self.ps.dp_rank
                ),
            )

    def emit_kv_metrics(self):
        if not self.enable_kv_cache_events:
            return

        kv_metrics = KvMetrics()
        kv_metrics.request_active_slots = self.get_stats().num_running_reqs.total
        kv_metrics.request_total_slots = self.max_running_requests
        kv_metrics.kv_active_blocks = int(
            self.get_stats().token_usage * self.max_total_num_tokens
        )
        kv_metrics.kv_total_blocks = self.max_total_num_tokens
        kv_metrics.num_requests_waiting = self.get_stats().num_queue_reqs.total
        kv_metrics.gpu_cache_usage_perc = self.get_stats().token_usage
        kv_metrics.gpu_prefix_cache_hit_rate = self.get_stats().cache_hit_rate
        kv_metrics.data_parallel_rank = (
            self.ps.dp_rank if self.ps.dp_rank is not None else 0
        )

        if not self.send_metrics_from_scheduler.closed:
            sock_send(self.send_metrics_from_scheduler, kv_metrics)

    def publish_kv_events(self):
        if not self.enable_kv_cache_events:
            return

        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)
