from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)
from urllib.parse import urlsplit

import msgspec
import zmq

from sglang.srt.disaggregation.kv_events import (
    EventPublisherFactory,
    KVEventBatch,
    KVEventsConfig,
    is_kv_publisher_rank,
    select_kv_publisher_dp_rank,
)
from sglang.srt.managers.io_struct import hook_custom_types, sock_send
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
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
    tree_cache: BasePrefixCache
    send_metrics_from_scheduler: Optional[zmq.Socket]
    max_running_requests: int
    max_total_num_tokens: int
    get_stats: Callable
    enable_kv_cache_events: bool = False
    kv_event_publisher: Any = None

    def __post_init__(self) -> None:
        self.init_kv_events(self.kv_events_config)

    def init_kv_events(self, kv_events_config: Optional[str]):
        self.enable_kv_cache_events = is_kv_publisher_rank(kv_events_config)

        if self.enable_kv_cache_events:
            parallel = get_parallel()
            sources = self.kv_hint_sources()
            if sources:
                config = KVEventsConfig.from_cli(kv_events_config)
                if config.format != "dynamo" or config.publisher != "zmq":
                    raise ValueError(
                        "KVCR linker inventory requires Dynamo-format ZMQ events"
                    )
                config.topic = hashlib.sha256(
                    sources[0]["cache_owner_id"].encode()
                ).hexdigest()[:16]
                kv_events_config = config.model_dump_json()
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config,
                select_kv_publisher_dp_rank(
                    parallel.attn_dp_size, parallel.attn_dp_rank, parallel.dp_rank
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
            get_parallel().dp_rank if get_parallel().dp_rank is not None else 0
        )

        if not self.send_metrics_from_scheduler.closed:
            sock_send(self.send_metrics_from_scheduler, kv_metrics)

    def kv_hint_sources(self) -> list[dict[str, Any]]:
        linker = getattr(getattr(self, "tree_cache", None), "linker", None)
        backend = getattr(linker, "cache_linker", None)
        return getattr(backend, "get_kv_hint_sources", lambda: [])()

    def local_kv_event_sources(self, block_size: int) -> list[dict[str, Any]]:
        if self.kv_event_publisher is None:
            return []
        hints = self.kv_hint_sources()
        if hints:
            host = urlsplit(hints[0]["source_control_endpoint"]).hostname
            source = self.kv_event_publisher.describe_local_source(block_size, host)
            if source is not None:
                source["format"] = KVEventsConfig.from_cli(self.kv_events_config).format
        else:
            source = self.kv_event_publisher.describe_local_source(block_size)
        return [source] if source is not None else []

    def publish_kv_events(self):
        if not self.enable_kv_cache_events:
            return

        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)
