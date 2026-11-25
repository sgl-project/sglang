from __future__ import annotations

import logging
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
    KVEventsConfig,
    build_advertised_publisher_entry,
    is_kv_publisher_rank,
    offset_advertisable_tcp,
    select_kv_publisher_dp_rank,
)
from sglang.srt.managers.io_struct import hook_custom_types, sock_send
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


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
    metrics_drop_warned: bool = False

    def __post_init__(self) -> None:
        self.init_kv_events(self.kv_events_config)

    def init_kv_events(self, kv_events_config: Optional[str]):
        self.enable_kv_cache_events = is_kv_publisher_rank(kv_events_config)

        if self.enable_kv_cache_events:
            parallel = get_parallel()
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
            # Auxiliary telemetry must not stall the scheduler when the receiver
            # is unavailable or cannot keep up. KV events use a separate publisher.
            try:
                sock_send(
                    self.send_metrics_from_scheduler, kv_metrics, flags=zmq.NOBLOCK
                )
            except zmq.Again:
                if not self.metrics_drop_warned:
                    logger.warning(
                        "KV metrics receiver unavailable or queue full; dropping auxiliary metrics"
                    )
                    self.metrics_drop_warned = True

    def local_kv_event_sources(self, block_size: int) -> list[dict[str, Any]]:
        if self.kv_event_publisher is None:
            return []
        source = self.kv_event_publisher.describe_local_source(block_size)
        return [source] if source is not None else []

    def describe_advertised_publisher(
        self,
        load_endpoint: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Describe this rank's routable KV-event (and optional replay/load)
        publisher for ``/server_info["kv_events"]["publishers"]``.

        Returns ``None`` on non-publishing ranks, when publishing is disabled,
        or when the configured endpoint is not a routable TCP address.
        """
        if (
            not self.enable_kv_cache_events
            or self.kv_event_publisher is None
            or not self.kv_events_config
        ):
            return None
        try:
            cfg = KVEventsConfig.from_cli(self.kv_events_config)
        except Exception:
            return None
        if cfg.publisher == "null" or not cfg.endpoint:
            return None
        parallel = get_parallel()
        dp_rank = select_kv_publisher_dp_rank(
            parallel.attn_dp_size, parallel.attn_dp_rank, parallel.dp_rank
        )
        return build_advertised_publisher_entry(
            dp_rank=dp_rank,
            kv_endpoint=offset_advertisable_tcp(cfg.endpoint, dp_rank),
            replay_endpoint=offset_advertisable_tcp(cfg.replay_endpoint, dp_rank),
            load_endpoint=load_endpoint,
            advertise_host=cfg.advertise_host,
        )

    def publish_kv_events(self):
        if not self.enable_kv_cache_events:
            return

        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)
