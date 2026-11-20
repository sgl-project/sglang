from __future__ import annotations

import hashlib
import ipaddress
import logging
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

logger = logging.getLogger(__name__)

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


def _is_loopback(host: str) -> bool:
    # Also true for wildcard and unspecified hosts: no remote peer can dial them.
    if host == "*" or host.rstrip(".").lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if address.version == 6 and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback or address.is_unspecified


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
    # Host of the router source of a storage backend that publishes its own
    # residency under a cache owner; None for every other configuration.
    storage_advertise_host: Optional[str] = None
    # Whether the launch-time storage backend publishes its own residency;
    # fixed for the scheduler's lifetime.
    storage_publishes_kv_events: bool = False

    def __post_init__(self) -> None:
        self.init_kv_events(self.kv_events_config)

    def init_kv_events(self, kv_events_config: Optional[str]):
        self.enable_kv_cache_events = is_kv_publisher_rank(kv_events_config)
        storage_publishes = self.tree_cache.storage_kv_events_enabled
        self.storage_publishes_kv_events = storage_publishes
        if storage_publishes and not kv_events_config:
            raise ValueError(
                "The HiCache storage backend publishes KV events; configure "
                "a publisher with --kv-events-config."
            )

        if self.enable_kv_cache_events:
            parallel = get_parallel()
            owner_source = self._storage_owner_source() if storage_publishes else None
            if owner_source is not None:
                config = KVEventsConfig.from_cli(kv_events_config)
                # A cache owner ID is Dynamo's CacheOwnerId (see
                # HiCacheStorage.get_kv_hint_source); its state agent hashes
                # only the dynamo event format.
                if config.format != "dynamo":
                    raise ValueError(
                        "The HiCache storage backend publishes KV events under a "
                        'cache owner; set "format": "dynamo" in --kv-events-config.'
                    )
                owner = hashlib.sha256(
                    owner_source["cache_owner_id"].encode()
                ).hexdigest()
                # The owner changes with each engine incarnation, so a reused
                # TCP port never mixes a replacement engine's stream into a
                # subscriber still attached to the previous owner.
                config.topic = f"{config.topic}/owner/{owner}"
                kv_events_config = config.model_dump_json()
                self.storage_advertise_host = urlsplit(
                    owner_source["source_control_endpoint"]
                ).hostname
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config,
                select_kv_publisher_dp_rank(
                    parallel.attn_dp_size, parallel.attn_dp_rank, parallel.dp_rank
                ),
            )
            if owner_source is not None:
                self._require_external_source()

    def _require_external_source(self) -> None:
        """Fail at launch unless a subscriber on another host can reach the events."""
        source = self.kv_event_publisher.describe_local_source(
            1, advertise_host=self.storage_advertise_host
        )
        host = None
        if source is not None and source["endpoint"].startswith("tcp://"):
            host = urlsplit(source["endpoint"]).hostname
        if host is None or _is_loopback(host):
            self.kv_event_publisher.shutdown()
            raise ValueError(
                "The HiCache storage backend publishes KV events for a subscriber "
                "on another host; --kv-events-config needs a zmq publisher bound "
                'to a wildcard or routable TCP endpoint, e.g. "tcp://*:5557".'
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

    def _storage_owner_source(self) -> Optional[dict]:
        """The router source naming the cache owner of the backend's events."""
        return next(
            (
                source
                for source in self.tree_cache.storage_kv_hint_sources()
                if "cache_owner_id" in source
            ),
            None,
        )

    def local_kv_event_sources(self, block_size: int) -> list[dict[str, Any]]:
        if self.kv_event_publisher is None:
            return []
        advertise_host = self.storage_advertise_host
        if advertise_host is None:
            source = self.kv_event_publisher.describe_local_source(block_size)
            return [source] if source is not None else []
        # An external state agent subscribes from another host and must know
        # how the events hash: report a reachable host and the event format.
        source = self.kv_event_publisher.describe_local_source(
            block_size, advertise_host=advertise_host
        )
        if source is None:
            return []
        source["format"] = KVEventsConfig.from_cli(self.kv_events_config).format
        return [source]

    def publish_kv_events(self):
        if not self.enable_kv_cache_events:
            return

        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)
