from __future__ import annotations  # noqa: F401

import dataclasses  # noqa: F401
import time  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from typing import Callable, Optional  # noqa: F401

from sglang.srt.disaggregation.kv_events import (  # noqa: F401
    EventPublisherFactory,
    KVEventBatch,
)


# ``SchedulerStats`` referenced only as a type hint in ``emit_kv_metrics`` —
# leave a forward-ref placeholder.
class SchedulerStats: ...  # type: ignore[no-redef]


@dataclasses.dataclass
class KvMetrics:
    request_active_slots: int = 0
    request_total_slots: int = 0
    kv_active_blocks: int = 0
    kv_total_blocks: int = 0
    num_requests_waiting: int = 0
    gpu_cache_usage_perc: float = 0.0
    gpu_prefix_cache_hit_rate: float = 0.0
    data_parallel_rank: int = 0


class SchedulerKvEventsPublisher:
    """KV cache event / metrics publication channel. Composition target on
    Scheduler (``self.kv_events_publisher``)."""

    def __init__(
        self,
        *,
        kv_events_config: Optional[str],
        attn_tp_rank: int,
        attn_cp_rank: int,
        attn_dp_rank: int,
        dp_rank: Optional[int],
        tree_cache,
        send_metrics_from_scheduler,
        max_running_requests: int,
        max_total_num_tokens: int,
        get_stats: Callable,
    ) -> None:
        self.tree_cache = tree_cache
        self.send_metrics_from_scheduler = send_metrics_from_scheduler
        self.dp_rank = dp_rank
        self.attn_tp_rank = attn_tp_rank
        self.attn_cp_rank = attn_cp_rank
        self.attn_dp_rank = attn_dp_rank
        self.max_running_requests = max_running_requests
        self.max_total_num_tokens = max_total_num_tokens
        self.get_stats = get_stats
        self.enable_kv_cache_events = bool(
            kv_events_config and attn_tp_rank == 0 and attn_cp_rank == 0
        )
        if self.enable_kv_cache_events:
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config, attn_dp_rank
            )
