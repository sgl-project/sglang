from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)

import zmq

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


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


@dataclass(kw_only=True, slots=True)
class SchedulerKvEventsPublisher:
    kv_events_config: Optional[str]
    ps: "ParallelState"
    attn_tp_rank: int
    attn_cp_rank: int
    attn_dp_rank: int
    dp_rank: Optional[int]
    tree_cache: "BasePrefixCache"
    send_metrics_from_scheduler: Optional["zmq.Socket"]
    max_running_requests: int
    max_total_num_tokens: int
    get_stats: Callable
    enable_kv_cache_events: bool = False
    kv_event_publisher: Any = None

    def __post_init__(self) -> None:
        from sglang.srt.observability.scheduler_metrics_mixin import (
            SchedulerMetricsMixin,
        )

        SchedulerMetricsMixin.init_kv_events(self, self.kv_events_config)
