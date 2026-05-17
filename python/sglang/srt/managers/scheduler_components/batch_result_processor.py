from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional


from sglang.srt.disaggregation.utils import DisaggregationMode

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
        DecodeKVCacheOffloadManager,
    )
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.scheduler_components.logprob_result_processor import (
        SchedulerLogprobResultProcessor,
    )
    from sglang.srt.managers.scheduler_components.metrics_reporter import (
        SchedulerMetricsReporter,
    )
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerBatchResultProcessor:
    is_generation: bool
    disaggregation_mode: "DisaggregationMode"
    enable_overlap: bool
    enable_overlap_mlx: bool
    server_args: "ServerArgs"
    model_config: "ModelConfig"
    token_to_kv_pool_allocator: "BaseTokenToKVPoolAllocator"
    tree_cache: "BasePrefixCache"
    hisparse_coordinator: Optional["HiSparseCoordinator"]
    req_to_token_pool: "ReqToTokenPool"
    decode_offload_manager: Optional["DecodeKVCacheOffloadManager"]
    metrics_collector: "SchedulerMetricsCollector"
    metrics_reporter: "SchedulerMetricsReporter"
    draft_worker: "BaseTpWorker"
    model_worker: "BaseTpWorker"
    logprob_result_processor: "SchedulerLogprobResultProcessor"
    output_streamer: "SchedulerOutputStreamer"
    abort_request: Callable
