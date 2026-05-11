from __future__ import annotations  # noqa: F401

import logging  # noqa: F401
import time  # noqa: F401
from typing import Callable, Optional  # noqa: F401

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: F401
from sglang.srt.managers.io_struct import (  # noqa: F401
    DisaggregationMetrics,
    GetLoadsReqInput,
    GetLoadsReqOutput,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)

logger = logging.getLogger(__name__)


class SchedulerLoadInquirer:
    """``/v1/loads`` RPC handler. Composition target on Scheduler
    (``self.load_inquirer``)."""

    def __init__(
        self,
        *,
        disaggregation_mode,
        ps,
        max_total_num_tokens: int,
        max_running_requests: int,
        enable_lora: bool,
        pool_stats_observer,
        tp_worker,
        token_to_kv_pool_allocator,
        spec_algorithm,
        get_running_batch: Callable,
        get_waiting_queue: Callable,
        get_stats: Callable,
        get_chunked_req: Callable,
        get_disagg_prefill_bootstrap_queue: Callable,
        get_disagg_prefill_inflight_queue: Callable,
        get_disagg_decode_prealloc_queue: Callable,
        get_disagg_decode_transfer_queue: Callable,
        get_spec_total_num_accepted_tokens: Callable,
        get_spec_total_num_forward_ct: Callable,
    ) -> None:
        self.disaggregation_mode = disaggregation_mode
        self.ps = ps
        self.max_total_num_tokens = max_total_num_tokens
        self.max_running_requests = max_running_requests
        self.enable_lora = enable_lora
        self.pool_stats_observer = pool_stats_observer
        self.tp_worker = tp_worker
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.spec_algorithm = spec_algorithm
        self.get_running_batch = get_running_batch
        self.get_waiting_queue = get_waiting_queue
        self.get_stats = get_stats
        self.get_chunked_req = get_chunked_req
        self.get_disagg_prefill_bootstrap_queue = get_disagg_prefill_bootstrap_queue
        self.get_disagg_prefill_inflight_queue = get_disagg_prefill_inflight_queue
        self.get_disagg_decode_prealloc_queue = get_disagg_decode_prealloc_queue
        self.get_disagg_decode_transfer_queue = get_disagg_decode_transfer_queue
        self.get_spec_total_num_accepted_tokens = get_spec_total_num_accepted_tokens
        self.get_spec_total_num_forward_ct = get_spec_total_num_forward_ct
