from __future__ import annotations  # noqa: F401

import logging  # noqa: F401
import warnings  # noqa: F401
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple  # noqa: F401

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: F401
from sglang.srt.environ import envs  # noqa: F401
from sglang.srt.managers.scheduler_components.pool_stats_observer import (  # noqa: F401
    PoolStats,
    SchedulerPoolStatsObserver,
)
from sglang.srt.utils.common import ceil_align, raise_error_or_warn  # noqa: F401

logger = logging.getLogger(__name__)


class SchedulerInvariantChecker:
    """KV pool / req pool / tree_cache memory invariant checks.
    Composition target on Scheduler (``self.invariant_checker``)."""

    def __init__(
        self,
        *,
        is_hybrid_swa: bool,
        is_hybrid_ssm: bool,
        disaggregation_mode,
        page_size: int,
        full_tokens_per_layer,
        swa_tokens_per_layer,
        max_total_num_tokens: int,
        server_args,
        tree_cache,
        token_to_kv_pool_allocator,
        req_to_token_pool,
        pool_stats_observer: SchedulerPoolStatsObserver,
        get_last_batch: Callable,
        get_running_batch: Callable,
        get_pool_stats: Callable,
    ) -> None:
        self.is_hybrid_swa = is_hybrid_swa
        self.is_hybrid_ssm = is_hybrid_ssm
        self.disaggregation_mode = disaggregation_mode
        self.page_size = page_size
        self.full_tokens_per_layer = full_tokens_per_layer
        self.swa_tokens_per_layer = swa_tokens_per_layer
        self.max_total_num_tokens = max_total_num_tokens
        self.server_args = server_args
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.req_to_token_pool = req_to_token_pool
        self.pool_stats_observer = pool_stats_observer
        self.get_last_batch = get_last_batch
        self.get_running_batch = get_running_batch
        self.get_pool_stats = get_pool_stats
        self.count_req_pool_leak_warnings: int = 0
        self.count_memory_leak_warnings: int = 0
