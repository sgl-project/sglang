from __future__ import annotations  # noqa: F401

import logging  # noqa: F401
from typing import TYPE_CHECKING, Callable, Optional  # noqa: F401

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: F401
from sglang.srt.observability.scheduler_metrics_mixin import (  # noqa: F401
    SchedulerMetricsMixin,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler  # noqa: F401


logger = logging.getLogger(__name__)


class SchedulerMetricsReporter:
    """Prometheus / Stats hot-path. Composition target on Scheduler
    (``self.metrics_reporter``)."""

    def __init__(
        self,
        *,
        ps,
        server_args,
        disaggregation_mode,
        spec_algorithm,
        metrics_collector,
        enable_priority_scheduling: bool,
        enable_lora,
        enable_hierarchical_cache: bool,
        max_running_requests: int,
        max_total_num_tokens: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank,
        attn_tp_rank: int,
        moe_ep_rank: int,
        device: str,
        model_config,
        max_running_requests_under_SLO,
        waiting_queue,
        grammar_manager,
        mm_receiver,
        tree_cache,
        tp_worker,
        draft_worker,
        disagg_prefill_bootstrap_queue,
        disagg_prefill_inflight_queue,
        disagg_decode_prealloc_queue,
        disagg_decode_transfer_queue,
        kv_events_publisher,
        pool_stats_observer,
        get_running_batch: Callable,
        get_forward_ct: Callable,
        get_running_mbs: Callable,
        get_last_batch: Callable,
        get_grammar_manager: Callable,
        get_disaggregation_mode: Callable,
        get_disagg_prefill_bootstrap_queue: Callable,
        get_disagg_prefill_inflight_queue: Callable,
        get_disagg_decode_prealloc_queue: Callable,
        get_disagg_decode_transfer_queue: Callable,
    ) -> None:
        # Owned counters (ownership migration from Scheduler).
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        # Stash deps + sisters + Callable getters.
        self.ps = ps
        self.server_args = server_args
        self.disaggregation_mode = disaggregation_mode
        self.spec_algorithm = spec_algorithm
        self.metrics_collector = metrics_collector
        self.enable_priority_scheduling = enable_priority_scheduling
        self.enable_lora = enable_lora
        self.enable_hierarchical_cache = enable_hierarchical_cache
        self.max_running_requests = max_running_requests
        self.max_total_num_tokens = max_total_num_tokens
        self.device = device
        self.model_config = model_config
        self.max_running_requests_under_SLO = max_running_requests_under_SLO
        self.waiting_queue = waiting_queue
        self.grammar_manager = grammar_manager
        self.mm_receiver = mm_receiver
        self.tree_cache = tree_cache
        self.tp_worker = tp_worker
        self.draft_worker = draft_worker
        self.disagg_prefill_bootstrap_queue = disagg_prefill_bootstrap_queue
        self.disagg_prefill_inflight_queue = disagg_prefill_inflight_queue
        self.disagg_decode_prealloc_queue = disagg_decode_prealloc_queue
        self.disagg_decode_transfer_queue = disagg_decode_transfer_queue
        self.kv_events_publisher = kv_events_publisher
        self.pool_stats_observer = pool_stats_observer
        self.get_running_batch = get_running_batch
        self.get_forward_ct = get_forward_ct
        self.get_running_mbs = get_running_mbs
        self.get_last_batch = get_last_batch
        self.get_grammar_manager = get_grammar_manager
        self.get_disaggregation_mode = get_disaggregation_mode
        self.get_disagg_prefill_bootstrap_queue = get_disagg_prefill_bootstrap_queue
        self.get_disagg_prefill_inflight_queue = get_disagg_prefill_inflight_queue
        self.get_disagg_decode_prealloc_queue = get_disagg_decode_prealloc_queue
        self.get_disagg_decode_transfer_queue = get_disagg_decode_transfer_queue
        # Run the original init_metrics body via the qualified staticmethod
        # form — methods still live on SchedulerMetricsMixin during prep;
        # the upcoming ``-move`` commit cuts + pastes them into this class
        # and the qualified prefix collapses to ``self.init_metrics(...)``.
        SchedulerMetricsMixin.init_metrics(self, tp_rank, pp_rank, dp_rank)
        # ``install_device_timer_on_runners`` was originally called from
        # Scheduler.__init__ right after init_model_worker; we invoke it
        # here so callers don't need a separate hook.
        SchedulerMetricsMixin.install_device_timer_on_runners(self)
