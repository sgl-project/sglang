from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.load_snapshot import (
    DisaggregationMetrics,
    LoadSnapshot,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)
from sglang.srt.runtime_context import get_disagg, get_lora, get_parallel, get_spec
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_policy import SchedulePolicy
    from sglang.srt.managers.scheduler_components.bundles import (
        KvCapacity,
        KvResources,
        Workers,
    )
    from sglang.srt.managers.scheduler_components.metrics_reporter import (
        MetricsSink,
    )
    from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
    from sglang.srt.managers.scheduler_components.pool_stats_observer import (
        SchedulerPoolStatsObserver,
    )
    from sglang.srt.managers.scheduler_components.request_registry import (
        RequestRegistry,
    )
    from sglang.srt.managers.scheduler_components.step_counters import StepCounters
logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLoadInquirer:
    kv: KvResources
    capacity: KvCapacity
    pool_stats_observer: SchedulerPoolStatsObserver
    workers: Workers
    policy: SchedulePolicy
    metrics_sink: MetricsSink
    request_registry: RequestRegistry
    mode_owners: ModeOwners
    step_counters: StepCounters
    spec_algorithm: SpeculativeAlgorithm = field(
        init=False,
        default_factory=lambda: SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        ),
    )

    def get_num_waiting_uncached_tokens(self) -> int:
        """Estimate input tokens waiting for prefill compute."""
        if (
            DisaggregationMode(get_disagg().disaggregation_mode)
            == DisaggregationMode.DECODE
        ):
            return 0
        registry = self.request_registry
        queued = registry.queued_reqs()
        waiting_queue_prefix_matched = self.policy.waiting_queue_prefix_matched(queued)
        cache_miss_rate = 1.0 - self.metrics_sink.recent_cache_hit_rate
        num_tokens = 0
        for req in queued:
            if waiting_queue_prefix_matched:
                matched = req.num_matched_prefix_tokens
                uncached = max(0, req.seqlen - matched)
                span = self.kv.tree_cache.waiting_storage_span(req.cache_request_handle)
                if span is not None:
                    storage_tokens = max(
                        0, min(span[1], req.seqlen) - max(span[0], matched)
                    )
                    uncached -= storage_tokens
            else:
                # The historical hit rate already includes admitted storage hits.
                uncached = int(req.seqlen * cache_miss_rate)
            num_tokens += uncached
        # The other waiting requests that own no KV yet (PD prefill
        # bootstrap) were never prefix-matched.
        for req in registry.reqs(
            lambda entry: (
                entry.counts_queued and not entry.holds_kv and not entry.queued
            )
        ):
            num_tokens += int(req.seqlen * cache_miss_rate)
        for cr in registry.reqs(lambda entry: entry.partly_prefilled):
            num_tokens += max(0, cr.seqlen - cr.prefix_len)
        return num_tokens

    def get_loads(self) -> LoadSnapshot:
        """Build the per-DP-rank load snapshot for DP balancing and /v1/loads."""
        disaggregation_mode = DisaggregationMode(get_disagg().disaggregation_mode)
        stats = self.metrics_sink.stats
        registry = self.request_registry
        num_running_reqs = len(registry.running_reqs())

        queued = registry.queued_reqs()
        num_waiting_reqs = len(registry.reqs(lambda entry: entry.counts_queued))
        # The waiting requests that own KV (PD decode's transfer and ready
        # requests) are in num_used_tokens already.
        pending_reqs = registry.reqs(
            lambda entry: entry.counts_queued and not entry.holds_kv
        )
        awaiting_kv_tokens = sum(
            req.seqlen for req in registry.reqs(lambda entry: entry.awaits_kv)
        )
        num_used_tokens, kv_token_usage = (
            self.pool_stats_observer.get_pool_stats().get_kv_token_stats()
        )
        num_total_tokens = num_used_tokens + sum(req.seqlen for req in pending_reqs)
        num_active_tokens = max(0, num_total_tokens - awaiting_kv_tokens)

        memory = None
        try:
            memory = MemoryMetrics(
                weight_gb=round(
                    self.workers.tp_worker.model_runner.weight_load_mem_usage, 3
                ),
                kv_cache_gb=round(
                    self.kv.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                ),
                graph_gb=round(
                    sum(self.workers.tp_worker.graph_memory_usage.values()), 3
                ),
                token_capacity=int(self.capacity.widened_max_total_num_tokens),
            )
        except (AttributeError, TypeError) as e:
            logger.debug(f"Memory metrics not available: {e}")

        speculative = None
        metrics_sink = self.metrics_sink
        if (
            not self.spec_algorithm.is_none()
            and metrics_sink.spec_total_num_forward_ct > 0
        ):
            speculative = SpeculativeMetrics(
                accept_length=(
                    metrics_sink.spec_total_num_accept_tokens
                    / metrics_sink.spec_total_num_forward_ct
                ),
                accept_rate=stats.spec_accept_rate,
            )

        lora = None
        if get_lora().enable_lora:
            lora = LoRAMetrics(
                slots_used=stats.lora_pool_slots_used,
                slots_total=stats.lora_pool_slots_total,
                utilization=stats.lora_pool_utilization,
            )

        mode_str = "null"
        queue_metrics = {}
        decode_prealloc_ready = 0
        num_prealloc_ready_tokens = 0
        if disaggregation_mode == DisaggregationMode.PREFILL:
            mode_str = "prefill"
            queue_metrics = self.mode_owners.disagg_prefill.load_queue_metrics()
        elif disaggregation_mode == DisaggregationMode.DECODE:
            mode_str = "decode"
            disagg_decode = self.mode_owners.disagg_decode
            queue_metrics = disagg_decode.load_queue_metrics()
            ready_reqs = disagg_decode.prealloc_ready_reqs()
            decode_prealloc_ready = len(ready_reqs)
            num_prealloc_ready_tokens = sum(req.seqlen for req in ready_reqs)
        disaggregation = DisaggregationMetrics(
            mode=mode_str,
            **queue_metrics,
            kv_transfer_speed_gb_s=stats.kv_transfer_speed_gb_s,
            kv_transfer_latency_ms=stats.kv_transfer_latency_ms,
        )

        queues = QueueMetrics(
            waiting=len(queued),
            grammar=stats.num_grammar_queue_reqs,
            paused=stats.num_paused_reqs,
            retracted=stats.num_retracted_reqs,
            prealloc_ready=decode_prealloc_ready,
        )

        totals = self.step_counters.decode_moment_totals
        decode_moments = list(totals) if totals[0] > 0 else None

        return LoadSnapshot(
            dp_rank=int(get_parallel().dp_rank)
            if get_parallel().dp_rank is not None
            else 0,
            timestamp=time.time(),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_waiting_uncached_tokens=self.get_num_waiting_uncached_tokens(),
            num_used_tokens=num_used_tokens,
            num_total_tokens=num_total_tokens,
            num_active_tokens=num_active_tokens,
            num_prealloc_ready_tokens=num_prealloc_ready_tokens,
            max_total_num_tokens=self.capacity.widened_max_total_num_tokens,
            max_running_requests=self.capacity.max_running_requests,
            token_usage=round(kv_token_usage, 4),
            gen_throughput=round(stats.gen_throughput, 2),
            cache_hit_rate=round(stats.cache_hit_rate, 4),
            utilization=round(stats.utilization, 4),
            memory=memory,
            speculative=speculative,
            lora=lora,
            disaggregation=disaggregation,
            queues=queues,
            total_prefill_uncached_tokens=self.step_counters.total_prefill_uncached_tokens,
            total_prefill_busy_us=self.step_counters.total_prefill_busy_us,
            decode_moments=decode_moments,
        )
