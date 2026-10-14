from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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
from sglang.srt.runtime_context import get_disagg, get_lora, get_parallel

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_policy import SchedulePolicy
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
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLoadInquirer:
    max_total_num_tokens: int
    max_running_requests: int
    pool_stats_observer: SchedulerPoolStatsObserver
    tp_worker: BaseTpWorker
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    spec_algorithm: SpeculativeAlgorithm
    policy: SchedulePolicy
    metrics_sink: MetricsSink
    request_registry: RequestRegistry
    mode_owners: ModeOwners
    step_counters: StepCounters

    def get_num_waiting_uncached_tokens(self) -> int:
        """Estimate input tokens waiting for prefill compute."""
        if (
            DisaggregationMode(get_disagg().disaggregation_mode)
            == DisaggregationMode.DECODE
        ):
            return 0
        lists = self.request_registry.request_lists()
        waiting_queue_prefix_matched = self.policy.waiting_queue_prefix_matched(
            lists["waiting"].reqs
        )
        cache_miss_rate = 1.0 - self.metrics_sink.recent_cache_hit_rate
        num_tokens = 0
        for req in lists["waiting"].reqs:
            if waiting_queue_prefix_matched:
                num_tokens += max(0, req.seqlen - req.num_matched_prefix_tokens)
            else:
                num_tokens += int(req.seqlen * cache_miss_rate)
        for cr in lists["chunked"].reqs:
            num_tokens += max(0, cr.seqlen - cr.prefix_len)
        return num_tokens

    def get_loads(self) -> LoadSnapshot:
        """Build the per-DP-rank load snapshot for DP balancing and /v1/loads."""
        disaggregation_mode = DisaggregationMode(get_disagg().disaggregation_mode)
        stats = self.metrics_sink.stats
        lists = self.request_registry.request_lists()
        num_running_reqs = len(lists["decoding"].reqs)

        queued = [req for entry in lists.values() if entry.queued for req in entry.reqs]
        waiting_queues = [queued]
        pending_token_queues = [queued]
        awaiting_kv_tokens = 0
        if disaggregation_mode == DisaggregationMode.PREFILL:
            prefill_bootstrap_queue = lists["bootstrap"].reqs
            waiting_queues.append(prefill_bootstrap_queue)
            pending_token_queues.append(prefill_bootstrap_queue)
        elif disaggregation_mode == DisaggregationMode.DECODE:
            decode_prealloc_queue = lists["prealloc"].reqs
            decode_transfer_queue = lists["transfer"].reqs
            decode_retracted_queue = lists["retracted"].reqs
            waiting_queues.append(decode_prealloc_queue)
            waiting_queues.append(decode_transfer_queue)
            waiting_queues.append(decode_retracted_queue)
            # In disaggregated decode, transfer-queue requests and ready
            # requests have already pre-allocated decode-side KV slots, so
            # they are already included in num_used_tokens.
            pending_token_queues = [decode_prealloc_queue, decode_retracted_queue]
            # KV not yet arrived from the prefill side.
            awaiting_kv_tokens = sum(
                req.seqlen
                for queue in (decode_prealloc_queue, decode_transfer_queue)
                for req in queue
            )

        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)
        num_used_tokens, kv_token_usage = (
            self.pool_stats_observer.get_pool_stats().get_kv_token_stats()
        )
        num_total_tokens = num_used_tokens + sum(
            req.seqlen for queue in pending_token_queues for req in queue
        )
        num_active_tokens = max(0, num_total_tokens - awaiting_kv_tokens)

        memory = None
        try:
            memory = MemoryMetrics(
                weight_gb=round(self.tp_worker.model_runner.weight_load_mem_usage, 3),
                kv_cache_gb=round(
                    self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                ),
                graph_gb=round(sum(self.tp_worker.graph_memory_usage.values()), 3),
                token_capacity=int(self.max_total_num_tokens),
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
        prefill_bootstrap = prefill_inflight = 0
        decode_prealloc = decode_transfer = decode_retracted = 0
        decode_prealloc_ready = 0
        num_prealloc_ready_tokens = 0
        if disaggregation_mode == DisaggregationMode.PREFILL:
            mode_str = "prefill"
            prefill_bootstrap = len(lists["bootstrap"].reqs)
            prefill_inflight = len(lists["transfer"].reqs)
        elif disaggregation_mode == DisaggregationMode.DECODE:
            mode_str = "decode"
            decode_prealloc = len(lists["prealloc"].reqs)
            decode_transfer = len(lists["transfer"].reqs)
            decode_retracted = len(lists["retracted"].reqs)
            ready_reqs = [
                decode_req.req
                for decode_req in self.mode_owners.disagg_decode.prealloc_queue.queue
                if decode_req.waiting_for_input
            ]
            decode_prealloc_ready = len(ready_reqs)
            num_prealloc_ready_tokens = sum(req.seqlen for req in ready_reqs)
        disaggregation = DisaggregationMetrics(
            mode=mode_str,
            prefill_bootstrap_queue_reqs=prefill_bootstrap,
            prefill_inflight_queue_reqs=prefill_inflight,
            decode_prealloc_queue_reqs=decode_prealloc,
            decode_transfer_queue_reqs=decode_transfer,
            decode_retracted_queue_reqs=decode_retracted,
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
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
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
