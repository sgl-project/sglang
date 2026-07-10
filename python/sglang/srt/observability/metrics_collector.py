# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for Prometheus Metrics Collection."""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.utils import exponential_buckets, generate_buckets
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.gauge_histogram import GaugeHistogram

if TYPE_CHECKING:
    from prometheus_client import Gauge

    from sglang.srt.managers.schedule_batch import Req

SGLANG_TEST_REQUEST_TIME_STATS = get_bool_env_var("SGLANG_TEST_REQUEST_TIME_STATS")

logger = logging.getLogger(__name__)


@dataclass
class QueueCount:
    """Holds both the total count and optional per-priority breakdown for a queue."""

    total: int = 0
    by_priority: Optional[Dict[int, int]] = None

    @classmethod
    def from_reqs(cls, reqs: List[Req], enable_priority_scheduling: bool = False):
        # NOTE: If requests have priority=None (no --default-priority-value set),
        # Counter will produce {None: N}, resulting in priority="None" Prometheus labels.
        # Set --default-priority-value when enabling priority scheduling to avoid this.
        by_priority = (
            dict(Counter(req.priority for req in reqs))
            if enable_priority_scheduling
            else None
        )
        return cls(total=len(reqs), by_priority=by_priority)


@dataclass
class SchedulerStats:
    # Basics
    num_running_reqs: QueueCount = field(default_factory=QueueCount)
    num_queue_reqs: QueueCount = field(default_factory=QueueCount)
    num_grammar_queue_reqs: int = 0
    gen_throughput: float = 0.0
    cache_hit_rate: float = 0.0
    decode_sum_seq_lens: int = 0

    # Memory pool usage ratios (0.0–1.0).
    # Each pool tracks: used = total - available - evictable, usage = used / total.
    #
    # token_usage:      max(full, swa, mamba) — the bottleneck across all pools.
    #                   FIXME: misleadingly named "token_usage"; rename requires API deprecation.
    # full_token_usage: full-attention KV cache pool usage (always active).
    # swa_token_usage:  sliding-window attention KV cache pool usage (hybrid SWA models only, e.g. Gemma2).
    # mamba_usage:      Mamba SSM state pool usage (hybrid SSM models only, e.g. Jamba).
    token_usage: float = 0.0
    full_token_usage: float = 0.0
    swa_token_usage: float = 0.0
    mamba_usage: float = 0.0

    # Absolute token counts for the full-attention KV cache pool.
    # Invariant: kv_available_tokens + kv_evictable_tokens + kv_used_tokens <= max_total_num_tokens
    # (the gap accounts for protected/session-held tokens not exposed here).
    # max_total_num_tokens is emitted once at startup via emit_constants.
    #
    # kv_available_tokens:  free (unallocated) slots in the pool.
    # kv_evictable_tokens:  slots holding radix-cached KV data that can be evicted for new requests.
    # kv_used_tokens:       actively used slots (locked by running requests). Equals full_num_used.
    # num_used_tokens:      max(full_num_used, swa_num_used) for hybrid-SWA models, else full_num_used.
    #                       Does NOT include the mamba pool.
    num_used_tokens: int = 0
    kv_available_tokens: int = 0
    kv_evictable_tokens: int = 0
    kv_used_tokens: int = 0

    swa_available_tokens: int = 0
    swa_evictable_tokens: int = 0
    swa_used_tokens: int = 0
    mamba_available_tokens: int = 0
    mamba_evictable_tokens: int = 0
    mamba_used_tokens: int = 0

    # Speculative decoding
    spec_accept_length: float = 0.0
    spec_accept_rate: float = 0.0
    # Adaptive speculative decoding (currently active tier).
    spec_num_steps: int = 0
    spec_num_draft_tokens: int = 0

    # Retract
    num_retracted_reqs: int = 0
    num_paused_reqs: int = 0

    # PD disaggregation
    num_prefill_bootstrap_queue_reqs: QueueCount = field(default_factory=QueueCount)
    num_prefill_inflight_queue_reqs: QueueCount = field(default_factory=QueueCount)
    num_decode_prealloc_queue_reqs: QueueCount = field(default_factory=QueueCount)
    num_decode_transfer_queue_reqs: QueueCount = field(default_factory=QueueCount)
    kv_transfer_speed_gb_s: float = 0.0
    kv_transfer_latency_ms: float = 0.0
    pending_prealloc_token_usage: float = 0.0

    # Utilization
    utilization: float = 0.0
    fwd_occupancy: float = float("nan")

    # Scheduler policy
    new_token_ratio: float = 0.0

    # CUDA graph
    is_cuda_graph: int = 0

    # LoRA pool metrics
    lora_pool_slots_used: int = 0
    lora_pool_slots_total: int = 0
    lora_pool_utilization: float = 0.0

    # HiCache metrics
    hicache_host_used_tokens: int = 0
    hicache_host_total_tokens: int = 0

    # Streaming session metrics
    num_streaming_sessions: int = 0
    streaming_session_held_tokens: int = 0

    # Routing key metrics
    num_unique_running_routing_keys: int = 0
    routing_key_running_req_counts: List[int] = field(default_factory=list)
    routing_key_all_req_counts: List[int] = field(default_factory=list)


ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS = [1, 2, 3, 5, 7, 10, 20, 50, 100, 200]


def compute_routing_key_stats(routing_keys: List[Optional[str]]) -> tuple:
    """Returns (num_unique_keys, per_key_counts)."""
    from collections import Counter

    key_counts = Counter(k for k in routing_keys if k is not None)
    return len(key_counts), list(key_counts.values())


@dataclass
class DPCooperationInfo:
    # Users can derive that, except for cases with idle, num_decode_ranks=world_size-num_prefill_ranks
    # We do not provide `num_decode_ranks` to avoid cardinality explosion.
    num_prefill_ranks: int

    @staticmethod
    def create(forward_modes: List[int]):
        return DPCooperationInfo(
            # Count ranks that are doing any extend-like work.
            # With overlap scheduling, prefill can appear as MIXED rather than EXTEND.
            num_prefill_ranks=sum(
                1 for mode in forward_modes if ForwardMode(mode).is_extend()
            ),
        )

    def to_labels(self):
        return dataclasses.asdict(self)


# Role keys used by ServerArgs.stat_loggers to look up collector overrides.
# Embedded-use callers (e.g. Ray Serve LLM) pass {"scheduler": MyClass, ...} on
# ServerArgs and the five collector instantiation sites pick the right class.
STAT_LOGGER_ROLE_SCHEDULER = "scheduler"
STAT_LOGGER_ROLE_TOKENIZER = "tokenizer"
STAT_LOGGER_ROLE_STORAGE = "storage"
STAT_LOGGER_ROLE_RADIX_CACHE = "radix_cache"
STAT_LOGGER_ROLE_EXPERT_DISPATCH = "expert_dispatch"


def resolve_collector_class(
    server_args: Optional[ServerArgs], role: str, default_cls: type
) -> type:
    """Return the subclass registered for `role` on `server_args.stat_loggers`,
    or `default_cls` if none is registered. Tolerates `server_args=None` and
    `stat_loggers=None`."""
    if server_args is None:
        return default_cls
    stat_loggers = getattr(server_args, "stat_loggers", None)
    if not stat_loggers:
        return default_cls
    return stat_loggers.get(role, default_cls)


class _StatLoggerDIMixin:
    """Shared DI override hooks for all *MetricsCollector classes.

    Subclasses (e.g. a Ray-backed wrapper) replace these class attributes with
    classes that mirror the prometheus_client API but emit through a different
    backend. ``None`` keeps the prometheus_client default.
    """

    _counter_cls = None
    _gauge_cls = None
    _histogram_cls = None
    _summary_cls = None


@dataclass(kw_only=True, frozen=True, slots=True)
class SchedulerMetricsCollectorContext:
    enable_metrics: bool
    is_stats_logging_rank: bool
    current_scheduler_metrics_enabled: bool
    enable_kv_cache_events: bool
    collector: Optional[SchedulerMetricsCollector]


class SchedulerMetricsCollector(_StatLoggerDIMixin):

    def __init__(
        self,
        labels: Dict[str, str],
        enable_lora: bool = False,
        enable_hierarchical_cache: bool = False,
        enable_streaming_session: bool = False,
        server_args: Optional[ServerArgs] = None,
    ) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter as _PromCounter
        from prometheus_client import Gauge as _PromGauge
        from prometheus_client import Histogram as _PromHistogram
        from prometheus_client import Summary as _PromSummary

        Counter = self._counter_cls or _PromCounter
        Gauge = self._gauge_cls or _PromGauge
        Histogram = self._histogram_cls or _PromHistogram
        Summary = self._summary_cls or _PromSummary

        self.labels = labels
        self.enable_lora = enable_lora
        self.enable_hierarchical_cache = enable_hierarchical_cache
        self.enable_streaming_session = enable_streaming_session
        self.last_log_time = time.perf_counter()
        self._known_priorities: Set[int] = set()

        # =================================================================
        # Basics
        # =================================================================
        self.num_running_reqs = Gauge(
            name="sglang:num_running_reqs",
            documentation="The number of running requests.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_queue_reqs = Gauge(
            name="sglang:num_queue_reqs",
            documentation="The number of requests in the waiting queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_grammar_queue_reqs = Gauge(
            name="sglang:num_grammar_queue_reqs",
            documentation="The number of requests in the grammar waiting queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.gen_throughput = Gauge(
            name="sglang:gen_throughput",
            documentation="The generation throughput (token/s).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.cache_hit_rate = Gauge(
            name="sglang:cache_hit_rate",
            documentation="The prefix cache hit rate.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.decode_sum_seq_lens = Gauge(
            name="sglang:decode_sum_seq_lens",
            documentation="The sum of all sequence lengths in decode.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # Memory pool usage ratios
        # =================================================================
        self.token_usage = Gauge(
            name="sglang:token_usage",
            documentation="The token usage.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.full_token_usage = Gauge(
            name="sglang:full_token_usage",
            documentation="The token usage for full attention layers.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.swa_token_usage = Gauge(
            name="sglang:swa_token_usage",
            documentation="The token usage for SWA layers.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.mamba_usage = Gauge(
            name="sglang:mamba_usage",
            documentation="The token usage for Mamba layers.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # Absolute token counts
        # =================================================================
        self.num_used_tokens = Gauge(
            name="sglang:num_used_tokens",
            documentation="The number of used tokens.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_available_tokens = Gauge(
            name="sglang:kv_available_tokens",
            documentation="Number of free token slots in the KV cache pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_evictable_tokens = Gauge(
            name="sglang:kv_evictable_tokens",
            documentation="Number of evictable (radix-cached) token slots in the KV cache pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_used_tokens = Gauge(
            name="sglang:kv_used_tokens",
            documentation="Number of actively used token slots in the KV cache pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.swa_available_tokens = Gauge(
            name="sglang:swa_available_tokens",
            documentation="Number of free token slots in the SWA pool (hybrid-SWA only).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.swa_evictable_tokens = Gauge(
            name="sglang:swa_evictable_tokens",
            documentation="Number of evictable (radix-cached) token slots in the SWA pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.swa_used_tokens = Gauge(
            name="sglang:swa_used_tokens",
            documentation="Number of actively used token slots in the SWA pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.mamba_available_tokens = Gauge(
            name="sglang:mamba_available_tokens",
            documentation="Number of free state slots in the mamba SSM pool (hybrid-SSM only).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.mamba_evictable_tokens = Gauge(
            name="sglang:mamba_evictable_tokens",
            documentation="Number of evictable (radix-cached) state slots in the mamba SSM pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.mamba_used_tokens = Gauge(
            name="sglang:mamba_used_tokens",
            documentation="Number of actively used state slots in the mamba SSM pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # Weight update
        # =================================================================
        self.weight_load_duration_seconds = Gauge(
            name="sglang:weight_load_duration_seconds",
            documentation=(
                "Wall time of the most recent update_weights_from_<source> call on "
                "this scheduler rank (seconds). `source` label is one of: disk, "
                "distributed, tensor, ipc. Event-detection via "
                "changes(...[<range>]) > 0 — no separate counter needed."
            ),
            labelnames=[*labels.keys(), "source"],
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # Speculative decoding
        # =================================================================
        self.spec_accept_length = Gauge(
            name="sglang:spec_accept_length",
            documentation="Mean acceptance length of speculative decoding (accepted drafts + bonus token per forward).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.spec_accept_rate = Gauge(
            name="sglang:spec_accept_rate",
            documentation="Speculative acceptance rate (`accepted drafts / proposed drafts` in batch).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.spec_num_steps = Gauge(
            name="sglang:spec_num_steps",
            documentation="Currently active speculative_num_steps.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.spec_num_draft_tokens = Gauge(
            name="sglang:spec_num_draft_tokens",
            documentation="Currently active speculative_num_draft_tokens (decouples from steps under topk>1).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # Retract
        # =================================================================
        # TODO maybe remove this old gauge in favor of the new counter
        self.num_retracted_reqs = Gauge(
            name="sglang:num_retracted_reqs",
            documentation="The number of retracted requests.",
            labelnames=labels.keys(),
        )
        self.num_retracted_reqs_total = Counter(
            # The name is `requests` instead of `reqs` to avoid dup name error
            name="sglang:num_retracted_requests_total",
            documentation="Total number of retracted requests.",
            labelnames=labels.keys(),
        )
        self.num_retracted_input_tokens_total = Counter(
            name="sglang:num_retracted_input_tokens_total",
            documentation="Total number of retracted input tokens.",
            labelnames=labels.keys(),
        )
        self.num_retracted_output_tokens_total = Counter(
            name="sglang:num_retracted_output_tokens_total",
            documentation="Total number of retracted output tokens.",
            labelnames=labels.keys(),
        )
        self.num_paused_reqs = Gauge(
            name="sglang:num_paused_reqs",
            documentation="The number of paused requests by async weight sync.",
            labelnames=labels.keys(),
        )

        # =================================================================
        # PD disaggregation
        # =================================================================
        self.num_prefill_bootstrap_queue_reqs = Gauge(
            name="sglang:num_prefill_bootstrap_queue_reqs",
            documentation="The number of requests in the prefill bootstrap queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_prefill_inflight_queue_reqs = Gauge(
            name="sglang:num_prefill_inflight_queue_reqs",
            documentation="The number of requests in the prefill inflight queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_decode_prealloc_queue_reqs = Gauge(
            name="sglang:num_decode_prealloc_queue_reqs",
            documentation="The number of requests in the decode prealloc queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_decode_transfer_queue_reqs = Gauge(
            name="sglang:num_decode_transfer_queue_reqs",
            documentation="The number of requests in the decode transfer queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_transfer_speed_gb_s = Histogram(
            name="sglang:kv_transfer_speed_gb_s",
            documentation="Histogram of KV cache transfer speed in GB/s.",
            labelnames=labels.keys(),
            buckets=(0.1, 0.5, 1, 5, 10, 25, 50, 100, 200, 400),
        )
        self.kv_transfer_latency_ms = Histogram(
            name="sglang:kv_transfer_latency_ms",
            documentation="Histogram of KV cache transfer latency in ms.",
            labelnames=labels.keys(),
            buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
        )
        self.pending_prealloc_token_usage = Gauge(
            name="sglang:pending_prealloc_token_usage",
            documentation="The token usage for pending preallocated tokens (not preallocated yet).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_bootstrap_failed_reqs = Counter(
            name="sglang:num_bootstrap_failed_reqs_total",
            documentation="The number of bootstrap failed requests.",
            labelnames=labels.keys(),
        )
        self.num_transfer_failed_reqs = Counter(
            name="sglang:num_transfer_failed_reqs_total",
            documentation="The number of transfer failed requests.",
            labelnames=labels.keys(),
        )
        self.num_prefill_retries_total = Counter(
            name="sglang:num_prefill_retries_total",
            documentation="Total number of prefill retries.",
            labelnames=labels.keys(),
        )
        self.kv_transfer_bootstrap_ms = Histogram(
            name="sglang:kv_transfer_bootstrap_ms",
            documentation="Histogram of KV transfer bootstrap time in ms.",
            labelnames=labels.keys(),
            buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500),
        )
        self.kv_transfer_alloc_ms = Histogram(
            name="sglang:kv_transfer_alloc_ms",
            documentation="Histogram of KV transfer allocation waiting time in ms.",
            labelnames=labels.keys(),
            buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500),
        )
        self.kv_transfer_total_mb = Histogram(
            name="sglang:kv_transfer_total_mb",
            documentation="Histogram of KV cache transfer size in MB.",
            labelnames=labels.keys(),
            buckets=(1, 5, 10, 50, 100, 500, 1000, 5000, 10000),
        )

        # =================================================================
        # FutureMap overlap relay
        # =================================================================
        self.num_future_map_stash_total = Counter(
            name="sglang:num_future_map_stash_total",
            documentation="Total number of FutureMap stash (write) operations.",
            labelnames=labels.keys(),
        )
        self.num_future_map_publish_total = Counter(
            name="sglang:num_future_map_publish_total",
            documentation="Total number of FutureMap publish operations.",
            labelnames=labels.keys(),
        )
        self.num_future_map_resolve_total = Counter(
            name="sglang:num_future_map_resolve_total",
            documentation="Total number of FutureMap resolve (read) operations.",
            labelnames=labels.keys(),
        )
        self.future_map_relay_latency_ms = Histogram(
            name="sglang:future_map_relay_latency_ms",
            documentation="Histogram of FutureMap relay operation latency in ms.",
            labelnames=labels.keys(),
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500),
        )

        # =================================================================
        # MinFreeSlotsDelayer
        # =================================================================
        self.min_free_slots_delay_total = Counter(
            name="sglang:min_free_slots_delay_total",
            documentation="Total number of prefill admissions delayed by MinFreeSlotsDelayer.",
            labelnames=labels.keys(),
        )
        self.min_free_slots_checks_total = Counter(
            name="sglang:min_free_slots_checks_total",
            documentation="Total number of MinFreeSlotsDelayer.should_delay checks.",
            labelnames=labels.keys(),
        )
        self.min_free_slots_running_bs = Histogram(
            name="sglang:min_free_slots_running_bs",
            documentation="Histogram of running batch size at MinFreeSlotsDelayer check.",
            labelnames=labels.keys(),
            buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128),
        )
        self.min_free_slots_allocatable = Histogram(
            name="sglang:min_free_slots_allocatable",
            documentation="Histogram of allocatable request slots at MinFreeSlotsDelayer check.",
            labelnames=labels.keys(),
            buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128),
        )

        # =================================================================
        # Scheduler health dashboard
        # =================================================================
        self.scheduler_loop_iterations_total = Counter(
            name="sglang:scheduler_loop_iterations_total",
            documentation="Total number of scheduler event loop iterations.",
            labelnames=labels.keys(),
        )
        self.scheduler_loop_batch_dispatches_total = Counter(
            name="sglang:scheduler_loop_batch_dispatches_total",
            documentation="Total number of event loop iterations that dispatched a batch.",
            labelnames=labels.keys(),
        )
        self.scheduler_loop_idle_total = Counter(
            name="sglang:scheduler_loop_idle_total",
            documentation="Total number of event loop iterations that went idle (no batch).",
            labelnames=labels.keys(),
        )
        self.scheduler_loop_iteration_lag_ms = Histogram(
            name="sglang:scheduler_loop_iteration_lag_ms",
            documentation="Histogram of scheduler event loop iteration duration in ms.",
            labelnames=labels.keys(),
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000),
        )
        self.scheduler_aborts_total = Counter(
            name="sglang:scheduler_aborts_total",
            documentation="Total number of aborted requests by reason.",
            labelnames={**labels, "reason": ""},
        )

        # =================================================================
        # Utilization
        # =================================================================
        self.utilization = Gauge(
            name="sglang:utilization",
            documentation="The utilization.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.fwd_occupancy = Gauge(
            name="sglang:fwd_occupancy",
            documentation="Forward pass GPU occupancy percentage.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # Scheduler policy
        # =================================================================
        self.new_token_ratio = Gauge(
            name="sglang:new_token_ratio",
            documentation="The new token ratio.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # =================================================================
        # CUDA graph
        # =================================================================
        # TODO maybe remove this old gauge in favor of the new counter
        self.is_cuda_graph = Gauge(
            name="sglang:is_cuda_graph",
            documentation="Whether the batch is using CUDA graph.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.cuda_graph_passes_total = Counter(
            name="sglang:cuda_graph_passes_total",
            documentation="Total number of forward passes categorized by CUDA graph.",
            labelnames=list(labels.keys()) + ["mode"],
        )

        # =================================================================
        # LoRA pool metrics (only created when LoRA is enabled)
        # =================================================================
        if self.enable_lora:
            self.lora_pool_slots_used = Gauge(
                name="sglang:lora_pool_slots_used",
                documentation="Number of LoRA adapter slots currently occupied in GPU memory.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )
            self.lora_pool_slots_total = Gauge(
                name="sglang:lora_pool_slots_total",
                documentation="Total number of LoRA adapter slots available (max_loras_per_batch).",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )
            self.lora_pool_utilization = Gauge(
                name="sglang:lora_pool_utilization",
                documentation="LoRA pool utilization ratio (used/total). 1.0 means pool is full.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )

        # =================================================================
        # HiCache metrics (only created when hierarchical cache is enabled)
        # =================================================================
        if self.enable_hierarchical_cache:
            self.hicache_host_used_tokens = Gauge(
                name="sglang:hicache_host_used_tokens",
                documentation="Number of tokens currently used in the host KV cache.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )
            self.hicache_host_total_tokens = Gauge(
                name="sglang:hicache_host_total_tokens",
                documentation="Total capacity of the host KV cache in tokens.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )

        # =================================================================
        # Streaming session metrics (only created when streaming sessions are enabled)
        # =================================================================
        if self.enable_streaming_session:
            self.num_streaming_sessions = Gauge(
                name="sglang:num_streaming_sessions",
                documentation="The number of streaming sessions.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )
            self.streaming_session_held_tokens = Gauge(
                name="sglang:streaming_session_held_tokens",
                documentation="The number of KV tokens currently held by streaming session slots.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )

        # =================================================================
        # Routing key metrics
        # =================================================================
        self.num_unique_running_routing_keys = Gauge(
            name="sglang:num_unique_running_routing_keys",
            documentation="Number of unique routing keys in running batch.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.routing_key_running_req_count = GaugeHistogram(
            name="sglang:routing_key_running_req_count",
            documentation="Distribution of routing keys by running request count (gt < count <= le).",
            labelnames=list(labels.keys()),
            bucket_bounds=ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS,
        )
        self.routing_key_all_req_count = GaugeHistogram(
            name="sglang:routing_key_all_req_count",
            documentation="Distribution of routing keys by running+waiting request count (gt < count <= le).",
            labelnames=list(labels.keys()),
            bucket_bounds=ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS,
        )

        # =================================================================
        # Request latency
        # =================================================================
        self.queue_time = Histogram(
            name="sglang:queue_time_seconds",
            documentation="Histogram of queueing time in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.000,
                0.001,
                0.005,
                0.010,
                0.050,
                0.100,
                0.200,
                0.500,
                1,
                2,
                3,
                4,
                5,
                10,
                15,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                1200,
                1400,
                1600,
                1800,
                2000,
                2500,
                3000,
            ],
        )
        self.per_stage_req_latency_seconds = Histogram(
            name="sglang:per_stage_req_latency_seconds",
            documentation="The latency of each stage of requests.",
            # captures latency in range [1ms - ~1191s]
            buckets=exponential_buckets(start=0.001, width=1.62, length=30),
            labelnames=list(labels.keys()) + ["stage"],
        )

        # =================================================================
        # Grammar
        # =================================================================
        self.grammar_compilation_time = Histogram(
            name="sglang:grammar_compilation_time_seconds",
            documentation="Histogram of grammar compilation time in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.0,
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.5,
                1,
                2,
                5,
                10,
                20,
                30,
                60,
                90,
                120,
                240,
            ],
        )
        self.num_grammar_cache_hit = Counter(
            name="sglang:num_grammar_cache_hit_total",
            documentation="Number of grammar cache hits.",
            labelnames=labels.keys(),
        )
        self.num_grammar_aborted = Counter(
            name="sglang:num_grammar_aborted_total",
            documentation="Number of grammar aborted requests.",
            labelnames=labels.keys(),
        )
        self.num_grammar_timeout = Counter(
            name="sglang:num_grammar_timeout_total",
            documentation="Number of grammar timeouts.",
            labelnames=labels.keys(),
        )
        self.num_grammar_total = Counter(
            name="sglang:num_grammar_total",
            documentation="Number of the total grammar requests.",
            labelnames=labels.keys(),
        )
        self.grammar_schema_count = Histogram(
            name="sglang:grammar_schema_count",
            documentation="Histogram of grammar schema count.",
            labelnames=labels.keys(),
            buckets=[
                0,
                1,
                2,
                5,
                10,
                20,
                30,
                40,
                60,
                80,
                100,
                120,
                140,
                160,
                180,
                200,
                300,
                400,
                500,
                700,
                1000,
            ],
        )
        self.grammar_ebnf_size = Histogram(
            name="sglang:grammar_ebnf_size",
            documentation="Histogram of grammar EBNF size.",
            labelnames=labels.keys(),
            buckets=[
                0,
                50,
                100,
                200,
                300,
                500,
                1000,
                2000,
                3000,
                5000,
                10000,
                20000,
                30000,
                50000,
                100000,
            ],
        )

        tree_traversal_time_buckets = [
            0.0,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1,
            2,
            5,
            10,
            15,
            30,
            60,
            90,
            120,
            240,
        ]
        self.grammar_tree_traversal_time_avg = Histogram(
            name="sglang:grammar_tree_traversal_time_avg",
            documentation="Histogram of average grammar tree traversal time in seconds.",
            labelnames=labels.keys(),
            buckets=tree_traversal_time_buckets,
        )
        self.grammar_tree_traversal_time_max = Histogram(
            name="sglang:grammar_tree_traversal_time_max",
            documentation="Histogram of max grammar tree traversal time in seconds.",
            labelnames=labels.keys(),
            buckets=tree_traversal_time_buckets,
        )

        # =================================================================
        # Execution
        # =================================================================
        if (
            labels["moe_ep_rank"] == 0
        ) and envs.SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC.get():
            self.eplb_balancedness = Summary(
                name="sglang:eplb_balancedness",
                documentation="Balancedness of MoE in expert parallelism.",
                labelnames=list(labels.keys()) + ["forward_mode"],
            )

        self.realtime_tokens_total = Counter(
            name="sglang:realtime_tokens_total",
            documentation=(
                "Total number of tokens processed (updated on each log interval). "
                "mode: prefill_compute, prefill_cache, decode."
            ),
            labelnames=list(labels.keys()) + ["mode"],
        )
        self.forward_execution_seconds_total = Counter(
            name="sglang:forward_execution_seconds_total",
            documentation=(
                "Total time that GPU is busy executing model forward passes. "
                "Refer to ForwardMode for category labels."
            ),
            labelnames=list(labels.keys()) + ["category"],
        )
        self.estimated_flops_per_gpu_total = Counter(
            name="sglang:estimated_flops_per_gpu_total",
            documentation=(
                "Estimated number of floating point operations per GPU "
                "(for Model FLOPs Utilization calculations)."
            ),
            labelnames=labels.keys(),
        )
        self.estimated_read_bytes_per_gpu_total = Counter(
            name="sglang:estimated_read_bytes_per_gpu_total",
            documentation=(
                "Estimated number of bytes read from memory per GPU "
                "(for Model FLOPs Utilization calculations)."
            ),
            labelnames=labels.keys(),
        )
        self.estimated_write_bytes_per_gpu_total = Counter(
            name="sglang:estimated_write_bytes_per_gpu_total",
            documentation=(
                "Estimated number of bytes written to memory per GPU "
                "(for Model FLOPs Utilization calculations)."
            ),
            labelnames=labels.keys(),
        )

        self.dp_cooperation_realtime_tokens_total = Counter(
            name="sglang:dp_cooperation_realtime_tokens_total",
            documentation=(
                "Total number of tokens processed with labels about DP cooperation. "
                "mode: prefill_compute, prefill_cache, decode."
            ),
            labelnames=list(labels.keys()) + ["mode", "num_prefill_ranks"],
        )
        self.dp_cooperation_forward_execution_seconds_total = Counter(
            name="sglang:dp_cooperation_forward_execution_seconds_total",
            documentation=(
                "Total time that GPU is busy executing model forward passes, "
                "with labels about DP cooperation. "
                "Refer to ForwardMode for category labels."
            ),
            labelnames=list(labels.keys()) + ["category", "num_prefill_ranks"],
        )

        # =================================================================
        # Prefill delayer
        # =================================================================
        max_delay = server_args.prefill_delayer_max_delay_passes
        self.prefill_delayer_wait_forward_passes = Histogram(
            name="sglang:prefill_delayer_wait_forward_passes",
            documentation="Histogram of forward passes waited by prefill delayer.",
            labelnames=labels.keys(),
            buckets=sorted(
                set(
                    x
                    for x in (
                        server_args.prefill_delayer_forward_passes_buckets
                        or [5, 20, 50, 100, 200]
                    )
                    if x < max_delay
                )
                # Need bucket "<=0" for zero-delay cases, and "max_delay-1" to distinguish "max_delay" timeout passes
                | {0, max_delay - 1}
            ),
        )
        self.prefill_delayer_wait_seconds = Histogram(
            name="sglang:prefill_delayer_wait_seconds",
            documentation="Histogram of wait time in seconds by prefill delayer.",
            labelnames=labels.keys(),
            buckets=sorted(
                set(
                    server_args.prefill_delayer_wait_seconds_buckets
                    or [1, 2, 5, 10, 20, 50, 100, 200, 500]
                )
                # Need bucket "<=0" for zero-delay cases
                | {0}
            ),
        )
        self.prefill_delayer_outcomes_total = Counter(
            name="sglang:prefill_delayer_outcomes_total",
            documentation="Prefill delayer outcome counts.",
            labelnames=[
                *labels.keys(),
                "input_estimation",
                "output_allow",
                "output_reason",
                "actual_execution",
            ],
        )

        # =================================================================
        # Constants (set once at startup via emit_constants)
        # =================================================================
        self.max_total_num_tokens = Gauge(
            name="sglang:max_total_num_tokens",
            documentation="Maximum total number of tokens in the KV cache pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.max_running_requests_under_SLO = Gauge(
            name="sglang:max_running_requests_under_SLO",
            documentation="The maximum number of running requests under SLO.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.engine_startup_time = Gauge(
            name="sglang:engine_startup_time",
            documentation="The time taken for the engine to start up.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.engine_load_weights_time = Gauge(
            name="sglang:engine_load_weights_time",
            documentation="The time taken for the engine to load weights.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.page_size = Gauge(
            name="sglang:page_size",
            documentation="KV cache page size in tokens.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_pages = Gauge(
            name="sglang:num_pages",
            documentation="Number of KV cache pages.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.context_len = Gauge(
            name="sglang:context_len",
            documentation="Maximum context length.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.startup_available_gpu_memory_gb = Gauge(
            name="sglang:startup_available_gpu_memory_gb",
            documentation="Available GPU memory in GB at startup.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

    @classmethod
    def init_new(
        cls,
        *,
        server_args: ServerArgs,
        ps: Any,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        enable_priority_scheduling: bool,
        enable_lora: bool,
        enable_hierarchical_cache: bool,
    ) -> SchedulerMetricsCollectorContext:
        enable_metrics = server_args.enable_metrics
        is_stats_logging_rank = ps.attn_tp_rank == 0
        current_scheduler_metrics_enabled = enable_metrics and (
            is_stats_logging_rank or server_args.enable_metrics_for_all_schedulers
        )
        enable_kv_cache_events = bool(
            server_args.kv_events_config
            and ps.pp_rank == 0
            and ps.attn_tp_rank == 0
            and ps.attn_cp_rank == 0
        )
        collector: Optional[SchedulerMetricsCollector] = None
        if enable_metrics:
            engine_type = DisaggregationMode.to_engine_type(
                server_args.disaggregation_mode
            )
            labels = {
                "model_name": server_args.served_model_name,
                "engine_type": engine_type,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "moe_ep_rank": ps.moe_ep_rank,
            }
            if enable_priority_scheduling:
                labels["priority"] = ""
            if dp_rank is not None:
                labels["dp_rank"] = dp_rank
            if server_args.extra_metric_labels:
                labels.update(server_args.extra_metric_labels)
            scheduler_collector_cls = resolve_collector_class(
                server_args, STAT_LOGGER_ROLE_SCHEDULER, cls
            )
            collector = scheduler_collector_cls(
                labels=labels,
                enable_lora=enable_lora,
                enable_hierarchical_cache=enable_hierarchical_cache,
                enable_streaming_session=server_args.enable_streaming_session,
                server_args=server_args,
            )
        return SchedulerMetricsCollectorContext(
            enable_metrics=enable_metrics,
            is_stats_logging_rank=is_stats_logging_rank,
            current_scheduler_metrics_enabled=current_scheduler_metrics_enabled,
            enable_kv_cache_events=enable_kv_cache_events,
            collector=collector,
        )

    def _log_gauge(self, gauge: Gauge, data: Union[int, float]) -> None:
        # Convenience function for logging a scalar to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_gauge_queue_count(self, gauge: Gauge, data: QueueCount) -> None:
        # Log a QueueCount to gauge: total under default labels, per-priority breakdown under priority="<int>".
        # NOTE: When priority scheduling is enabled, the total is recorded under
        # priority="" (the default label value). Per-priority breakdowns are recorded
        # with priority="<int>". Grafana queries should use priority="" for totals.
        gauge.labels(**self.labels).set(data.total)
        if data.by_priority is not None:
            self._known_priorities.update(data.by_priority.keys())
            for priority in self._known_priorities:
                value = data.by_priority.get(priority, 0)
                labels = dict(self.labels)
                labels["priority"] = str(priority)
                gauge.labels(**labels).set(value)

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        histogram.labels(**self.labels).observe(data)

    def increment_bootstrap_failed_reqs(self) -> None:
        self.num_bootstrap_failed_reqs.labels(**self.labels).inc(1)

    def increment_transfer_failed_reqs(self) -> None:
        self.num_transfer_failed_reqs.labels(**self.labels).inc(1)

    def increment_prefill_retries(self, count: int) -> None:
        if count > 0:
            self.num_prefill_retries_total.labels(**self.labels).inc(count)

    def observe_kv_transfer_metrics(
        self,
        latency_ms: float,
        total_mb: float,
        speed_gb_s: float,
    ) -> None:
        self._log_histogram(self.kv_transfer_latency_ms, latency_ms)
        self._log_histogram(self.kv_transfer_total_mb, total_mb)
        self._log_histogram(self.kv_transfer_speed_gb_s, speed_gb_s)

    def observe_kv_transfer_bootstrap(
        self,
        bootstrap_ms: float,
        alloc_ms: float,
    ) -> None:
        self._log_histogram(self.kv_transfer_bootstrap_ms, bootstrap_ms)
        self._log_histogram(self.kv_transfer_alloc_ms, alloc_ms)

    def observe_future_map_stash(self) -> None:
        self.num_future_map_stash_total.labels(**self.labels).inc(1)

    def observe_future_map_publish(self) -> None:
        self.num_future_map_publish_total.labels(**self.labels).inc(1)

    def observe_future_map_resolve(self, latency_ms: float) -> None:
        self.num_future_map_resolve_total.labels(**self.labels).inc(1)
        self._log_histogram(self.future_map_relay_latency_ms, latency_ms)

    def observe_min_free_slots_check(
        self, *, running_bs: int, num_allocatable_reqs: int, delayed: bool
    ) -> None:
        self.min_free_slots_checks_total.labels(**self.labels).inc(1)
        self._log_histogram(self.min_free_slots_running_bs, running_bs)
        self._log_histogram(self.min_free_slots_allocatable, num_allocatable_reqs)
        if delayed:
            self.min_free_slots_delay_total.labels(**self.labels).inc(1)

    def observe_scheduler_loop_iteration(
        self, dispatched_batch: bool, lag_ms: float
    ) -> None:
        self.scheduler_loop_iterations_total.labels(**self.labels).inc(1)
        if dispatched_batch:
            self.scheduler_loop_batch_dispatches_total.labels(**self.labels).inc(1)
        else:
            self.scheduler_loop_idle_total.labels(**self.labels).inc(1)
        self._log_histogram(self.scheduler_loop_iteration_lag_ms, lag_ms)

    def observe_scheduler_abort(self, reason: str) -> None:
        self.scheduler_aborts_total.labels(**self.labels, reason=reason).inc(1)

    def observe_per_stage_req_latency(self, stage: str, latency: float) -> None:
        labels_with_stage = {**self.labels, "stage": stage}
        self.per_stage_req_latency_seconds.labels(**labels_with_stage).observe(latency)

    def observe_queue_time(self, latency: float) -> None:
        self._log_histogram(self.queue_time, latency)

    def observe_weight_load(self, duration_seconds: float, source: str) -> None:
        # Edge-triggered: engine is paused during the update, so log_stats
        # won't fire — write the gauge inline at end of update_weights_from_*.
        # `source` is "disk" | "distributed" | "tensor" | "ipc".
        self.weight_load_duration_seconds.labels(**self.labels, source=source).set(
            duration_seconds
        )

    def observe_prefill_delayer_outcome(
        self,
        forward_passes: int,
        wait_seconds: float,
        input_estimation: str,
        output_allow: bool,
        output_reason: str,
        actual_execution: bool,
    ) -> None:
        if output_allow and actual_execution:
            self._log_histogram(
                self.prefill_delayer_wait_forward_passes, forward_passes
            )
            self._log_histogram(self.prefill_delayer_wait_seconds, wait_seconds)

        self.prefill_delayer_outcomes_total.labels(
            **self.labels,
            input_estimation=input_estimation,
            output_allow=str(output_allow).lower(),
            output_reason=output_reason,
            actual_execution=str(actual_execution).lower(),
        ).inc(1)

    def increment_retracted_reqs(
        self,
        num_retracted_reqs: int,
        num_retracted_input_tokens: int,
        num_retracted_output_tokens: int,
    ) -> None:
        self.num_retracted_reqs_total.labels(**self.labels).inc(num_retracted_reqs)
        self.num_retracted_input_tokens_total.labels(**self.labels).inc(
            num_retracted_input_tokens
        )
        self.num_retracted_output_tokens_total.labels(**self.labels).inc(
            num_retracted_output_tokens
        )

    def increment_decode_cuda_graph_pass(self, value: bool) -> None:
        mode = "decode_cuda_graph" if value else "decode_none"
        self.cuda_graph_passes_total.labels(**self.labels, mode=mode).inc(1)

    def increment_prefill_cuda_graph_pass(self, value: bool) -> None:
        mode = "prefill_cuda_graph" if value else "prefill_none"
        self.cuda_graph_passes_total.labels(**self.labels, mode=mode).inc(1)

    def increment_eplb_balancedness(
        self, forward_mode: str, balancedness: float
    ) -> None:
        self.eplb_balancedness.labels(**self.labels, forward_mode=forward_mode).observe(
            balancedness
        )

    def increment_realtime_tokens(
        self,
        dp_cooperation_info: Optional[DPCooperationInfo],
        prefill_compute_tokens=0,
        prefill_cache_tokens=0,
        decode_tokens=0,
    ):
        for mode, delta in [
            ("prefill_compute", prefill_compute_tokens),
            ("prefill_cache", prefill_cache_tokens),
            ("decode", decode_tokens),
        ]:
            if delta == 0:
                continue
            self.realtime_tokens_total.labels(**self.labels, mode=mode).inc(delta)
            if dp_cooperation_info is not None:
                self.dp_cooperation_realtime_tokens_total.labels(
                    **self.labels,
                    mode=mode,
                    **dp_cooperation_info.to_labels(),
                ).inc(delta)

    def increment_forward_execution_seconds(
        self,
        category: str,
        t: float,
        dp_cooperation_info: Optional[DPCooperationInfo] = None,
    ):
        self.forward_execution_seconds_total.labels(
            **self.labels, category=category
        ).inc(t)
        if dp_cooperation_info is not None:
            self.dp_cooperation_forward_execution_seconds_total.labels(
                **self.labels,
                category=category,
                **dp_cooperation_info.to_labels(),
            ).inc(t)

    def increment_estimated_perf(
        self,
        num_flops_per_gpu: float = 0.0,
        num_read_bytes_per_gpu: float = 0.0,
        num_write_bytes_per_gpu: float = 0.0,
    ) -> None:
        if num_flops_per_gpu > 0:
            self.estimated_flops_per_gpu_total.labels(**self.labels).inc(
                num_flops_per_gpu
            )
        if num_read_bytes_per_gpu > 0:
            self.estimated_read_bytes_per_gpu_total.labels(**self.labels).inc(
                num_read_bytes_per_gpu
            )
        if num_write_bytes_per_gpu > 0:
            self.estimated_write_bytes_per_gpu_total.labels(**self.labels).inc(
                num_write_bytes_per_gpu
            )

    def log_stats(self, stats: SchedulerStats) -> None:
        # Basics
        self._log_gauge_queue_count(self.num_running_reqs, stats.num_running_reqs)
        self._log_gauge_queue_count(self.num_queue_reqs, stats.num_queue_reqs)
        self._log_gauge(self.num_grammar_queue_reqs, stats.num_grammar_queue_reqs)
        self._log_gauge(self.gen_throughput, stats.gen_throughput)
        self._log_gauge(self.cache_hit_rate, stats.cache_hit_rate)
        self._log_gauge(self.decode_sum_seq_lens, stats.decode_sum_seq_lens)

        # Memory pool usage ratios
        self._log_gauge(self.token_usage, stats.token_usage)
        self._log_gauge(self.full_token_usage, stats.full_token_usage)
        self._log_gauge(self.swa_token_usage, stats.swa_token_usage)
        self._log_gauge(self.mamba_usage, stats.mamba_usage)

        # Absolute token counts
        self._log_gauge(self.num_used_tokens, stats.num_used_tokens)
        self._log_gauge(self.kv_available_tokens, stats.kv_available_tokens)
        self._log_gauge(self.kv_evictable_tokens, stats.kv_evictable_tokens)
        self._log_gauge(self.kv_used_tokens, stats.kv_used_tokens)
        self._log_gauge(self.swa_available_tokens, stats.swa_available_tokens)
        self._log_gauge(self.swa_evictable_tokens, stats.swa_evictable_tokens)
        self._log_gauge(self.swa_used_tokens, stats.swa_used_tokens)
        self._log_gauge(self.mamba_available_tokens, stats.mamba_available_tokens)
        self._log_gauge(self.mamba_evictable_tokens, stats.mamba_evictable_tokens)
        self._log_gauge(self.mamba_used_tokens, stats.mamba_used_tokens)

        # Speculative decoding
        self._log_gauge(self.spec_accept_length, stats.spec_accept_length)
        self._log_gauge(self.spec_accept_rate, stats.spec_accept_rate)
        self._log_gauge(self.spec_num_steps, stats.spec_num_steps)
        self._log_gauge(self.spec_num_draft_tokens, stats.spec_num_draft_tokens)

        # Retract
        self._log_gauge(self.num_retracted_reqs, stats.num_retracted_reqs)
        self._log_gauge(self.num_paused_reqs, stats.num_paused_reqs)

        # PD disaggregation
        self._log_gauge_queue_count(
            self.num_prefill_bootstrap_queue_reqs,
            stats.num_prefill_bootstrap_queue_reqs,
        )
        self._log_gauge_queue_count(
            self.num_prefill_inflight_queue_reqs, stats.num_prefill_inflight_queue_reqs
        )
        self._log_gauge_queue_count(
            self.num_decode_prealloc_queue_reqs, stats.num_decode_prealloc_queue_reqs
        )
        self._log_gauge_queue_count(
            self.num_decode_transfer_queue_reqs, stats.num_decode_transfer_queue_reqs
        )
        self._log_gauge(
            self.pending_prealloc_token_usage, stats.pending_prealloc_token_usage
        )

        # Utilization
        self._log_gauge(self.utilization, stats.utilization)
        self._log_gauge(self.fwd_occupancy, stats.fwd_occupancy)

        # Scheduler policy
        self._log_gauge(self.new_token_ratio, stats.new_token_ratio)

        # CUDA graph
        self._log_gauge(self.is_cuda_graph, stats.is_cuda_graph)

        # LoRA pool metrics
        if self.enable_lora:
            self._log_gauge(self.lora_pool_slots_used, stats.lora_pool_slots_used)
            self._log_gauge(self.lora_pool_slots_total, stats.lora_pool_slots_total)
            self._log_gauge(self.lora_pool_utilization, stats.lora_pool_utilization)

        # HiCache metrics
        if self.enable_hierarchical_cache:
            self._log_gauge(
                self.hicache_host_used_tokens, stats.hicache_host_used_tokens
            )
            self._log_gauge(
                self.hicache_host_total_tokens, stats.hicache_host_total_tokens
            )

        # Streaming session metrics
        if self.enable_streaming_session:
            self._log_gauge(self.num_streaming_sessions, stats.num_streaming_sessions)
            self._log_gauge(
                self.streaming_session_held_tokens, stats.streaming_session_held_tokens
            )

        # Routing key metrics
        self._log_gauge(
            self.num_unique_running_routing_keys, stats.num_unique_running_routing_keys
        )
        self.routing_key_running_req_count.set_by_current_observations(
            self.labels, stats.routing_key_running_req_counts
        )
        self.routing_key_all_req_count.set_by_current_observations(
            self.labels, stats.routing_key_all_req_counts
        )

        self.last_log_time = time.perf_counter()

    def log_grammar_stats(self, grammar_stats) -> None:
        if grammar_stats.compilation_time is not None:
            self._log_histogram(
                self.grammar_compilation_time, grammar_stats.compilation_time
            )
        if grammar_stats.schema_count is not None:
            self._log_histogram(self.grammar_schema_count, grammar_stats.schema_count)
        if grammar_stats.ebnf_size is not None:
            self._log_histogram(self.grammar_ebnf_size, grammar_stats.ebnf_size)
        tree_times = grammar_stats.tree_traversal_time
        if tree_times:
            max_time = max(tree_times)
            avg_time = sum(tree_times) / len(tree_times)
            self._log_histogram(self.grammar_tree_traversal_time_max, max_time)
            self._log_histogram(self.grammar_tree_traversal_time_avg, avg_time)
        if grammar_stats.is_cache_hit:
            self.num_grammar_cache_hit.labels(**self.labels).inc(1)
        if grammar_stats.is_grammar_aborted:
            self.num_grammar_aborted.labels(**self.labels).inc(1)
        if grammar_stats.num_timeout > 0:
            self.num_grammar_timeout.labels(**self.labels).inc(
                grammar_stats.num_timeout
            )
        self.num_grammar_total.labels(**self.labels).inc(1)

    def emit_constants(
        self,
        max_total_num_tokens: int,
        max_running_requests_under_SLO: Optional[int],
        engine_startup_time: float,
        engine_load_weights_time: float,
        page_size: int,
        num_pages: int,
        context_len: int,
        startup_available_gpu_memory_gb: float,
    ) -> None:
        self._log_gauge(self.max_total_num_tokens, max_total_num_tokens)
        if max_running_requests_under_SLO is not None:
            self._log_gauge(
                self.max_running_requests_under_SLO, max_running_requests_under_SLO
            )
        self._log_gauge(self.engine_startup_time, engine_startup_time)
        self._log_gauge(self.engine_load_weights_time, engine_load_weights_time)
        self._log_gauge(self.page_size, page_size)
        self._log_gauge(self.num_pages, num_pages)
        self._log_gauge(self.context_len, context_len)
        self._log_gauge(
            self.startup_available_gpu_memory_gb, startup_available_gpu_memory_gb
        )


class TokenizerMetricsCollector(_StatLoggerDIMixin):
    def __init__(
        self,
        server_args: Optional[ServerArgs] = None,
        labels: Dict[str, str] = None,
        bucket_time_to_first_token: Optional[List[float]] = None,
        bucket_inter_token_latency: Optional[List[float]] = None,
        bucket_e2e_request_latency: Optional[List[float]] = None,
    ) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter as _PromCounter
        from prometheus_client import Histogram as _PromHistogram

        Counter = self._counter_cls or _PromCounter
        Histogram = self._histogram_cls or _PromHistogram

        self.labels = labels or {}

        self.prompt_tokens_total = Counter(
            name="sglang:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labels.keys(),
        )
        self.generation_tokens_total = Counter(
            name="sglang:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labels.keys(),
        )
        self.spec_verify_calls_total = Counter(
            name="sglang:spec_verify_calls_total",
            documentation="Number of speculative decoding verification calls.",
            labelnames=labels.keys(),
        )

        default_bucket_prompt_tokens = [
            100,
            300,
            500,
            700,
            1000,
            1500,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
            12500,
            15000,
            17500,
            20000,
            22500,
            25000,
            27500,
            30000,
            35000,
            40000,
            60000,
            80000,
            100000,
            200000,
            300000,
            400000,
            600000,
            800000,
            1000000,
            1100000,
        ]
        self.prompt_tokens_histogram = Histogram(
            name="sglang:prompt_tokens_histogram",
            documentation="Histogram of prompt token length.",
            labelnames=labels.keys(),
            buckets=generate_buckets(
                server_args.prompt_tokens_buckets, default_bucket_prompt_tokens
            ),
        )
        self.uncached_prompt_tokens_histogram = Histogram(
            name="sglang:uncached_prompt_tokens_histogram",
            documentation="Histogram of uncached (compute) prompt token length.",
            labelnames=labels.keys(),
            buckets=generate_buckets(
                server_args.prompt_tokens_buckets, default_bucket_prompt_tokens
            ),
        )
        self.generation_tokens_histogram = Histogram(
            name="sglang:generation_tokens_histogram",
            documentation="Histogram of generation token length.",
            labelnames=labels.keys(),
            buckets=generate_buckets(
                server_args.generation_tokens_buckets,
                default_bucket_prompt_tokens,
            ),
        )

        self.cached_tokens_total = Counter(
            name="sglang:cached_tokens_total",
            documentation="Number of cached prompt tokens by source (device/host/storage).",
            labelnames=list(labels.keys()) + ["cache_source"],
        )

        self.num_requests_total = Counter(
            name="sglang:num_requests_total",
            documentation="Number of requests processed.",
            labelnames=labels.keys(),
        )

        self.get_loads_duration_seconds = Histogram(
            name="sglang:get_loads_duration_seconds",
            documentation="Time spent serving /v1/loads requests (seconds).",
            labelnames=labels.keys(),
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
        )

        self.num_so_requests_total = Counter(
            name="sglang:num_so_requests_total",
            documentation="Number of structured output requests processed.",
            labelnames=labels.keys(),
        )

        self.num_aborted_requests_total = Counter(
            name="sglang:num_aborted_requests_total",
            documentation="Number of requests aborted.",
            labelnames=labels.keys(),
        )

        if bucket_time_to_first_token is None:
            bucket_time_to_first_token = [
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                100,
                200,
                400,
            ]

        if bucket_e2e_request_latency is None:
            bucket_e2e_request_latency = [
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                100,
                200,
                400,
                600,
                1200,
                1800,
                2400,
            ]

        if bucket_inter_token_latency is None:
            bucket_inter_token_latency = [
                0.002,
                0.004,
                0.006,
                0.008,
                0.010,
                0.015,
                0.020,
                0.025,
                0.030,
                0.035,
                0.040,
                0.060,
                0.080,
                0.100,
                0.200,
                0.400,
                0.600,
                0.800,
                1.000,
                2.000,
                4.000,
                6.000,
                8.000,
            ]

        self.histogram_time_to_first_token = Histogram(
            name="sglang:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_time_to_first_token,
        )

        self.histogram_inter_token_latency = Histogram(
            name="sglang:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_inter_token_latency,
        )

        self.histogram_e2e_request_latency = Histogram(
            name="sglang:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=labels.keys(),
            buckets=bucket_e2e_request_latency,
        )

    def observe_one_finished_request(
        self,
        labels: Dict[str, str],
        prompt_tokens: int,
        generation_tokens: int,
        cached_tokens: int,
        e2e_latency: float,
        has_grammar: bool,
        cached_tokens_details: Optional[Dict[str, Any]] = None,
        spec_verify_ct: int = 0,
    ):
        self.prompt_tokens_total.labels(**labels).inc(prompt_tokens)
        self.generation_tokens_total.labels(**labels).inc(generation_tokens)
        if spec_verify_ct > 0:
            self.spec_verify_calls_total.labels(**labels).inc(spec_verify_ct)

        # Report cached tokens with detailed source breakdown
        if cached_tokens > 0:
            if cached_tokens_details:
                # Report by cache source (device/host, and storage if L3 enabled)
                def report_cache_source(source: str, value: int):
                    if value > 0:
                        source_labels = {**labels, "cache_source": source}
                        self.cached_tokens_total.labels(**source_labels).inc(value)

                report_cache_source("device", cached_tokens_details.get("device", 0))
                report_cache_source("host", cached_tokens_details.get("host", 0))

                # Storage fields are only present when L3 storage backend is enabled
                if "storage" in cached_tokens_details:
                    storage_tokens = cached_tokens_details.get("storage", 0)
                    if storage_tokens > 0:
                        backend = (
                            cached_tokens_details.get("storage_backend") or "unknown"
                        )
                        report_cache_source(f"storage_{backend}", storage_tokens)
            else:
                # Fallback for backward compatibility
                labels_total = {**labels, "cache_source": "total"}
                self.cached_tokens_total.labels(**labels_total).inc(cached_tokens)

        self.num_requests_total.labels(**labels).inc(1)
        if has_grammar:
            self.num_so_requests_total.labels(**labels).inc(1)
        self.histogram_e2e_request_latency.labels(**labels).observe(float(e2e_latency))
        self.prompt_tokens_histogram.labels(**labels).observe(float(prompt_tokens))
        self.uncached_prompt_tokens_histogram.labels(**labels).observe(
            float(prompt_tokens - cached_tokens)
        )
        self.generation_tokens_histogram.labels(**labels).observe(
            float(generation_tokens)
        )

    def observe_time_to_first_token(self, labels: Dict[str, str], value: float):
        self.histogram_time_to_first_token.labels(**labels).observe(value)

    def check_time_to_first_token_straggler(self, value: float) -> bool:
        his = self.histogram_time_to_first_token.labels(**self.labels)
        total_observations = sum(bucket._value for bucket in his._buckets)
        if total_observations < 100:
            return False
        p99_threshold = total_observations * 0.99
        cumulative_count = 0
        for i, bucket in enumerate(his._buckets):
            cumulative_count += bucket._value
            if cumulative_count > p99_threshold:
                return value >= his._upper_bounds[i]
        return False

    def observe_inter_token_latency(
        self, labels: Dict[str, str], internval: float, num_new_tokens: int
    ):
        adjusted_interval = internval / num_new_tokens

        # A faster version of the Histogram::observe which observes multiple values at the same time.
        # reference: https://github.com/prometheus/client_python/blob/v0.21.1/prometheus_client/metrics.py#L639
        his = self.histogram_inter_token_latency.labels(**labels)
        his._sum.inc(internval)

        for i, bound in enumerate(his._upper_bounds):
            if adjusted_interval <= bound:
                his._buckets[i].inc(num_new_tokens)
                break

    def observe_one_aborted_request(self, labels: Dict[str, str]):
        self.num_aborted_requests_total.labels(**labels).inc(1)


@dataclass
class StorageMetrics:
    prefetch_pgs: List[int] = field(default_factory=list)
    backup_pgs: List[int] = field(default_factory=list)
    prefetch_bandwidth: List[float] = field(default_factory=list)
    backup_bandwidth: List[float] = field(default_factory=list)


class StorageMetricsCollector(_StatLoggerDIMixin):
    def __init__(
        self,
        labels: Dict[str, str],
    ):
        from prometheus_client import Counter as _PromCounter
        from prometheus_client import Histogram as _PromHistogram

        Counter = self._counter_cls or _PromCounter
        Histogram = self._histogram_cls or _PromHistogram

        self.labels = labels

        self.prefetched_tokens_total = Counter(
            name="sglang:prefetched_tokens_total",
            documentation="Number of prefetched prompt tokens.",
            labelnames=labels.keys(),
        )

        self.backuped_tokens_total = Counter(
            name="sglang:backuped_tokens_total",
            documentation="Number of backuped tokens.",
            labelnames=labels.keys(),
        )

        bucket_io = [
            1,
            5,
            10,
            50,
            100,
        ]

        bucket_bandwidth = [
            0.1,
            0.5,
            1,
            5,
            10,
            50,
            100,
        ]

        self.histogram_prefetch_pgs = Histogram(
            name="sglang:prefetch_pgs",
            documentation="Histogram of prefetch pages of batches.",
            labelnames=labels.keys(),
            buckets=bucket_io,
        )

        self.histogram_backup_pgs = Histogram(
            name="sglang:backup_pgs",
            documentation="Histogram of backup pages of batches.",
            labelnames=labels.keys(),
            buckets=bucket_io,
        )

        self.histogram_prefetch_bandwidth = Histogram(
            name="sglang:prefetch_bandwidth",
            documentation="Histogram of prefetch bandwidth in GB/s.",
            labelnames=labels.keys(),
            buckets=bucket_bandwidth,
        )

        self.histogram_backup_bandwidth = Histogram(
            name="sglang:backup_bandwidth",
            documentation="Histogram of backup bandwidth in GB/s.",
            labelnames=labels.keys(),
            buckets=bucket_bandwidth,
        )

    def log_prefetched_tokens(self, prefetched_tokens: int):
        if prefetched_tokens > 0:
            self.prefetched_tokens_total.labels(**self.labels).inc(prefetched_tokens)

    def log_backuped_tokens(self, backuped_tokens: int):
        if backuped_tokens > 0:
            self.backuped_tokens_total.labels(**self.labels).inc(backuped_tokens)

    def _log_histogram(self, histogram, data: Union[int, float]):
        histogram.labels(**self.labels).observe(data)

    def log_storage_metrics(self, storage_metrics: Optional[StorageMetrics] = None):
        if storage_metrics is None:
            return

        assert isinstance(storage_metrics, StorageMetrics)

        for v in storage_metrics.prefetch_pgs:
            self._log_histogram(self.histogram_prefetch_pgs, v)
        for v in storage_metrics.backup_pgs:
            self._log_histogram(self.histogram_backup_pgs, v)
        for v in storage_metrics.prefetch_bandwidth:
            self._log_histogram(self.histogram_prefetch_bandwidth, v)
        for v in storage_metrics.backup_bandwidth:
            self._log_histogram(self.histogram_backup_bandwidth, v)


class ExpertDispatchCollector(_StatLoggerDIMixin):
    def __init__(self, ep_size: int) -> None:
        from prometheus_client import Histogram as _PromHistogram

        Histogram = self._histogram_cls or _PromHistogram

        ep_size_buckets = [i for i in range(ep_size)]
        self.eplb_gpu_physical_count = Histogram(
            name="sglang:eplb_gpu_physical_count",
            documentation="The selected count of physical experts on each layer and GPU rank.",
            labelnames={"layer"},
            buckets=ep_size_buckets,
        )


class RadixCacheMetricsCollector(_StatLoggerDIMixin):
    def __init__(
        self,
        labels: Dict[str, str],
    ) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter as _PromCounter
        from prometheus_client import Histogram as _PromHistogram

        Counter = self._counter_cls or _PromCounter
        Histogram = self._histogram_cls or _PromHistogram

        self.labels = labels

        bucket_eviction_duration = get_histogram_conf_from_env(
            "SGLANG_BUCKET_EVICTION_DURATION"
        )
        if bucket_eviction_duration is None:
            bucket_eviction_duration = [
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.1,
                0.2,
                0.5,
                1.0,
            ]
        bucket_load_back_duration = get_histogram_conf_from_env(
            "SGLANG_BUCKET_LOAD_BACK_DURATION"
        )
        if bucket_load_back_duration is None:
            bucket_load_back_duration = [
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.1,
                0.2,
                0.5,
                1.0,
            ]
        self.eviction_duration_seconds = Histogram(
            name="sglang:eviction_duration_seconds",
            documentation="Time taken to evict memory from GPU to CPU in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_eviction_duration,
        )

        self.eviction_num_tokens = Counter(
            name="sglang:evicted_tokens_total",
            documentation="The number of tokens evicted from GPU to CPU.",
            labelnames=labels.keys(),
        )

        self.load_back_duration_seconds = Histogram(
            name="sglang:load_back_duration_seconds",
            documentation="Time taken to load memory from CPU to GPU in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_load_back_duration,
        )

        self.load_back_num_tokens = Counter(
            name="sglang:load_back_tokens_total",
            documentation="The number of tokens loaded from CPU to GPU.",
            labelnames=labels.keys(),
        )

    def increment_eviction_num_tokens(self, num_tokens: int) -> None:
        self.eviction_num_tokens.labels(**self.labels).inc(num_tokens)

    def increment_load_back_num_tokens(self, num_tokens: int) -> None:
        self.load_back_num_tokens.labels(**self.labels).inc(num_tokens)

    def observe_eviction_duration(self, duration_seconds: float) -> None:
        self.eviction_duration_seconds.labels(**self.labels).observe(duration_seconds)

    def observe_load_back_duration(self, duration_seconds: float) -> None:
        self.load_back_duration_seconds.labels(**self.labels).observe(duration_seconds)


class EncoderMetricsCollector(_StatLoggerDIMixin):
    """Metrics collector for the EPD encoder server (--encoder-only)."""

    def __init__(self, labels: Dict[str, str]) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter as _PromCounter
        from prometheus_client import Gauge as _PromGauge
        from prometheus_client import Histogram as _PromHistogram

        Counter = self._counter_cls or _PromCounter
        Gauge = self._gauge_cls or _PromGauge
        Histogram = self._histogram_cls or _PromHistogram

        self.labels = labels

        self.cache_evictions_total = Counter(
            name="sglang:encoder_cache_evictions_total",
            documentation="Total cache evictions.",
            labelnames=list(labels.keys()) + ["modality"],
        )
        self.cache_size_mb = Gauge(
            name="sglang:encoder_cache_size_mb",
            documentation="Current cache size in MB.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.cache_entries = Gauge(
            name="sglang:encoder_cache_entries",
            documentation="Current number of cache entries.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.cache_hit_tokens_total = Counter(
            name="sglang:encoder_cache_hit_tokens_total",
            documentation="Total tokens served from cache (cache hits).",
            labelnames=list(labels.keys()) + ["modality"],
        )
        self.cache_total_tokens_total = Counter(
            name="sglang:encoder_cache_total_tokens_total",
            documentation="Total tokens processed (hit + miss).",
            labelnames=list(labels.keys()) + ["modality"],
        )
        self.cache_hit_files_total = Counter(
            name="sglang:encoder_cache_hit_files_total",
            documentation="Total files served from cache.",
            labelnames=list(labels.keys()) + ["modality"],
        )
        self.cache_total_files_total = Counter(
            name="sglang:encoder_cache_total_files_total",
            documentation="Total files processed (hit + miss).",
            labelnames=list(labels.keys()) + ["modality"],
        )

        # Total encoder requests by modality and status
        self.requests_total = Counter(
            name="sglang:encoder_requests_total",
            documentation="Total encoder requests by modality and status.",
            labelnames=list(labels.keys()) + ["modality", "status"],
        )

        # Total requests received per DP rank (incremented at receive time, before processing).
        # Use rate(sglang:encoder_requests_received_total[1m]) for per-encoder QPS.
        self.requests_received_total = Counter(
            name="sglang:encoder_requests_received_total",
            documentation="Total requests received by encoder (at receive time), per DP rank.",
            labelnames=list(labels.keys()) + ["modality"],
        )

        # Multimodal items per batch histogram
        self.mm_items_per_batch = Histogram(
            name="sglang:encoder_mm_items_per_batch",
            documentation="Histogram of multimodal items processed per encoder batch.",
            labelnames=list(labels.keys()) + ["modality"],
            buckets=[
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                32,
                64,
                128,
            ],
        )

        # Multimodal items per request histogram
        self.mm_items_per_request = Histogram(
            name="sglang:encoder_mm_items_per_request",
            documentation="Histogram of multimodal items per individual encoder request.",
            labelnames=list(labels.keys()) + ["modality"],
            buckets=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 64],
        )

        # Per-request E2E encoder latency
        self.encoder_request_e2e_latency_seconds = Histogram(
            name="sglang:encoder_request_e2e_latency_seconds",
            documentation="Histogram of per-request end-to-end encoder latency in seconds (queue wait + encode).",
            labelnames=list(labels.keys()) + ["modality"],
            buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60],
        )

        # --- Latency breakdown histograms ---

        # Queue wait: time spent in scheduler queue before batch processing starts
        self.queue_wait_seconds = Histogram(
            name="sglang:encoder_queue_wait_seconds",
            documentation="Time request spent waiting in scheduler queue.",
            labelnames=list(labels.keys()) + ["modality"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
        )

        # Preprocess: CPU data loading + processor (image decode, video frame sampling, etc.)
        self.preprocess_seconds = Histogram(
            name="sglang:encoder_preprocess_seconds",
            documentation="Data loading and preprocessing latency.",
            labelnames=list(labels.keys()) + ["modality"],
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30],
        )

        #  Model forward: model forward pass latency
        self.model_forward_seconds = Histogram(
            name="sglang:encoder_model_forward_seconds",
            documentation="GPU model forward pass latency.",
            labelnames=list(labels.keys()) + ["modality"],
            buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
        )

        # Embedding transfer: embedding transfer to prefill node (zmq or mooncake)
        self.transfer_seconds = Histogram(
            name="sglang:encoder_transfer_seconds",
            documentation="Embedding transfer latency to prefill node.",
            labelnames=list(labels.keys()) + ["backend"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2],
        )

    def _inc_cache_counter(self, counter, modality: str, count: int = 1) -> None:
        counter.labels(**self.labels, modality=modality).inc(count)

    def inc_cache_evictions(self, modality: str = "image", count: int = 1) -> None:
        self._inc_cache_counter(self.cache_evictions_total, modality, count)

    def record_cache_tokens(
        self, hit_tokens: int, total_tokens: int, modality: str = "image"
    ) -> None:
        self._inc_cache_counter(self.cache_total_tokens_total, modality, total_tokens)
        if hit_tokens > 0:
            self._inc_cache_counter(self.cache_hit_tokens_total, modality, hit_tokens)

    def record_cache_files(
        self, hit_files: int, total_files: int, modality: str = "image"
    ) -> None:
        self._inc_cache_counter(self.cache_total_files_total, modality, total_files)
        if hit_files > 0:
            self._inc_cache_counter(self.cache_hit_files_total, modality, hit_files)

    def set_cache_state(self, current_size: int, num_entries: int) -> None:
        self.cache_size_mb.labels(**self.labels).set(current_size / (1024 * 1024))
        self.cache_entries.labels(**self.labels).set(num_entries)

    def observe_queue_wait(
        self, latency_seconds: float, modality: str = "image"
    ) -> None:
        """Record time spent waiting in the scheduler queue."""
        self.queue_wait_seconds.labels(**self.labels, modality=modality).observe(
            latency_seconds
        )

    def observe_preprocess(
        self, latency_seconds: float, modality: str = "image"
    ) -> None:
        """Record data loading and preprocessing latency."""
        self.preprocess_seconds.labels(**self.labels, modality=modality).observe(
            latency_seconds
        )

    def observe_model_forward(
        self, latency_seconds: float, modality: str = "image"
    ) -> None:
        """Record model forward pass latency."""
        self.model_forward_seconds.labels(**self.labels, modality=modality).observe(
            latency_seconds
        )

    def observe_transfer(self, latency_seconds: float, backend: str = "zmq") -> None:
        """Record embedding transfer latency."""
        self.transfer_seconds.labels(**self.labels, backend=backend).observe(
            latency_seconds
        )

    def observe_mm_items_per_batch(self, count: int, modality: str = "image") -> None:
        """Record the number of multimodal items processed in a batch."""
        self.mm_items_per_batch.labels(**self.labels, modality=modality).observe(count)

    def observe_mm_items_per_request(self, count: int, modality: str = "image") -> None:
        """Record the number of multimodal items in a single request."""
        self.mm_items_per_request.labels(**self.labels, modality=modality).observe(
            count
        )

    def inc_requests_total(self, modality: str, status: str) -> None:
        """Increment encoder request counter. status: 'success' | 'error'."""
        self.requests_total.labels(
            **self.labels, modality=modality, status=status
        ).inc()

    def inc_requests_received(self, modality: str = "image") -> None:
        """Increment the received-requests counter at request-arrival time.

        dp_rank is supplied via self.labels (set per process at construction).
        """
        self.requests_received_total.labels(**self.labels, modality=modality).inc()

    def observe_request_e2e_latency(
        self, latency_seconds: float, modality: str = "image"
    ) -> None:
        """Record per-request end-to-end encoder latency in seconds."""
        self.encoder_request_e2e_latency_seconds.labels(
            **self.labels, modality=modality
        ).observe(latency_seconds)


def get_histogram_conf_from_env(env_var_name: str) -> Optional[List[float]]:
    """
    Get the histogram configuration from the environment variable.
    env value should be like "0.1,0.2,0.5,1,2"
    """
    if env_var_name not in os.environ:
        return None
    # if the env var is not set or empty, return None
    env_var_value = os.environ[env_var_name]
    if not env_var_value:
        return None
    return [float(x) for x in env_var_value.split(",")]
