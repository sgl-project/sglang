"""Frozen bundles the scheduler hands to its components: each groups values
that travel together and never change after the scheduler is built."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler_components.metrics_reporter import MetricsSink
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.observability.scheduler_stage_metrics import (
        SchedulerStageMetricsRecorder,
    )
    from sglang.srt.sampling.penalties import PenaltyStore


@dataclass(frozen=True, kw_only=True, slots=True)
class Workers:
    """The model workers: the target, the draft under speculative decoding,
    and the one the scheduler runs its forwards through."""

    tp_worker: Any
    # None without speculative decoding, and on a PP+spec stage before the last.
    draft_worker: Any
    model_worker: Any


@dataclass(frozen=True, kw_only=True, slots=True)
class KvResources:
    """The KV memory the components share."""

    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache
    penalty_store: PenaltyStore
    # The draft model's KV pool under speculative decoding.
    draft_token_to_kv_pool: Optional[KVCache]


@dataclass(frozen=True, kw_only=True, slots=True)
class KvCapacity:
    """The KV limits and the pools' layout, known once the pools are built."""

    # Request-token capacity as the worker reports it, in logical tokens.
    max_total_num_tokens: int
    # Logical-to-physical KV capacity ratio of the page-interleaved shards.
    kv_shard_widening: int
    # The request limit that keeps hybrid SWA's separate pool bounds.
    effective_max_total_num_tokens: int
    # The SWA pool's capacity; None without hybrid SWA.
    swa_max_total_num_tokens: Optional[int]
    max_token_pool_size: int
    max_running_requests: int
    # The layout the KV cache was built with: a hybrid SWA pool (with its
    # window and the per-layer token counts of both pools) or a hybrid SSM one.
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    sliding_window_size: Optional[int]
    full_tokens_per_layer: Optional[int]
    swa_tokens_per_layer: Optional[int]

    @property
    def widened_max_total_num_tokens(self) -> int:
        """The capacity in the units the allocator and radix counters use."""
        return self.max_total_num_tokens * self.kv_shard_widening


@dataclass(frozen=True, kw_only=True, slots=True)
class RequestLimits:
    """The request and prefill-step limits, known once the workers report
    theirs and the chunk size is resolved."""

    # The longest request, and its longest input, in tokens.
    max_req_len: int
    max_req_input_len: int
    # The waiting queue's cap; None without one.
    max_queued_requests: Optional[int]
    # The tokens one prefill step may take.
    max_prefill_tokens: int
    # The prefill chunk; None without chunked prefill.
    chunked_prefill_size: Optional[int]
    # A prefill chunk may share its step with decoding requests.
    is_mixed_chunk: bool
    # The chunk alignment a truncated prefill keeps; None without one.
    truncation_align_size: Optional[int]


@dataclass(frozen=True, kw_only=True, slots=True)
class Streams:
    """The device, its torch module and the streams the scheduler loop runs
    on."""

    device: str
    device_module: Any
    schedule_stream: Any
    forward_stream: Any
    copy_stream: Any


@dataclass(frozen=True, kw_only=True, slots=True)
class SchedulerServices:
    """The leaf services every owner writes to."""

    metrics_sink: MetricsSink
    output_streamer: SchedulerOutputStreamer
    scheduler_stage_metrics: SchedulerStageMetricsRecorder


@dataclass(frozen=True, kw_only=True, slots=True)
class ComponentClasses:
    """The classes a `Scheduler` subclass swaps through its
    `component_classes` override point, read once by the assembly. Whoever
    builds one of these objects, the assembly or a component, takes its class
    from here; a component takes it as a constructor argument
    (`prefill_adder_class=...`)."""

    rust_server: type
    output_streamer: type
    # The step planner builds one per prefill admission pass.
    prefill_adder: type
    # Builds each generation request through its `create_generation_req`.
    request_intake: type
