"""Config fields of the ``schedule`` namespace.

One class per namespace. The class *is* the namespace: a field declared here
lands in the ``schedule`` bag, which is what ``get_schedule()`` returns, so a reader
spells it exactly as before. ``ServerArgs`` composes these classes, so the
record stays one flat object -- the split moves where declarations live, not
how config is shaped at runtime.
"""

from __future__ import annotations

import dataclasses
from typing import (
    List,
    Optional,
)

from sglang.srt.arg_groups.arg_utils import (
    A,
    Arg,
)
from sglang.srt.utils.common import human_readable_int


@dataclasses.dataclass
class Schedule:
    """Namespace ``schedule``."""

    _NS_PATH = "schedule"

    # -------------------------------------------------------------------------
    # Memory and scheduling
    # -------------------------------------------------------------------------
    mem_fraction_static: A[
        Optional[float],
        "The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
    ] = None
    max_running_requests: A[
        Optional[int],
        "The maximum number of running requests.",
    ] = None
    max_queued_requests: A[
        Optional[int],
        "The maximum number of queued requests. This option is ignored when using disaggregation-mode.",
    ] = None
    max_total_tokens: A[
        Optional[int],
        Arg(
            help=(
                "The maximum number of tokens in the memory pool. If not "
                "specified, it will be automatically calculated based on the "
                "memory usage fraction. This option is typically used for "
                "development and debugging purposes."
                + f"\n\n{human_readable_int.__doc__}"
            ),
            type_parser=human_readable_int,
        ),
    ] = None
    chunked_prefill_size: A[
        Optional[int],
        "The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill.",
    ] = None
    prefill_decode_interval: A[
        int,
        "The number of decode rounds to run after a prefill batch before scheduling the next prefill. In data-parallel attention mode, the interval is synchronized across all DP ranks. Set to 0 to disable.",
    ] = 0
    enable_dynamic_chunking: A[
        bool,
        "Enable dynamic chunk size adjustment for pipeline parallelism. When enabled, chunk sizes are dynamically calculated based on fitted function to maintain consistent execution time across chunks.",
    ] = False
    max_prefill_tokens: A[
        int,
        Arg(
            help=(
                "The maximum number of tokens in a prefill batch. The real bound "
                "will be the maximum of this value and the model's maximum "
                "context length." + f"\n\n{human_readable_int.__doc__}"
            ),
            type_parser=human_readable_int,
        ),
    ] = 16384
    prefill_max_requests: A[
        Optional[int],
        "The maximum number of requests in a prefill batch. If not specified, there is no limit.",
    ] = None
    schedule_policy: A[
        str,
        Arg(
            help="The scheduling policy of the requests.",
            choices=[
                "lpm",
                "random",
                "fcfs",
                "dfs-weight",
                "lof",
                "priority",
                "routing-key",
            ],
        ),
    ] = "fcfs"
    enable_priority_scheduling: A[
        bool,
        "Enable priority scheduling. Requests with higher priority integer values will be scheduled first by default.",
    ] = False
    disable_priority_preemption: A[
        bool,
        "Disable priority scheduling preemption.",
    ] = False
    default_priority_value: A[
        Optional[int], "Default priority for requests without explicit priority."
    ] = None
    abort_on_priority_when_disabled: A[
        bool,
        "If set, abort requests that specify a priority when priority scheduling is disabled.",
    ] = False
    schedule_low_priority_values_first: A[
        bool,
        "If specified with --enable-priority-scheduling, the scheduler will schedule requests with lower priority integer values first.",
    ] = False
    priority_scheduling_preemption_threshold: A[
        int,
        "Minimum difference in priorities for an incoming request to have to preempt running request(s).",
    ] = 10
    retraction_policy: A[
        str,
        Arg(
            help=(
                "The decode retraction policy to use when the KV cache is full. "
                "'length' preserves the existing behavior and retracts short-output, "
                "long-input requests first. 'priority' retracts lower-priority "
                "requests first, using the same priority direction as priority "
                "scheduling."
            ),
            choices=["length", "priority"],
        ),
    ] = "length"
    schedule_conservativeness: A[
        float,
        "How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
    ] = 1.0
    page_size: A[
        Optional[int], Arg(help="The number of tokens in a page.", resolvable=True)
    ] = None
    c128_page_size: A[
        int,
        "The physical page size of the NPU DSV4 C128 KV cache. Must be a positive multiple of 16.",
    ] = 16
    swa_full_tokens_ratio: A[
        Optional[float],
        Arg(
            help=(
                "The ratio of SWA layer KV tokens / full layer KV tokens, regardless "
                "of the number of swa:full layers. It should be between 0 and 1. "
                "E.g. 0.5 means if each swa layer has 50 tokens, then each full "
                "layer has 100 tokens."
            ),
            resolvable=True,
            fallback=0.8,
        ),
    ] = None
    disable_hybrid_swa_memory: A[
        bool, Arg(help="Disable the hybrid SWA memory pool.", resolvable=True)
    ] = False
    prefill_only_disable_kv_cache: A[
        bool,
        "Skip the physical KV cache allocation for embedding-mode prefill-only workloads. Currently only valid with --is-embedding, --chunked-prefill-size=-1, --disable-radix-cache, an FA prefill backend, and non-FP4 KV cache so the fa_skip_kv_cache path is active (no layer reads or writes the cache). Other prefill-only workloads such as scoring/MIS may benefit from this later once their attention paths stop using paged KV. Scheduler admission accounting is unchanged; per-layer K/V tensors are sized to (page_size, head_num, head_dim) placeholders so GPU memory is not wasted.",
    ] = False
    disable_chunked_prefix_cache: A[
        bool,
        "Disable chunked prefix cache feature for deepseek, which should save overhead for short sequences.",
    ] = False
    disable_overlap_schedule: A[
        bool,
        Arg(
            help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
            resolvable=True,
        ),
    ] = False
    num_continuous_decode_steps: A[
        int,
        "Run multiple continuous decoding steps to reduce scheduling overhead. This can potentially increase throughput but may also increase time-to-first-token latency. The default value is 1, meaning only run one decoding step at a time.",
    ] = 1
    scheduler_recv_interval: A[
        int,
        "The interval to poll requests in scheduler. Can be set to >1 to reduce the overhead of this.",
    ] = 1
    enable_mixed_chunk: A[
        bool,
        "Enabling mixing prefill and decode in a batch when using chunked prefill.",
    ] = False

    # -------------------------------------------------------------------------
    # Mamba cache and linear attn
    # -------------------------------------------------------------------------
    max_mamba_cache_size: A[
        Optional[int],
        "The maximum size of the mamba cache.",
    ] = None
    mamba_full_memory_ratio: A[
        Optional[float],
        Arg(
            help="The ratio of mamba state memory to full kv cache memory.",
            resolvable=True,
            fallback=0.9,
        ),
    ] = None

    # -------------------------------------------------------------------------
    # Prefill delayer
    # -------------------------------------------------------------------------
    enable_prefill_delayer: A[
        bool, "Enable prefill delayer for DP attention to reduce idle time."
    ] = False
    prefill_delayer_max_delay_passes: A[
        int,
        "Maximum forward passes to delay prefill.",
    ] = 30
    prefill_delayer_token_usage_low_watermark: A[
        Optional[float], "Token usage low watermark for prefill delayer."
    ] = None
    prefill_delayer_forward_passes_buckets: A[
        Optional[List[float]],
        "Custom buckets for prefill delayer forward passes histogram. 0 and max_delay_passes-1 will be auto-added.",
    ] = None
    prefill_delayer_wait_seconds_buckets: A[
        Optional[List[float]],
        "Custom buckets for prefill delayer wait seconds histogram. 0 will be auto-added.",
    ] = None
    prefill_delayer_queue_min_ratio: A[
        Optional[float],
        (
            "Opt-in to the adaptive queue-based delay trigger (independent of the "
            "slot-based one). Delays prefill until the waiting queue reaches "
            "min(running_req * ratio, prefill_max_requests), falling back to the "
            "observed max_prefill_bs when no request limit is set. Unset (default) "
            "keeps the original slot-only behavior. Typical: 0.1 ~ 0.5."
        ),
    ] = None
    prefill_delayer_max_delay_ms: A[
        Optional[float],
        (
            "Wall-clock cap (ms) on a single queue-trigger delay; once exceeded, "
            "prefill is force-released to bound worst-case TTFT. Only consulted "
            "when --prefill-delayer-queue-min-ratio is set. Typical: 1000 ~ "
            "5000; defaults to 5000 if unset."
        ),
    ] = None

    # -------------------------------------------------------------------------
    # Min free slots delay (prefill refill batching)
    # -------------------------------------------------------------------------
    min_free_slots_delay: A[
        Optional[int],
        (
            "Hold new prefills until at least N running-request slots have freed "
            "up, so they are admitted in one batch instead of one at a time. "
            "Useful when each admission is disproportionately expensive, e.g. "
            "speculative decoding with a separate draft prefill pass. An "
            "explicit value always wins, capped by max-running-requests "
            "(1 disables). When unset, DFlash workloads auto-enable the "
            "formula; other workloads stay disabled. Not supported with "
            "pipeline parallelism."
        ),
    ] = None
