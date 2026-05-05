from typing import Any, Dict, List, Optional, Type, Union

import msgspec
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.observability.req_time_stats import (
    SchedulerReqTimeStats,
)


class LoRAMetrics(msgspec.Struct, tag=True):
    """LoRA adapter pool metrics."""

    slots_used: int
    slots_total: int
    utilization: float


class MemoryMetrics(msgspec.Struct, tag=True):
    """Memory breakdown metrics."""

    weight_gb: float
    kv_cache_gb: float
    graph_gb: float
    token_capacity: int


class SpeculativeMetrics(msgspec.Struct, tag=True):
    """Speculative decoding metrics."""

    accept_length: float
    accept_rate: float


class DisaggregationMetrics(msgspec.Struct, tag=True):
    """PD disaggregation metrics."""

    mode: str  # "prefill", "decode", or "null" - not a metric
    prefill_prealloc_queue_reqs: int
    prefill_inflight_queue_reqs: int
    decode_prealloc_queue_reqs: int
    decode_transfer_queue_reqs: int
    decode_retracted_queue_reqs: int
    kv_transfer_speed_gb_s: float
    kv_transfer_latency_ms: float


class QueueMetrics(msgspec.Struct, tag=True):
    """Detailed queue breakdown."""

    waiting: int
    grammar: int
    paused: int
    retracted: int


class GetLoadsReqOutput(msgspec.Struct, tag=True):
    """Per-DP-rank load metrics for /v1/loads endpoint."""

    # From GetLoadsReqOutput dataclass
    dp_rank: Optional[int] = None
    timestamp: Optional[float] = None

    num_running_reqs: Optional[int] = None
    num_waiting_reqs: Optional[int] = None
    num_used_tokens: Optional[int] = None
    # num_used_tokens + pending prefill tokens (waiting-queue seqlen, incl.
    # disagg bootstrap/prealloc/transfer queues). Used for DP balance.
    num_total_tokens: Optional[int] = None
    max_total_num_tokens: Optional[int] = None
    # FIXME: token_usage is actually max usage across all pools (KV, SWA, mamba),
    # not just KV token usage. Rename requires API deprecation.
    token_usage: Optional[float] = None
    gen_throughput: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    utilization: Optional[float] = None
    max_running_requests: Optional[int] = None

    memory: Optional[MemoryMetrics] = None
    speculative: Optional[SpeculativeMetrics] = None
    lora: Optional[LoRAMetrics] = None
    disaggregation: Optional[DisaggregationMetrics] = None
    queues: Optional[QueueMetrics] = None

    # From BaseBatchReq dataclass
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class FreezeGCReq(msgspec.Struct, tag=True):
    pass


class FlushCacheReqInput(msgspec.Struct, tag=True):
    timeout_s: Optional[float] = None


class FlushCacheReqOutput(msgspec.Struct, tag=True):
    success: bool
    message: str = ""


class BatchTokenIDOutput(msgspec.Struct, tag=True):

    # From SpeculativeDecodingMetricsMixin dataclass
    # Verify count: number of verification forward passes
    spec_verify_ct: List[int]

    # Accepted tokens: Number of accepted tokens during speculative decoding
    spec_accepted_tokens: List[int]

    # Acceptance histogram: List of lists, where each inner list represents histogram counts.
    # List index = number of accepted tokens in a step, List value = count of steps with that many accepted tokens.
    # Example: histogram[0] = 5 means 5 steps with 0 accepted tokens, histogram[3] = 10 means 10 steps with 3 accepted tokens.
    # Empty list [] when speculative decoding is disabled.
    spec_acceptance_histogram: List[List[int]]

    # From BatchTokenIDOutput dataclass
    # The finish reason
    finished_reasons: List[Optional[Dict[str, Any]]]
    # For incremental decoding
    decoded_texts: List[str]
    decode_ids: List[Union[List[int], int]]
    read_offsets: List[int]

    # Detokenization configs
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    no_stop_trim: List[bool]

    # Token counts
    prompt_tokens: List[int]
    reasoning_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    # Only used when `--skip-tokenizer-init` is on
    output_ids: Optional[List[Union[int, List[int]]]] = None

    # Logprobs
    input_token_logprobs_val: Optional[List[float]] = None
    input_token_logprobs_idx: Optional[List[int]] = None
    output_token_logprobs_val: Optional[List[float]] = None
    output_token_logprobs_idx: Optional[List[int]] = None
    input_top_logprobs_val: Optional[List[List]] = None
    input_top_logprobs_idx: Optional[List[List]] = None
    output_top_logprobs_val: Optional[List[List]] = None
    output_top_logprobs_idx: Optional[List[List]] = None
    input_token_ids_logprobs_val: Optional[List[List]] = None
    input_token_ids_logprobs_idx: Optional[List[List]] = None
    output_token_ids_logprobs_val: Optional[List[List]] = None
    output_token_ids_logprobs_idx: Optional[List[List]] = None
    output_token_entropy_val: Optional[List[float]] = None

    # Hidden states
    output_hidden_states: Optional[List[List[float]]] = None

    # The routed experts for each token, including both input and output tokens
    # routed_experts[i] is a tensor of shape (token, layer, top_k) for request i
    routed_experts: Optional[List[Optional[torch.Tensor]]] = None

    # The information of placeholder tokens (e.g., image token)
    # idx is the index of the token in the prompt after expansion.
    # val is the length of padded tokens after expansion.
    placeholder_tokens_idx: Optional[List[Optional[List[int]]]] = None
    placeholder_tokens_val: Optional[List[Optional[List[int]]]] = None

    # Number of times each request was retracted.
    retraction_counts: Optional[List[int]] = None

    # The trainer step id. Used to know which step's weights are used for sampling.
    token_steps: Optional[List[List[int]]] = None

    # Load for DP balance
    load: Optional[GetLoadsReqOutput] = None
    # Customized info
    customized_info: Optional[Dict[str, List[Any]]] = None
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None
    # DP rank of the scheduler that processed each request
    dp_ranks: Optional[List[Optional[int]]] = None

    # For observability
    time_stats: Optional[List[Any]] = None  # Optional[List[SchedulerReqTimeStats]]

    # From BaseBatchReq dataclass
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[str]] = None


class BatchStrOutput(msgspec.Struct, tag=True):

    # From SpeculativeDecodingMetricsMixin dataclass
    # Verify count: number of verification forward passes
    spec_verify_ct: List[int]

    # Accepted tokens: Number of accepted tokens during speculative decoding
    spec_accepted_tokens: List[int]

    # Acceptance histogram: List of lists, where each inner list represents histogram counts.
    # List index = number of accepted tokens in a step, List value = count of steps with that many accepted tokens.
    # Example: histogram[0] = 5 means 5 steps with 0 accepted tokens, histogram[3] = 10 means 10 steps with 3 accepted tokens.
    # Empty list [] when speculative decoding is disabled.
    spec_acceptance_histogram: List[List[int]]

    # From BatchStrOutput dataclass
    # The finish reason
    finished_reasons: List[Optional[Dict[str, Any]]]
    # The output decoded strings
    output_strs: List[str]

    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    reasoning_tokens: List[int]
    cached_tokens: List[int]

    # The token ids
    output_ids: Optional[List[Union[int, List[int]]]] = None

    # Logprobs
    input_token_logprobs_val: Optional[List[float]] = None
    input_token_logprobs_idx: Optional[List[int]] = None
    output_token_logprobs_val: Optional[List[float]] = None
    output_token_logprobs_idx: Optional[List[int]] = None
    input_top_logprobs_val: Optional[List[List]] = None
    input_top_logprobs_idx: Optional[List[List]] = None
    output_top_logprobs_val: Optional[List[List]] = None
    output_top_logprobs_idx: Optional[List[List]] = None
    input_token_ids_logprobs_val: Optional[List[List]] = None
    input_token_ids_logprobs_idx: Optional[List[List]] = None
    output_token_ids_logprobs_val: Optional[List[List]] = None
    output_token_ids_logprobs_idx: Optional[List[List]] = None
    output_token_entropy_val: Optional[List[float]] = None

    # Hidden states
    output_hidden_states: Optional[List[List[float]]] = None

    # The routed experts for each token, including both input and output tokens
    # routed_experts[i] is a tensor of shape (token, layer, top_k) for request i
    routed_experts: Optional[List[Optional[torch.Tensor]]] = None

    # The information of placeholder tokens (e.g., image token)
    # idx is the index of the token in the prompt after expansion.
    # val is the length of padded tokens after expansion.
    placeholder_tokens_idx: Optional[List[Optional[List[int]]]] = None
    placeholder_tokens_val: Optional[List[Optional[List[int]]]] = None

    # Number of times each request was retracted.
    retraction_counts: Optional[List[int]] = None

    # The trainer step id. Used to know which step's weights are used for sampling.
    token_steps: Optional[List[List[int]]] = None

    # Load for DP balance
    load: Optional[GetLoadsReqOutput] = None

    # Customized info
    customized_info: Optional[Dict[str, List[Any]]] = None
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None
    # DP rank of the scheduler that processed each request
    dp_ranks: Optional[List[Optional[int]]] = None

    # For observability
    time_stats: Optional[List[Any]] = None  # Optional[List[SchedulerReqTimeStats]]

    # From BaseBatchReq dataclass
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[str]] = None


class BatchEmbeddingOutput(msgspec.Struct, tag=True):

    # From BatchEmbeddingOutput dataclass
    # The finish reason
    finished_reasons: List[Optional[Dict[str, Any]]]
    # The output embedding
    embeddings: List[Any]  # Union[List[List[float]], List[Dict[int, float]]]
    # Token counts
    prompt_tokens: List[int]
    cached_tokens: List[int]
    # Placeholder token info
    placeholder_tokens_idx: List[Optional[List[int]]] = []
    placeholder_tokens_val: List[Optional[List[int]]] = []

    # Number of times each request was retracted.
    retraction_counts: List[int] = []
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None

    # For observability
    time_stats: Optional[List[Any]] = None  # Optional[List[SchedulerReqTimeStats]]

    # Optional pooled hidden states (pre-head transformer output).
    # Sent as a single stacked tensor to minimize pickle overhead.
    pooled_hidden_states: Optional[Any] = (
        None  # Union[List[Optional[torch.Tensor]], torch.Tensor]
    )

    # From BaseBatchReq dataclass
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[str]] = None


class SchedulerReqTimeStatsIPC(msgspec.Struct, tag=True):

    # From ReqTimeStatsBase
    enable_metrics: bool = False
    disagg_mode: DisaggregationMode = DisaggregationMode.NULL

    # From SchedulerReqTimeStats
    # Placeholder: not used currently
    # propagated from tokenizer/grpc_server or dp controller
    created_time: float = 0.0
    api_server_dispatch_time: float = 0.0
    dpc_dispatch_time: float = 0.0

    # common, get by time.perf_counter()
    wait_queue_entry_time: float = 0.0
    forward_entry_time: float = 0.0
    prefill_run_batch_start_time: float = 0.0
    prefill_run_batch_end_time: float = 0.0
    prefill_finished_time: float = 0.0
    completion_time: float = 0.0

    # prefill node, get by time.perf_counter()
    prefill_bootstrap_queue_entry_time: float = 0.0
    prefill_transfer_queue_entry_time: float = 0.0
    prefill_kv_transfer_finish_time: float = 0.0

    # decode node, get by time.perf_counter()
    decode_prealloc_queue_entry_time: float = 0.0
    decode_transfer_queue_entry_time: float = 0.0
    decode_prebuilt_finish_time: float = 0.0

    # bootstrap sub-phase tracking (PD disagg)
    bootstrap_done_time: float = 0.0

    # only for request tracing
    scheduler_recv_time: float = 0.0
    last_chunked_prefill_finish_time: float = 0.0
    last_decode_finish_time: float = 0.0
    decode_ct: int = 0
    last_decode_scheduled_time: float = 0.0
    last_forward_entry_time: float = 0.0
    last_prefill_finished_time: float = 0.0

    # speculative decoding
    spec_draft_start_time: float = 0.0
    spec_verify_start_time: float = 0.0
    spec_draft_extend_start_time: float = 0.0

    # other
    transfer_speed_gb_s: float = 0.0
    transfer_total_mb: float = 0.0
    # Number of prefill retries for this request
    prefill_retry_count: int = 0

    @classmethod
    def from_req_time_stats(cls, req_stats) -> "SchedulerReqTimeStatsIPC":
        """Convert from SchedulerReqTimeStats to SchedulerReqTimeStatsIPC."""
        if req_stats is None:
            return cls()

        # Create a new instance and copy all matching fields
        ipc_stats = cls()
        for field_name in cls.__struct_fields__:
            if hasattr(req_stats, field_name):
                value = getattr(req_stats, field_name)
                setattr(ipc_stats, field_name, value)

        return ipc_stats

    def to_req_time_stats(self):
        """Convert to SchedulerReqTimeStats instance."""
        from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats

        req_stats = SchedulerReqTimeStats()

        # Copy all matching fields
        for field_name in self.__struct_fields__:
            if hasattr(req_stats, field_name):
                value = getattr(self, field_name)
                setattr(req_stats, field_name, value)

        return req_stats


def enc_hook(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        # encode torch tensor as Tuple(shape, dtype, data)
        tensor_dtype = str(obj.dtype).split(".")[-1]  # e.g., "float32"
        raw_data = obj.flatten().cpu().contiguous().view(torch.uint8).numpy().data
        return (obj.shape, tensor_dtype, raw_data)
    elif isinstance(obj, SchedulerReqTimeStats):
        return SchedulerReqTimeStatsIPC.from_req_time_stats(obj)
    else:
        # Raise a NotImplementedError for other types
        raise NotImplementedError(
            f"Encode objects of type {type(obj)} are not supported"
        )


def dec_hook(type: Type, obj: Any) -> Any:
    # `type` here is the value of the custom type annotation being decoded.
    if type is torch.Tensor:
        # Convert ``obj`` (which should be a ``tuple``) to a torch.Tensor
        shape, dtype, data = obj
        tensor_dtype = getattr(torch, dtype)

        return torch.frombuffer(data, dtype=tensor_dtype).reshape(shape)
    elif type is SchedulerReqTimeStats:
        return SchedulerReqTimeStatsIPC(**obj).to_req_time_stats()
    else:
        # Raise a NotImplementedError for other types
        raise NotImplementedError(f"Decode objects of type {type} are not supported")


_unified_struct = Union[
    FreezeGCReq,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    BatchTokenIDOutput,
    BatchStrOutput,
    BatchEmbeddingOutput,
    GetLoadsReqOutput,
    MemoryMetrics,
    SpeculativeMetrics,
    LoRAMetrics,
    DisaggregationMetrics,
    QueueMetrics,
]

_msgpack_encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
_msgpack_decoder = msgspec.msgpack.Decoder(_unified_struct, dec_hook=dec_hook)


def serialize(obj: msgspec.Struct) -> bytes:
    return _msgpack_encoder.encode(obj)


def deserialize(data: bytes) -> msgspec.Struct:
    return _msgpack_decoder.decode(data)
