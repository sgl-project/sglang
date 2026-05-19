from enum import Enum
import msgspec
from typing import Any, Dict, List, Optional, Type, Union

import torch

from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.observability.req_time_stats import (
    SchedulerReqTimeStats,
    ReqTimeStatsBase,
)

class TensorIPC(msgspec.Struct, tag=True):
    """Compact wire-format for a CPU torch.Tensor.

    Designed so a Rust scheduler / worker can decode it natively without
    needing pickle. Encode/decode helpers live in ``tensor_to_ipc`` and
    ``ipc_to_tensor`` (see ``managers/io_struct/__init__.py``).
    """

    # Raw little-endian byte representation of the tensor's storage.
    data: bytes
    # Tensor shape (use [] for a 0-d scalar tensor).
    shape: List[int]
    # Torch dtype name without the "torch." prefix (e.g. "int64", "int32",
    # "float32"). Both sides use ``torch.{dtype}`` to resolve.
    dtype: str


class DeferredAllocIPC(msgspec.Struct, tag=True):
    """Worker → scheduler GPU-allocated KV slot indices (CpuScheduler path)."""

    mode: str  # "decode" | "extend"
    req_pool_indices: TensorIPC
    out_cache_loc: TensorIPC
    # Decode-only fields.
    seq_lens_minus1: Optional[TensorIPC] = None
    # Extend-only fields.
    prefix_lens: Optional[TensorIPC] = None
    extend_lens: Optional[TensorIPC] = None
    free_pages_remaining: int = 0


class DecodeStepControl(msgspec.Struct, tag=True):
    """M_DECODE_STEP control message: scheduler → worker.

    Delta-only payload for the decode fast path. Stable fields
    (req_pool_indices, lora_ids, forward_mode, etc.) live on the worker's
    cached batch and are not re-sent.

    Rust-decodable wire format. ``sampling_info_pickle`` is the one
    holdout — sampling_info has nested object references and rare
    branches we haven't fully audited, so we ferry it as an opaque blob.
    """

    seq_lens: TensorIPC
    seq_lens_cpu: TensorIPC
    orig_seq_lens: TensorIPC
    seq_lens_sum: int
    # None when input_slot is set — worker resolves input_ids from the
    # FutureMap slot in that case.
    input_ids: Optional[TensorIPC] = None
    indices_to_free: Optional[TensorIPC] = None
    sampling_info_pickle: Optional[bytes] = None
    mamba_track_indices: Optional[TensorIPC] = None
    mamba_track_mask: Optional[TensorIPC] = None
    mamba_track_seqlens: Optional[TensorIPC] = None
    global_num_tokens: Optional[List[int]] = None
    global_num_tokens_for_logprob: Optional[List[int]] = None
    # FutureMap pipeline contract (see TpWorkerServer._future_tokens).
    input_slot: Optional[int] = None
    output_slot: Optional[int] = None


class DecodeForwardReplySlim(msgspec.Struct, tag=True):
    """Slim reply for M_DECODE_STEP / M_FORWARD_GENERATION on the
    CpuScheduler path. Reconstructed into a GenerationBatchResult on the
    scheduler side (see ``TpWorkerClientGroup._maybe_rehydrate_decode_reply``).

    Designed for native decoding from Rust — the rich-Python-object
    fallback fields (``logits_output``, ``routed_experts_output``,
    ``expert_distribution_metrics``) are carried as opaque pickle blobs
    because they only show up on rare paths (logprob streaming, MoE
    metrics) the Rust scheduler will skip for now.
    """

    next_token_ids: TensorIPC
    deferred_alloc: Optional[DeferredAllocIPC] = None
    accept_lens: Optional[TensorIPC] = None
    can_run_cuda_graph: bool = False
    num_accepted_drafts: int = 0
    num_accepted_drafts_per_req_cpu: Optional[List[int]] = None
    # Opaque blobs for the rare paths.
    logits_output_pickle: Optional[bytes] = None
    routed_experts_output_pickle: Optional[bytes] = None
    expert_distribution_metrics_pickle: Optional[bytes] = None
    next_draft_input_pickle: Optional[bytes] = None


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


class GetLoadsReqInput(msgspec.Struct, tag=True):
    # From GetLoadsReqInput dataclass
    """Request for /v1/loads endpoint."""

    VALID_SECTIONS = frozenset(
        {"core", "memory", "spec", "lora", "disagg", "queues", "all"}
    )

    include: List[str] = msgspec.field(default_factory=lambda: ["all"])
    dp_rank: Optional[int] = None

    def __post_init__(self):
        """Validate include sections."""
        if self.include:
            invalid = set(self.include) - self.VALID_SECTIONS
            if invalid:
                raise ValueError(
                    f"Invalid include sections: {invalid}. "
                    f"Valid options: {sorted(self.VALID_SECTIONS)}"
                )


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

    # Accepted drafts tokens: Number of accepted drafts tokens during speculative decoding
    spec_accepted_drafts: List[int]

    # Acceptance histogram: List of lists, where each inner list represents histogram counts.
    # List index = number of accepted drafts tokens in a step, List value = count of steps with that many accepted drafts tokens.
    # Example: histogram[0] = 5 means 5 steps with 0 accepted drafts tokens, histogram[3] = 10 means 10 steps with 3 accepted drafts tokens.
    # Empty list [] when speculative decoding is disabled.
    spec_acceptance_histogram: List[List[int]]

    # From BatchTokenIDOutput dataclass
    # The finish reason
    finished_reasons: List[Optional[Dict[str, Any]]] # List[BaseFinishReason]
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
    time_stats: Optional[List[SchedulerReqTimeStats]] = None

    # From BaseBatchReq dataclass
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[str]] = None


class BatchStrOutput(msgspec.Struct, tag=True):

    # From SpeculativeDecodingMetricsMixin dataclass
    # Verify count: number of verification forward passes
    spec_verify_ct: List[int]

    # Accepted drafts: Number of accepted drafts tokens during speculative decoding
    spec_accepted_drafts: List[int]

    # Acceptance histogram: List of lists, where each inner list represents histogram counts.
    # List index = number of accepted tokens in a step, List value = count of steps with that many accepted tokens.
    # Example: histogram[0] = 5 means 5 steps with 0 accepted tokens, histogram[3] = 10 means 10 steps with 3 accepted tokens.
    # Empty list [] when speculative decoding is disabled.
    spec_acceptance_histogram: List[List[int]]

    # From BatchStrOutput dataclass
    # The finish reason
    finished_reasons: List[Optional[Dict[str, Any]]] # List[BaseFinishReason]
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
    time_stats: Optional[List[SchedulerReqTimeStats]] = None

    # From BaseBatchReq dataclass
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[str]] = None


class BatchEmbeddingOutput(msgspec.Struct, tag=True):

    # From BatchEmbeddingOutput dataclass
    # The finish reason
    finished_reasons: List[Optional[Dict[str, Any]]] # List[BaseFinishReason]
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
    time_stats: Optional[List[SchedulerReqTimeStats]] = None

    # Optional pooled hidden states (pre-head transformer output).
    # Sent as a single stacked tensor to minimize pickle overhead.
    pooled_hidden_states: Optional[Any] = (
        None  # Union[List[Optional[torch.Tensor]], torch.Tensor]
    )

    # From BaseBatchReq dataclass
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[str]] = None


class SessionParams(msgspec.Struct, tag=True):
    id: Optional[str] = None
    rid: Optional[str] = None
    offset: Optional[int] = None
    replace: Optional[bool] = None
    drop_previous_output: Optional[bool] = None

class SamplingParamsIPC(msgspec.Struct, tag=True):
    max_new_tokens: int = 128
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    stop_regex: Optional[Union[str, List[str]]] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    n: int = 1
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    no_stop_trim: bool = False
    custom_params: Optional[Dict[str, Any]] = None
    stream_interval: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    sampling_seed: Optional[int] = None
    stop_str_max_len: int = 0
    stop_regex_max_len: int = 0

    @classmethod
    def from_sampling_params(cls, params: SamplingParams) -> "SamplingParamsIPC":
        return cls(
            max_new_tokens=params.max_new_tokens,
            stop=params.stop_strs,
            stop_token_ids=params.stop_token_ids,
            stop_regex=params.stop_regex_strs,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            min_p=params.min_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            repetition_penalty=params.repetition_penalty,
            min_new_tokens=params.min_new_tokens,
            n=params.n,
            json_schema=params.json_schema,
            regex=params.regex,
            ebnf=params.ebnf,
            structural_tag=params.structural_tag,
            ignore_eos=params.ignore_eos,
            skip_special_tokens=params.skip_special_tokens,
            spaces_between_special_tokens=params.spaces_between_special_tokens,
            no_stop_trim=params.no_stop_trim,
            custom_params=params.custom_params,
            stream_interval=params.stream_interval,
            logit_bias=params.logit_bias,
            sampling_seed=params.sampling_seed,
            stop_str_max_len=params.stop_str_max_len,
            stop_regex_max_len=params.stop_regex_max_len,
        )

    def to_sampling_params(self) -> "SamplingParams":
        param = SamplingParams(
            max_new_tokens=self.max_new_tokens,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            stop_regex=self.stop_regex,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            min_new_tokens=self.min_new_tokens,
            n=self.n,
            json_schema=self.json_schema,
            regex=self.regex,
            ebnf=self.ebnf,
            structural_tag=self.structural_tag,
            ignore_eos=self.ignore_eos,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            no_stop_trim=self.no_stop_trim,
            custom_params=self.custom_params,
            stream_interval=self.stream_interval,
            logit_bias=self.logit_bias,
            sampling_seed=self.sampling_seed,
        )
        param.stop_str_max_len = self.stop_str_max_len
        param.stop_regex_max_len = self.stop_regex_max_len
        return param


class TokenizedGenerateReqInput(msgspec.Struct, tag=True):
    # The input token ids
    input_ids: List[int]
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the logprobs
    return_logprob: bool
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: int
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: int
    # Whether to stream output
    stream: bool

    # The input text
    input_text: Optional[str] = None
    # If return logprobs, the token id to return logprob for
    token_ids_logprob: Optional[List[int]] = None
    # The multimodal inputs
    mm_inputs: Optional[Dict[str, Any]] = None #TODO: @rainj-me need to fix the type of mm_inputs.
    # Whether to return hidden states
    return_hidden_states: bool = False

    # Whether to return captured routed experts
    return_routed_experts: bool = False
    # The start location in the prompt for returning routed experts.
    routed_experts_start_len: int = 0

    # The input embeds
    input_embeds: Optional[List[List[Any]]] = (
        None  # Optional[Union[List[List[List[float]]], List[List[float]]]]
    )

    # Embedding overrides to place at specific token positions.
    positional_embed_overrides: Optional[PositionalEmbeds] = None

    # Session info for continual prompting
    session_params: Optional[SessionParams] = None

    # LoRA related
    lora_id: Optional[str] = None  # None means just use the base model

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[str] = None

    # For disaggregated inference
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    bootstrap_room: Optional[int] = None
    bootstrap_pair_key: Optional[str] = None
    decode_tp_size: Optional[int] = None

    # Require reasoning for the request (hybrid reasoning model only)
    require_reasoning: bool = False

    # For DP routing
    routed_dp_rank: Optional[int] = None
    # For PD disagg — hint telling decode which prefill DP worker has the KV cache
    disagg_prefill_dp_rank: Optional[int] = None

    # Priority for the request
    priority: Optional[int] = None

    # Extra key for classifying the request (e.g. cache_salt)
    extra_key: Optional[str] = None

    # Routing key for routing-key schedule policy
    routing_key: Optional[str] = None

    # Whether to disallow logging for this request (e.g. due to ZDR)
    no_logs: bool = False

    # (Internal) Whether to return bytes for image generation
    return_bytes: bool = False

    # Whether to return entropy
    return_entropy: bool = False

    token_type_ids: Optional[List[int]] = None

    need_wait_for_mm_inputs: Optional[bool] = None
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None

    # Pre-computed delimiter indices for multi-item scoring
    multi_item_delimiter_indices: Optional[List[int]] = None

    # For observability
    time_stats: Optional[ReqTimeStatsBase] = (
        None # Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]] 
    )

    # From BaseReq dataclass
    rid: Optional[str] = None
    http_worker_ipc: Optional[str] = None


class BatchTokenizedGenerateReqInput(msgspec.Struct, tag=True):
    reqs: List[TokenizedGenerateReqInput]

    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[str]] = None


class TokenizedEmbeddingReqInput(msgspec.Struct, tag=True):
    # The input token ids
    input_ids: List[int]
    # Dummy sampling params for compatibility
    sampling_params: SamplingParams  # SamplingParams

    # The input text
    input_text: Optional[str] = None
    # The image inputs
    image_inputs: Optional[dict] = None
    # The token type ids
    token_type_ids: Optional[List[int]] = None
    # Embedding overrides to place at specific token positions.
    positional_embed_overrides: Optional[PositionalEmbeds] = None
    # For DP routing
    routed_dp_rank: Optional[int] = None
    # Priority for the request
    priority: Optional[int] = None
    # The number of dimensions the resulting output embeddings should have. It is applicable for Matryoshka Embeddings.
    dimensions: Optional[int] = None

    # LoRA related
    lora_id: Optional[str] = None  # None means just use the base model
    # Pre-computed delimiter indices for multi-item scoring
    multi_item_delimiter_indices: Optional[List[int]] = None
    # For observability
    time_stats: Optional[ReqTimeStatsBase] = (
        None # Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]] 
    )

    # Whether to return pooled hidden states (pre-head transformer output)
    return_pooled_hidden_states: bool = False

    # From BaseReq dataclass
    rid: Optional[str] = None
    http_worker_ipc: Optional[str] = None


class BatchTokenizedEmbeddingReqInput(msgspec.Struct, tag=True):
    reqs: List[TokenizedEmbeddingReqInput]

    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[str]] = None


class AbortReq(msgspec.Struct, tag=True):
    # From AbortReq dataclass
    abort_all: bool = False
    finished_reason: Optional[Dict[str, Any]] = None
    abort_message: Optional[str] = None

    # From BaseReq dataclass
    rid: Optional[str] = None
    http_worker_ipc: Optional[str] = None


class OpenSessionReqOutput(msgspec.Struct, tag=True):
    # From OpenSessionReqOutput dataclass
    session_id: Optional[str]
    success: bool


class OpenSessionReqInput(msgspec.Struct, tag=True):
    # From OpenSessionReqInput dataclass
    capacity_of_str_len: int
    session_id: Optional[str] = None
    streaming: Optional[bool] = None
    timeout: Optional[float] = None

class CloseSessionReqInput(msgspec.Struct, tag=True):
    # From CloseSessionReqInput dataclass
    session_id: str


class UpdateWeightFromDiskReqInput(msgspec.Struct, tag=True):
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: Optional[str] = None
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Whether to update weights asynchronously
    is_async: bool = False
    # Whether to empty torch cache
    torch_empty_cache: bool = False
    # Whether to keep the scheduler paused after weight update
    keep_pause: bool = False
    # Whether to recapture cuda graph after weight update
    recapture_cuda_graph: bool = False
    # The trainer step id. Used to know which step's weights are used for sampling.
    token_step: int = 0
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Tensor metadata
    manifest: Optional[Dict[str, Any]] = None


class UpdateWeightFromDiskReqOutput(msgspec.Struct, tag=True):
    # From UpdateWeightFromDiskReqOutput dataclass
    success: bool
    message: str
    # Number of paused requests during weight sync.
    num_paused_requests: Optional[int] = 0


class HealthCheckOutput(msgspec.Struct, tag=True):
    # From HealthCheckOutput dataclass
    pass


class ActiveRanksOutput(msgspec.Struct, tag=True):
    # From ActiveRanksOutput dataclass
    status: List[bool]


class InitWeightsUpdateGroupReqInput(msgspec.Struct, tag=True):
    # From InitWeightsUpdateGroupReqInput dataclass
    # The master address
    master_address: str
    # The master port
    master_port: int
    # The rank offset
    rank_offset: int
    # The world size
    world_size: int
    # The group name
    group_name: str = "weight_update_group"
    # The backend
    backend: str = "nccl"


class InitWeightsUpdateGroupReqOutput(msgspec.Struct, tag=True):
    # From InitWeightsUpdateGroupReqOutput dataclass
    success: bool
    message: str


class DestroyWeightsUpdateGroupReqInput(msgspec.Struct, tag=True):
    # From DestroyWeightsUpdateGroupReqInput dataclass
    group_name: str = "weight_update_group"


class DestroyWeightsUpdateGroupReqOutput(msgspec.Struct, tag=True):
    # From DestroyWeightsUpdateGroupReqOutput dataclass
    success: bool
    message: str


class UpdateWeightsFromDistributedReqInput(msgspec.Struct, tag=True):
    # From UpdateWeightsFromDistributedReqInput dataclass
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    # The group name
    group_name: str = "weight_update_group"
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Optional format specification for loading
    load_format: Optional[str] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


class UpdateWeightsFromDistributedReqOutput(msgspec.Struct, tag=True):
    # From UpdateWeightsFromDistributedReqOutput dataclass
    success: bool
    message: str


class InitWeightsSendGroupForRemoteInstanceReqInput(msgspec.Struct, tag=True):
    # From InitWeightsSendGroupForRemoteInstanceReqInput dataclass
    # The master address
    master_address: str
    # The master port
    master_port: int
    # The rank offset
    rank_offset: int
    # The world size
    world_size: int
    # The group name
    group_name: str = "weight_send_group"
    # The backend
    backend: str = "nccl"


class InitWeightsSendGroupForRemoteInstanceReqOutput(msgspec.Struct, tag=True):
    # From InitWeightsSendGroupForRemoteInstanceReqOutput dataclass
    success: bool
    message: str


class SendWeightsToRemoteInstanceReqInput(msgspec.Struct, tag=True):
    # From SendWeightsToRemoteInstanceReqInput dataclass
    # The master address
    master_address: str
    # The ports for each rank's communication group
    ports: str
    # The group name
    group_name: str = "weight_send_group"


class SendWeightsToRemoteInstanceReqOutput(msgspec.Struct, tag=True):
    # From SendWeightsToRemoteInstanceReqOutput dataclass
    success: bool
    message: str


class UpdateWeightsFromTensorReqInput(msgspec.Struct, tag=True):
    # From UpdateWeightsFromTensorReqInput dataclass
    """Update model weights from tensor input.

    - Tensors are serialized for transmission
    - Data is structured in JSON for easy transmission over HTTP
    """

    serialized_named_tensors: List[bytes]
    # Optional format specification for loading
    load_format: Optional[str] = None
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Optional: Determine whether to disable updating the draft model
    disable_draft_model: Optional[bool] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


class UpdateWeightsFromTensorReqOutput(msgspec.Struct, tag=True):
    # From UpdateWeightsFromTensorReqOutput dataclass
    success: bool
    message: str


class UpdateWeightsFromIPCReqInput(msgspec.Struct, tag=True):
    # From UpdateWeightsFromIPCReqInput dataclass
    # ZMQ socket paths for each device UUID
    zmq_handles: Dict[str, str]
    # Whether to flush cache after weight update
    flush_cache: bool = True
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


class UpdateWeightsFromIPCReqOutput(msgspec.Struct, tag=True):
    # From UpdateWeightsFromIPCReqOutput dataclass
    success: bool
    message: str


class GetWeightsByNameReqInput(msgspec.Struct, tag=True):
    # From GetWeightsByNameReqInput dataclass
    name: str
    truncate_size: int = 100


class GetWeightsByNameReqOutput(msgspec.Struct, tag=True):
    # From GetWeightsByNameReqOutput dataclass
    parameter: list


class ReleaseMemoryOccupationReqInput(msgspec.Struct, tag=True):
    # From ReleaseMemoryOccupationReqInput dataclass
    # Optional tags to identify the memory region, which is primarily used for RL
    # Currently we only support `weights` and `kv_cache`
    tags: Optional[List[str]] = None


class ReleaseMemoryOccupationReqOutput(msgspec.Struct, tag=True):
    # From ReleaseMemoryOccupationReqOutput dataclass
    pass


class ResumeMemoryOccupationReqInput(msgspec.Struct, tag=True):
    # From ResumeMemoryOccupationReqInput dataclass
    # Optional tags to identify the memory region, which is primarily used for RL
    # Currently we only support `weights` and `kv_cache`
    tags: Optional[List[str]] = None


class ResumeMemoryOccupationReqOutput(msgspec.Struct, tag=True):
    # From ResumeMemoryOccupationReqOutput dataclass
    pass


class CheckWeightsReqInput(msgspec.Struct, tag=True):
    # From CheckWeightsReqInput dataclass
    action: str


class CheckWeightsReqOutput(msgspec.Struct, tag=True):
    # From CheckWeightsReqOutput dataclass
    success: bool
    message: str


class SlowDownReqInput(msgspec.Struct, tag=True):
    # From SlowDownReqInput dataclass
    forward_sleep_time: Optional[float]


class SlowDownReqOutput(msgspec.Struct, tag=True):
    # From SlowDownReqOutput dataclass
    pass


class FlushCacheReqInput(msgspec.Struct, tag=True):
    # From FlushCacheReqInput dataclass
    timeout_s: Optional[float] = None


class FlushCacheReqOutput(msgspec.Struct, tag=True):
    # From FlushCacheReqOutput dataclass
    success: bool
    message: str


class AddExternalCorpusReqInput(msgspec.Struct, tag=True):
    # From AddExternalCorpusReqInput dataclass
    corpus_id: Optional[str] = None
    file_path: Optional[str] = None
    documents: Optional[List[str]] = None
    token_chunks: Optional[List[List[int]]] = None


class AddExternalCorpusReqOutput(msgspec.Struct, tag=True):
    # From AddExternalCorpusReqOutput dataclass
    success: bool
    corpus_id: str = ""
    message: str = ""
    loaded_token_count: int = 0


class RemoveExternalCorpusReqInput(msgspec.Struct, tag=True):
    # From RemoveExternalCorpusReqInput dataclass
    corpus_id: str


class RemoveExternalCorpusReqOutput(msgspec.Struct, tag=True):
    # From RemoveExternalCorpusReqOutput dataclass
    success: bool
    message: str = ""


class ListExternalCorporaReqInput(msgspec.Struct, tag=True):
    pass


class ListExternalCorporaReqOutput(msgspec.Struct, tag=True):
    # From ListExternalCorporaReqOutput dataclass
    success: bool
    corpus_token_counts: Dict[str, int]
    message: str = ""


class ClearHiCacheReqInput(msgspec.Struct, tag=True):
    pass


class ClearHiCacheReqOutput(msgspec.Struct, tag=True):
    # From ClearHiCacheReqOutput dataclass
    success: bool


class AttachHiCacheStorageReqInput(msgspec.Struct, tag=True):
    # From AttachHiCacheStorageReqInput dataclass
    """Dynamically attach (enable) HiCache storage backend at runtime.

    Note: `hicache_storage_backend_extra_config_json` is a JSON string. It may contain both:
    - backend-specific configs (e.g., mooncake master address)
    - prefetch-related knobs (prefetch_threshold, prefetch_timeout_*, hicache_storage_pass_prefix_keys)
    """

    hicache_storage_backend: str
    hicache_storage_backend_extra_config_json: Optional[str] = None
    hicache_storage_prefetch_policy: Optional[str] = None
    hicache_write_policy: Optional[str] = None

    def __post_init__(self):
        if self.hicache_storage_prefetch_policy is None:
            pass
        else:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if self.hicache_storage_prefetch_policy not in allowed:
                raise ValueError(
                    f"Invalid hicache_storage_prefetch_policy: {self.hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}."
                )

        if self.hicache_write_policy is None:
            return
        allowed = ["write_back", "write_through", "write_through_selective"]
        if self.hicache_write_policy not in allowed:
            raise ValueError(
                f"Invalid hicache_write_policy: {self.hicache_write_policy!r}. "
                f"Expected one of {allowed}."
            )


class AttachHiCacheStorageReqOutput(msgspec.Struct, tag=True):
    # From AttachHiCacheStorageReqOutput dataclass
    success: bool
    message: str = ""


class DetachHiCacheStorageReqInput(msgspec.Struct, tag=True):
    # From DetachHiCacheStorageReqInput dataclass
    pass


class DetachHiCacheStorageReqOutput(msgspec.Struct, tag=True):
    # From DetachHiCacheStorageReqOutput dataclass
    success: bool
    message: str = ""


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


class ProfileReq(msgspec.Struct, tag=True):
    # From ProfileReq dataclass
    profile_type: ProfileReqType
    output_dir: Optional[str] = None
    start_step: Optional[int] = None
    num_steps: Optional[int] = None
    activities: Optional[List[str]] = None
    profile_by_stage: bool = False
    with_stack: Optional[bool] = None
    record_shapes: Optional[bool] = None
    profile_id: Optional[str] = None
    merge_profiles: bool = False
    profile_prefix: Optional[str] = None
    profile_stages: Optional[List[str]] = None


class ProfileReqOutput(msgspec.Struct, tag=True):
    # From ProfileReqOutput dataclass
    success: bool
    message: str = ""


class GetInternalStateReq(msgspec.Struct, tag=True):
    pass


class GetInternalStateReqOutput(msgspec.Struct, tag=True):
    # From GetInternalStateReqOutput dataclass
    internal_state: Dict[Any, Any]


class SetInternalStateReq(msgspec.Struct, tag=True):
    server_args: Dict[str, Any]


class SetInternalStateReqOutput(msgspec.Struct, tag=True):
    # From SetInternalStateReqOutput dataclass
    updated: bool
    server_args: Dict[str, Any]


class ExpertDistributionReqType(Enum):
    START_RECORD = 1
    STOP_RECORD = 2
    DUMP_RECORD = 3


class ExpertDistributionReq(msgspec.Struct, tag=True):
    # From ExpertDistributionReq dataclass
    action: ExpertDistributionReqType


class ExpertDistributionReqOutput(msgspec.Struct, tag=True):
    # From ExpertDistributionReqOutput dataclass
    pass


class LoadLoRAAdapterReqInput(msgspec.Struct, tag=True):
    # From LoadLoRAAdapterReqInput dataclass
    # The name of the lora module to newly loaded.
    lora_name: str
    # The path of loading.
    lora_path: str
    # Whether to pin the LoRA adapter in memory.
    pinned: bool = False
    # The unique identifier for the LoRA adapter, which automatically generated in the `TokenizerManager`.
    lora_id: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path=self.lora_path,
            pinned=self.pinned,
        )


class UnloadLoRAAdapterReqInput(msgspec.Struct, tag=True):
    # From UnloadLoRAAdapterReqInput dataclass
    # The name of lora module to unload.
    lora_name: str
    # The unique identifier for the LoRA adapter, which automatically generated in the `TokenizerManager`.
    lora_id: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
        )


class LoadLoRAAdapterFromTensorsReqInput(msgspec.Struct, tag=True):
    # From LoadLoRAAdapterFromTensorsReqInput dataclass
    lora_name: str
    config_dict: Dict[str, Any]
    serialized_tensors: str
    pinned: bool = False
    added_tokens_config: Optional[Dict[str, Any]] = None
    lora_id: Optional[str] = None
    load_format: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path="__tensor__",
            pinned=self.pinned,
        )


class LoRAUpdateOutput(msgspec.Struct, tag=True):
    # From LoRAUpdateOutput dataclass
    success: bool
    error_message: Optional[str] = None
    loaded_adapters: Optional[Dict[str, LoRARef]] = None


class DumperControlReqInput(msgspec.Struct, tag=True):
    # From DumperControlReqInput dataclass
    method: str
    body: Dict[str, Any]


class DumperControlReqOutput(msgspec.Struct, tag=True):
    # From DumperControlReqOutput dataclass
    success: bool
    response: List[Dict[str, Any]]
    error: str = ""

class WatchLoadUpdateReq(msgspec.Struct, tag=True):
    # From WatchLoadUpdateReq dataclass
    loads: List[GetLoadsReqOutput]


class ContinueGenerationReqInput(msgspec.Struct, tag=True):
    # From ContinueGenerationReqInput dataclass
    pass


class PauseGenerationReqInput(msgspec.Struct, tag=True):
    # From PauseGenerationReqInput dataclass
    mode: str = "abort"

class RpcReqInput(msgspec.Struct, tag=True):
    # From RpcReqInput dataclass
    method: str
    parameters: Optional[Dict] = None

class RpcReqOutput(msgspec.Struct, tag=True):
    # From RpcReqOutput dataclass
    success: bool
    message: str

class BlockReqType(Enum):
    BLOCK = 1
    UNBLOCK = 2

class BlockReqInput(msgspec.Struct, tag=True):
    # From BlockReqInput dataclass
    block_type: BlockReqType


class UpdateExpertBackupReq(msgspec.Struct, tag=True):
    # From UpdateExpertBackupReq dataclass
    pass



class BackupDramReq(msgspec.Struct, tag=True):
    # From BackupDramReq dataclass
    rank: int
    weight_pointer_map: Dict[str, Any]
    session_id: str
    buffer_size: int


def enc_hook(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        # encode torch tensor as Tuple(shape, dtype, data)
        tensor_dtype = str(obj.dtype).split(".")[-1]  # e.g., "float32"
        raw_data = obj.flatten().cpu().contiguous().view(torch.uint8).numpy().data
        return (obj.shape, tensor_dtype, raw_data)
    elif isinstance(obj, SamplingParams):
        return SamplingParamsIPC.from_sampling_params(obj)
    else:
        # Raise a NotImplementedError for other types
        raise NotImplementedError(
            f"Encode objects of type {type(obj)} and value {str(obj)[:32]} are not supported"
        )


def dec_hook(type: Type, obj: Any) -> Any:
    # `type` here is the value of the custom type annotation being decoded.
    if type is torch.Tensor:
        # Convert ``obj`` (which should be a ``tuple``) to a torch.Tensor
        shape, dtype, data = obj
        tensor_dtype = getattr(torch, dtype)

        return torch.frombuffer(data, dtype=tensor_dtype).reshape(shape)
    elif type is SamplingParams:
        obj.pop("type", None)
        return SamplingParamsIPC(**obj).to_sampling_params()
    else:
        # Raise a NotImplementedError for other types
        raise NotImplementedError(
            f"Decode objects of type {type} and value {obj} are not supported"
        )


_unified_struct = Union[
    FreezeGCReq,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    BatchTokenIDOutput,
    BatchStrOutput,
    BatchEmbeddingOutput,
    GetLoadsReqInput,
    GetLoadsReqOutput,
    MemoryMetrics,
    SpeculativeMetrics,
    LoRAMetrics,
    DisaggregationMetrics,
    QueueMetrics,
    SessionParams,
    TokenizedGenerateReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    BatchTokenizedEmbeddingReqInput,
    AbortReq,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    CloseSessionReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    HealthCheckOutput,
    ActiveRanksOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    ProfileReq,
    ProfileReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    LoadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoRAUpdateOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    WatchLoadUpdateReq,
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    RpcReqInput,
    RpcReqOutput,
    BlockReqInput,
    UpdateExpertBackupReq,
    BackupDramReq,
]

_msgpack_encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
_msgpack_decoder = msgspec.msgpack.Decoder(_unified_struct, dec_hook=dec_hook)


def serialize(obj: msgspec.Struct) -> bytes:
    return _msgpack_encoder.encode(obj)


def deserialize(data: bytes) -> msgspec.Struct:
    return _msgpack_decoder.decode(data)
