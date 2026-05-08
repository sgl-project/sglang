from typing import Any, Dict, List, Optional, Type, Union

import msgspec
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.observability.req_time_stats import (
    SchedulerReqTimeStats,
    ReqTimeStatsBase,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.sampling.sampling_params import SamplingParams


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
]

_msgpack_encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
_msgpack_decoder = msgspec.msgpack.Decoder(_unified_struct, dec_hook=dec_hook)


def serialize(obj: msgspec.Struct) -> bytes:
    return _msgpack_encoder.encode(obj)


def deserialize(data: bytes) -> msgspec.Struct:
    return _msgpack_decoder.decode(data)
