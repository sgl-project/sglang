import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SamplingParams(_message.Message):
    __slots__ = ("temperature", "top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty", "max_new_tokens", "stop", "stop_token_ids", "skip_special_tokens", "spaces_between_special_tokens", "regex", "json_schema", "ebnf_grammar", "lora_path", "n", "token_healing", "min_new_tokens", "ignore_eos", "no_stop_trim", "stream_interval", "logit_bias", "structural_tag", "custom_params")
    class LogitBiasEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    MIN_P_FIELD_NUMBER: _ClassVar[int]
    FREQUENCY_PENALTY_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    REPETITION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    MAX_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    STOP_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    SKIP_SPECIAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SPACES_BETWEEN_SPECIAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    REGEX_FIELD_NUMBER: _ClassVar[int]
    JSON_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    EBNF_GRAMMAR_FIELD_NUMBER: _ClassVar[int]
    LORA_PATH_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    TOKEN_HEALING_FIELD_NUMBER: _ClassVar[int]
    MIN_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    IGNORE_EOS_FIELD_NUMBER: _ClassVar[int]
    NO_STOP_TRIM_FIELD_NUMBER: _ClassVar[int]
    STREAM_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    LOGIT_BIAS_FIELD_NUMBER: _ClassVar[int]
    STRUCTURAL_TAG_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_PARAMS_FIELD_NUMBER: _ClassVar[int]
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float
    max_new_tokens: int
    stop: _containers.RepeatedScalarFieldContainer[str]
    stop_token_ids: _containers.RepeatedScalarFieldContainer[int]
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    regex: str
    json_schema: str
    ebnf_grammar: str
    lora_path: str
    n: int
    token_healing: bool
    min_new_tokens: int
    ignore_eos: bool
    no_stop_trim: bool
    stream_interval: int
    logit_bias: _containers.ScalarMap[str, float]
    structural_tag: str
    custom_params: _struct_pb2.Struct
    def __init__(self, temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., top_k: _Optional[int] = ..., min_p: _Optional[float] = ..., frequency_penalty: _Optional[float] = ..., presence_penalty: _Optional[float] = ..., repetition_penalty: _Optional[float] = ..., max_new_tokens: _Optional[int] = ..., stop: _Optional[_Iterable[str]] = ..., stop_token_ids: _Optional[_Iterable[int]] = ..., skip_special_tokens: bool = ..., spaces_between_special_tokens: bool = ..., regex: _Optional[str] = ..., json_schema: _Optional[str] = ..., ebnf_grammar: _Optional[str] = ..., lora_path: _Optional[str] = ..., n: _Optional[int] = ..., token_healing: bool = ..., min_new_tokens: _Optional[int] = ..., ignore_eos: bool = ..., no_stop_trim: bool = ..., stream_interval: _Optional[int] = ..., logit_bias: _Optional[_Mapping[str, float]] = ..., structural_tag: _Optional[str] = ..., custom_params: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class SessionParams(_message.Message):
    __slots__ = ("session_id", "request_id", "offset", "replace", "drop_previous_output")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    REPLACE_FIELD_NUMBER: _ClassVar[int]
    DROP_PREVIOUS_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    request_id: str
    offset: int
    replace: bool
    drop_previous_output: bool
    def __init__(self, session_id: _Optional[str] = ..., request_id: _Optional[str] = ..., offset: _Optional[int] = ..., replace: bool = ..., drop_previous_output: bool = ...) -> None: ...

class DisaggregatedParams(_message.Message):
    __slots__ = ("bootstrap_host", "bootstrap_port", "bootstrap_room")
    BOOTSTRAP_HOST_FIELD_NUMBER: _ClassVar[int]
    BOOTSTRAP_PORT_FIELD_NUMBER: _ClassVar[int]
    BOOTSTRAP_ROOM_FIELD_NUMBER: _ClassVar[int]
    bootstrap_host: str
    bootstrap_port: int
    bootstrap_room: int
    def __init__(self, bootstrap_host: _Optional[str] = ..., bootstrap_port: _Optional[int] = ..., bootstrap_room: _Optional[int] = ...) -> None: ...

class InitializeRequest(_message.Message):
    __slots__ = ("client_id", "client_version", "mode")
    class Mode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        REGULAR: _ClassVar[InitializeRequest.Mode]
        PREFILL: _ClassVar[InitializeRequest.Mode]
        DECODE: _ClassVar[InitializeRequest.Mode]
    REGULAR: InitializeRequest.Mode
    PREFILL: InitializeRequest.Mode
    DECODE: InitializeRequest.Mode
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    client_version: str
    mode: InitializeRequest.Mode
    def __init__(self, client_id: _Optional[str] = ..., client_version: _Optional[str] = ..., mode: _Optional[_Union[InitializeRequest.Mode, str]] = ...) -> None: ...

class InitializeResponse(_message.Message):
    __slots__ = ("success", "scheduler_version", "model_info", "capabilities", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    SCHEDULER_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODEL_INFO_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    scheduler_version: str
    model_info: ModelInfo
    capabilities: ServerCapabilities
    error_message: str
    def __init__(self, success: bool = ..., scheduler_version: _Optional[str] = ..., model_info: _Optional[_Union[ModelInfo, _Mapping]] = ..., capabilities: _Optional[_Union[ServerCapabilities, _Mapping]] = ..., error_message: _Optional[str] = ...) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ("model_name", "max_context_length", "vocab_size", "supports_tool_calling", "supports_vision", "special_tokens", "model_type", "num_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "tokenizer_type", "eos_token_ids", "pad_token_id", "bos_token_id")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MAX_CONTEXT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    VOCAB_SIZE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_TOOL_CALLING_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_VISION_FIELD_NUMBER: _ClassVar[int]
    SPECIAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    NUM_LAYERS_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_SIZE_FIELD_NUMBER: _ClassVar[int]
    NUM_ATTENTION_HEADS_FIELD_NUMBER: _ClassVar[int]
    NUM_KEY_VALUE_HEADS_FIELD_NUMBER: _ClassVar[int]
    TOKENIZER_TYPE_FIELD_NUMBER: _ClassVar[int]
    EOS_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    PAD_TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    BOS_TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    max_context_length: int
    vocab_size: int
    supports_tool_calling: bool
    supports_vision: bool
    special_tokens: _containers.RepeatedScalarFieldContainer[str]
    model_type: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    tokenizer_type: str
    eos_token_ids: _containers.RepeatedScalarFieldContainer[int]
    pad_token_id: int
    bos_token_id: int
    def __init__(self, model_name: _Optional[str] = ..., max_context_length: _Optional[int] = ..., vocab_size: _Optional[int] = ..., supports_tool_calling: bool = ..., supports_vision: bool = ..., special_tokens: _Optional[_Iterable[str]] = ..., model_type: _Optional[str] = ..., num_layers: _Optional[int] = ..., hidden_size: _Optional[int] = ..., num_attention_heads: _Optional[int] = ..., num_key_value_heads: _Optional[int] = ..., tokenizer_type: _Optional[str] = ..., eos_token_ids: _Optional[_Iterable[int]] = ..., pad_token_id: _Optional[int] = ..., bos_token_id: _Optional[int] = ...) -> None: ...

class ServerCapabilities(_message.Message):
    __slots__ = ("continuous_batching", "disaggregated_serving", "speculative_decoding", "max_batch_size", "max_num_batched_tokens", "max_prefill_tokens", "attention_backend", "supports_lora", "supports_grammar", "supports_multimodal", "supported_modalities", "supports_custom_logit_processor", "supports_session", "num_gpus", "gpu_type", "total_gpu_memory", "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")
    CONTINUOUS_BATCHING_FIELD_NUMBER: _ClassVar[int]
    DISAGGREGATED_SERVING_FIELD_NUMBER: _ClassVar[int]
    SPECULATIVE_DECODING_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAX_NUM_BATCHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MAX_PREFILL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_BACKEND_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_LORA_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_GRAMMAR_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_MULTIMODAL_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_MODALITIES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_CUSTOM_LOGIT_PROCESSOR_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_SESSION_FIELD_NUMBER: _ClassVar[int]
    NUM_GPUS_FIELD_NUMBER: _ClassVar[int]
    GPU_TYPE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_GPU_MEMORY_FIELD_NUMBER: _ClassVar[int]
    TENSOR_PARALLEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_PARALLEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    DATA_PARALLEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    continuous_batching: bool
    disaggregated_serving: bool
    speculative_decoding: bool
    max_batch_size: int
    max_num_batched_tokens: int
    max_prefill_tokens: int
    attention_backend: str
    supports_lora: bool
    supports_grammar: bool
    supports_multimodal: bool
    supported_modalities: _containers.RepeatedScalarFieldContainer[str]
    supports_custom_logit_processor: bool
    supports_session: bool
    num_gpus: int
    gpu_type: str
    total_gpu_memory: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    def __init__(self, continuous_batching: bool = ..., disaggregated_serving: bool = ..., speculative_decoding: bool = ..., max_batch_size: _Optional[int] = ..., max_num_batched_tokens: _Optional[int] = ..., max_prefill_tokens: _Optional[int] = ..., attention_backend: _Optional[str] = ..., supports_lora: bool = ..., supports_grammar: bool = ..., supports_multimodal: bool = ..., supported_modalities: _Optional[_Iterable[str]] = ..., supports_custom_logit_processor: bool = ..., supports_session: bool = ..., num_gpus: _Optional[int] = ..., gpu_type: _Optional[str] = ..., total_gpu_memory: _Optional[int] = ..., tensor_parallel_size: _Optional[int] = ..., pipeline_parallel_size: _Optional[int] = ..., data_parallel_size: _Optional[int] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("request_id", "text", "tokenized", "mm_inputs", "sampling_params", "return_logprob", "logprob_start_len", "top_logprobs_num", "token_ids_logprob", "return_hidden_states", "session_params", "disaggregated_params", "custom_logit_processor", "timestamp", "log_metrics", "input_embeds", "lora_id", "data_parallel_rank", "dp_balance_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOKENIZED_FIELD_NUMBER: _ClassVar[int]
    MM_INPUTS_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMS_FIELD_NUMBER: _ClassVar[int]
    RETURN_LOGPROB_FIELD_NUMBER: _ClassVar[int]
    LOGPROB_START_LEN_FIELD_NUMBER: _ClassVar[int]
    TOP_LOGPROBS_NUM_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_LOGPROB_FIELD_NUMBER: _ClassVar[int]
    RETURN_HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    SESSION_PARAMS_FIELD_NUMBER: _ClassVar[int]
    DISAGGREGATED_PARAMS_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_LOGIT_PROCESSOR_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LOG_METRICS_FIELD_NUMBER: _ClassVar[int]
    INPUT_EMBEDS_FIELD_NUMBER: _ClassVar[int]
    LORA_ID_FIELD_NUMBER: _ClassVar[int]
    DATA_PARALLEL_RANK_FIELD_NUMBER: _ClassVar[int]
    DP_BALANCE_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    text: str
    tokenized: TokenizedInput
    mm_inputs: MultimodalInputs
    sampling_params: SamplingParams
    return_logprob: bool
    logprob_start_len: int
    top_logprobs_num: int
    token_ids_logprob: _containers.RepeatedScalarFieldContainer[int]
    return_hidden_states: bool
    session_params: SessionParams
    disaggregated_params: DisaggregatedParams
    custom_logit_processor: str
    timestamp: _timestamp_pb2.Timestamp
    log_metrics: bool
    input_embeds: _containers.RepeatedScalarFieldContainer[float]
    lora_id: str
    data_parallel_rank: int
    dp_balance_id: int
    def __init__(self, request_id: _Optional[str] = ..., text: _Optional[str] = ..., tokenized: _Optional[_Union[TokenizedInput, _Mapping]] = ..., mm_inputs: _Optional[_Union[MultimodalInputs, _Mapping]] = ..., sampling_params: _Optional[_Union[SamplingParams, _Mapping]] = ..., return_logprob: bool = ..., logprob_start_len: _Optional[int] = ..., top_logprobs_num: _Optional[int] = ..., token_ids_logprob: _Optional[_Iterable[int]] = ..., return_hidden_states: bool = ..., session_params: _Optional[_Union[SessionParams, _Mapping]] = ..., disaggregated_params: _Optional[_Union[DisaggregatedParams, _Mapping]] = ..., custom_logit_processor: _Optional[str] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., log_metrics: bool = ..., input_embeds: _Optional[_Iterable[float]] = ..., lora_id: _Optional[str] = ..., data_parallel_rank: _Optional[int] = ..., dp_balance_id: _Optional[int] = ...) -> None: ...

class TokenizedInput(_message.Message):
    __slots__ = ("original_text", "input_ids")
    ORIGINAL_TEXT_FIELD_NUMBER: _ClassVar[int]
    INPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    original_text: str
    input_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, original_text: _Optional[str] = ..., input_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class MultimodalInputs(_message.Message):
    __slots__ = ("image_urls", "video_urls", "audio_urls", "processed_features", "image_data", "video_data", "audio_data", "modalities")
    IMAGE_URLS_FIELD_NUMBER: _ClassVar[int]
    VIDEO_URLS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_URLS_FIELD_NUMBER: _ClassVar[int]
    PROCESSED_FEATURES_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    VIDEO_DATA_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    MODALITIES_FIELD_NUMBER: _ClassVar[int]
    image_urls: _containers.RepeatedScalarFieldContainer[str]
    video_urls: _containers.RepeatedScalarFieldContainer[str]
    audio_urls: _containers.RepeatedScalarFieldContainer[str]
    processed_features: _struct_pb2.Struct
    image_data: _containers.RepeatedScalarFieldContainer[bytes]
    video_data: _containers.RepeatedScalarFieldContainer[bytes]
    audio_data: _containers.RepeatedScalarFieldContainer[bytes]
    modalities: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, image_urls: _Optional[_Iterable[str]] = ..., video_urls: _Optional[_Iterable[str]] = ..., audio_urls: _Optional[_Iterable[str]] = ..., processed_features: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., image_data: _Optional[_Iterable[bytes]] = ..., video_data: _Optional[_Iterable[bytes]] = ..., audio_data: _Optional[_Iterable[bytes]] = ..., modalities: _Optional[_Iterable[str]] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("request_id", "chunk", "complete", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    chunk: GenerateStreamChunk
    complete: GenerateComplete
    error: GenerateError
    def __init__(self, request_id: _Optional[str] = ..., chunk: _Optional[_Union[GenerateStreamChunk, _Mapping]] = ..., complete: _Optional[_Union[GenerateComplete, _Mapping]] = ..., error: _Optional[_Union[GenerateError, _Mapping]] = ...) -> None: ...

class GenerateStreamChunk(_message.Message):
    __slots__ = ("token_id", "text", "prompt_tokens", "completion_tokens", "cached_tokens", "logprobs", "hidden_states", "generation_time", "queue_time")
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    GENERATION_TIME_FIELD_NUMBER: _ClassVar[int]
    QUEUE_TIME_FIELD_NUMBER: _ClassVar[int]
    token_id: int
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    logprobs: LogProbs
    hidden_states: _containers.RepeatedScalarFieldContainer[float]
    generation_time: float
    queue_time: int
    def __init__(self, token_id: _Optional[int] = ..., text: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ..., logprobs: _Optional[_Union[LogProbs, _Mapping]] = ..., hidden_states: _Optional[_Iterable[float]] = ..., generation_time: _Optional[float] = ..., queue_time: _Optional[int] = ...) -> None: ...

class GenerateComplete(_message.Message):
    __slots__ = ("output_ids", "output_text", "finish_reason", "prompt_tokens", "completion_tokens", "cached_tokens", "total_generation_time", "time_to_first_token", "tokens_per_second", "spec_verify_count", "all_logprobs", "all_hidden_states")
    class FinishReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STOP: _ClassVar[GenerateComplete.FinishReason]
        LENGTH: _ClassVar[GenerateComplete.FinishReason]
        EOS_TOKEN: _ClassVar[GenerateComplete.FinishReason]
        STOP_STR: _ClassVar[GenerateComplete.FinishReason]
        ABORT: _ClassVar[GenerateComplete.FinishReason]
    STOP: GenerateComplete.FinishReason
    LENGTH: GenerateComplete.FinishReason
    EOS_TOKEN: GenerateComplete.FinishReason
    STOP_STR: GenerateComplete.FinishReason
    ABORT: GenerateComplete.FinishReason
    OUTPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TEXT_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_GENERATION_TIME_FIELD_NUMBER: _ClassVar[int]
    TIME_TO_FIRST_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TOKENS_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    SPEC_VERIFY_COUNT_FIELD_NUMBER: _ClassVar[int]
    ALL_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    ALL_HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    output_ids: _containers.RepeatedScalarFieldContainer[int]
    output_text: str
    finish_reason: GenerateComplete.FinishReason
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    total_generation_time: float
    time_to_first_token: float
    tokens_per_second: float
    spec_verify_count: int
    all_logprobs: _containers.RepeatedCompositeFieldContainer[LogProbs]
    all_hidden_states: _containers.RepeatedCompositeFieldContainer[HiddenStates]
    def __init__(self, output_ids: _Optional[_Iterable[int]] = ..., output_text: _Optional[str] = ..., finish_reason: _Optional[_Union[GenerateComplete.FinishReason, str]] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ..., total_generation_time: _Optional[float] = ..., time_to_first_token: _Optional[float] = ..., tokens_per_second: _Optional[float] = ..., spec_verify_count: _Optional[int] = ..., all_logprobs: _Optional[_Iterable[_Union[LogProbs, _Mapping]]] = ..., all_hidden_states: _Optional[_Iterable[_Union[HiddenStates, _Mapping]]] = ...) -> None: ...

class GenerateError(_message.Message):
    __slots__ = ("message", "http_status_code", "details")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    HTTP_STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    message: str
    http_status_code: str
    details: str
    def __init__(self, message: _Optional[str] = ..., http_status_code: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...

class LogProbs(_message.Message):
    __slots__ = ("token_logprobs", "token_ids", "top_logprobs", "token_texts")
    TOKEN_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    TOP_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_TEXTS_FIELD_NUMBER: _ClassVar[int]
    token_logprobs: _containers.RepeatedScalarFieldContainer[float]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    top_logprobs: _containers.RepeatedCompositeFieldContainer[TopLogProbs]
    token_texts: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, token_logprobs: _Optional[_Iterable[float]] = ..., token_ids: _Optional[_Iterable[int]] = ..., top_logprobs: _Optional[_Iterable[_Union[TopLogProbs, _Mapping]]] = ..., token_texts: _Optional[_Iterable[str]] = ...) -> None: ...

class TopLogProbs(_message.Message):
    __slots__ = ("values", "token_ids", "token_texts")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_TEXTS_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    token_texts: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, values: _Optional[_Iterable[float]] = ..., token_ids: _Optional[_Iterable[int]] = ..., token_texts: _Optional[_Iterable[str]] = ...) -> None: ...

class HiddenStates(_message.Message):
    __slots__ = ("values", "layer", "position")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    LAYER_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    layer: int
    position: int
    def __init__(self, values: _Optional[_Iterable[float]] = ..., layer: _Optional[int] = ..., position: _Optional[int] = ...) -> None: ...

class EmbedRequest(_message.Message):
    __slots__ = ("request_id", "text", "tokenized", "mm_inputs", "sampling_params", "log_metrics", "token_type_ids", "data_parallel_rank", "is_cross_encoder", "texts")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOKENIZED_FIELD_NUMBER: _ClassVar[int]
    MM_INPUTS_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMS_FIELD_NUMBER: _ClassVar[int]
    LOG_METRICS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_TYPE_IDS_FIELD_NUMBER: _ClassVar[int]
    DATA_PARALLEL_RANK_FIELD_NUMBER: _ClassVar[int]
    IS_CROSS_ENCODER_FIELD_NUMBER: _ClassVar[int]
    TEXTS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    text: str
    tokenized: TokenizedInput
    mm_inputs: MultimodalInputs
    sampling_params: SamplingParams
    log_metrics: bool
    token_type_ids: _containers.RepeatedScalarFieldContainer[int]
    data_parallel_rank: int
    is_cross_encoder: bool
    texts: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, request_id: _Optional[str] = ..., text: _Optional[str] = ..., tokenized: _Optional[_Union[TokenizedInput, _Mapping]] = ..., mm_inputs: _Optional[_Union[MultimodalInputs, _Mapping]] = ..., sampling_params: _Optional[_Union[SamplingParams, _Mapping]] = ..., log_metrics: bool = ..., token_type_ids: _Optional[_Iterable[int]] = ..., data_parallel_rank: _Optional[int] = ..., is_cross_encoder: bool = ..., texts: _Optional[_Iterable[str]] = ...) -> None: ...

class EmbedResponse(_message.Message):
    __slots__ = ("request_id", "complete", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    complete: EmbedComplete
    error: EmbedError
    def __init__(self, request_id: _Optional[str] = ..., complete: _Optional[_Union[EmbedComplete, _Mapping]] = ..., error: _Optional[_Union[EmbedError, _Mapping]] = ...) -> None: ...

class EmbedComplete(_message.Message):
    __slots__ = ("embedding", "prompt_tokens", "cached_tokens", "embedding_dim", "generation_time", "batch_embeddings")
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIM_FIELD_NUMBER: _ClassVar[int]
    GENERATION_TIME_FIELD_NUMBER: _ClassVar[int]
    BATCH_EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    embedding: _containers.RepeatedScalarFieldContainer[float]
    prompt_tokens: int
    cached_tokens: int
    embedding_dim: int
    generation_time: float
    batch_embeddings: _containers.RepeatedCompositeFieldContainer[Embedding]
    def __init__(self, embedding: _Optional[_Iterable[float]] = ..., prompt_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ..., embedding_dim: _Optional[int] = ..., generation_time: _Optional[float] = ..., batch_embeddings: _Optional[_Iterable[_Union[Embedding, _Mapping]]] = ...) -> None: ...

class Embedding(_message.Message):
    __slots__ = ("values", "index")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    index: int
    def __init__(self, values: _Optional[_Iterable[float]] = ..., index: _Optional[int] = ...) -> None: ...

class EmbedError(_message.Message):
    __slots__ = ("message", "code", "details")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    message: str
    code: str
    details: str
    def __init__(self, message: _Optional[str] = ..., code: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ("include_detailed_metrics",)
    INCLUDE_DETAILED_METRICS_FIELD_NUMBER: _ClassVar[int]
    include_detailed_metrics: bool
    def __init__(self, include_detailed_metrics: bool = ...) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("healthy", "num_requests_running", "num_requests_waiting", "gpu_cache_usage", "gpu_memory_usage", "kv_cache_total_blocks", "kv_cache_used_blocks", "kv_cache_hit_rate", "num_grammar_queue_requests", "generation_throughput", "average_queue_time", "average_generation_time", "cpu_usage", "memory_usage", "num_prefill_requests", "num_decode_requests", "detailed_metrics")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    NUM_REQUESTS_RUNNING_FIELD_NUMBER: _ClassVar[int]
    NUM_REQUESTS_WAITING_FIELD_NUMBER: _ClassVar[int]
    GPU_CACHE_USAGE_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_USAGE_FIELD_NUMBER: _ClassVar[int]
    KV_CACHE_TOTAL_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    KV_CACHE_USED_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    KV_CACHE_HIT_RATE_FIELD_NUMBER: _ClassVar[int]
    NUM_GRAMMAR_QUEUE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    GENERATION_THROUGHPUT_FIELD_NUMBER: _ClassVar[int]
    AVERAGE_QUEUE_TIME_FIELD_NUMBER: _ClassVar[int]
    AVERAGE_GENERATION_TIME_FIELD_NUMBER: _ClassVar[int]
    CPU_USAGE_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USAGE_FIELD_NUMBER: _ClassVar[int]
    NUM_PREFILL_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    NUM_DECODE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    DETAILED_METRICS_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    num_requests_running: int
    num_requests_waiting: int
    gpu_cache_usage: float
    gpu_memory_usage: float
    kv_cache_total_blocks: int
    kv_cache_used_blocks: int
    kv_cache_hit_rate: float
    num_grammar_queue_requests: int
    generation_throughput: float
    average_queue_time: float
    average_generation_time: float
    cpu_usage: float
    memory_usage: int
    num_prefill_requests: int
    num_decode_requests: int
    detailed_metrics: _struct_pb2.Struct
    def __init__(self, healthy: bool = ..., num_requests_running: _Optional[int] = ..., num_requests_waiting: _Optional[int] = ..., gpu_cache_usage: _Optional[float] = ..., gpu_memory_usage: _Optional[float] = ..., kv_cache_total_blocks: _Optional[int] = ..., kv_cache_used_blocks: _Optional[int] = ..., kv_cache_hit_rate: _Optional[float] = ..., num_grammar_queue_requests: _Optional[int] = ..., generation_throughput: _Optional[float] = ..., average_queue_time: _Optional[float] = ..., average_generation_time: _Optional[float] = ..., cpu_usage: _Optional[float] = ..., memory_usage: _Optional[int] = ..., num_prefill_requests: _Optional[int] = ..., num_decode_requests: _Optional[int] = ..., detailed_metrics: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class AbortRequest(_message.Message):
    __slots__ = ("request_id", "reason")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    reason: str
    def __init__(self, request_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class AbortResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class FlushCacheRequest(_message.Message):
    __slots__ = ("flush_all", "session_ids")
    FLUSH_ALL_FIELD_NUMBER: _ClassVar[int]
    SESSION_IDS_FIELD_NUMBER: _ClassVar[int]
    flush_all: bool
    session_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, flush_all: bool = ..., session_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class FlushCacheResponse(_message.Message):
    __slots__ = ("success", "num_entries_flushed", "memory_freed", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    NUM_ENTRIES_FLUSHED_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FREED_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    num_entries_flushed: int
    memory_freed: int
    message: str
    def __init__(self, success: bool = ..., num_entries_flushed: _Optional[int] = ..., memory_freed: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class LoadLoRARequest(_message.Message):
    __slots__ = ("adapter_id", "adapter_path", "rank")
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_PATH_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    adapter_id: str
    adapter_path: str
    rank: int
    def __init__(self, adapter_id: _Optional[str] = ..., adapter_path: _Optional[str] = ..., rank: _Optional[int] = ...) -> None: ...

class LoadLoRAResponse(_message.Message):
    __slots__ = ("success", "adapter_id", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    adapter_id: str
    message: str
    def __init__(self, success: bool = ..., adapter_id: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class UnloadLoRARequest(_message.Message):
    __slots__ = ("adapter_id",)
    ADAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    adapter_id: str
    def __init__(self, adapter_id: _Optional[str] = ...) -> None: ...

class UnloadLoRAResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class UpdateWeightsRequest(_message.Message):
    __slots__ = ("disk_path", "tensor_data", "remote_url", "weight_name")
    DISK_PATH_FIELD_NUMBER: _ClassVar[int]
    TENSOR_DATA_FIELD_NUMBER: _ClassVar[int]
    REMOTE_URL_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_NAME_FIELD_NUMBER: _ClassVar[int]
    disk_path: str
    tensor_data: bytes
    remote_url: str
    weight_name: str
    def __init__(self, disk_path: _Optional[str] = ..., tensor_data: _Optional[bytes] = ..., remote_url: _Optional[str] = ..., weight_name: _Optional[str] = ...) -> None: ...

class UpdateWeightsResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class GetInternalStateRequest(_message.Message):
    __slots__ = ("state_keys",)
    STATE_KEYS_FIELD_NUMBER: _ClassVar[int]
    state_keys: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, state_keys: _Optional[_Iterable[str]] = ...) -> None: ...

class GetInternalStateResponse(_message.Message):
    __slots__ = ("state",)
    STATE_FIELD_NUMBER: _ClassVar[int]
    state: _struct_pb2.Struct
    def __init__(self, state: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class SetInternalStateRequest(_message.Message):
    __slots__ = ("state",)
    STATE_FIELD_NUMBER: _ClassVar[int]
    state: _struct_pb2.Struct
    def __init__(self, state: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class SetInternalStateResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...
