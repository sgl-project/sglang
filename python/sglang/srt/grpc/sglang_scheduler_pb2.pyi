import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SamplingParams(_message.Message):
    __slots__ = ("temperature", "top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty", "max_new_tokens", "stop", "stop_token_ids", "skip_special_tokens", "spaces_between_special_tokens", "regex", "json_schema", "ebnf_grammar", "structural_tag", "n", "min_new_tokens", "ignore_eos", "no_stop_trim", "stream_interval", "logit_bias", "custom_params")
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
    STRUCTURAL_TAG_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    MIN_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    IGNORE_EOS_FIELD_NUMBER: _ClassVar[int]
    NO_STOP_TRIM_FIELD_NUMBER: _ClassVar[int]
    STREAM_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    LOGIT_BIAS_FIELD_NUMBER: _ClassVar[int]
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
    structural_tag: str
    n: int
    min_new_tokens: int
    ignore_eos: bool
    no_stop_trim: bool
    stream_interval: int
    logit_bias: _containers.ScalarMap[str, float]
    custom_params: _struct_pb2.Struct
    def __init__(self, temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., top_k: _Optional[int] = ..., min_p: _Optional[float] = ..., frequency_penalty: _Optional[float] = ..., presence_penalty: _Optional[float] = ..., repetition_penalty: _Optional[float] = ..., max_new_tokens: _Optional[int] = ..., stop: _Optional[_Iterable[str]] = ..., stop_token_ids: _Optional[_Iterable[int]] = ..., skip_special_tokens: bool = ..., spaces_between_special_tokens: bool = ..., regex: _Optional[str] = ..., json_schema: _Optional[str] = ..., ebnf_grammar: _Optional[str] = ..., structural_tag: _Optional[str] = ..., n: _Optional[int] = ..., min_new_tokens: _Optional[int] = ..., ignore_eos: bool = ..., no_stop_trim: bool = ..., stream_interval: _Optional[int] = ..., logit_bias: _Optional[_Mapping[str, float]] = ..., custom_params: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class DisaggregatedParams(_message.Message):
    __slots__ = ("bootstrap_host", "bootstrap_port", "bootstrap_room")
    BOOTSTRAP_HOST_FIELD_NUMBER: _ClassVar[int]
    BOOTSTRAP_PORT_FIELD_NUMBER: _ClassVar[int]
    BOOTSTRAP_ROOM_FIELD_NUMBER: _ClassVar[int]
    bootstrap_host: str
    bootstrap_port: int
    bootstrap_room: int
    def __init__(self, bootstrap_host: _Optional[str] = ..., bootstrap_port: _Optional[int] = ..., bootstrap_room: _Optional[int] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("request_id", "tokenized", "mm_inputs", "sampling_params", "return_logprob", "logprob_start_len", "top_logprobs_num", "token_ids_logprob", "return_hidden_states", "disaggregated_params", "custom_logit_processor", "timestamp", "log_metrics", "input_embeds", "lora_id", "data_parallel_rank", "stream")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TOKENIZED_FIELD_NUMBER: _ClassVar[int]
    MM_INPUTS_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMS_FIELD_NUMBER: _ClassVar[int]
    RETURN_LOGPROB_FIELD_NUMBER: _ClassVar[int]
    LOGPROB_START_LEN_FIELD_NUMBER: _ClassVar[int]
    TOP_LOGPROBS_NUM_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_LOGPROB_FIELD_NUMBER: _ClassVar[int]
    RETURN_HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    DISAGGREGATED_PARAMS_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_LOGIT_PROCESSOR_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LOG_METRICS_FIELD_NUMBER: _ClassVar[int]
    INPUT_EMBEDS_FIELD_NUMBER: _ClassVar[int]
    LORA_ID_FIELD_NUMBER: _ClassVar[int]
    DATA_PARALLEL_RANK_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    tokenized: TokenizedInput
    mm_inputs: MultimodalInputs
    sampling_params: SamplingParams
    return_logprob: bool
    logprob_start_len: int
    top_logprobs_num: int
    token_ids_logprob: _containers.RepeatedScalarFieldContainer[int]
    return_hidden_states: bool
    disaggregated_params: DisaggregatedParams
    custom_logit_processor: str
    timestamp: _timestamp_pb2.Timestamp
    log_metrics: bool
    input_embeds: _containers.RepeatedScalarFieldContainer[float]
    lora_id: str
    data_parallel_rank: int
    stream: bool
    def __init__(self, request_id: _Optional[str] = ..., tokenized: _Optional[_Union[TokenizedInput, _Mapping]] = ..., mm_inputs: _Optional[_Union[MultimodalInputs, _Mapping]] = ..., sampling_params: _Optional[_Union[SamplingParams, _Mapping]] = ..., return_logprob: bool = ..., logprob_start_len: _Optional[int] = ..., top_logprobs_num: _Optional[int] = ..., token_ids_logprob: _Optional[_Iterable[int]] = ..., return_hidden_states: bool = ..., disaggregated_params: _Optional[_Union[DisaggregatedParams, _Mapping]] = ..., custom_logit_processor: _Optional[str] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., log_metrics: bool = ..., input_embeds: _Optional[_Iterable[float]] = ..., lora_id: _Optional[str] = ..., data_parallel_rank: _Optional[int] = ..., stream: bool = ...) -> None: ...

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
    __slots__ = ("token_ids", "prompt_tokens", "completion_tokens", "cached_tokens", "output_logprobs", "hidden_states", "input_logprobs", "index")
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    INPUT_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    output_logprobs: OutputLogProbs
    hidden_states: _containers.RepeatedScalarFieldContainer[float]
    input_logprobs: InputLogProbs
    index: int
    def __init__(self, token_ids: _Optional[_Iterable[int]] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ..., output_logprobs: _Optional[_Union[OutputLogProbs, _Mapping]] = ..., hidden_states: _Optional[_Iterable[float]] = ..., input_logprobs: _Optional[_Union[InputLogProbs, _Mapping]] = ..., index: _Optional[int] = ...) -> None: ...

class GenerateComplete(_message.Message):
    __slots__ = ("output_ids", "finish_reason", "prompt_tokens", "completion_tokens", "cached_tokens", "output_logprobs", "all_hidden_states", "matched_token_id", "matched_stop_str", "input_logprobs", "index")
    OUTPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    ALL_HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    MATCHED_TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    MATCHED_STOP_STR_FIELD_NUMBER: _ClassVar[int]
    INPUT_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    output_ids: _containers.RepeatedScalarFieldContainer[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    output_logprobs: OutputLogProbs
    all_hidden_states: _containers.RepeatedCompositeFieldContainer[HiddenStates]
    matched_token_id: int
    matched_stop_str: str
    input_logprobs: InputLogProbs
    index: int
    def __init__(self, output_ids: _Optional[_Iterable[int]] = ..., finish_reason: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ..., output_logprobs: _Optional[_Union[OutputLogProbs, _Mapping]] = ..., all_hidden_states: _Optional[_Iterable[_Union[HiddenStates, _Mapping]]] = ..., matched_token_id: _Optional[int] = ..., matched_stop_str: _Optional[str] = ..., input_logprobs: _Optional[_Union[InputLogProbs, _Mapping]] = ..., index: _Optional[int] = ...) -> None: ...

class GenerateError(_message.Message):
    __slots__ = ("message", "http_status_code", "details")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    HTTP_STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    message: str
    http_status_code: str
    details: str
    def __init__(self, message: _Optional[str] = ..., http_status_code: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...

class OutputLogProbs(_message.Message):
    __slots__ = ("token_logprobs", "token_ids", "top_logprobs")
    TOKEN_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    TOP_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    token_logprobs: _containers.RepeatedScalarFieldContainer[float]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    top_logprobs: _containers.RepeatedCompositeFieldContainer[TopLogProbs]
    def __init__(self, token_logprobs: _Optional[_Iterable[float]] = ..., token_ids: _Optional[_Iterable[int]] = ..., top_logprobs: _Optional[_Iterable[_Union[TopLogProbs, _Mapping]]] = ...) -> None: ...

class InputLogProbs(_message.Message):
    __slots__ = ("token_logprobs", "token_ids", "top_logprobs")
    TOKEN_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    TOP_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    token_logprobs: _containers.RepeatedCompositeFieldContainer[InputTokenLogProb]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    top_logprobs: _containers.RepeatedCompositeFieldContainer[TopLogProbs]
    def __init__(self, token_logprobs: _Optional[_Iterable[_Union[InputTokenLogProb, _Mapping]]] = ..., token_ids: _Optional[_Iterable[int]] = ..., top_logprobs: _Optional[_Iterable[_Union[TopLogProbs, _Mapping]]] = ...) -> None: ...

class InputTokenLogProb(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: float
    def __init__(self, value: _Optional[float] = ...) -> None: ...

class TopLogProbs(_message.Message):
    __slots__ = ("values", "token_ids")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[float]] = ..., token_ids: _Optional[_Iterable[int]] = ...) -> None: ...

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
    __slots__ = ("request_id", "tokenized", "mm_inputs", "sampling_params", "log_metrics", "token_type_ids", "data_parallel_rank", "is_cross_encoder", "texts")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TOKENIZED_FIELD_NUMBER: _ClassVar[int]
    MM_INPUTS_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMS_FIELD_NUMBER: _ClassVar[int]
    LOG_METRICS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_TYPE_IDS_FIELD_NUMBER: _ClassVar[int]
    DATA_PARALLEL_RANK_FIELD_NUMBER: _ClassVar[int]
    IS_CROSS_ENCODER_FIELD_NUMBER: _ClassVar[int]
    TEXTS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    tokenized: TokenizedInput
    mm_inputs: MultimodalInputs
    sampling_params: SamplingParams
    log_metrics: bool
    token_type_ids: _containers.RepeatedScalarFieldContainer[int]
    data_parallel_rank: int
    is_cross_encoder: bool
    texts: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, request_id: _Optional[str] = ..., tokenized: _Optional[_Union[TokenizedInput, _Mapping]] = ..., mm_inputs: _Optional[_Union[MultimodalInputs, _Mapping]] = ..., sampling_params: _Optional[_Union[SamplingParams, _Mapping]] = ..., log_metrics: bool = ..., token_type_ids: _Optional[_Iterable[int]] = ..., data_parallel_rank: _Optional[int] = ..., is_cross_encoder: bool = ..., texts: _Optional[_Iterable[str]] = ...) -> None: ...

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
    __slots__ = ("embedding", "prompt_tokens", "cached_tokens", "embedding_dim", "batch_embeddings")
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIM_FIELD_NUMBER: _ClassVar[int]
    BATCH_EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    embedding: _containers.RepeatedScalarFieldContainer[float]
    prompt_tokens: int
    cached_tokens: int
    embedding_dim: int
    batch_embeddings: _containers.RepeatedCompositeFieldContainer[Embedding]
    def __init__(self, embedding: _Optional[_Iterable[float]] = ..., prompt_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ..., embedding_dim: _Optional[int] = ..., batch_embeddings: _Optional[_Iterable[_Union[Embedding, _Mapping]]] = ...) -> None: ...

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
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("healthy", "message")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    message: str
    def __init__(self, healthy: bool = ..., message: _Optional[str] = ...) -> None: ...

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

class GetModelInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetModelInfoResponse(_message.Message):
    __slots__ = ("model_path", "tokenizer_path", "is_generation", "preferred_sampling_params", "weight_version", "served_model_name", "max_context_length", "vocab_size", "supports_vision", "model_type", "eos_token_ids", "pad_token_id", "bos_token_id", "max_req_input_len")
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    TOKENIZER_PATH_FIELD_NUMBER: _ClassVar[int]
    IS_GENERATION_FIELD_NUMBER: _ClassVar[int]
    PREFERRED_SAMPLING_PARAMS_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_VERSION_FIELD_NUMBER: _ClassVar[int]
    SERVED_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MAX_CONTEXT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    VOCAB_SIZE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_VISION_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    EOS_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    PAD_TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    BOS_TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_REQ_INPUT_LEN_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    tokenizer_path: str
    is_generation: bool
    preferred_sampling_params: str
    weight_version: str
    served_model_name: str
    max_context_length: int
    vocab_size: int
    supports_vision: bool
    model_type: str
    eos_token_ids: _containers.RepeatedScalarFieldContainer[int]
    pad_token_id: int
    bos_token_id: int
    max_req_input_len: int
    def __init__(self, model_path: _Optional[str] = ..., tokenizer_path: _Optional[str] = ..., is_generation: bool = ..., preferred_sampling_params: _Optional[str] = ..., weight_version: _Optional[str] = ..., served_model_name: _Optional[str] = ..., max_context_length: _Optional[int] = ..., vocab_size: _Optional[int] = ..., supports_vision: bool = ..., model_type: _Optional[str] = ..., eos_token_ids: _Optional[_Iterable[int]] = ..., pad_token_id: _Optional[int] = ..., bos_token_id: _Optional[int] = ..., max_req_input_len: _Optional[int] = ...) -> None: ...

class GetServerInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetServerInfoResponse(_message.Message):
    __slots__ = ("server_args", "scheduler_info", "active_requests", "is_paused", "last_receive_timestamp", "uptime_seconds", "sglang_version", "server_type", "start_time")
    SERVER_ARGS_FIELD_NUMBER: _ClassVar[int]
    SCHEDULER_INFO_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    IS_PAUSED_FIELD_NUMBER: _ClassVar[int]
    LAST_RECEIVE_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SGLANG_VERSION_FIELD_NUMBER: _ClassVar[int]
    SERVER_TYPE_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    server_args: _struct_pb2.Struct
    scheduler_info: _struct_pb2.Struct
    active_requests: int
    is_paused: bool
    last_receive_timestamp: float
    uptime_seconds: float
    sglang_version: str
    server_type: str
    start_time: _timestamp_pb2.Timestamp
    def __init__(self, server_args: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., scheduler_info: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., active_requests: _Optional[int] = ..., is_paused: bool = ..., last_receive_timestamp: _Optional[float] = ..., uptime_seconds: _Optional[float] = ..., sglang_version: _Optional[str] = ..., server_type: _Optional[str] = ..., start_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
