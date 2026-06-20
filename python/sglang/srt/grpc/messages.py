"""AUTO-GENERATED from proto via proto/generate_msgspec.sh. Do not edit by hand.

msgpack codec for the SGLang runtime API, generated from
proto/sglang/runtime/v1/sglang.proto. Wire format: msgpack map keyed by proto
field name, default-valued fields omitted. Interoperates with the Rust codec
(rust/sglang-grpc/src/msgpack.rs).

    from sglang.srt.grpc import messages
    blob = messages.encode(req)
    req = messages.decode(blob, messages.TextGenerateRequest)
"""

from __future__ import annotations

import msgspec


class SamplingParams(msgspec.Struct, omit_defaults=True):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int | None = None
    min_new_tokens: int | None = None
    stop: list[str] = msgspec.field(default_factory=list)
    stop_token_ids: list[int] = msgspec.field(default_factory=list)
    ignore_eos: bool | None = None
    n: int | None = None
    json_schema: str | None = None
    regex: str | None = None


class TextGenerateRequest(msgspec.Struct, omit_defaults=True):
    text: str = ""
    sampling_params: SamplingParams | None = None
    stream: bool | None = None
    return_logprob: bool | None = None
    top_logprobs_num: int | None = None
    logprob_start_len: int | None = None
    return_text_in_logprobs: bool | None = None
    rid: str | None = None
    lora_path: str | None = None
    routing_key: str | None = None
    routed_dp_rank: int | None = None
    trace_headers: dict[str, str] = msgspec.field(default_factory=dict)


class TextGenerateResponse(msgspec.Struct, omit_defaults=True):
    text: str = ""
    meta_info: dict[str, str] = msgspec.field(default_factory=dict)
    finished: bool = False


class GenerateRequest(msgspec.Struct, omit_defaults=True):
    input_ids: list[int] = msgspec.field(default_factory=list)
    sampling_params: SamplingParams | None = None
    stream: bool | None = None
    return_logprob: bool | None = None
    top_logprobs_num: int | None = None
    logprob_start_len: int | None = None
    rid: str | None = None
    lora_path: str | None = None
    routing_key: str | None = None
    routed_dp_rank: int | None = None
    trace_headers: dict[str, str] = msgspec.field(default_factory=dict)


class GenerateResponse(msgspec.Struct, omit_defaults=True):
    output_ids: list[int] = msgspec.field(default_factory=list)
    meta_info: dict[str, str] = msgspec.field(default_factory=dict)
    finished: bool = False


class TextEmbedRequest(msgspec.Struct, omit_defaults=True):
    text: str = ""
    rid: str | None = None
    routing_key: str | None = None
    trace_headers: dict[str, str] = msgspec.field(default_factory=dict)


class TextEmbedResponse(msgspec.Struct, omit_defaults=True):
    embedding: list[float] = msgspec.field(default_factory=list)
    meta_info: dict[str, str] = msgspec.field(default_factory=dict)


class EmbedRequest(msgspec.Struct, omit_defaults=True):
    input_ids: list[int] = msgspec.field(default_factory=list)
    rid: str | None = None
    routing_key: str | None = None
    trace_headers: dict[str, str] = msgspec.field(default_factory=dict)


class EmbedResponse(msgspec.Struct, omit_defaults=True):
    embedding: list[float] = msgspec.field(default_factory=list)
    meta_info: dict[str, str] = msgspec.field(default_factory=dict)


class HealthCheckRequest(msgspec.Struct, omit_defaults=True):
    pass


class HealthCheckResponse(msgspec.Struct, omit_defaults=True):
    healthy: bool = False


class GetModelInfoRequest(msgspec.Struct, omit_defaults=True):
    pass


class GetModelInfoResponse(msgspec.Struct, omit_defaults=True):
    model_path: str = ""
    json_info: str = ""


class GetServerInfoRequest(msgspec.Struct, omit_defaults=True):
    pass


class GetServerInfoResponse(msgspec.Struct, omit_defaults=True):
    json_info: str = ""


class AbortRequest(msgspec.Struct, omit_defaults=True):
    rid: str = ""
    abort_all: bool = False


class AbortResponse(msgspec.Struct, omit_defaults=True):
    success: bool = False


class ClassifyRequest(msgspec.Struct, omit_defaults=True):
    text: str = ""
    input_ids: list[int] = msgspec.field(default_factory=list)
    rid: str | None = None
    routing_key: str | None = None
    trace_headers: dict[str, str] = msgspec.field(default_factory=dict)


class ClassifyResponse(msgspec.Struct, omit_defaults=True):
    embedding: list[float] = msgspec.field(default_factory=list)
    meta_info: dict[str, str] = msgspec.field(default_factory=dict)


class TokenizeRequest(msgspec.Struct, omit_defaults=True):
    text: str = ""
    add_special_tokens: bool | None = None


class TokenizeResponse(msgspec.Struct, omit_defaults=True):
    tokens: list[int] = msgspec.field(default_factory=list)
    count: int = 0
    max_model_len: int = 0
    input_text: str = ""


class DetokenizeRequest(msgspec.Struct, omit_defaults=True):
    tokens: list[int] = msgspec.field(default_factory=list)


class DetokenizeResponse(msgspec.Struct, omit_defaults=True):
    text: str = ""


class ListModelsRequest(msgspec.Struct, omit_defaults=True):
    pass


class ModelCard(msgspec.Struct, omit_defaults=True):
    id: str = ""
    root: str = ""
    parent: str | None = None
    max_model_len: int | None = None


class ListModelsResponse(msgspec.Struct, omit_defaults=True):
    models: list[ModelCard] = msgspec.field(default_factory=list)


class GetLoadRequest(msgspec.Struct, omit_defaults=True):
    dp_rank: int | None = None


class GetLoadResponse(msgspec.Struct, omit_defaults=True):
    json_info: str = ""


class FlushCacheRequest(msgspec.Struct, omit_defaults=True):
    pass


class FlushCacheResponse(msgspec.Struct, omit_defaults=True):
    success: bool = False
    message: str = ""


class PauseGenerationRequest(msgspec.Struct, omit_defaults=True):
    mode: str = ""


class PauseGenerationResponse(msgspec.Struct, omit_defaults=True):
    message: str = ""


class ContinueGenerationRequest(msgspec.Struct, omit_defaults=True):
    pass


class ContinueGenerationResponse(msgspec.Struct, omit_defaults=True):
    message: str = ""


class OpenAIRequest(msgspec.Struct, omit_defaults=True):
    json_body: bytes = b""
    trace_headers: dict[str, str] = msgspec.field(default_factory=dict)


class OpenAIStreamChunk(msgspec.Struct, omit_defaults=True):
    json_chunk: bytes = b""
    finished: bool = False


class OpenAIResponse(msgspec.Struct, omit_defaults=True):
    json_body: bytes = b""
    status_code: int = 0


class StartProfileRequest(msgspec.Struct, omit_defaults=True):
    output_dir: str | None = None


class StartProfileResponse(msgspec.Struct, omit_defaults=True):
    message: str = ""


class StopProfileRequest(msgspec.Struct, omit_defaults=True):
    pass


class StopProfileResponse(msgspec.Struct, omit_defaults=True):
    message: str = ""


class UpdateWeightsRequest(msgspec.Struct, omit_defaults=True):
    model_path: str = ""
    load_format: str | None = None


class UpdateWeightsResponse(msgspec.Struct, omit_defaults=True):
    success: bool = False
    message: str = ""


_ENCODER = msgspec.msgpack.Encoder()
_DECODERS: dict = {}


def encode(obj) -> bytes:
    """Serialize a generated message Struct to msgpack bytes."""
    return _ENCODER.encode(obj)


def decode(data: bytes, typ):
    """Deserialize msgpack bytes into the given generated message type."""
    dec = _DECODERS.get(typ)
    if dec is None:
        dec = _DECODERS[typ] = msgspec.msgpack.Decoder(typ)
    return dec.decode(data)
