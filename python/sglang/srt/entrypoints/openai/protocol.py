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
"""Pydantic models for OpenAI API protocol"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypeAlias, Union

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response import ToolChoice
from openai.types.responses.tool import Tool
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)
from typing_extensions import Literal

try:
    from xgrammar import StructuralTag
except:
    StructuralTag = Any

from sglang.utils import convert_json_schema_to_str

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "default"


class ModelCard(BaseModel):
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sglang"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None


class ModelList(BaseModel):
    """Model list consists of model cards."""

    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class TopLogprob(BaseModel):
    token: str
    bytes: List[int]
    logprob: float


class ChatCompletionTokenLogprob(BaseModel):
    token: str
    bytes: List[int]
    logprob: float
    top_logprobs: List[TopLogprob]


class ChoiceLogprobs(BaseModel):
    # build for v1/chat/completions response
    content: List[ChatCompletionTokenLogprob]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    # only used to return cached tokens when --enable-cache-report is set
    prompt_tokens_details: Optional[Dict[str, int]] = None
    reasoning_tokens: Optional[int] = 0


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False
    continuous_usage_stats: Optional[bool] = False


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: Optional[str] = None
    # use alias to workaround pydantic conflict
    schema_: Optional[Dict[str, object]] = Field(alias="schema", default=None)
    strict: Optional[bool] = False


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


class StructuresResponseFormat(BaseModel):
    begin: str
    schema_: Optional[Dict[str, object]] = Field(alias="schema", default=None)
    end: str


# NOTE(dark): keep this for backward compatibility
class LegacyStructuralTagResponseFormat(BaseModel):
    type: Literal["structural_tag"]
    structures: List[StructuresResponseFormat]
    triggers: List[str]


StructuralTagResponseFormat: TypeAlias = Union[
    LegacyStructuralTagResponseFormat, StructuralTag
]

ToolCallConstraint: TypeAlias = Union[
    Tuple[Literal["structural_tag"], StructuralTagResponseFormat],
    Tuple[Literal["json_schema"], Any],  # json_schema can be dict/str/None
]


class FileRequest(BaseModel):
    # https://platform.openai.com/docs/api-reference/files/create
    file: bytes  # The File object (not file name) to be uploaded
    purpose: str = (
        "batch"  # The intended purpose of the uploaded file, default is "batch"
    )


class FileResponse(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str


class FileDeleteResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool


class BatchRequest(BaseModel):
    input_file_id: (
        str  # The ID of an uploaded file that contains requests for the new batch
    )
    endpoint: str  # The endpoint to be used for all requests in the batch
    completion_window: str  # The time frame within which the batch should be processed
    metadata: Optional[dict] = None  # Optional custom metadata for the batch


class BatchResponse(BaseModel):
    id: str
    object: str = "batch"
    endpoint: str
    errors: Optional[dict] = None
    input_file_id: str
    completion_window: str
    status: str = "validating"
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    created_at: int
    in_progress_at: Optional[int] = None
    expires_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expired_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    request_counts: Optional[dict] = None
    metadata: Optional[dict] = None


class CompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Model name. Supports LoRA adapters via 'base-model:adapter-name' syntax.",
    )
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: bool = False
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: int = 16
    n: int = 1
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: float = 1.0
    top_p: float = 1.0
    user: Optional[str] = None
    return_hidden_states: bool = False

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    top_k: int = -1
    min_p: float = 0.0
    min_tokens: int = 0
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    repetition_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = None
    stop_regex: Optional[Union[str, List[str]]] = None
    no_stop_trim: bool = False
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    session_params: Optional[Dict] = None
    response_format: Optional[Union[ResponseFormat, StructuralTagResponseFormat]] = None
    custom_params: Optional[Dict] = None
    custom_logit_processor: Optional[str] = None

    # For PD disaggregation
    bootstrap_host: Optional[Union[List[str], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[int], int]] = None

    # For request id
    rid: Optional[Union[List[str], str]] = None
    # Extra key for classifying the request (e.g. cache_salt)
    extra_key: Optional[Union[List[str], str]] = None
    # Cache salt for request caching
    cache_salt: Optional[Union[List[str], str]] = None
    # Priority for the request
    priority: Optional[int] = None

    # For custom metric labels
    custom_labels: Optional[Dict[str, str]] = None

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter", "abort"]] = None
    matched_stop: Union[None, int, str] = None
    hidden_states: Optional[object] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        return data


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo
    metadata: Optional[Dict[str, Any]] = None


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter", "abort"]] = None
    matched_stop: Union[None, int, str] = None
    hidden_states: Optional[object] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        return data


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class ChatCompletionMessageContentTextPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatCompletionMessageContentImageURL(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class ChatCompletionMessageContentVideoURL(BaseModel):
    url: str


class ChatCompletionMessageContentAudioURL(BaseModel):
    url: str


class ChatCompletionMessageContentImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ChatCompletionMessageContentImageURL
    modalities: Optional[Literal["image", "multi-images", "video"]] = "image"


class ChatCompletionMessageContentVideoPart(BaseModel):
    type: Literal["video_url"]
    video_url: ChatCompletionMessageContentVideoURL


class ChatCompletionMessageContentAudioPart(BaseModel):
    type: Literal["audio_url"]
    audio_url: ChatCompletionMessageContentAudioURL


ChatCompletionMessageContentPart = Union[
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentVideoPart,
    ChatCompletionMessageContentAudioPart,
]


class FunctionResponse(BaseModel):
    """Function response."""

    name: Optional[str] = None
    arguments: Optional[str | Dict[str, Any]] = None


class ToolCall(BaseModel):
    """Tool call response."""

    id: Optional[str] = None
    index: Optional[int] = None
    type: Literal["function"] = "function"
    function: FunctionResponse


class ChatCompletionMessageGenericParam(BaseModel):
    role: Literal["system", "assistant", "tool", "function"]
    content: Union[str, List[ChatCompletionMessageContentTextPart], None] = Field(
        default=None
    )
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, v):
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower not in {"system", "assistant", "tool", "function"}:
                raise ValueError(
                    "'role' must be one of 'system', 'assistant', 'tool', or 'function' (case-insensitive)."
                )
            return v_lower
        raise ValueError("'role' must be a string")


class ChatCompletionMessageUserParam(BaseModel):
    role: Literal["user"]
    content: Union[str, List[ChatCompletionMessageContentPart]]


ChatCompletionMessageParam = Union[
    ChatCompletionMessageGenericParam, ChatCompletionMessageUserParam
]


class Function(BaseModel):
    """Function descriptions."""

    description: Optional[str] = Field(default=None, examples=[None])
    name: str
    parameters: Optional[object] = None
    strict: bool = False


class Tool(BaseModel):
    """Function wrapper."""

    type: str = Field(default="function", examples=["function"])
    function: Function


class ToolChoiceFuncName(BaseModel):
    """The name of tool choice function."""

    name: Optional[str] = None


class ToolChoice(BaseModel):
    """The tool choice definition."""

    function: ToolChoiceFuncName
    type: Literal["function"] = Field(default="function", examples=["function"])


class ChatCompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]
    model: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Model name. Supports LoRA adapters via 'base-model:adapter-name' syntax.",
    )
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: bool = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
        description="The maximum number of tokens that can be generated in the chat completion. ",
    )
    max_completion_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of completion tokens for a chat completion request, "
        "including visible output tokens and reasoning tokens. Input tokens are not included. ",
    )
    n: int = 1
    presence_penalty: float = 0.0
    response_format: Optional[Union[ResponseFormat, StructuralTagResponseFormat]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = Field(
        default="auto", examples=["none"]
    )  # noqa
    return_hidden_states: bool = False
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(
        default="medium",
        description="Constrains effort on reasoning for reasoning models. "
        "'low' is the least effort, 'high' is the most effort. Reducing reasoning effort can "
        "result in faster responses and fewer tokens used on reasoning in a response. "
        "Currently only supported for OpenAI models in the harmony path, i.e GPT-OSS models.",
    )

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    min_tokens: int = 0
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    repetition_penalty: Optional[float] = None
    stop_token_ids: Optional[List[int]] = None
    stop_regex: Optional[Union[str, List[str]]] = None
    no_stop_trim: bool = False
    ignore_eos: bool = False
    continue_final_message: bool = False
    skip_special_tokens: bool = True
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    session_params: Optional[Dict] = None
    separate_reasoning: bool = True
    stream_reasoning: bool = True
    chat_template_kwargs: Optional[Dict] = None

    # Custom logit processor for advanced sampling control
    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None
    custom_params: Optional[Dict] = None

    # For request id
    rid: Optional[Union[List[str], str]] = None
    # Extra key for classifying the request (e.g. cache_salt)
    extra_key: Optional[Union[List[str], str]] = None
    # Cache salt for request caching
    cache_salt: Optional[Union[List[str], str]] = None
    # Priority for the request
    priority: Optional[int] = None

    # For PD disaggregation
    bootstrap_host: Optional[Union[List[str], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[int], int]] = None

    # For data parallel rank routing
    data_parallel_rank: Optional[int] = None
    decode_dp_rank: Optional[int] = None

    # OpenAI/SGLang default sampling parameters
    _DEFAULT_SAMPLING_PARAMS = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }

    @model_validator(mode="before")
    @classmethod
    def set_tool_choice_default(cls, values):
        if values.get("tool_choice") is None:
            if values.get("tools") is None:
                values["tool_choice"] = "none"
            else:
                values["tool_choice"] = "auto"
        return values

    @model_validator(mode="before")
    @classmethod
    def normalize_reasoning_inputs(cls, values: Dict):
        r = values.get("reasoning")
        if r is None:
            return values

        if isinstance(r, dict):
            effort = r.get("effort") or r.get("reasoning_effort")
            if effort in {"low", "medium", "high"}:
                values["reasoning_effort"] = effort

            enabled = (
                r.get("enabled")
                if r.get("enabled") is not None
                else r.get("enable", False)
            )
            if isinstance(enabled, str):
                enabled = enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
            if enabled:
                ctk = values.get("chat_template_kwargs")
                if not isinstance(ctk, dict):
                    ctk = {}
                ctk.setdefault("thinking", True)
                values["chat_template_kwargs"] = ctk

        return values

    @model_validator(mode="before")
    @classmethod
    def set_json_schema(cls, values):
        response_format = values.get("response_format")
        if not response_format:
            return values

        if response_format.get("type") != "json_schema":
            return values

        schema = response_format.pop("schema", None)
        json_schema = response_format.get("json_schema")

        if json_schema:
            return values

        if schema:
            name_ = schema.get("title", "Schema")
            strict_ = False
            if "properties" in schema and "strict" in schema["properties"]:
                item = schema["properties"].pop("strict", None)
                if item and item.get("default", False):
                    strict_ = True

            response_format["json_schema"] = {
                "name": name_,
                "schema": schema,
                "strict": strict_,
            }

        return values

    def to_sampling_params(
        self,
        stop: List[str],
        model_generation_config: Dict[str, Any],
        tool_call_constraint: Optional[ToolCallConstraint] = None,
    ) -> Dict[str, Any]:
        """
        Convert request to sampling parameters.
        Priority: user value > model generation_config > OpenAI defaults
        """

        def get_param(param_name: str):
            value = getattr(self, param_name)
            if value is None:
                return model_generation_config.get(
                    param_name, self._DEFAULT_SAMPLING_PARAMS[param_name]
                )
            return value

        sampling_params = {
            "temperature": get_param("temperature"),
            "max_new_tokens": self.max_tokens or self.max_completion_tokens,
            "min_new_tokens": self.min_tokens,
            "stop": stop,
            "stop_token_ids": self.stop_token_ids,
            "stop_regex": self.stop_regex,
            "top_p": get_param("top_p"),
            "top_k": get_param("top_k"),
            "min_p": get_param("min_p"),
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": get_param("repetition_penalty"),
            "regex": self.regex,
            "ebnf": self.ebnf,
            "n": self.n,
            "no_stop_trim": self.no_stop_trim,
            "ignore_eos": self.ignore_eos,
            "skip_special_tokens": self.skip_special_tokens,
            "logit_bias": self.logit_bias,
            "custom_params": self.custom_params,
        }

        if self.response_format and self.response_format.type == "json_schema":
            sampling_params["json_schema"] = convert_json_schema_to_str(
                self.response_format.json_schema.schema_
            )
        elif self.response_format and self.response_format.type == "json_object":
            sampling_params["json_schema"] = '{"type": "object"}'
        elif self.response_format and self.response_format.type == "structural_tag":
            sampling_params["structural_tag"] = convert_json_schema_to_str(
                self.response_format.model_dump(by_alias=True)
            )

        # Check if there are already existing output constraints
        has_existing_constraints = (
            sampling_params.get("regex")
            or sampling_params.get("ebnf")
            or sampling_params.get("structural_tag")
            or sampling_params.get("json_schema")
        )

        if tool_call_constraint and has_existing_constraints:
            logger.warning("Constrained decoding is not compatible with tool calls.")
        elif tool_call_constraint:
            constraint_type, constraint_value = tool_call_constraint
            if constraint_type == "structural_tag":
                sampling_params[constraint_type] = convert_json_schema_to_str(
                    constraint_value.model_dump(by_alias=True)
                )
            elif constraint_type == "json_schema":
                sampling_params[constraint_type] = convert_json_schema_to_str(
                    constraint_value  # type: ignore
                )
            else:
                sampling_params[constraint_type] = constraint_value

        return sampling_params


class ChatMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None
    finish_reason: Optional[
        Literal[
            "stop", "length", "tool_calls", "content_filter", "function_call", "abort"
        ]
    ] = None
    matched_stop: Union[None, int, str] = None
    hidden_states: Optional[object] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        return data


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    metadata: Optional[Dict[str, Any]] = None


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])
    hidden_states: Optional[object] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        return data


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None
    finish_reason: Optional[
        Literal[
            "stop", "length", "tool_calls", "content_filter", "function_call", "abort"
        ]
    ] = None
    matched_stop: Union[None, int, str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class MultimodalEmbeddingInput(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None


EmbeddingInput = Union[
    List[int], List[List[int]], str, List[str], List[MultimodalEmbeddingInput]
]


class EmbeddingRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings/create
    input: EmbeddingInput
    model: str = DEFAULT_MODEL_NAME
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None

    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Priority for the request
    priority: Optional[int] = None


class EmbeddingObject(BaseModel):
    embedding: List[float]
    index: int
    object: str = "embedding"


ClassifyInput = Union[str, List[str], List[int]]


class ClassifyRequest(BaseModel):
    # OpenAI-compatible classification request
    model: str = DEFAULT_MODEL_NAME
    input: ClassifyInput
    user: Optional[str] = None

    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Priority for the request
    priority: Optional[int] = None


class ClassifyData(BaseModel):
    index: int
    label: str
    probs: List[float]
    num_classes: int


class ClassifyResponse(BaseModel):
    id: str
    object: str = "list"
    created: int
    model: str
    data: List[ClassifyData]
    usage: UsageInfo


class EmbeddingResponse(BaseModel):
    data: List[EmbeddingObject]
    model: str
    object: str = "list"
    usage: Optional[UsageInfo] = None


class ScoringRequest(BaseModel):
    query: Optional[Union[str, List[int]]] = (
        None  # Query text or pre-tokenized token IDs
    )
    items: Optional[Union[str, List[str], List[List[int]]]] = (
        None  # Item text(s) or pre-tokenized token IDs
    )
    label_token_ids: Optional[List[int]] = (
        None  # Token IDs to compute probabilities for
    )
    apply_softmax: bool = False
    item_first: bool = False
    model: str = DEFAULT_MODEL_NAME


class ScoringResponse(BaseModel):
    scores: List[
        List[float]
    ]  # List of lists of probabilities, each in the order of label_token_ids
    model: str
    usage: Optional[UsageInfo] = None
    object: str = "scoring"


class V1RerankReqInput(BaseModel):
    query: str
    documents: List[str]


class RerankResponse(BaseModel):
    score: float
    document: str
    index: int
    meta_info: Optional[dict] = None


class TokenizeRequest(BaseModel):
    """Request schema for the /tokenize endpoint."""

    model: str = DEFAULT_MODEL_NAME
    prompt: Union[str, List[str]]
    add_special_tokens: bool = Field(
        default=True,
        description="whether to add model-specific special tokens (e.g. BOS/EOS) during encoding.",
    )


class TokenizeResponse(BaseModel):
    """Response schema for the /tokenize endpoint."""

    tokens: Union[List[int], List[List[int]]]
    count: Union[int, List[int]]
    max_model_len: int


class DetokenizeRequest(BaseModel):
    """Request schema for the /detokenize endpoint."""

    model: str = DEFAULT_MODEL_NAME
    tokens: Union[List[int], List[List[int]]]
    skip_special_tokens: bool = Field(
        default=True,
        description="whether to exclude special tokens (e.g. padding or EOS) during decoding.",
    )


class DetokenizeResponse(BaseModel):
    """Response schema for the /detokenize endpoint."""

    text: Union[str, List[str]]


OpenAIServingRequest = Union[
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    ClassifyRequest,
    ScoringRequest,
    V1RerankReqInput,
    TokenizeRequest,
    DetokenizeRequest,
]


# Response API protocol definitions
class ResponseReasoningParam(BaseModel):
    """Reasoning parameters for responses."""

    effort: Optional[Literal["low", "medium", "high"]] = Field(
        default="medium",
        description="Constrains effort on reasoning for reasoning models.",
    )


class ResponseTool(BaseModel):
    """Tool definition for responses."""

    type: Literal["web_search_preview", "code_interpreter"] = Field(
        description="Type of tool to enable"
    )


ResponseInputOutputItem: TypeAlias = Union[
    ResponseInputItemParam,
    "ResponseReasoningItem",
    ResponseFunctionToolCall,
]


class ResponsesRequest(BaseModel):
    """Request body for v1/responses endpoint."""

    # Core OpenAI API fields (ordered by official documentation)
    background: Optional[bool] = False
    include: Optional[
        List[
            Literal[
                "code_interpreter_call.outputs",
                "computer_call_output.output.image_url",
                "file_search_call.results",
                "message.input_image.image_url",
                "message.output_text.logprobs",
                "reasoning.encrypted_content",
            ]
        ]
    ] = None
    input: Union[str, List[ResponseInputOutputItem]]
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    model: Optional[str] = None  # Made optional to match vLLM
    parallel_tool_calls: Optional[bool] = True
    previous_response_id: Optional[str] = None
    reasoning: Optional[ResponseReasoningParam] = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] = "auto"
    store: Optional[bool] = True
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    tool_choice: Literal["auto", "required", "none"] = "auto"
    tools: List[ResponseTool] = Field(default_factory=list)
    top_logprobs: Optional[int] = 0
    top_p: Optional[float] = None
    truncation: Optional[Literal["auto", "disabled"]] = "disabled"
    user: Optional[str] = None

    # Extra SGLang parameters
    request_id: str = Field(
        default_factory=lambda: f"resp_{uuid.uuid4().hex}",
        description="The request_id related to this request. If the caller does not set it, a random uuid will be generated.",
    )
    priority: int = Field(default=0, description="Request priority")
    extra_key: Optional[str] = Field(
        default=None,
        description="Extra key for classifying the request (e.g. cache_salt)",
    )
    cache_salt: Optional[str] = Field(
        default=None, description="Cache salt for request caching"
    )

    # SGLang-specific sampling parameters
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    # Default sampling parameters
    _DEFAULT_SAMPLING_PARAMS = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }

    def to_sampling_params(
        self, default_max_tokens: int, default_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Convert to sampling parameters for generation."""
        if default_params is None:
            default_params = {}

        # Use max_output_tokens if available, otherwise use max_tokens for backwards compatibility
        if self.max_output_tokens is not None:
            max_tokens = min(self.max_output_tokens, default_max_tokens)
        else:
            max_tokens = default_max_tokens

        # Avoid exceed the context length by minus 2 token
        max_tokens -= 2

        # Get parameters with defaults
        temperature = self.temperature
        if temperature is None:
            temperature = default_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"]
            )

        top_p = self.top_p
        if top_p is None:
            top_p = default_params.get("top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"])

        params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }

        # Apply any additional default parameters
        for key, value in default_params.items():
            if key not in params or params[key] is None:
                params[key] = value

        return params


class PromptTokenUsageInfo(BaseModel):
    """Prompt token usage details."""

    cached_tokens: int = 0


class ResponsesResponse(BaseModel):
    """Response body for v1/responses endpoint."""

    id: str = Field(default_factory=lambda: f"resp_{time.time()}")
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str

    output: List[
        Union[ResponseOutputItem, ResponseReasoningItem, ResponseFunctionToolCall]
    ] = Field(default_factory=list)
    status: Literal["queued", "in_progress", "completed", "failed", "cancelled"]
    usage: Optional[UsageInfo] = None
    parallel_tool_calls: bool = True
    tool_choice: str = "auto"
    tools: List[ResponseTool] = Field(default_factory=list)

    # OpenAI compatibility fields. not all are used at the moment.
    # Recommend checking https://platform.openai.com/docs/api-reference/responses
    error: Optional[dict] = None
    incomplete_details: Optional[dict] = None  # TODO(v) support this input
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    previous_response_id: Optional[str] = None
    reasoning: Optional[dict] = (
        # Unused. No model supports this. For GPT-oss, system prompt sets
        # the field, not server args.
        None  # {"effort": Optional[str], "summary": Optional[str]}
    )
    store: Optional[bool] = None
    temperature: Optional[float] = None
    text: Optional[dict] = None  # e.g. {"format": {"type": "text"}}
    top_p: Optional[float] = None
    truncation: Optional[str] = None
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_request(
        cls,
        request: ResponsesRequest,
        sampling_params: Any,
        model_name: str,
        created_time: int,
        output: List[
            Union[ResponseOutputItem, ResponseReasoningItem, ResponseFunctionToolCall]
        ],
        status: str,
        usage: Optional[UsageInfo],
    ) -> "ResponsesResponse":
        """Create a response from a request."""

        # Determine if the output is plain text only to set text.format
        def _is_text_only(
            items: List[
                Union[
                    ResponseOutputItem, ResponseReasoningItem, ResponseFunctionToolCall
                ]
            ],
        ) -> bool:
            if not items:
                return False
            for it in items:
                # tool call -> not pure text.
                if isinstance(it, ResponseReasoningItem) or isinstance(
                    it, ResponseFunctionToolCall
                ):
                    return False
                try:
                    if isinstance(it, ResponseOutputText):
                        continue
                    elif isinstance(it, ResponseOutputMessage):
                        if not it.content:
                            continue
                        for c in it.content:
                            if not isinstance(c, ResponseOutputText):
                                return False
                    else:
                        # Unknown type, not considered text-only
                        return False
                except AttributeError:
                    return False
            return True

        text_format = {"format": {"type": "text"}} if _is_text_only(output) else None

        return cls(
            id=request.request_id,
            created_at=created_time,
            model=model_name,
            output=output,
            status=status,
            usage=usage,
            parallel_tool_calls=request.parallel_tool_calls or True,
            tool_choice=request.tool_choice,
            tools=request.tools,
            # fields for parity with v1/responses
            error=None,
            incomplete_details=None,
            instructions=request.instructions,
            max_output_tokens=request.max_output_tokens,
            previous_response_id=request.previous_response_id,  # TODO(v): ensure this is propagated if retrieved from store
            reasoning={
                "effort": request.reasoning.effort if request.reasoning else None,
                "summary": None,  # unused
            },
            store=request.store,
            temperature=request.temperature,
            text=text_format,  # TODO(v): Expand coverage per https://platform.openai.com/docs/api-reference/responses/list
            top_p=request.top_p,
            truncation=request.truncation,
            user=request.user,
            metadata=request.metadata or {},
        )


class RequestResponseMetadata(BaseModel):
    """Metadata for request/response tracking."""

    request_id: str
    final_usage_info: Optional[UsageInfo] = None


@dataclass
class MessageProcessingResult:
    """Result of processing chat messages and applying templates.

    This dataclass encapsulates all the outputs from message processing including
    prompt generation, multimodal data extraction, and constraint preparation.
    Used internally by OpenAIServingChat to pass processed data between methods.

    Args:
        prompt: The final text prompt after applying chat template
        prompt_ids: Either the text prompt (str) or tokenized IDs (List[int])
        image_data: Extracted image data from messages, if any
        audio_data: Extracted audio data from messages, if any
        modalities: List of modality types present in the messages
        stop: Combined stop strings from template and request
        tool_call_constraint: Optional constraint for structured tool calls
    """

    prompt: str
    prompt_ids: Union[str, List[int]]
    image_data: Optional[Any]
    audio_data: Optional[Any]
    video_data: Optional[Any]
    modalities: List[str]
    stop: List[str]
    tool_call_constraint: Optional[ToolCallConstraint] = None


class ToolCallProcessingResult(NamedTuple):
    """Result of processing tool calls in a response."""

    tool_calls: Optional[
        List[Any]
    ]  # List of ToolCall objects or None if parsing failed
    remaining_text: str  # Text remaining after parsing tool calls
    finish_reason: Dict[str, Any]  # Updated finish reason dictionary


class ResponseReasoningTextContent(BaseModel):
    text: str
    type: Literal["reasoning_text"] = "reasoning_text"


ResponseInputOutputItem: TypeAlias = Union[
    ResponseInputItemParam, "ResponseReasoningItem", ResponseFunctionToolCall
]
