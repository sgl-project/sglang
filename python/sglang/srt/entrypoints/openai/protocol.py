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

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeAlias, Union

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseOutputItem,
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

DEFAULT_MODEL_NAME = "default"


class ModelCard(BaseModel):
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sglang"
    root: Optional[str] = None
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


class StructuralTagResponseFormat(BaseModel):
    type: Literal["structural_tag"]
    structures: List[StructuresResponseFormat]
    triggers: List[str]


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
    model: str = DEFAULT_MODEL_NAME
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
    no_stop_trim: bool = False
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    session_params: Optional[Dict] = None
    response_format: Optional[Union[ResponseFormat, StructuralTagResponseFormat]] = None

    # For PD disaggregation
    bootstrap_host: Optional[Union[List[str], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[int], int]] = None

    # For request id
    rid: Optional[Union[List[str], str]] = None

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
    arguments: Optional[str] = None


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
    name: Optional[str] = None
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
    model: str = DEFAULT_MODEL_NAME
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
    temperature: float = 0.7
    top_p: float = 1.0
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
        "Currently only supported for OpenAI models.",
    )

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

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    top_k: int = -1
    min_p: float = 0.0
    min_tokens: int = 0
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    repetition_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = None
    no_stop_trim: bool = False
    ignore_eos: bool = False
    continue_final_message: bool = False
    skip_special_tokens: bool = True
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    session_params: Optional[Dict] = None
    separate_reasoning: bool = True
    stream_reasoning: bool = True
    chat_template_kwargs: Optional[Dict] = None

    # For request id
    rid: Optional[Union[List[str], str]] = None

    # For PD disaggregation
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    bootstrap_room: Optional[int] = None


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


class EmbeddingObject(BaseModel):
    embedding: List[float]
    index: int
    object: str = "embedding"


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


OpenAIServingRequest = Union[
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    ScoringRequest,
    V1RerankReqInput,
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
    tool_call_constraint: Optional[Any] = None


class ResponseReasoningTextContent(BaseModel):
    text: str
    type: Literal["reasoning_text"] = "reasoning_text"


ResponseInputOutputItem: TypeAlias = Union[
    ResponseInputItemParam, "ResponseReasoningItem", ResponseFunctionToolCall
]
