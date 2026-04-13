"""Pydantic models for Anthropic Messages API protocol"""

import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ANTHROPIC_WEB_SEARCH_TOOL_TYPES = frozenset(
    {"web_search_20250305", "web_search_20260209"}
)


def validate_search_result_parts(
    source: object, title: object, content: object
) -> list[str]:
    if not isinstance(source, str):
        raise ValueError("search_result source must be a string")
    if not source:
        raise ValueError("search_result source must be non-empty")
    if not isinstance(title, str):
        raise ValueError("search_result title must be a string")
    if not title:
        raise ValueError("search_result title must be non-empty")
    if not isinstance(content, list):
        raise ValueError("search_result content must be a list")

    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            raise ValueError("search_result content blocks must be dictionaries")
        if item.get("type") != "text":
            raise ValueError("search_result content blocks must be text")
        text = item.get("text")
        if not isinstance(text, str):
            raise ValueError("search_result text must be a string")
        if not text:
            raise ValueError("search_result text must be non-empty")
        text_parts.append(text)

    if not text_parts:
        raise ValueError("search_result content must include at least one text block")

    return text_parts


class AnthropicError(BaseModel):
    """Error structure for Anthropic API"""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response structure for Anthropic API"""

    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    """Token usage information"""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicContentBlock(BaseModel):
    """Content block in message"""

    type: Literal[
        "text",
        "image",
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
        "search_result",
    ]
    text: Optional[str] = None
    # For image content
    source: Optional[dict[str, Any] | str] = None
    title: Optional[str] = None
    # For tool use/result
    id: Optional[str] = None
    tool_use_id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict[str, Any]] = None
    content: Optional[str | list[dict[str, Any]]] = None
    is_error: Optional[bool] = None
    # For thinking content
    thinking: Optional[str] = None
    signature: Optional[str] = None

    @model_validator(mode="after")
    def validate_search_result(self) -> "AnthropicContentBlock":
        if self.type != "search_result":
            return self

        validate_search_result_parts(self.source, self.title, self.content)
        return self


class AnthropicMessage(BaseModel):
    """Message structure"""

    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    input_schema: Optional[dict[str, Any]] = None
    allowed_callers: Optional[list[str]] = None
    allowed_domains: Optional[list[str]] = None
    blocked_domains: Optional[list[str]] = None
    cache_control: Optional[dict[str, Any]] = None
    defer_loading: Optional[bool] = None
    max_uses: Optional[int] = None
    strict: Optional[bool] = None
    user_location: Optional[dict[str, Any]] = None

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v: object) -> dict[str, Any] | None:
        if v is None:
            return v
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"
        return v

    @model_validator(mode="after")
    def validate_tool_shape(self) -> "AnthropicTool":
        if not self.name:
            raise ValueError("tool name must be non-empty")

        if self.type in ANTHROPIC_WEB_SEARCH_TOOL_TYPES:
            return self

        if self.input_schema is None:
            raise ValueError("input_schema is required for custom tools")

        return self


class AnthropicToolChoice(BaseModel):
    """Tool Choice definition"""

    type: Literal["auto", "any", "tool", "none"]
    name: Optional[str] = None


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic Count Tokens API request"""

    model: str
    messages: list[AnthropicMessage]
    system: Optional[str | list[AnthropicContentBlock]] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    tools: Optional[list[AnthropicTool]] = None


class AnthropicCountTokensResponse(BaseModel):
    """Anthropic Count Tokens API response"""

    input_tokens: int


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request"""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: Optional[dict[str, Any]] = None
    stop_sequences: Optional[list[str]] = None
    stream: Optional[bool] = False
    system: Optional[str | list[AnthropicContentBlock]] = None
    temperature: Optional[float] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    tools: Optional[list[AnthropicTool]] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class AnthropicDelta(BaseModel):
    """Delta for streaming responses"""

    type: Optional[Literal["text_delta", "input_json_delta"]] = None
    text: Optional[str] = None
    partial_json: Optional[str] = None

    # Message delta fields
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None


class AnthropicStreamEvent(BaseModel):
    """Streaming event"""

    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: Optional["AnthropicMessagesResponse"] = None
    delta: Optional[AnthropicDelta] = None
    content_block: Optional[AnthropicContentBlock] = None
    index: Optional[int] = None
    error: Optional[AnthropicError] = None
    usage: Optional[AnthropicUsage] = None


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response"""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Optional[AnthropicUsage] = None
