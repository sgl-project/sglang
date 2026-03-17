"""Pydantic models for Anthropic Messages API protocol"""

import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


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
        "text", "image", "tool_use", "tool_result", "thinking", "redacted_thinking"
    ]
    text: Optional[str] = None
    # For image content
    source: Optional[dict[str, Any]] = None
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


class AnthropicMessage(BaseModel):
    """Message structure"""

    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: Optional[str] = None
    input_schema: dict[str, Any]

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v):
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"
        return v


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
