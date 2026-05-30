"""Pydantic models for Anthropic Messages API protocol"""

import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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
        "tool_reference",
        "thinking",
        "redacted_thinking",
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

    role: Literal["user", "assistant", "system"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: Optional[str] = None
    input_schema: dict[str, Any]
    defer_loading: Optional[bool] = None

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

    @model_validator(mode="before")
    @classmethod
    def move_mid_conversation_system_messages(cls, values: dict) -> dict:
        """
        Handle mid-conversation system messages by moving them to the system field.

        Some clients (e.g., Claude Code) send system messages mid-conversation
        (after user/assistant messages), which violates Anthropic's API spec.
        This validator extracts those messages and appends them to the
        top-level system field, maintaining compatibility while respecting
        the API structure.
        """
        messages = values.get("messages", [])
        if not messages:
            return values

        clean_messages = []
        extracted_system_texts = []

        for msg in messages:
            if msg.get("role") == "system":
                # Extract system content
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    extracted_system_texts.append(content.strip())
                elif isinstance(content, list):
                    # Handle structured system content
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "").strip()
                            if text:
                                extracted_system_texts.append(text)
            else:
                # Keep non-system messages in place
                clean_messages.append(msg)

        # Append extracted system texts to existing system field
        if extracted_system_texts:
            existing_system = values.get("system")
            combined_system = []

            # Add existing system content first
            if existing_system:
                if isinstance(existing_system, str):
                    if existing_system.strip():
                        combined_system.append(existing_system.strip())
                elif isinstance(existing_system, list):
                    for block in existing_system:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "").strip()
                            if text:
                                combined_system.append(text)

            # Add extracted system texts
            combined_system.extend(extracted_system_texts)

            # Set as string or list based on content type
            if len(combined_system) == 1:
                values["system"] = combined_system[0]
            elif combined_system:
                values["system"] = combined_system

        # Update messages with system messages removed
        values["messages"] = clean_messages
        return values

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
