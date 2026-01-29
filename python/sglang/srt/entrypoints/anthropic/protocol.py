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
"""Incomplete mapping of https://api.anthropic.com/v1/messages API as of November 2025

Supports anthropic-version header validation. Only version 2023-06-01 is supported;
requests with other versions will log a warning but proceed with 2023-06-01 behavior.
"""

import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

ANTHROPIC_API_VERSION = "2023-06-01"


def _generate_message_id() -> str:
    """Generate a unique message ID with msg_ prefix"""
    return f"msg_{uuid.uuid4().hex}"


class AnthropicFinishReason(BaseModel):
    """Internal finish reason from SGLang scheduler"""

    type: Literal["stop", "length", "abort"]
    matched: Optional[Union[int, List[int], str]] = Field(
        default=None, description="Matched stop token or string"
    )
    length: Optional[int] = Field(default=None, description="Maximum length reached")
    message: Optional[str] = Field(default=None, description="Abort message")
    status_code: Optional[int] = Field(
        default=None, description="HTTP status code for abort"
    )
    err_type: Optional[str] = Field(default=None, description="Error type for abort")


class AnthropicSystemBlock(BaseModel):
    """System message content block"""

    type: Literal["text"] = "text"
    text: str = Field(description="Text content of system message")
    cache_control: Optional[Dict[str, Any]] = Field(
        default=None, description="Cache control for prompt caching"
    )


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


# --- Discriminated Content Block Types ---
# Each block type has its own model with required fields properly typed


class TextContentBlock(BaseModel):
    """Text content block"""

    type: Literal["text"] = "text"
    text: str


class ImageContentBlock(BaseModel):
    """Image content block"""

    type: Literal["image"] = "image"
    source: Dict[str, Any]


class ToolUseContentBlock(BaseModel):
    """Tool use content block - represents a tool call from the assistant"""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class ToolResultContentBlock(BaseModel):
    """Tool result content block - represents a tool execution result from the user"""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]]]
    is_error: Optional[bool] = None


class ThinkingContentBlock(BaseModel):
    """Thinking content block - extended reasoning"""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = Field(
        default="N/A",
        description="Cryptographic signature for verification. NOTE: Not implemented.",
    )


class DocumentContentBlock(BaseModel):
    """Document content block"""

    type: Literal["document"] = "document"
    source: Dict[str, Any]


class ServerToolUseContentBlock(BaseModel):
    """Server-side tool use content block"""

    type: Literal["server_tool_use"] = "server_tool_use"
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class ServerToolResultContentBlock(BaseModel):
    """Server-side tool result content block"""

    type: Literal["server_tool_result"] = "server_tool_result"
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]]]


# Discriminated union - Pydantic will parse into correct type based on "type" field
AnthropicContentBlock = Annotated[
    Union[
        TextContentBlock,
        ImageContentBlock,
        ToolUseContentBlock,
        ToolResultContentBlock,
        ThinkingContentBlock,
        DocumentContentBlock,
        ServerToolUseContentBlock,
        ServerToolResultContentBlock,
    ],
    Field(discriminator="type"),
]


class AnthropicMessage(BaseModel):
    """Message structure"""

    role: Literal["user", "assistant"]
    content: Union[str, List[AnthropicContentBlock]]


class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]
    strict: Optional[bool] = Field(
        default=None,
        description="Enable strict schema validation for tool parameters. NOTE: Not implemented, parameter is accepted but ignored.",
    )

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v):
        if "type" not in v:
            v["type"] = "object"
        return v


class AnthropicToolChoice(BaseModel):
    """Tool Choice definition"""

    type: Literal["auto", "any", "tool"]
    name: Optional[str] = Field(
        default=None,
        description="Tool name. Required when type='tool', ignored otherwise.",
    )


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request"""

    model: str = Field(
        min_length=1, description="Model identifier to use for generation"
    )
    messages: List[AnthropicMessage] = Field(
        min_items=1, description="Conversation messages"
    )
    max_tokens: int = Field(gt=0, description="Maximum number of tokens to generate")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata for tracking and attribution"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None, description="Custom strings that will stop generation"
    )
    stream: Optional[bool] = Field(
        default=False, description="Enable server-sent event streaming"
    )
    system: Optional[Union[str, List[AnthropicSystemBlock]]] = Field(
        default=None,
        description="System prompt as string or list of system content blocks. Supports both formats for compatibility.",
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Sampling temperature"
    )
    tool_choice: Optional[AnthropicToolChoice] = Field(
        default=None, description="How the model should use tools"
    )
    tools: Optional[List[AnthropicTool]] = Field(
        default=None, description="Available tools for the model to use"
    )
    top_k: Optional[int] = Field(
        default=None, gt=0, description="Sample from top K options"
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    service_tier: Optional[str] = Field(
        default=None,
        description='Service tier selection: "auto" or "standard_only". NOTE: Not implemented, parameter is accepted but ignored.',
    )
    thinking: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Extended thinking configuration for reasoning models. Supports {"type": "enabled"/"disabled", "budget_tokens": int}. Requires server configured with --reasoning-parser.',
    )
    container: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Container specifications for code execution. NOTE: Not implemented, parameter is accepted but ignored.",
    )
    context_management: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Context handling strategy configuration. NOTE: Not implemented, parameter is accepted but ignored.",
    )
    output_format: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Structured output format with JSON schema (type: "json_schema", schema: {...}). Requires beta header. NOTE: Not implemented, parameter is accepted but ignored.',
    )


class AnthropicDelta(BaseModel):
    """Delta for streaming responses"""

    type: Literal["text_delta", "input_json_delta", "thinking_delta"]
    text: Optional[str] = None
    partial_json: Optional[str] = None
    thinking: Optional[str] = None


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


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response"""

    id: str = Field(
        default_factory=_generate_message_id, description="Unique message identifier"
    )
    type: Literal["message"] = Field(default="message", description="Object type")
    role: Literal["assistant"] = Field(
        default="assistant", description="Conversational role"
    )
    content: List[AnthropicContentBlock] = Field(description="Generated content blocks")
    model: str = Field(description="Model used for generation")
    stop_reason: Optional[
        Literal[
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
            "model_context_window_exceeded",
        ]
    ] = Field(
        default=None,
        description="Reason why generation stopped: end_turn (natural completion), max_tokens (limit reached), "
        "stop_sequence (custom stop sequence matched), tool_use (tool invocation), pause_turn (long turn paused), "
        "refusal (policy intervention), model_context_window_exceeded (context overflow)",
    )
    stop_sequence: Optional[str] = Field(
        default=None, description="The stop sequence that triggered completion, if any"
    )
    usage: AnthropicUsage = Field(description="Token usage information")
    context_management: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Context management information. NOTE: Not implemented, always None.",
    )
    container: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Container execution information. NOTE: Not implemented, always None.",
    )


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic Count Tokens API request

    Accepts the same parameters as Messages API but only counts tokens without generation.
    See: https://platform.claude.com/docs/en/build-with-claude/token-counting
    """

    model: str = Field(min_length=1, description="Model identifier for tokenization")
    messages: List[AnthropicMessage] = Field(
        min_items=1, description="Conversation messages to count tokens for"
    )
    system: Optional[Union[str, List[AnthropicSystemBlock]]] = Field(
        default=None,
        description="System prompt as string or list of system content blocks",
    )
    tools: Optional[List[AnthropicTool]] = Field(
        default=None, description="Tools to include in token count"
    )
    tool_choice: Optional[AnthropicToolChoice] = Field(
        default=None, description="Tool choice configuration"
    )
    thinking: Optional[Dict[str, Any]] = Field(
        default=None, description="Extended thinking configuration"
    )


class AnthropicCountTokensResponse(BaseModel):
    """Anthropic Count Tokens API response"""

    input_tokens: int = Field(description="Number of input tokens")


# Forward reference resolution
AnthropicStreamEvent.model_rebuild()
