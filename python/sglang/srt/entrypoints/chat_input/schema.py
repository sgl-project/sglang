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
"""Shared message, tool, and formatting schemas for chat input preparation."""

from __future__ import annotations

from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypeAlias,
    Union,
    get_args,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    StrictBool,
    field_validator,
    model_serializer,
    model_validator,
)
from typing_extensions import Literal

try:
    from xgrammar import StructuralTag
except:
    StructuralTag = Any


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: Optional[str] = None
    # use alias to workaround pydantic conflict
    schema_: Optional[Dict[str, object]] = Field(alias="schema", default=None)
    # The OpenAI wire contract accepts JSON booleans only; StrictBool rejects
    # the values lax pydantic would coerce ("yes", "on", 0, 1, ...), matching
    # OpenAI's 422 behavior. Omitted (None) keeps its meaning.
    strict: Optional[StrictBool] = None


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
    at_least_one: bool = False


StructuralTagResponseFormat: TypeAlias = Union[
    LegacyStructuralTagResponseFormat, StructuralTag
]

ToolCallConstraint: TypeAlias = Union[
    Tuple[Literal["structural_tag"], StructuralTagResponseFormat],
    Tuple[Literal["json_schema"], Any],  # json_schema can be dict/str/None
]


class ChatCompletionMessageContentTextPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatCompletionMessageContentThinkingPart(BaseModel):
    type: Literal["thinking", "reasoning"]
    thinking: Optional[str] = None
    text: Optional[str] = None

    @model_validator(mode="after")
    def validate_payload(self):
        if (self.thinking is None) == (self.text is None):
            raise ValueError(
                "thinking parts require exactly one of 'thinking' or 'text'"
            )
        return self


class ChatCompletionMessageContentImageURL(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"
    max_dynamic_patch: Optional[int] = None
    min_dynamic_patch: Optional[int] = None
    content_hash: Optional[str] = None

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: Optional[str]) -> Optional[str]:
        from sglang.srt.multimodal.cache import parse_content_hash

        return parse_content_hash(value)


class ChatCompletionMessageContentVideoURL(BaseModel):
    url: str
    max_dynamic_patch: Optional[int] = None
    min_dynamic_patch: Optional[int] = None
    fps: Optional[float] = None
    max_frames: Optional[int] = None
    max_tokens_per_frame: Optional[int] = None
    max_image_tokens: Optional[int] = None


class ChatCompletionMessageContentAudioURL(BaseModel):
    url: str


class ChatCompletionMessageContentImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ChatCompletionMessageContentImageURL
    modalities: Optional[Literal["image", "multi-images", "video"]] = "image"


class ChatCompletionMessageContentVideoPart(BaseModel):
    type: Literal["video_url"]
    video_url: ChatCompletionMessageContentVideoURL


class ChatCompletionMessageContentInputAudio(BaseModel):
    data: str
    format: Literal["wav", "mp3"]


_AUDIO_FORMAT_TO_MIME_TYPE = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
}


class ChatCompletionMessageContentAudioURLPart(BaseModel):
    type: Literal["audio_url"]
    audio_url: ChatCompletionMessageContentAudioURL


class ChatCompletionMessageContentAudioInlinePart(BaseModel):
    type: Literal["input_audio"]
    input_audio: ChatCompletionMessageContentInputAudio


def _to_audio_url_part(
    part: Union[
        ChatCompletionMessageContentAudioURLPart,
        ChatCompletionMessageContentAudioInlinePart,
    ],
) -> ChatCompletionMessageContentAudioURLPart:
    if isinstance(part, ChatCompletionMessageContentAudioURLPart):
        return part

    audio = part.input_audio
    return ChatCompletionMessageContentAudioURLPart(
        type="audio_url",
        audio_url=ChatCompletionMessageContentAudioURL(
            url=f"data:{_AUDIO_FORMAT_TO_MIME_TYPE[audio.format]};base64,{audio.data}"
        ),
    )


# Audio arrives by reference as `audio_url`, holding a URL or a data URI, or
# inline as OpenAI's `input_audio`, holding base64. Inline audio is converted to
# the equivalent data URI as it validates.
ChatCompletionMessageContentAudioPart = Annotated[
    Union[
        ChatCompletionMessageContentAudioURLPart,
        ChatCompletionMessageContentAudioInlinePart,
    ],
    AfterValidator(_to_audio_url_part),
]


class ChatCompletionMessageContentToolReferenceBlock(BaseModel):
    # GLM-specific extension used alongside `defer_loading` tools. The chat
    # template looks up `tools[*].function.name == tr.name` and renders the
    # referenced tool schemas inline for the current turn. Not part of any
    # OpenAI API; included here so Pydantic accepts the content through the
    # Chat Completions path (the Anthropic endpoint translates its
    # `tool_name` field to `name` before forwarding).
    type: Literal["tool_reference"]
    name: str


ChatCompletionMessageContentPart = Union[
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentThinkingPart,
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentVideoPart,
    ChatCompletionMessageContentAudioPart,
    ChatCompletionMessageContentToolReferenceBlock,
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


_GenericMessageRole = Literal[
    "system", "assistant", "tool", "function", "developer", "latest_reminder"
]
_GENERIC_MESSAGE_ROLES: Tuple[str, ...] = get_args(_GenericMessageRole)


class ChatCompletionMessageGenericParam(BaseModel):
    role: _GenericMessageRole
    content: Union[str, List[ChatCompletionMessageContentPart], None] = Field(
        default=None
    )
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, v):
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower not in _GENERIC_MESSAGE_ROLES:
                allowed = ", ".join(repr(r) for r in _GENERIC_MESSAGE_ROLES)
                raise ValueError(f"'role' must be one of {allowed} (case-insensitive).")
            return v_lower
        raise ValueError("'role' must be a string")

    @model_validator(mode="after")
    def validate_thinking_parts_role(self):
        if self.role != "assistant" and isinstance(self.content, list):
            for part in self.content:
                if isinstance(part, ChatCompletionMessageContentThinkingPart):
                    raise ValueError(
                        "thinking content parts are only valid in assistant messages"
                    )
        return self


class ChatCompletionMessageUserParam(BaseModel):
    role: Literal["user"]
    content: Union[str, List[ChatCompletionMessageContentPart]]

    @model_validator(mode="after")
    def validate_thinking_parts_role(self):
        if isinstance(self.content, list):
            for part in self.content:
                if isinstance(part, ChatCompletionMessageContentThinkingPart):
                    raise ValueError(
                        "thinking content parts are only valid in assistant messages"
                    )
        return self


ChatCompletionMessageParam = Union[
    ChatCompletionMessageGenericParam, ChatCompletionMessageUserParam
]


class Function(BaseModel):
    """Function descriptions."""

    description: Optional[str] = Field(default=None, examples=[None])
    name: str
    parameters: Optional[object] = None
    strict: bool = False
    defer_loading: Optional[bool] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.defer_loading is None:
            data.pop("defer_loading", None)
        return data


class Tool(BaseModel):
    """Function wrapper."""

    type: str = Field(default="function", examples=["function"])
    function: Function
    defer_loading: Optional[bool] = None

    @model_validator(mode="after")
    def _propagate_defer_loading(self) -> Tool:
        if self.defer_loading is not None and self.function.defer_loading is None:
            self.function.defer_loading = self.defer_loading
        return self


# Tool is defined after the message params that reference it, so the forward
# reference has to be resolved explicitly.
ChatCompletionMessageGenericParam.model_rebuild()


class ToolChoiceFuncName(BaseModel):
    """The name of tool choice function."""

    name: Optional[str] = None


class ToolChoice(BaseModel):
    """The tool choice definition."""

    function: ToolChoiceFuncName
    type: Literal["function"] = Field(default="function", examples=["function"])


ReasoningEffortTier = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max"
]
# Chat Completions and /v1/tokenize additionally accept a fine-grained float in
# [0.0, 0.99] as an sglang extension (not part of the OpenAI schema, so the
# /v1/responses surface deliberately keeps the string tiers only). Single-sourced
# so these surfaces cannot drift apart.
ReasoningEffortType = Optional[
    Union[
        ReasoningEffortTier,
        Annotated[float, Field(ge=0.0, le=0.99, allow_inf_nan=False)],
    ]
]
