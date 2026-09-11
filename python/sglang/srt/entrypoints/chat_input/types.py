from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from pydantic import BaseModel

from sglang.srt.entrypoints.chat_input.schema import (
    ChatCompletionMessageParam,
    ReasoningEffortType,
    ResponseFormat,
    StructuralTagResponseFormat,
    Tool,
    ToolCallConstraint,
    ToolChoice,
)


class ChatInput(BaseModel):
    messages: List[ChatCompletionMessageParam]
    tools: Optional[List[Tool]] = None
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "none"
    parallel_tool_calls: bool = True
    response_format: Optional[Union[ResponseFormat, StructuralTagResponseFormat]] = None
    reasoning_effort: ReasoningEffortType = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    continue_final_message: bool = False
    input_ids: Optional[List[int]] = None
    stop: Optional[Union[str, List[str]]] = None
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    task: Optional[
        Literal["action", "query", "authority", "domain", "title", "read_url"]
    ] = None

    def effective_tools(self) -> List[Tool]:
        tools = list(self.tools or [])
        for message in self.messages:
            if message.role in ("system", "developer"):
                tools.extend(getattr(message, "tools", None) or [])
        return tools


@dataclass(frozen=True)
class TextPrompt:
    text: str
    cached_token_ids: Optional[List[int]] = None

    def to_generate_kwargs(self) -> Dict[str, str]:
        return {"text": self.text}

    def tokenize(self, tokenizer) -> List[int]:
        if self.cached_token_ids is not None:
            return list(self.cached_token_ids)
        return tokenizer.encode(self.text, add_special_tokens=False)


@dataclass(frozen=True)
class TokenPrompt:
    token_ids: List[int]

    def to_generate_kwargs(self) -> Dict[str, List[int]]:
        return {"input_ids": list(self.token_ids)}

    def tokenize(self, tokenizer) -> List[int]:
        return list(self.token_ids)


@dataclass(frozen=True)
class RenderedPrompt:
    prompt: Union[TextPrompt, TokenPrompt]
    image_data: Optional[Any] = None
    audio_data: Optional[Any] = None
    video_data: Optional[Any] = None
    modalities: List[str] = field(default_factory=list)
    template_stop: Optional[Union[str, List[str]]] = None


@dataclass(frozen=True)
class PreparedChat:
    prompt: Union[TextPrompt, TokenPrompt]
    image_data: Optional[Any]
    audio_data: Optional[Any]
    video_data: Optional[Any]
    modalities: List[str]
    stop: Optional[Union[str, List[str]]]
    tool_call_constraint: Optional[ToolCallConstraint]
    skip_special_tokens: bool
    require_reasoning: bool
    reasoning_end_token_ids: Optional[List[int]]
    chat_template_kwargs: Optional[Dict[str, Any]]
    reasoning_effort: ReasoningEffortType


class MessageRenderer(Protocol):
    def render(
        self, request: ChatInput, tools: Optional[List[Dict]], require_reasoning: bool
    ) -> RenderedPrompt: ...


@dataclass(frozen=True)
class ChatModelConfig:
    tokenizer: Any
    template_manager: Any
    is_multimodal: bool
    model_name: str = ""
    chat_encoding_spec: Optional[str] = None
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    reasoning_detector: Any = None
    default_chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    dsv4_reasoning_effort_profile: Optional[str] = None
    inkling_default_reasoning_effort: Optional[float] = 0.9
    tokenizer_auto_adds_specials: bool = True
    is_gpt_oss: bool = False
    is_gemma4: bool = False
