"""Unregistered chat encoding specs retain the existing serving branches;
concrete adapters load only when selected."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol

import msgspec

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ToolCall
    from sglang.srt.function_call.core_types import ToolCallItem

EncodingBackend = Literal["builtin", "recipe"]


class ChatOutputHooks(msgspec.Struct, frozen=True, kw_only=True):
    """Chat-handler policies an output adapter needs, bound to one handler."""

    tool_call_parsing_active: Callable[[ChatCompletionRequest], bool]
    history_tool_calls_count: Callable[[ChatCompletionRequest], int]
    tool_call_id: Callable[[ToolCallItem, int], str]


class ResponsesOutputStream(Protocol):
    def step(
        self, chunk: dict[str, Any], finish_reason: Optional[dict[str, Any]]
    ) -> tuple[str, str, list[ToolCallItem]]: ...


class ChatEncodingOutputAdapter(Protocol):
    def parse_chat_response(
        self,
        *,
        request: ChatCompletionRequest,
        output_ids: list[int],
        finish_reason: dict[str, Any],
    ) -> tuple[Optional[str], str, Optional[list[ToolCall]], dict[str, Any]]: ...

    def stream_chat_chunks(
        self,
        *,
        content: dict[str, Any],
        index: int,
        request: ChatCompletionRequest,
        created: int,
        parser_dict: dict[int, Any],
        has_tool_calls: dict[int, bool],
        choice_logprobs: Optional[dict],
        finish_reason_type: Optional[str],
        usage: Optional[dict[str, Any]],
    ) -> list[str]: ...

    def parse_responses_output(
        self,
        *,
        output_ids: list[int],
        finish_reason: Optional[dict[str, Any]],
        separate_reasoning: bool,
        parse_tool_calls: bool,
    ) -> tuple[str, str, list[ToolCallItem]]: ...

    def new_responses_stream(
        self, *, separate_reasoning: bool, parse_tool_calls: bool
    ) -> ResponsesOutputStream: ...


class ChatEncodingAdapter(Protocol):
    collects_openai_media: bool
    """Whether media still has to be collected from the OpenAI messages after
    ``encode_messages`` returns prompt ids."""

    @property
    def output(self) -> Optional[ChatEncodingOutputAdapter]:
        """None when the configured parsers leave output to the text parsers."""
        ...

    def encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
    ) -> list[int]: ...


class ChatEncodingAdapterContext(msgspec.Struct, frozen=True, kw_only=True):
    """Everything a builder may read; a builder takes only what its encoding needs.

    Consumers:
      tokenizer: prompt encoding (Inkling).
      reasoning_parser, tool_call_parser, chat_hooks: adapter-owned output parsing.
      hf_config: config-dependent encoding (DeepSeek V4.1 image_token_id).
      model_path, revision: checkpoint files (DeepSeek tokenizer.json and the
        encoder file for effort-profile detection).
      backend: selects the builder; "recipe" is an explicit opt-in.
    """

    chat_encoding_spec: str
    tokenizer: Any
    reasoning_parser: Optional[str]
    tool_call_parser: Optional[str]
    chat_hooks: ChatOutputHooks
    hf_config: Any
    model_path: str
    revision: Optional[str]
    backend: EncodingBackend = "builtin"

    def require(self, field: str) -> Any:
        value = getattr(self, field)
        if value is None:
            raise ValueError(
                f"chat encoding spec {self.chat_encoding_spec!r} ({self.backend} backend) "
                f"requires {field}, which this server does not provide"
            )
        return value


def _build_inkling(ctx: ChatEncodingAdapterContext) -> ChatEncodingAdapter:
    from sglang.srt.entrypoints.openai.serving_inkling import InklingChatEncodingAdapter

    return InklingChatEncodingAdapter(tokenizer=ctx.tokenizer)


_BUILDERS: dict[
    str,
    dict[EncodingBackend, Callable[[ChatEncodingAdapterContext], ChatEncodingAdapter]],
] = {
    "inkling": {"builtin": _build_inkling},
}


def has_chat_encoding_adapter(chat_encoding_spec: Optional[str]) -> bool:
    return chat_encoding_spec in _BUILDERS


def build_chat_encoding_adapter(
    ctx: ChatEncodingAdapterContext,
) -> Optional[ChatEncodingAdapter]:
    builders = _BUILDERS.get(ctx.chat_encoding_spec, {})
    builder = builders.get(ctx.backend)
    if builder is not None:
        return builder(ctx)
    if ctx.backend != "builtin":
        raise ValueError(
            f"chat encoding spec {ctx.chat_encoding_spec!r} has no {ctx.backend} backend; "
            f"available: {sorted(builders) or ['builtin']}"
        )
    return None
