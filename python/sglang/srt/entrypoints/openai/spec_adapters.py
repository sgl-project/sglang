"""Unregistered specs retain the existing serving branches;
concrete adapters load only when selected."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

import msgspec

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ToolCall
    from sglang.srt.function_call.core_types import ToolCallItem


class ChatOutputHooks(msgspec.Struct, frozen=True, kw_only=True):
    """Chat-handler policies an output adapter needs, bound to one handler."""

    tool_call_parsing_active: Callable[[ChatCompletionRequest], bool]
    history_tool_calls_count: Callable[[ChatCompletionRequest], int]
    tool_call_id: Callable[[ToolCallItem, int], str]


class ResponsesOutputStream(Protocol):
    def step(
        self, chunk: dict[str, Any], finish_reason: Optional[dict[str, Any]]
    ) -> tuple[str, str, list[ToolCallItem]]: ...


class SpecOutputAdapter(Protocol):
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


class SpecAdapter(Protocol):
    collects_openai_media: bool
    """Whether media still has to be collected from the OpenAI messages after
    ``encode_messages`` returns prompt ids."""

    @property
    def output(self) -> Optional[SpecOutputAdapter]:
        """None when the configured parsers leave output to the text parsers."""
        ...

    def encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
    ) -> list[int]: ...


class SpecAdapterContext(msgspec.Struct, frozen=True, kw_only=True):
    tokenizer: Any
    reasoning_parser: Optional[str]
    tool_call_parser: Optional[str]
    chat_hooks: ChatOutputHooks


def _build_inkling(ctx: SpecAdapterContext) -> SpecAdapter:
    from sglang.srt.entrypoints.openai.serving_inkling import InklingSpecAdapter

    return InklingSpecAdapter(
        tokenizer=ctx.tokenizer,
        reasoning_parser=ctx.reasoning_parser,
        tool_call_parser=ctx.tool_call_parser,
        chat_hooks=ctx.chat_hooks,
    )


_BUILDERS: dict[str, Callable[[SpecAdapterContext], SpecAdapter]] = {
    "inkling": _build_inkling
}


def build_spec_adapter(
    spec: Optional[str], ctx: SpecAdapterContext
) -> Optional[SpecAdapter]:
    builder = _BUILDERS.get(spec) if spec is not None else None
    if builder is None:
        return None
    return builder(ctx)
