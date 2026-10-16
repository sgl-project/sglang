"""Serving adapters for chat encoding specs that own prompt rendering.

``build_spec_adapter`` returns None for a spec with no registered adapter; that
spec keeps its existing rendering branches in ``serving_chat``. Concrete
adapters are imported lazily, so selecting another spec never loads them.

The output protocols re-derive encode-time state (such as a prompt ending inside
an open text block) from the request. A spec whose output parsing depends on a
decision the encoder made, such as whether the prompt opened a thinking block,
should return it from encoding as an ``EncodedPrompt`` (prompt ids plus that
state) and take it in the output methods, rather than deciding twice.
"""

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
    """Parses sampled token IDs into API output, replacing the text parsers."""

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
    """Everything a builder may read; a new builder input is a field here, so the
    builder signature and the one construction site in serving_chat stay stable."""

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
