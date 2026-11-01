"""Unregistered chat encoding specs retain the existing serving branches;
concrete adapters load only when selected."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol

import msgspec

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ResponsesRequest,
        Tool,
        ToolCall,
    )
    from sglang.srt.function_call.core_types import ToolCallItem

EncodingBackend = Literal["builtin", "recipe"]


class ChatOutputHooks(msgspec.Struct, frozen=True, kw_only=True):
    """Chat-handler policies an output adapter needs, bound to one handler."""

    thinking_requested: Callable[[ChatCompletionRequest], bool]
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
        request: ResponsesRequest,
        require_reasoning: bool,
        output_ids: list[int],
        finish_reason: Optional[dict[str, Any]],
        separate_reasoning: bool,
        parse_tool_calls: bool,
    ) -> tuple[str, str, list[ToolCallItem]]: ...

    def new_responses_stream(
        self,
        *,
        request: ResponsesRequest,
        require_reasoning: bool,
        separate_reasoning: bool,
        parse_tool_calls: bool,
    ) -> ResponsesOutputStream: ...


class ChatEncodingAdapter(Protocol):
    collects_openai_media: bool
    """Whether media still has to be collected from the OpenAI messages after
    ``encode_messages`` returns prompt ids."""

    strict_tool_call_arguments: bool
    """Whether history tool-call ``arguments`` must parse as JSON before
    ``encode_messages`` sees them; False when the encoding serializes them itself."""

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
        thinking: bool,
        image_data: list,
    ) -> list[int]:
        """``thinking`` is the handler's thinking decision for ``request``. An
        encoding that collects media itself appends the prompt's images to
        ``image_data`` in placeholder order."""
        ...


class NativeTextParser(Protocol):
    """The native /separate_reasoning and /parse_function_call endpoints, for a
    parser whose output the served adapter parses."""

    def separate_reasoning_text(self, text: str) -> tuple[str, str]: ...

    def parse_function_call_text(
        self, text: str, tools: list[Tool]
    ) -> tuple[str, list[ToolCallItem]]: ...


# The reasoning and tool-call parser names whose output the spec's adapter parses.
ADAPTER_OUTPUT_PARSERS: dict[str, tuple[str, str]] = {
    "dsv4": ("deepseek-v4", "deepseekv4"),
    "dsv41": ("deepseek-v41", "deepseekv41"),
}


def native_text_parser(
    parser_name: Optional[str],
    *,
    chat_encoding_spec: Optional[str],
    adapter: Optional[ChatEncodingAdapter],
) -> Optional[NativeTextParser]:
    """The served adapter when it parses `parser_name` output; None for the text
    parsers. A parser whose adapter is not served raises ValueError: the adapter
    parses with the served checkpoint's tokenizer."""
    owners = [
        spec for spec, names in ADAPTER_OUTPUT_PARSERS.items() if parser_name in names
    ]
    if not owners:
        return None
    if adapter is None or chat_encoding_spec not in owners:
        raise ValueError(
            f"{parser_name!r} output is parsed with its model's tokenizer; this "
            "server does not serve that model"
        )
    return adapter


def select_new_output_ids(
    *,
    output_ids: list[int],
    num_consumed_tokens: int,
    completion_tokens: int,
    finish_reason_type: Optional[str],
    incremental: bool,
) -> list[int]:
    if not incremental:
        # Intermediate chunks share the live output list; completion_tokens is
        # this chunk's snapshot, so later tokens wait for their own chunk.
        return output_ids[num_consumed_tokens:completion_tokens]
    if finish_reason_type == "abort":
        # The abort chunk re-sends already-streamed tokens.
        return output_ids[: max(completion_tokens - num_consumed_tokens, 0)]
    return output_ids


def token_output_sse_chunks(
    *,
    chunk_id: str,
    created: int,
    model: str,
    index: int,
    usage: Optional[dict[str, Any]],
    logprobs: Optional[dict],
    reasoning: str,
    content: str,
    tool_calls: list[ToolCall],
) -> list[str]:
    """One streamed step of a token-level output parser: a reasoning chunk, a content
    chunk, then one chunk per tool call; the step's logprobs ride on the first."""
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionResponseStreamChoice,
        ChatCompletionStreamResponse,
        DeltaMessage,
    )
    from sglang.srt.entrypoints.openai.sse_utils import build_sse_content

    fields = dict(
        chunk_id=chunk_id, created=created, model=model, index=index, usage=usage
    )
    chunks = []
    if reasoning:
        chunks.append(
            build_sse_content(reasoning_content=reasoning, logprobs=logprobs, **fields)
        )
        logprobs = None
    if content:
        chunks.append(build_sse_content(content=content, logprobs=logprobs, **fields))
        logprobs = None
    for tool_call in tool_calls:
        chunk = ChatCompletionStreamResponse(
            id=chunk_id,
            created=created,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=index,
                    delta=DeltaMessage(tool_calls=[tool_call]),
                    finish_reason=None,
                )
            ],
            model=model,
            usage=usage,
        )
        chunks.append(f"data: {chunk.model_dump_json()}\n\n")
    if logprobs is not None:
        chunks.append(build_sse_content(logprobs=logprobs, **fields))
    return chunks


class ChatEncodingAdapterContext(msgspec.Struct, frozen=True, kw_only=True):
    """Everything a builder may read; a builder takes only what its encoding needs.

    Consumers:
      tokenizer: prompt encoding (Inkling, DeepSeek-V4 / V4.1).
      reasoning_parser, tool_call_parser, chat_hooks: adapter-owned output parsing.
      hf_config: config-dependent encoding (DeepSeek V4.1 image_token_id).
      model_path, revision: checkpoint files.
      default_chat_template_kwargs: the workers' --default-chat-template-kwargs
        (deepseek-recipe validates the default effort at startup).
      backend: selects the builder; every spec has a "builtin" one.
    """

    chat_encoding_spec: str
    tokenizer: Any
    reasoning_parser: Optional[str]
    tool_call_parser: Optional[str]
    chat_hooks: ChatOutputHooks
    hf_config: Any
    model_path: str
    revision: Optional[str]
    default_chat_template_kwargs: Optional[dict[str, Any]] = None
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

    return InklingChatEncodingAdapter(
        tokenizer=ctx.tokenizer,
        reasoning_parser=ctx.reasoning_parser,
        tool_call_parser=ctx.tool_call_parser,
        chat_hooks=ctx.chat_hooks,
    )


def _build_deepseek_recipe(
    ctx: ChatEncodingAdapterContext,
) -> Optional[ChatEncodingAdapter]:
    if ctx.tokenizer is None:
        return None  # --skip-tokenizer-init: the server encodes no chat prompts.
    from sglang.srt.entrypoints.openai.serving_deepseek_recipe import (
        DeepSeekRecipeChatEncodingAdapter,
    )

    return DeepSeekRecipeChatEncodingAdapter(
        chat_encoding_spec=ctx.chat_encoding_spec,
        tokenizer=ctx.tokenizer,
        reasoning_parser=ctx.reasoning_parser,
        tool_call_parser=ctx.tool_call_parser,
        chat_hooks=ctx.chat_hooks,
        default_chat_template_kwargs=ctx.default_chat_template_kwargs,
    )


_BUILDERS: dict[
    str,
    dict[
        EncodingBackend,
        Callable[[ChatEncodingAdapterContext], Optional[ChatEncodingAdapter]],
    ],
] = {
    "inkling": {"builtin": _build_inkling},
    "dsv4": {"builtin": _build_deepseek_recipe},
    "dsv41": {"builtin": _build_deepseek_recipe},
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
