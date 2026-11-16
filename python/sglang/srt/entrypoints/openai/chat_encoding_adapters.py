"""Unregistered chat encoding specs retain the existing serving branches;
concrete adapters load only when selected."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol

import msgspec

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ErrorResponse,
        ResponsesRequest,
        Tool,
        ToolCall,
        UsageInfo,
    )
    from sglang.srt.function_call.core_types import ToolCallItem

EncodingBackend = Literal["builtin", "recipe"]
HistoryToolCallArguments = Literal["strict", "lenient", "as_sent"]


class ChatOutputHooks(msgspec.Struct, frozen=True, kw_only=True):
    """Chat-handler policies an adapter needs, bound to one handler."""

    thinking_requested: Callable[[ChatCompletionRequest], bool]
    tool_call_parsing_active: Callable[[ChatCompletionRequest], bool]
    history_tool_calls_count: Callable[[ChatCompletionRequest], int]
    tool_call_id: Callable[[ToolCallItem, int], str]
    effective_tools: Callable[[ChatCompletionRequest], list[Tool]]
    split_assistant_prefix: Callable[
        [list[dict[str, Any]], ChatCompletionRequest],
        tuple[list[dict[str, Any]], Optional[str]],
    ]
    append_assistant_prefix: Callable[[list[int], str], list[int]]


class ChatWireResponse(Protocol):
    """How one /v1/chat/completions response goes on the wire."""

    serialization_context: Optional[dict[str, Any]]
    """The pydantic serialization context of its stream chunks."""

    omit_null_delta: bool

    continuous_usage: bool
    """Whether increment frames may carry continuous usage stats."""

    split_tool_call_opening: bool
    """Whether a streamed tool call opens with empty arguments, the arguments
    the parser already has following in a chunk of their own."""

    errors_end_choices: bool
    """Whether a mid-stream error ends every open choice with a finish reason
    instead of sending an error frame."""

    def response_id(self, engine_rid: str) -> str: ...

    def finish_reason(self, reason: Optional[str]) -> Optional[str]: ...

    def finalize_usage(self, usage: UsageInfo, *, cached_tokens: int) -> UsageInfo: ...

    def choice_usage(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ) -> Optional[dict[str, Any]]:
        """The usage of one choice's end frame; None for none."""
        ...

    def headers(self) -> dict[str, str]:
        """Response headers, read once the first engine event was observed."""
        ...

    def check_abort(
        self, finish_reason: dict[str, Any], *, stream_started: bool
    ) -> None:
        """May raise for an abort that carries an error status."""
        ...

    def observe(self, index: int, content: dict[str, Any]) -> None:
        """Sees each engine event of choice ``index`` before its chunks."""
        ...

    def ends_reasoning(
        self, index: int, detector: Any, reasoning_text: str, *, finishing: bool
    ) -> bool:
        """Whether to send the end-of-thinking frame after this parse step."""
        ...

    def internal_content(
        self,
        index: int,
        source: Optional[str],
        *,
        kind: Literal["reasoning", "content", "tool"],
    ) -> Optional[list[dict[str, Any]]]:
        """The ``internal_content`` of a chunk that streams ``source``."""
        ...


class ChatWire(Protocol):
    def new_response(
        self, request: ChatCompletionRequest, *, created: int
    ) -> ChatWireResponse: ...

    def error_body(self, error: ErrorResponse) -> Optional[dict[str, Any]]:
        """The body of an error response; None for the handler's own."""
        ...


class _OpenAIChatWire:
    """OpenAI's shape; stateless, so it is also its own per-response object."""

    serialization_context = None
    omit_null_delta = False
    continuous_usage = True
    split_tool_call_opening = False
    errors_end_choices = False

    def new_response(self, request, *, created):
        return self

    def error_body(self, error):
        return None

    def response_id(self, engine_rid):
        return engine_rid

    def finish_reason(self, reason):
        return reason

    def finalize_usage(self, usage, *, cached_tokens):
        return usage

    def choice_usage(
        self, *, prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens
    ):
        return None

    def headers(self):
        return {}

    def check_abort(self, finish_reason, *, stream_started):
        return None

    def observe(self, index, content):
        return None

    def ends_reasoning(self, index, detector, reasoning_text, *, finishing):
        return False

    def internal_content(self, index, source, *, kind):
        return None


OPENAI_CHAT_WIRE = _OpenAIChatWire()


def reported_prompt_tokens(meta_info: dict[str, Any], uncounted: int) -> int:
    return max(0, meta_info.get("prompt_tokens", 0) - uncounted)


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

    history_tool_call_arguments: HistoryToolCallArguments
    """How history tool-call ``arguments`` reach ``encode_messages``: "strict"
    parsed as JSON, rejecting any that do not parse; "lenient" parsed where
    they parse, the encoding serializing the rest itself; "as_sent" unparsed,
    the encoding parsing them itself."""

    renders_response_format: bool
    """Whether the prompt carries ``response_format`` instead of a grammar."""

    uncounted_prompt_tokens: int
    """Prompt tokens the encoding adds that reported usage leaves out."""

    wire: ChatWire
    """The shape of /v1/chat/completions responses."""

    def validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        """The error of a request the encoding rejects, or None."""
        ...

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
      tokenizer: prompt encoding (Inkling, DeepSeek-V4 / V4.1, Kimi-K3).
      reasoning_parser, tool_call_parser, chat_hooks: adapter-owned output parsing.
      hf_config: config-dependent encoding (DeepSeek V4.1 image_token_id, Kimi-K3.1
        effort levels).
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


def _build_kimi_k3(ctx: ChatEncodingAdapterContext) -> ChatEncodingAdapter:
    from sglang.srt.entrypoints.openai.serving_kimi_k3 import (
        KimiK3ChatEncodingAdapter,
    )

    return KimiK3ChatEncodingAdapter(
        tokenizer=ctx.tokenizer, hf_config=ctx.hf_config, chat_hooks=ctx.chat_hooks
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
    "kimi_k3": {"builtin": _build_kimi_k3},
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
