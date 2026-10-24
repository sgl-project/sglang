"""Inkling serving glue: prompt rendering and token-level output parsing for the
Chat Completions and Responses APIs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.entrypoints.openai.chat_encoding_adapters import (
    select_new_output_ids,
    token_output_sse_chunks,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    FunctionResponse,
    ResponsesRequest,
    ToolCall,
)
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.parser.inkling_output import (
    InklingOutputDelta,
    InklingOutputParser,
    InklingToolCall,
)
from sglang.srt.parser.inkling_renderer import (
    load_tml_renderers,
    render_inkling_assistant_prefix,
    render_inkling_messages,
)
from sglang.srt.parser.inkling_tokenizer import INKLING_SPECIAL_TOKEN_IDS
from sglang.srt.runtime_context import get_serving

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.chat_encoding_adapters import ChatOutputHooks


class InklingChatEncodingAdapter:
    collects_openai_media = True
    strict_tool_call_arguments = True

    def __init__(
        self,
        *,
        tokenizer: Any,
        reasoning_parser: str | None,
        tool_call_parser: str | None,
        chat_hooks: ChatOutputHooks,
    ):
        # Validate the frozen effort default at startup,
        # so configuration errors fail before requests.
        self._default_reasoning_effort = get_default_reasoning_effort()
        load_tml_renderers()
        if tokenizer is not None:
            _check_served_tokenizer(tokenizer)
        parsers = {reasoning_parser, tool_call_parser} - {None}
        if "inkling" in parsers and parsers != {"inkling"}:
            # The token parser emits Inkling calls while constraints would come
            # from the other parser's format.
            raise ValueError(
                "Inkling token parsing requires --reasoning-parser and "
                "--tool-call-parser to be inkling or unset, got "
                f"{reasoning_parser!r} and {tool_call_parser!r}"
            )
        self._token_output = "inkling" in parsers
        self._separate_reasoning = bool(reasoning_parser)
        self._hooks = chat_hooks

    @property
    def output(self) -> Optional[InklingChatEncodingAdapter]:
        return self if self._token_output else None

    def encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
        tools: list[dict] | None,
        thinking: bool,
        image_data: list,
    ) -> list[int]:
        # Inkling's thinking is effort-conditioned and the handler collects its
        # media, so `thinking` and `image_data` are not read.
        reasoning_effort = parse_reasoning_effort(request.reasoning_effort)
        if reasoning_effort is None:
            reasoning_effort = self._default_reasoning_effort
        assistant_prefix = _pop_assistant_prefix(messages, request)
        prompt_ids = render_inkling_messages(
            messages, tools=tools, reasoning_effort=reasoning_effort
        )
        if assistant_prefix is not None:
            prompt_ids += render_inkling_assistant_prefix(assistant_prefix)
        return prompt_ids

    def parse_chat_response(
        self,
        *,
        request: ChatCompletionRequest,
        output_ids: list[int],
        finish_reason: dict[str, Any],
    ) -> tuple[str | None, str, list[ToolCall] | None, dict[str, Any]]:
        parsed = self._new_output_parser(request).finish(
            output_ids,
            matched_stop=finish_reason.get("matched"),
            keep_matched_stop=request.no_stop_trim,
        )
        if not parsed.tool_calls:
            return parsed.reasoning or None, parsed.content, None, finish_reason
        history_tool_calls_cnt = self._hooks.history_tool_calls_count(request)
        tool_calls = [
            self._tool_call(call, history_tool_calls_cnt) for call in parsed.tool_calls
        ]
        if finish_reason["type"] == "stop":
            finish_reason = {**finish_reason, "type": "tool_calls", "matched": None}
        return parsed.reasoning or None, parsed.content, tool_calls, finish_reason

    def stream_chat_chunks(
        self,
        *,
        content: dict[str, Any],
        index: int,
        request: ChatCompletionRequest,
        created: int,
        parser_dict: dict,
        has_tool_calls: dict[int, bool],
        choice_logprobs: dict | None,
        finish_reason_type: str | None,
        usage: dict[str, Any] | None,
    ) -> list[str]:
        if index not in parser_dict:
            parser_dict[index] = self._new_output_parser(request)
        parser = parser_dict[index]
        new_output_ids = select_new_output_ids(
            output_ids=content["output_ids"],
            num_consumed_tokens=parser.num_consumed_tokens,
            completion_tokens=content["meta_info"]["completion_tokens"],
            finish_reason_type=finish_reason_type,
            incremental=get_serving().incremental_streaming_output,
        )
        if finish_reason_type is None:
            delta = parser.feed(new_output_ids)
        else:
            delta = parser.finish(
                new_output_ids,
                matched_stop=content["meta_info"]["finish_reason"].get("matched"),
                keep_matched_stop=request.no_stop_trim,
            )

        history_tool_calls_cnt = self._hooks.history_tool_calls_count(request)
        if delta.tool_calls:
            has_tool_calls[index] = True
        return token_output_sse_chunks(
            chunk_id=content["meta_info"]["id"],
            created=created,
            model=request.model,
            index=index,
            usage=usage,
            logprobs=choice_logprobs,
            reasoning=delta.reasoning,
            content=delta.content,
            tool_calls=[
                self._tool_call(call, history_tool_calls_cnt)
                for call in delta.tool_calls
            ],
        )

    def parse_responses_output(
        self,
        *,
        request: ResponsesRequest,
        require_reasoning: bool,
        output_ids: list[int],
        finish_reason: dict[str, Any] | None,
        separate_reasoning: bool,
        parse_tool_calls: bool,
    ) -> tuple[str, str, list[ToolCallItem]]:
        parser = InklingOutputParser(
            separate_reasoning=separate_reasoning, parse_tool_calls=parse_tool_calls
        )
        delta = parser.finish(
            output_ids,
            matched_stop=(finish_reason or {}).get("matched"),
            keep_matched_stop=False,
        )
        return delta.reasoning, delta.content, _tool_call_items(delta)

    def new_responses_stream(
        self,
        *,
        request: ResponsesRequest,
        require_reasoning: bool,
        separate_reasoning: bool,
        parse_tool_calls: bool,
    ) -> InklingResponsesStream:
        return InklingResponsesStream(
            separate_reasoning=separate_reasoning, parse_tool_calls=parse_tool_calls
        )

    def _new_output_parser(self, request: ChatCompletionRequest) -> InklingOutputParser:
        return InklingOutputParser(
            separate_reasoning=self._separate_reasoning and request.separate_reasoning,
            parse_tool_calls=self._hooks.tool_call_parsing_active(request),
            stream_reasoning=request.stream_reasoning,
            continues_text_block=bool(
                request.continue_final_message
                and request.messages
                and _is_continuable_message(request.messages[-1].model_dump())
            ),
        )

    def _tool_call(
        self, call: InklingToolCall, history_tool_calls_cnt: int
    ) -> ToolCall:
        call_item = ToolCallItem(
            tool_index=call.index, name=call.name, parameters=call.arguments
        )
        return ToolCall(
            id=self._hooks.tool_call_id(call_item, history_tool_calls_cnt),
            index=call.index,
            function=FunctionResponse(name=call.name, arguments=call.arguments),
        )


class InklingResponsesStream:
    def __init__(self, *, separate_reasoning: bool, parse_tool_calls: bool):
        self._parser = InklingOutputParser(
            separate_reasoning=separate_reasoning, parse_tool_calls=parse_tool_calls
        )
        self._incremental = get_serving().incremental_streaming_output

    def step(
        self, chunk: dict[str, Any], finish_reason: dict[str, Any] | None
    ) -> tuple[str, str, list[ToolCallItem]]:
        new_output_ids = select_new_output_ids(
            output_ids=chunk.get("output_ids") or [],
            num_consumed_tokens=self._parser.num_consumed_tokens,
            completion_tokens=chunk["meta_info"]["completion_tokens"],
            finish_reason_type=finish_reason["type"] if finish_reason else None,
            incremental=self._incremental,
        )
        if finish_reason is None:
            delta = self._parser.feed(new_output_ids)
        else:
            delta = self._parser.finish(
                new_output_ids,
                matched_stop=finish_reason.get("matched"),
                keep_matched_stop=False,
            )
        return delta.reasoning, delta.content, _tool_call_items(delta)


def _tool_call_items(delta: InklingOutputDelta) -> list[ToolCallItem]:
    return [
        ToolCallItem(tool_index=call.index, name=call.name, parameters=call.arguments)
        for call in delta.tool_calls
    ]


_TOKENIZER_PROBE = "Hello, 世界! def f(x): return x**2"


def _check_served_tokenizer(tokenizer: Any) -> None:
    # Prompts and outputs use tml-renderers' tokenizer, while the scheduler matches
    # stop strings and detokenizes with the served one; they must agree.
    tml_tokenizer = load_tml_renderers().tokenizer
    # encode() is the lookup both the HF and the tiktoken tokenizer wrappers share.
    mismatched = [
        token
        for token in INKLING_SPECIAL_TOKEN_IDS
        if tokenizer.encode(token, add_special_tokens=False)
        != [tml_tokenizer.encode_special(token[2:-2])]
    ]
    if tokenizer.encode(_TOKENIZER_PROBE, add_special_tokens=False) != list(
        tml_tokenizer.encode_ordinary(_TOKENIZER_PROBE)
    ):
        mismatched.append("ordinary text")
    if mismatched:
        raise ValueError(
            "The served tokenizer disagrees with tml-renderers' o200k_base_chat on "
            f"{', '.join(mismatched)}; Inkling prompts and outputs would be misencoded."
        )


def _pop_assistant_prefix(
    messages: list[dict[str, Any]],
    request: ChatCompletionRequest,
) -> str | None:
    """Pops only a continuable text turn; callers pass a message-list copy."""
    if not request.continue_final_message or not messages:
        return None
    if not _is_continuable_message(messages[-1]):
        return None
    return messages.pop()["content"]


def _is_continuable_message(message: dict[str, Any]) -> bool:
    return (
        message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
        and not message.get("tool_calls")
        and not message.get("reasoning_content")
    )


def parse_reasoning_effort(
    value: str | float | None,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Inkling reasoning_effort must not be a boolean")
    if isinstance(value, (int, float)):
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
            raise ValueError("Inkling reasoning_effort must be in [0.0, 0.99]")
        return parsed
    _EFFORT_MAP = {
        "none": 0.0,
        "minimal": 0.1,
        "low": 0.2,
        "medium": 0.7,
        "high": 0.9,
        "xhigh": 0.99,
        "max": 0.99,
    }
    if value in _EFFORT_MAP:
        return _EFFORT_MAP[value]
    try:
        parsed = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid Inkling reasoning_effort: {value!r}") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
        raise ValueError("Inkling reasoning_effort must be in [0.0, 0.99]")
    return parsed


def get_default_reasoning_effort() -> float:
    from sglang.srt.environ import envs

    val = envs.SGLANG_INKLING_DEFAULT_REASONING_EFFORT.get()
    if not val:
        return 0.9
    try:
        parsed = float(val)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "SGLANG_INKLING_DEFAULT_REASONING_EFFORT must be numeric"
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
        raise ValueError(
            "SGLANG_INKLING_DEFAULT_REASONING_EFFORT must be in [0.0, 0.99]"
        )
    return parsed
