"""Inkling chat completions: prompt rendering and token-level output parsing."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    FunctionResponse,
    ToolCall,
)
from sglang.srt.entrypoints.openai.sse_utils import build_sse_content
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.parser.inkling_output import InklingOutputParser, InklingToolCall
from sglang.srt.parser.inkling_renderer import (
    load_tml_renderers,
    render_inkling_assistant_prefix,
    render_inkling_messages,
)
from sglang.srt.runtime_context import get_serving


class InklingChatAdapter:
    def __init__(
        self,
        *,
        reasoning_parser: str | None,
        tool_call_parser: str | None,
        tool_call_parsing_active: Callable[[ChatCompletionRequest], bool],
        history_tool_calls_cnt: Callable[[ChatCompletionRequest], int],
        tool_call_id: Callable[[ToolCallItem, int], str],
    ):
        # Resolve the env-configured Inkling effort default once: the env var is
        # frozen for the server's lifetime, and a misconfigured value should
        # fail at boot, not 400 every request.
        self._default_reasoning_effort = get_default_reasoning_effort()
        load_tml_renderers()
        self.token_output = "inkling" in (reasoning_parser, tool_call_parser)
        self._separate_reasoning = bool(reasoning_parser)
        self._tool_call_parsing_active = tool_call_parsing_active
        self._history_tool_calls_cnt = history_tool_calls_cnt
        self._tool_call_id = tool_call_id

    def encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
        tools: list[dict] | None,
    ) -> list[int]:
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

    def parse_response(
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
        history_tool_calls_cnt = self._history_tool_calls_cnt(request)
        tool_calls = [
            self._tool_call(call, history_tool_calls_cnt) for call in parsed.tool_calls
        ]
        if finish_reason["type"] == "stop":
            finish_reason = {**finish_reason, "type": "tool_calls", "matched": None}
        return parsed.reasoning or None, parsed.content, tool_calls, finish_reason

    def stream_chunks(
        self,
        *,
        content: dict[str, Any],
        index: int,
        request: ChatCompletionRequest,
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
            completion_tokens=content["meta_info"].get("completion_tokens", 0),
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

        chunk_fields = dict(
            chunk_id=content["meta_info"]["id"],
            created=int(time.time()),
            model=request.model,
            index=index,
            usage=usage,
        )

        chunks = []
        remaining_logprobs = choice_logprobs
        if delta.reasoning:
            chunks.append(
                build_sse_content(
                    reasoning_content=delta.reasoning,
                    logprobs=remaining_logprobs,
                    **chunk_fields,
                )
            )
            remaining_logprobs = None
        if delta.content:
            chunks.append(
                build_sse_content(
                    content=delta.content, logprobs=remaining_logprobs, **chunk_fields
                )
            )
            remaining_logprobs = None
        history_tool_calls_cnt = self._history_tool_calls_cnt(request)
        for call in delta.tool_calls:
            has_tool_calls[index] = True
            tool_call_chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=chunk_fields["created"],
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(
                            tool_calls=[self._tool_call(call, history_tool_calls_cnt)]
                        ),
                        finish_reason=None,
                    )
                ],
                model=request.model,
                usage=usage,
            )
            chunks.append(f"data: {tool_call_chunk.model_dump_json()}\n\n")
        if remaining_logprobs is not None:
            chunks.append(
                build_sse_content(logprobs=remaining_logprobs, **chunk_fields)
            )
        return chunks

    def _new_output_parser(self, request: ChatCompletionRequest) -> InklingOutputParser:
        return InklingOutputParser(
            separate_reasoning=self._separate_reasoning and request.separate_reasoning,
            parse_tool_calls=self._tool_call_parsing_active(request),
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
            id=self._tool_call_id(call_item, history_tool_calls_cnt),
            index=call.index,
            function=FunctionResponse(name=call.name, arguments=call.arguments),
        )


def select_new_output_ids(
    *,
    output_ids: list[int],
    num_consumed_tokens: int,
    completion_tokens: int,
    finish_reason_type: str | None,
    incremental: bool,
) -> list[int]:
    if not incremental:
        return output_ids[num_consumed_tokens:]
    if finish_reason_type == "abort":
        # The abort chunk re-sends already-streamed tokens.
        return output_ids[: max(completion_tokens - num_consumed_tokens, 0)]
    return output_ids


def _pop_assistant_prefix(
    messages: list[dict[str, Any]],
    request: ChatCompletionRequest,
) -> str | None:
    """Extract the trailing assistant text for ``continue_final_message``.

    Only a plain-string assistant message with no tool calls and no
    reasoning content can be continued; anything else renders as a closed
    historical turn. Mutates ``messages`` in place (callers pass a copy).
    """
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
    """Convert an OpenAI-style reasoning_effort to an Inkling float."""
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
    """Read the default Inkling reasoning effort from the environment."""
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
