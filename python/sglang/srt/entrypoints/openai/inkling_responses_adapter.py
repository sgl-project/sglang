"""Inkling Responses API: token-level parsing of output items."""

from __future__ import annotations

from typing import Any, Optional

from sglang.srt.entrypoints.openai.inkling_chat_adapter import select_new_output_ids
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.parser.inkling_output import InklingOutputDelta, InklingOutputParser
from sglang.srt.runtime_context import get_serving


def parse_output(
    *,
    output_ids: list[int],
    finish_reason: Optional[dict[str, Any]],
    separate_reasoning: bool,
    parse_tool_calls: bool,
) -> InklingOutputDelta:
    parser = InklingOutputParser(
        separate_reasoning=separate_reasoning, parse_tool_calls=parse_tool_calls
    )
    return parser.finish(
        output_ids,
        matched_stop=(finish_reason or {}).get("matched"),
        keep_matched_stop=False,
    )


class InklingResponsesStream:
    """One streamed response's parse, shaped like the text parsers' output:
    each engine chunk yields (reasoning, text, tool calls)."""

    def __init__(self, *, separate_reasoning: bool, parse_tool_calls: bool):
        self._parser = InklingOutputParser(
            separate_reasoning=separate_reasoning, parse_tool_calls=parse_tool_calls
        )
        self._incremental = get_serving().incremental_streaming_output
        self._finished = False

    def step(
        self, chunk: dict[str, Any], finish_reason: Optional[dict[str, Any]]
    ) -> tuple[str, str, list[ToolCallItem]]:
        # finish_reason stays set on later chunks; the parser finishes once.
        if self._finished:
            return "", "", []
        new_output_ids = select_new_output_ids(
            output_ids=chunk.get("output_ids") or [],
            num_consumed_tokens=self._parser.num_consumed_tokens,
            completion_tokens=(chunk.get("meta_info") or {}).get(
                "completion_tokens", 0
            ),
            finish_reason_type=finish_reason["type"] if finish_reason else None,
            incremental=self._incremental,
        )
        if finish_reason is None:
            delta = self._parser.feed(new_output_ids)
        else:
            self._finished = True
            delta = self._parser.finish(
                new_output_ids,
                matched_stop=finish_reason.get("matched"),
                keep_matched_stop=False,
            )
        calls = [
            ToolCallItem(
                tool_index=call.index, name=call.name, parameters=call.arguments
            )
            for call in delta.tool_calls
        ]
        return delta.reasoning, delta.content, calls
