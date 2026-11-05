import json
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.hermes_detector import HermesDetector


class RippleDetector(HermesDetector):
    """
    Detector for Ricursive Ripple tool calls: Hermes in the chat template's layout.

    Format:
        {content}\n<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>

    The newline before a call comes from the template, which adds it back when an
    assistant message is sent again, so it stays out of content.
    """

    def __init__(self):
        super().__init__()
        self._held_newline = ""
        self._in_calls = False
        self._seen_text = ""
        self._streamed_calls = {}

    def _record_calls(self, calls: List[ToolCallItem]) -> None:
        for call in calls:
            streamed = self._streamed_calls.setdefault(call.tool_index, ["", ""])
            streamed[0] += call.name or ""
            streamed[1] += call.parameters or ""
            while len(self.streamed_args_for_tool) <= call.tool_index:
                self.streamed_args_for_tool.append("")
            self.streamed_args_for_tool[call.tool_index] = streamed[1]

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        result = super().detect_and_parse(text, tools)
        idx = text.find(self.bot_token)
        if idx != -1 and result.calls:
            result.normal_text = text[:idx].removesuffix("\n")
        return result

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._seen_text += new_text
        result = super().parse_streaming_increment(new_text, tools)
        self._record_calls(result.calls)
        if self._in_calls:
            result.normal_text = ""
            return result
        self._in_calls = bool(
            self._buffer.startswith(self.bot_token)
            or self.current_tool_id >= 0
            or result.calls
        )
        if self._in_calls:
            result.normal_text = (self._held_newline + result.normal_text).removesuffix(
                "\n"
            )
            self._held_newline = ""
        elif result.normal_text:
            # Undecided until the next text arrives: this may be the template's newline.
            text = self._held_newline + result.normal_text
            self._held_newline = "\n" if text.endswith("\n") else ""
            result.normal_text = text.removesuffix("\n")
        return result

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        result = super().finish(tools)
        self._record_calls(result.calls)
        if self.bot_token in self._seen_text:
            # One chunk can span whole calls, and Hermes leaves those in its buffer.
            parsed = self.detect_and_parse(self._seen_text, tools)
            self._in_calls = True
            for index, call in enumerate(parsed.calls):
                name, arguments = self._streamed_calls.get(index, ("", ""))
                suffix = call.parameters.removeprefix(arguments)
                if suffix or not name:
                    item = ToolCallItem(
                        tool_index=index,
                        name=None if name else call.name,
                        parameters=suffix,
                    )
                    result.calls.append(item)
                    self._record_calls([item])
        else:
            result.normal_text = self._held_newline + self._buffer + result.normal_text
            self._held_newline = self._buffer = ""
        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<tool_call>\n{{"name": {json.dumps(name, ensure_ascii=False)}, "arguments": ',
            end="}\n</tool_call>",
            trigger="<tool_call>",
        )
