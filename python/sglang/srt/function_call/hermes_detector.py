import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    StructureInfo,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

# Upper bound for the end-of-stream drain: each round releases at most one
# pending unit (a tool name or one argument diff).
_MAX_FINISH_DRAIN_ROUNDS = 1024


class HermesDetector(BaseFormatDetector):
    """
    Detector for Hermes tool call format.

    Format:
        <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
        )
        self._normal_text_buffer = ""

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            for match in self.tool_call_regex.findall(text):
                raw = match[0] or match[1]
                if not raw:
                    continue
                parsed = json.loads(raw.strip())
                if isinstance(parsed, list):
                    calls.extend(self.parse_base_json(parsed, tools))
                else:
                    calls.extend(self.parse_base_json(parsed, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def _clean_normal_text(self, text: str) -> str:
        if not text:
            return text

        self._normal_text_buffer += text

        if self.eot_token in self._normal_text_buffer:
            cleaned = self._normal_text_buffer.replace(self.eot_token, "")
            self._normal_text_buffer = ""
            return cleaned

        partial_len = self._ends_with_partial_token(
            self._normal_text_buffer, self.eot_token
        )
        if partial_len:
            safe_text = self._normal_text_buffer[:-partial_len]
            self._normal_text_buffer = self._normal_text_buffer[-partial_len:]
            return safe_text

        cleaned = self._normal_text_buffer
        self._normal_text_buffer = ""
        return cleaned

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming parsing: handle normal text, partial tags, and tool calls.
        """
        self._buffer += new_text
        current_text = self._buffer

        if self.bot_token not in current_text:
            partial_len = self._ends_with_partial_token(current_text, self.bot_token)
            if partial_len:
                safe_text = current_text[:-partial_len]
                self._buffer = current_text[-partial_len:]
            else:
                safe_text = current_text
                self._buffer = ""
            return StreamingParseResult(normal_text=self._clean_normal_text(safe_text))

        bot_pos = current_text.find(self.bot_token)
        if bot_pos > 0:
            normal_text = current_text[:bot_pos]
            self._buffer = current_text[bot_pos:]
            return StreamingParseResult(normal_text=normal_text)

        result = super().parse_streaming_increment(new_text="", tools=tools)
        if result.normal_text:
            result.normal_text = self._clean_normal_text(result.normal_text)
        return result

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Release tool-call state the last delta left pending.

        parse_streaming_increment emits at most one unit per call: a tool name
        with empty parameters, then that call's argument diff, then the next
        call. A delta carrying one or more complete calls therefore leaves work
        pending for the following delta, and when that delta was the last one
        (multi-token deltas from speculative decoding / MTP,
        ``stream_interval > 1``, or a call short enough to be generated in a
        single step) nothing picked it up: the second call of a pair was dropped
        and the first arrived with empty arguments, while detect_and_parse
        returned both.

        Re-running the parser with an empty delta drains that queue. The markup
        the parser releases along the way (``</tool_call>`` when a second call
        follows) is framing, not assistant text -- detect_and_parse reports no
        text for those inputs -- so protocol tokens are stripped before anything
        is returned as content.
        """
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        for _ in range(_MAX_FINISH_DRAIN_ROUNDS):
            result = self.parse_streaming_increment("", tools)
            if not result.calls and not result.normal_text:
                break
            if result.calls:
                calls.extend(result.calls)
            text = result.normal_text or ""
            for token in (self.bot_token, self.eot_token):
                if token:
                    text = text.replace(token, "")
            if text.strip():
                normal_parts.append(text)
        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>{"name":"' + name + '", "arguments":',
            end="}</tool_call>",
            trigger="<tool_call>",
        )
