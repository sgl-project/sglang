import json
import logging
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

# ``JSONDecoder`` is stateless and thread-safe; reuse one instance instead of
# allocating a fresh decoder on every parse.
_JSON_DECODER = json.JSONDecoder()


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
        self._normal_text_buffer = ""

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing: detect and parse every tool call in ``text``."""
        first = text.find(self.bot_token)
        if first == -1:
            return StreamingParseResult(normal_text=text, calls=[])
        normal_text = text[:first].strip()

        # Decode one JSON value after each ``<tool_call>`` with a string-aware
        # decoder instead of a ``<tool_call>(.*?)</tool_call>`` regex. The
        # non-greedy regex stops at the first ``</tool_call>``, so an argument
        # string that legitimately contains the literal end-token truncates the
        # payload and the whole call is silently dropped -- which also makes this
        # non-streaming path disagree with the streaming path. ``raw_decode``
        # correctly ignores ``</tool_call>`` occurring inside string values.
        calls = []
        pos = first
        try:
            while (start := text.find(self.bot_token, pos)) != -1:
                cursor = start + len(self.bot_token)
                while cursor < len(text) and text[cursor] in " \t\r\n":
                    cursor += 1
                try:
                    obj, pos = _JSON_DECODER.raw_decode(text, cursor)
                except json.JSONDecodeError:
                    # Truncated or malformed payload (e.g. a partial tail); the
                    # remaining tool calls, if any, cannot be parsed reliably.
                    break
                calls.extend(self.parse_base_json(obj, tools))
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

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>{"name":"' + name + '", "arguments":',
            end="}</tool_call>",
            trigger="<tool_call>",
        )
