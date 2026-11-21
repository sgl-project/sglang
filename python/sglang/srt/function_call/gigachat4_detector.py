from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.gigachat35_detector import (
    _GCML_INVOKE_RE,
    GigaChat35Detector,
    _strip_trailing_markers,
)


class GigaChat4Detector(GigaChat35Detector):
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        result = super().detect_and_parse(text, tools)
        leading, marker, body = text.partition(self.bot_token)
        if marker and _GCML_INVOKE_RE.search(body.partition(self.eot_token)[0]):
            # Preserve the block separator already emitted by streaming.
            result.normal_text = _strip_trailing_markers(leading)
        return result

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        result = self.parse_streaming_increment("", tools)
        leftover, self._buffer = self._buffer, ""
        if leftover and not self._emitted_invokes:
            result.normal_text += _strip_trailing_markers(leftover)
        return result
