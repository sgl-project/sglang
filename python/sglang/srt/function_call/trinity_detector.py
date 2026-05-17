import logging
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.qwen25_detector import Qwen25Detector

logger = logging.getLogger(__name__)


class TrinityDetector(Qwen25Detector):
    """
    Detector for Trinity models using Qwen-style function call format.

    This detector extends Qwen25Detector to handle tool calls that may appear
    inside <think> sections by stripping the think tags before parsing.

    Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
    """

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think> and </think> tags, keeping the content inside."""
        return text.replace("<think>", "").replace("</think>", "")

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a tool call."""
        return super().has_tool_call(self._strip_think_tags(text))

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.
        """
        return super().detect_and_parse(self._strip_think_tags(text), tools)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for tool calls.
        """
        return super().parse_streaming_increment(
            self._strip_think_tags(new_text), tools
        )
