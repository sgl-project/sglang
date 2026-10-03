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
    inside <think> sections by stripping think tags from normal text while
    preserving literal tags in tool arguments.

    Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
    """

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think> and </think> tags, keeping the content inside."""
        return text.replace("<think>", "").replace("</think>", "")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.
        """
        result = super().detect_and_parse(text, tools)
        result.normal_text = self._strip_think_tags(result.normal_text)
        if self.bot_token in text:
            result.normal_text = result.normal_text.strip()
        return result

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for tool calls.
        """
        result = super().parse_streaming_increment(new_text, tools)
        result.normal_text = self._strip_think_tags(result.normal_text)
        return result
