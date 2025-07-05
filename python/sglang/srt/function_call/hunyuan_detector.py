import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer

logger = logging.getLogger(__name__)


class HunyuanDetector(BaseFormatDetector):
    """
    Detector for Hunyuan models.

    Handles tool calls in format:
      <tool_calls>[{"name": "func", "arguments": {...}}]</tool_calls>

    Note: This detector works independently of the reasoning parser,
    which handles <think> and <answer> tags separately.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_calls>"
        self.eot_token = "</tool_calls>"
        # Hunyuan uses comma separation for multiple tools
        self.tool_call_separator = ", "

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains tool call markers."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Non-streaming parsing of tool calls."""
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        # Find tool calls section
        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            return StreamingParseResult(normal_text=text, calls=[])

        # Extract and parse tool calls
        calls = []
        try:
            tool_json = json.loads(match.group(1).strip())
            if not isinstance(tool_json, list):
                tool_json = [tool_json]
            calls = self.parse_base_json(tool_json, tools)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool calls: {e}")

        # Remove tool calls from text
        normal_text = text[: match.start()] + text[match.end() :]

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        """Return structural information for constrained generation."""
        return lambda name: StructureInfo(
            begin=f'{self.bot_token}[{{"name":"{name}", "arguments":',
            end=f"}}]{self.eot_token}",
            trigger=self.bot_token,
        )

    def build_ebnf(self, tools: List[Tool]) -> str:
        """Build EBNF grammar for tool calling."""
        return EBNFComposer.build_ebnf(
            tools,
            function_format="json",
            sequence_start_token=self.bot_token + "[",
            sequence_end_token="]" + self.eot_token,
            tool_call_separator=self.tool_call_separator,
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Override streaming to handle JSON array format."""
        # For now, use a simple approach: buffer until we have complete tags
        self._buffer += new_text

        # If we don't have a complete tool_calls block, keep buffering
        if self.bot_token in self._buffer and self.eot_token in self._buffer:
            # We have a complete block, parse it
            result = self.detect_and_parse(self._buffer, tools)
            self._buffer = ""
            return result
        elif self.bot_token not in self._buffer:
            # No tool calls yet, return as normal text
            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text, calls=[])
        else:
            # Partial tool call, keep buffering
            return StreamingParseResult(normal_text="", calls=[])
