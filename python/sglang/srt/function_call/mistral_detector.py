import json
import logging
import re
from typing import List

from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer
from sglang.srt.openai_api.protocol import Tool

logger = logging.getLogger(__name__)


class MistralDetector(BaseFormatDetector):
    """
    Detector for Mistral models.
    Assumes function call format:
      [TOOL_CALLS] [{"name":"func1", "arguments":{...}}, {"name":"func2", "arguments":{...}}]
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "[TOOL_CALLS] ["
        self.eot_token = "]"
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Mistral format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Extract the JSON array part from [TOOL_CALLS] [...]
        # Look for [TOOL_CALLS] followed by a space and then capture the JSON array including brackets
        match = re.search(r"\[TOOL_CALLS\] (\[.*?\])", text, re.DOTALL)
        if not match:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Get the complete JSON array (with brackets)
        json_array_str = match.group(1)
        calls = []
        try:
            function_call_arr = json.loads(json_array_str)
            # Handle both single object and array of objects
            if not isinstance(function_call_arr, list):
                function_call_arr = [function_call_arr]
            calls = self.parse_base_json(function_call_arr, tools)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON part: {json_array_str}, JSON parse error: {str(e)}"
            )

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='[TOOL_CALLS] [{"name":"' + name + '", "arguments":',
            end="}]",
            trigger="[TOOL_CALLS]",
        )

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            sequence_start_token=self.bot_token,
            sequence_end_token=self.eot_token,
            function_format="json",
            tool_call_separator=", ",
        )
