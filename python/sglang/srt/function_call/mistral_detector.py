import json
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


class MistralDetector(BaseFormatDetector):
    """
    Detector for Mistral models.
    Assumes function call format:
      [TOOL_CALLS] [{"name":"xxx", "arguments":{...}}]
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

    def _clean_text(self, text: str) -> str:
        """
        clean text to only leave ''[TOOL_CALLS] [{"name": xxx, "arguments": {xxx}}]'
        for example,
            text = '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}]\n\nToday\'s weather in Boston is :{function call result} (in Fahrenheit)\n\nIf you prefer Celsius, please let me know.'
            return '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}]'
        The key pattern is [TOOL_CALLS] [...]
        """
        # TODO: check if Mistral supports multiple tool calls, currently assume only support one tool call
        find_results = re.findall(r"\[TOOL_CALLS\] \[.*?\]", text, re.DOTALL)
        if len(find_results) > 0:
            return find_results[0]
        else:
            return ""

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        text = self._clean_text(text)
        tool_content = text.replace("[TOOL_CALLS]", "").strip()
        raw_tool_calls = self.tool_call_regex.findall(tool_content)
        calls = []
        if len(raw_tool_calls) > 0:
            raw_tool_call = raw_tool_calls[0]
            function_call_arr = json.loads(raw_tool_call)
            for match_result in function_call_arr:
                calls.extend(self.parse_base_json(match_result, tools))
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
            bot_token=self.bot_token,
            eot_token=self.eot_token,
            function_format="json",
        )
