import json
import logging
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


class Llama32Detector(BaseFormatDetector):
    """
    Detector for Llama 3.2 models.
    Assumes function call format:
      <|python_tag|>{"name":"xxx", "arguments":{...}}
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<|python_tag|>"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Llama 3.2 format tool call."""
        # depending on the prompt format the Llama model may or may not
        # prefix the output with the <|python_tag|> token
        return "<|python_tag|>" in text or text.startswith("{")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse function calls from text, handling multiple JSON objects."""
        if "<|python_tag|>" not in text and not text.startswith("{"):
            return StreamingParseResult(normal_text=text, calls=[])

        if "<|python_tag|>" in text:
            normal_text, action_text = text.split("<|python_tag|>")
        else:
            normal_text, action_text = "", text

        # Split by semicolon and process each part
        json_parts = [part.strip() for part in action_text.split(";") if part.strip()]
        all_actions = []
        for part in json_parts:
            try:
                # Parse each individual JSON object
                action = json.loads(part)
                all_actions.append(action)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON part: {part}")
                logger.warning(f"JSON parse error: {str(e)}")
                continue
        calls = []
        # Only process if we found valid JSON objects
        if all_actions:
            calls = self.parse_base_json(all_actions, tools)
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<|python_tag|>{"name":"' + name + '", "arguments":',
            end="}",
            trigger="<|python_tag|>",
        )

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            function_format="json",
            tool_call_separator=",",
        )
