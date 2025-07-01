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
    Assumes function call format:
      <answer>
      [content]
      <tool_calls>[{"name": "func", "arguments": {...}}, ...]</tool_calls>
      </answer>
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.answer_start = "<answer>"
        self.answer_end = "</answer>"
        self.tool_calls_start = "<tool_calls>"
        self.tool_calls_end = "</tool_calls>"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains Hunyuan format tool calls."""
        return self.tool_calls_start in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        # Look for answer block
        answer_pattern = (
            rf"{re.escape(self.answer_start)}(.*?){re.escape(self.answer_end)}"
        )
        answer_match = re.search(answer_pattern, text, re.DOTALL)

        if not answer_match:
            # No answer block, return as normal text
            return StreamingParseResult(normal_text=text, calls=[])

        answer_content = answer_match.group(1)

        # Look for tool calls within answer
        tool_pattern = (
            rf"{re.escape(self.tool_calls_start)}(.*?){re.escape(self.tool_calls_end)}"
        )
        tool_match = re.search(tool_pattern, answer_content, re.DOTALL)

        if not tool_match:
            # Answer block but no tool calls - extract just the answer content
            normal_text = (
                text[: answer_match.start()]
                + answer_content.strip()
                + text[answer_match.end() :]
            ).strip()
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Parse tool calls
        calls = []
        try:
            tool_calls_json = json.loads(tool_match.group(1).strip())
            if not isinstance(tool_calls_json, list):
                tool_calls_json = [tool_calls_json]
            calls = self.parse_base_json(tool_calls_json, tools)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON part: {tool_match.group(1)}, JSON parse error: {str(e)}"
            )

        # Extract normal text without tool calls
        normal_content = (
            answer_content[: tool_match.start()] + answer_content[tool_match.end() :]
        )
        normal_text = (
            text[: answer_match.start()]
            + normal_content.strip()
            + text[answer_match.end() :]
        ).strip()

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<answer>\n<tool_calls>[{"name":"' + name + '", "arguments":',
            end="}]</tool_calls>\n</answer>",
            trigger="<answer>",
        )

    def build_ebnf(self, tools: List[Tool]):
        # Build custom EBNF for Hunyuan format
        # We need to match Hunyuan's exact JSON format in tool_calls

        # Tool rules for each function
        tool_rules = []
        tool_definitions = []

        for i, tool in enumerate(tools):
            if tool.function and tool.function.name:
                name = tool.function.name
                tool_rules.append(f"tool_{i}")

                # For simplicity, we'll use the generic json rule for arguments
                # A more sophisticated approach would generate specific argument rules
                tool_definitions.append(
                    f'tool_{i} ::= "{{\\"name\\": \\"{name}\\", \\"arguments\\": " json "}}"'
                )

        tools_choice = " | ".join(tool_rules) if tool_rules else "json"

        # Build complete EBNF with answer tags and tool_calls
        rules = [
            'root ::= "<answer>" ws content ws "<tool_calls>" ws "[" ws tool_list ws "]" ws "</tool_calls>" ws "</answer>"',
            "content ::= [^<]*",
            (
                f'tool_list ::= {tools_choice} (ws "," ws {tools_choice})*'
                if tool_rules
                else "tool_list ::= json"
            ),
        ]

        # Add tool definitions
        rules.extend(tool_definitions)

        # Add JSON grammar
        rules.append(EBNFComposer.json_grammar_ebnf_str)

        return "\n".join(rules)
