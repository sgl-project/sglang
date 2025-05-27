import ast
import json
import logging
import re
from typing import List, Optional

from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer
from sglang.srt.openai_api.protocol import Tool

logger = logging.getLogger(__name__)


class PythonicDetector(BaseFormatDetector):
    """
    Detector for Llama-3.2 and Llama-4 models with pythonic tool call format.
    Assumes function call format:
      [tool1(arg1=val1, arg2=val2), tool2(arg1=val3)]
    Arguments are Python literals (not JSON).
    """

    def __init__(self):
        super().__init__()
        self.tool_call_regex = re.compile(
            r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return bool(self.tool_call_regex.match(text.strip()))

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        # Try parsing the text as a Python list of function calls
        text = text.strip()
        if not (text.startswith("[") and text.endswith("]")):
            # Not a pythonic tool call format
            return StreamingParseResult(normal_text=text, calls=[])
        try:
            module = ast.parse(text)
            parsed = getattr(module.body[0], "value", None)
            if not (
                isinstance(parsed, ast.List)
                and all(isinstance(e, ast.Call) for e in parsed.elts)
            ):
                return StreamingParseResult(normal_text=text, calls=[])
            calls = []
            tool_indices = {
                tool.function.name: i
                for i, tool in enumerate(tools)
                if tool.function.name
            }
            for call in parsed.elts:
                if not isinstance(call.func, ast.Name):
                    continue
                function_name = call.func.id
                arguments = {}
                for keyword in call.keywords:
                    arguments[keyword.arg] = self._get_parameter_value(keyword.value)
                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices.get(function_name, -1),
                        name=function_name,
                        parameters=json.dumps(arguments, ensure_ascii=False),
                    )
                )
            return StreamingParseResult(normal_text="", calls=calls)
        except Exception:
            logger.exception("Error in pythonic tool call parsing.")
            return StreamingParseResult(normal_text=text, calls=[])

    def _find_matching_bracket(self, buffer: str, start: int) -> int:
        """
        Find the matching closing bracket for the opening bracket at start position.
        Properly handles nested brackets.

        Args:
            buffer: The text buffer to search in
            start: Position of the opening bracket '['

        Returns:
            Position of the matching closing bracket ']', or -1 if not found
        """
        bracket_count = 0
        for i in range(start, len(buffer)):
            if buffer[i] == "[":
                bracket_count += 1
            elif buffer[i] == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    return i
        return -1  # No matching bracket found

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for pythonic tool calls.
        Buffers input until a complete pythonic tool call (from [ to ]) is found,
        then parses and emits any detected calls.
        """
        self._buffer += new_text
        start = self._buffer.find("[")

        if start == -1:
            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text)

        normal_text = self._buffer[:start] if start > 0 else ""

        end = self._find_matching_bracket(self._buffer, start)
        if end != -1:
            call_text = self._buffer[start : end + 1]
            result = self.detect_and_parse(call_text, tools)
            self._buffer = self._buffer[end + 1 :]

            # If we had normal text before the tool call, add it to the result
            if normal_text:
                result.normal_text = normal_text + (result.normal_text or "")

            return result

        # We have an opening bracket but no closing bracket yet
        if normal_text:
            self._buffer = self._buffer[start:]
            return StreamingParseResult(normal_text=normal_text)

        # Otherwise, we're still accumulating a potential tool call
        return StreamingParseResult(normal_text="")

    def _get_parameter_value(self, val):
        if isinstance(val, ast.Constant):
            return val.value
        elif isinstance(val, ast.Dict):
            return {
                k.value: self._get_parameter_value(v)
                for k, v in zip(val.keys, val.values)
            }
        elif isinstance(val, ast.List):
            return [self._get_parameter_value(v) for v in val.elts]
        else:
            raise ValueError("Tool call arguments must be literals")

    def structure_info(self) -> _GetInfoFunc:
        def info(name: str):
            return StructureInfo(begin=f"[{name}(", end=")]", trigger=f"[{name}(")

        return info

    def build_ebnf(self, tools: List[Tool]) -> Optional[str]:
        return EBNFComposer.build_ebnf(
            tools,
            sequence_start_token="[",
            sequence_end_token="]",
            tool_call_separator=",",
            function_format="pythonic",
        )
