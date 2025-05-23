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
        FUNC_CALL = r"""
            [a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)*       # Function name: dotted, Python-style identifiers
            \(                                      # Opening parenthesis for arguments
                (                                   # --- Optional repeated key=val pairs ending with comma ---
                    [a-zA-Z]+\w*=.*,\s*             #     Match key=val followed by comma and optional whitespace
                )*                                  # --- Zero or more such arguments
                (                                   # --- Optional last argument without trailing comma ---
                    [a-zA-Z]+\w*=.*\s*              #     Match final key=val (no comma)
                )?                                  # --- This part is optional
            \)                                      # Closing parenthesis
        """

        self.tool_call_regex = re.compile(
            rf"""
            \[                                      # Opening square bracket
                \s*
                {FUNC_CALL}                         # First function call
                (
                    \s*,\s*{FUNC_CALL}              # Additional function calls (comma-separated)
                )*                                  # Zero or more
                \s*
            \]                                      # Closing square bracket
            """,
            re.VERBOSE | re.DOTALL,
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
        end = self._buffer.find("]", start)
        if start != -1 and end != -1:
            call_text = self._buffer[start : end + 1]
            result = self.detect_and_parse(call_text, tools)
            self._buffer = self._buffer[end + 1 :]
            return result
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
            tool_calls_rule='"[" function_call ("," function_call)* "]"',
            call_rule_fmt='call_{name} ::= "{name}" "(" {arguments_rule} ")"',
            arguments_rule_fmt="{arg_rules}",
            key_value_fmt='"{key}" "=" {valrule}',
            is_pythonic=True,
        )
