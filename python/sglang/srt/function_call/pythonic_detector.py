import ast
import json
import logging
import re
from typing import List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer

logger = logging.getLogger(__name__)


class PythonicDetector(BaseFormatDetector):
    """
    Detector for Llama-4 models with Pythonic tool call format.

    The Pythonic format uses Python function call syntax within square brackets,
    with arguments as Python literals rather than JSON.

    Format Structure:
    ```
    [tool1(arg1=val1, arg2=val2), tool2(arg1=val3)]
    ```

    Reference: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct?chat_template=default
    """

    def __init__(self):
        super().__init__()
        self.tool_call_regex = re.compile(
            r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
            re.DOTALL,
        )

    @staticmethod
    def _text_strip(text: str) -> str:
        # Llama 4 model sometime will output <|python_start|> and <|python_end|> tokens
        # remove those tokens
        text = text.replace("<|python_start|>", "")
        text = text.replace("<|python_end|>", "")
        return text

    def has_tool_call(self, text: str) -> bool:
        return bool(self.tool_call_regex.search(self._text_strip(text.strip())))

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        # Try parsing the text as a Python list of function calls
        text = text.strip()

        # Remove unexpected <|python_start|> and <|python_end|> for llama4
        text = self._text_strip(text)

        match = self.tool_call_regex.search(text)
        if match is None:
            return StreamingParseResult(normal_text=text, calls=[])

        # Extract the tool call part and any text before/after it
        tool_call_start = match.start()
        tool_call_end = match.end()

        normal_text_before = text[:tool_call_start] if tool_call_start > 0 else ""
        tool_call_text = text[tool_call_start:tool_call_end]
        normal_text_after = text[tool_call_end:] if tool_call_end < len(text) else ""

        # Combine normal text
        normal_text = normal_text_before + normal_text_after

        try:
            module = ast.parse(tool_call_text)
            parsed = getattr(module.body[0], "value", None)
            if not (
                isinstance(parsed, ast.List)
                and all(isinstance(e, ast.Call) for e in parsed.elts)
            ):
                return StreamingParseResult(normal_text=normal_text, calls=[])

            calls = []
            tool_indices = self._get_tool_indices(tools)
            for call_index, call in enumerate(parsed.elts):
                if not isinstance(call.func, ast.Name):
                    continue
                function_name = call.func.id
                # Validate that the function exists in the tools
                if function_name not in tool_indices:
                    logger.warning(
                        f"Model attempted to call undefined function: {function_name}"
                    )
                    continue
                arguments = {}
                for keyword in call.keywords:
                    arguments[keyword.arg] = self._get_parameter_value(keyword.value)
                calls.append(
                    ToolCallItem(
                        tool_index=call_index,  # Use the call index in the response, not tool position
                        name=function_name,
                        parameters=json.dumps(arguments, ensure_ascii=False),
                    )
                )

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception:
            logger.exception("Error in pythonic tool call parsing.")
            return StreamingParseResult(normal_text=normal_text, calls=[])

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

    def _strip_and_split_buffer(self, buffer: str) -> tuple[str, str]:
        """
        Strip special tokens from buffer and split into safe_text and held_back_text.

        Returns:
            tuple of (safe_text_to_output, text_to_hold_in_buffer)
        """
        # Check if original buffer ends with a partial token at the end
        special_tokens = ["<|python_start|>", "<|python_end|>"]

        for token in special_tokens:
            partial_length = self._ends_with_partial_token(buffer, token)
            if partial_length > 0:
                # Split buffer: safe part + held back partial token
                safe_text = buffer[:-partial_length]
                held_back = buffer[-partial_length:]
                # Strip complete special tokens from safe part only
                safe_text = self._text_strip(safe_text)
                return safe_text, held_back

        # No partial tokens found, strip complete tokens from entire buffer
        safe_text = self._text_strip(buffer)
        return safe_text, ""

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for pythonic tool calls.
        Buffers input until a complete pythonic tool call (from [ to ]) is found,
        then parses and emits any detected calls.
        """
        self._buffer += new_text

        # Strip special tokens from entire buffer and handle partial tokens
        stripped_buffer, held_back = self._strip_and_split_buffer(self._buffer)

        start = stripped_buffer.find("[")

        if start == -1:
            # No tool call bracket found
            self._buffer = held_back
            return StreamingParseResult(normal_text=stripped_buffer)

        normal_text = stripped_buffer[:start] if start > 0 else ""

        end = self._find_matching_bracket(stripped_buffer, start)
        if end != -1:
            # Found complete tool call
            call_text = stripped_buffer[start : end + 1]
            result = self.detect_and_parse(call_text, tools)

            # Update buffer with remaining text after tool call plus any held back text
            remaining_text = stripped_buffer[end + 1 :] + held_back
            self._buffer = remaining_text

            # If we had normal text before the tool call, add it to the result
            if normal_text:
                result.normal_text = normal_text + (result.normal_text or "")

            return result

        # We have an opening bracket but no closing bracket yet
        # Put back everything from the bracket onwards plus held back text
        self._buffer = stripped_buffer[start:] + held_back

        if normal_text:
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

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def build_ebnf(self, tools: List[Tool]) -> Optional[str]:
        return EBNFComposer.build_ebnf(
            tools,
            sequence_start_token="[",
            sequence_end_token="]",
            tool_call_separator=",",
            function_format="pythonic",
        )
