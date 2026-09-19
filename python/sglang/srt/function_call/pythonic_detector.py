import ast
import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import safe_ast_parse

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
            module = safe_ast_parse(tool_call_text)
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
                    if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                        continue  # Skip unknown tools (default legacy behavior)

                # Convert each call on its own: an unconvertible argument used
                # to escape to the outer handler and drop every parseable
                # sibling call in the block.
                try:
                    arguments = {}
                    for keyword in call.keywords:
                        arguments[keyword.arg] = self._get_parameter_value(
                            keyword.value
                        )
                    # allow_nan=False: a non-finite float (e.g. the literal
                    # 1e999 overflowing to inf) would otherwise serialize as
                    # Infinity, which is not valid JSON for downstream clients.
                    parameters = json.dumps(
                        arguments, ensure_ascii=False, allow_nan=False
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping tool call {function_name}: {e}")
                    continue
                calls.append(
                    ToolCallItem(
                        tool_index=call_index,  # Use the call index in the response, not tool position
                        name=function_name,
                        parameters=parameters,
                    )
                )

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception:
            logger.exception("Error in pythonic tool call parsing.")
            return StreamingParseResult(normal_text=normal_text, calls=[])

    def _find_matching_bracket(self, buffer: str, start: int) -> int:
        """
        Find the matching closing bracket for the opening bracket at start position.
        Ignore brackets inside Python string literals, including escaped quotes
        and triple-quoted strings, while counting nested containers.

        Args:
            buffer: The text buffer to search in
            start: Position of the opening bracket '['

        Returns:
            Position of the matching closing bracket ']', or -1 if not found
        """
        # An apostrophe in ordinary bracketed prose (e.g. [O'Reilly]) is
        # not a Python string opener. Only apply string rules to call lists.
        is_call_list = re.match(r"\[[a-zA-Z]\w*\(", buffer[start:]) is not None
        bracket_count = 0
        quote = None
        i = start
        while i < len(buffer):
            char = buffer[i]
            if quote is not None:
                if char == "\\":
                    i += 2
                    continue
                if buffer.startswith(quote, i):
                    i += len(quote)
                    quote = None
                    continue
            elif is_call_list and char in ("'", '"'):
                quote = char * 3 if buffer.startswith(char * 3, i) else char
                i += len(quote)
                continue
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    return i
            i += 1
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

        normal_text = []
        calls = []
        position = 0
        while position < len(stripped_buffer):
            start = stripped_buffer.find("[", position)
            if start == -1:
                normal_text.append(stripped_buffer[position:])
                position = len(stripped_buffer)
                break

            normal_text.append(stripped_buffer[position:start])
            end = self._find_matching_bracket(stripped_buffer, start)
            if end == -1:
                position = start
                break

            call_text = stripped_buffer[start : end + 1]
            result = self.detect_and_parse(call_text, tools)
            normal_text.append(result.normal_text)
            for call in result.calls:
                # detect_and_parse numbers calls within one list. Streaming
                # indexes must remain unique across every list in the response.
                self.current_tool_id += 1
                call.tool_index = self.current_tool_id
                calls.append(call)
            position = end + 1

        # Only incomplete calls or special-token prefixes need another chunk.
        # Drain complete calls and trailing text now: a final chunk may contain
        # more than one call, and the serving layer need not invoke us again.
        self._buffer = stripped_buffer[position:] + held_back
        return StreamingParseResult(normal_text="".join(normal_text), calls=calls)

    def _get_parameter_value(self, val):
        if isinstance(val, ast.Constant):
            if val.value is None or isinstance(val.value, (str, int, float)):
                return val.value
            # bytes/Ellipsis/complex have no JSON form; raising here lets the
            # per-call handler skip this call instead of a TypeError inside
            # json.dumps dropping every sibling call in the block.
            raise ValueError(
                f"Constant has no JSON representation: {type(val.value).__name__}"
            )
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
