"""
Detector for LFM2 (Liquid Foundation Model 2) function call format.

Format Structure (Pythonic style):
```
<|tool_call_start|>[function_name(arg1="value1", arg2="value2")]<|tool_call_end|>
```

Multiple tool calls:
```
<|tool_call_start|>[func1(arg="val"), func2(arg="val")]<|tool_call_end|>
```

Also supports JSON format:
```
<|tool_call_start|>[{"name": "func_name", "arguments": {...}}]<|tool_call_end|>
```
"""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class Lfm2Detector(BaseFormatDetector):
    """
    Detector for LFM2 (Liquid Foundation Model 2) function call format.

    Supports both Pythonic and JSON formats:

    Pythonic:
    ```
    <|tool_call_start|>[calculator(expression="5 * 7")]<|tool_call_end|>
    ```

    JSON:
    ```
    <|tool_call_start|>[{"name": "calculator", "arguments": {"expression": "5 * 7"}}]<|tool_call_end|>
    ```
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<|tool_call_start|>"
        self.eot_token = "<|tool_call_end|>"
        self.tool_call_separator = ""

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains an LFM2 format tool call."""
        return self.bot_token in text

    def _get_parameter_value(self, val: ast.AST) -> Any:
        """
        Extract Python literal value from AST node.

        Handles constants, dicts, and lists recursively.
        Reuses pattern from PythonicDetector.
        """
        if isinstance(val, ast.Constant):
            return val.value
        elif isinstance(val, ast.Dict):
            return {
                self._get_parameter_value(k): self._get_parameter_value(v)
                for k, v in zip(val.keys, val.values)
                if k is not None  # Handle {**kwargs} case where key is None
            }
        elif isinstance(val, ast.List):
            return [self._get_parameter_value(v) for v in val.elts]
        elif isinstance(val, ast.Tuple):
            return tuple(self._get_parameter_value(v) for v in val.elts)
        elif isinstance(val, ast.Name):
            # Handle True, False, None as names in older Python
            if val.id == "True":
                return True
            elif val.id == "False":
                return False
            elif val.id == "None":
                return None
            else:
                raise ValueError(f"Unsupported name reference: {val.id}")
        elif isinstance(val, ast.UnaryOp) and isinstance(val.op, ast.USub):
            # Handle negative numbers like -5
            inner = self._get_parameter_value(val.operand)
            if isinstance(inner, (int, float)):
                return -inner
            raise ValueError(f"Cannot negate non-numeric value: {inner}")
        else:
            raise ValueError(
                f"Tool call arguments must be literals, got: {type(val).__name__}"
            )

    def _parse_pythonic_call(
        self, call: ast.Call, call_index: int, tool_indices: Dict[str, int]
    ) -> Optional[ToolCallItem]:
        """
        Parse a single AST Call node into a ToolCallItem.

        Args:
            call: AST Call node representing a function call
            call_index: Index of this call in the list of calls
            tool_indices: Mapping of tool names to their indices

        Returns:
            ToolCallItem if successful, None if the call should be skipped
        """
        if not isinstance(call.func, ast.Name):
            logger.warning(
                f"Tool call function must be a simple name, got: {type(call.func).__name__}"
            )
            return None

        function_name = call.func.id

        # Validate that the function exists in the tools
        if function_name not in tool_indices:
            logger.warning(
                f"Model attempted to call undefined function: {function_name}"
            )
            if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                return None  # Skip unknown tools (default legacy behavior)

        # Parse arguments
        arguments = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                # **kwargs unpacking - skip for now
                logger.warning("Tool call with **kwargs unpacking is not supported")
                continue
            try:
                arguments[keyword.arg] = self._get_parameter_value(keyword.value)
            except ValueError as e:
                logger.warning(f"Failed to parse argument {keyword.arg}: {e}")
                return None

        return ToolCallItem(
            tool_index=call_index,  # Use the call index in the response, not tool position
            name=function_name,
            parameters=json.dumps(arguments, ensure_ascii=False),
        )

    def _parse_pythonic_content(
        self, content: str, tools: List[Tool]
    ) -> Tuple[List[ToolCallItem], str]:
        """
        Parse Pythonic format tool calls using AST.

        Args:
            content: The content between tool call tags (without the tags)
            tools: List of available tools

        Returns:
            Tuple of (list of parsed calls, error message if any)
        """
        content = content.strip()
        tool_indices = self._get_tool_indices(tools)

        try:
            module = ast.parse(content)
            parsed = getattr(module.body[0], "value", None) if module.body else None

            if parsed is None:
                return [], "Empty or invalid Python expression"

            # Handle both single call and list of calls
            if isinstance(parsed, ast.List):
                call_nodes = parsed.elts
            elif isinstance(parsed, ast.Call):
                call_nodes = [parsed]
            else:
                return (
                    [],
                    f"Expected function call or list, got: {type(parsed).__name__}",
                )

            # Validate all elements are calls
            if not all(isinstance(e, ast.Call) for e in call_nodes):
                return [], "Not all elements in list are function calls"

            calls = []
            for call_index, call in enumerate(call_nodes):
                item = self._parse_pythonic_call(call, call_index, tool_indices)
                if item is not None:
                    calls.append(item)

            return calls, ""

        except SyntaxError as e:
            return [], f"Python syntax error: {e}"
        except Exception as e:
            logger.exception("Unexpected error in pythonic tool call parsing")
            return [], f"Unexpected error: {e}"

    def _parse_json_content(
        self, content: str, tools: List[Tool]
    ) -> Tuple[List[ToolCallItem], str]:
        """
        Parse JSON format tool calls.

        Uses parse_base_json from BaseFormatDetector for consistent handling
        of SGLANG_FORWARD_UNKNOWN_TOOLS and tool validation.

        Args:
            content: The content between tool call tags (without the tags)
            tools: List of available tools

        Returns:
            Tuple of (list of parsed calls, error message if any)
        """
        content = content.strip()

        try:
            parsed = json.loads(content)
            # parse_base_json handles list/dict normalization, tool validation,
            # and SGLANG_FORWARD_UNKNOWN_TOOLS consistently with other detectors
            calls = self.parse_base_json(parsed, tools)
            return calls, ""

        except json.JSONDecodeError as e:
            return [], f"JSON parse error: {e}"

    def _parse_tool_calls_content(
        self, content: str, tools: List[Tool]
    ) -> List[ToolCallItem]:
        """
        Parse the content between tool call tags.
        Handles both JSON and Pythonic formats.
        """
        content = content.strip()

        # First, try JSON format (faster check)
        if content.startswith("[{") or content.startswith("{"):
            calls, error = self._parse_json_content(content, tools)
            if calls:
                return calls
            # If JSON parsing failed but it looked like JSON, log the error
            if error:
                logger.debug(f"JSON parsing failed: {error}, trying Pythonic format")

        # Try Pythonic format
        calls, error = self._parse_pythonic_content(content, tools)
        if calls:
            return calls

        if error:
            logger.warning(f"Failed to parse tool calls: {error}")

        return []

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Find all <|tool_call_start|>...<|tool_call_end|> blocks
        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        match_result_list = re.findall(pattern, text, re.DOTALL)

        calls = []
        for match_result in match_result_list:
            parsed_calls = self._parse_tool_calls_content(match_result, tools)
            calls.extend(parsed_calls)

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _strip_special_tokens(self, text: str) -> str:
        """Remove special tokens from text."""
        return text.replace(self.bot_token, "").replace(self.eot_token, "")

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for LFM2 tool calls.

        This implementation properly handles Pythonic format by:
        1. Buffering until we see complete <|tool_call_start|>[...]<|tool_call_end|>
        2. Emitting normal text before tool calls immediately
        3. Parsing complete tool call blocks using detect_and_parse

        Based on PythonicDetector streaming logic.
        """
        self._buffer += new_text

        # Check for partial bot_token at the end
        partial_bot = self._ends_with_partial_token(self._buffer, self.bot_token)
        partial_eot = self._ends_with_partial_token(self._buffer, self.eot_token)

        # Find bot_token position
        bot_pos = self._buffer.find(self.bot_token)

        if bot_pos == -1:
            # No tool call start found
            if partial_bot:
                # Might be partial bot_token, hold back that part
                safe_text = self._buffer[:-partial_bot]
                self._buffer = self._buffer[-partial_bot:]
                return StreamingParseResult(normal_text=safe_text)
            else:
                # No tool call, emit all as normal text
                normal_text = self._strip_special_tokens(self._buffer)
                self._buffer = ""
                return StreamingParseResult(normal_text=normal_text)

        # We have bot_token - extract any normal text before it
        normal_text_before = self._buffer[:bot_pos] if bot_pos > 0 else ""

        # Look for the end token
        eot_pos = self._buffer.find(self.eot_token, bot_pos + len(self.bot_token))

        if eot_pos == -1:
            # No end token yet - check if we might have a partial one
            if partial_eot:
                # Hold back the partial token, but we need to keep buffering
                # Just emit any normal text before the tool call
                if normal_text_before:
                    self._buffer = self._buffer[bot_pos:]
                    return StreamingParseResult(normal_text=normal_text_before)
                # Keep buffering
                return StreamingParseResult(normal_text="")

            # No end token and no partial - keep buffering but emit normal text
            if normal_text_before:
                self._buffer = self._buffer[bot_pos:]
                return StreamingParseResult(normal_text=normal_text_before)

            # Just keep buffering
            return StreamingParseResult(normal_text="")

        # We have a complete tool call block
        tool_call_block = self._buffer[bot_pos : eot_pos + len(self.eot_token)]
        remaining = self._buffer[eot_pos + len(self.eot_token) :]

        # Parse the complete block
        result = self.detect_and_parse(tool_call_block, tools)

        # Update buffer with remaining text
        self._buffer = remaining

        # Add any normal text before the tool call
        if normal_text_before:
            result.normal_text = normal_text_before + (result.normal_text or "")

        return result

    def supports_structural_tag(self) -> bool:
        """
        Return False because LFM2 uses Pythonic format which is not JSON-compatible.

        structural_tag only supports JSON-compatible content between begin and end,
        so it cannot parse Pythonic function call syntax like `func(arg="val")`.
        """
        return False

    def structure_info(self) -> _GetInfoFunc:
        """
        Return structure info for constrained generation.

        Note: This is provided for completeness but won't be used since
        supports_structural_tag() returns False.
        """
        return lambda name: StructureInfo(
            begin="<|tool_call_start|>[" + name + "(",
            end=")]<|tool_call_end|>",
            trigger="<|tool_call_start|>",
        )
