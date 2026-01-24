import ast
import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import infer_type_from_json_schema

logger = logging.getLogger(__name__)


class StreamState(str, Enum):
    """State machine states for XML to JSON streaming conversion."""

    INIT = "INIT"
    BETWEEN = "BETWEEN"
    IN_KEY = "IN_KEY"
    WAITING_VALUE = "WAITING_VALUE"
    IN_VALUE = "IN_VALUE"


def get_argument_type(
    func_name: str, arg_key: str, defined_tools: List[Tool]
) -> Optional[str]:
    """Get the expected type of a function argument from tool definitions.

    Supports complex JSON Schema definitions including:
    - Direct type field (including type arrays)
    - anyOf/oneOf: parameter can be any of multiple types
    - enum: parameter must be one of enum values
    - allOf: parameter must satisfy all type definitions
    - properties: inferred as object type
    - items: inferred as array type

    Args:
        func_name: Name of the function/tool
        arg_key: Name of the argument
        defined_tools: List of available tools

    Returns:
        The type string (e.g., 'string', 'number', 'object') or None if not found
    """
    name2tool = {tool.function.name: tool for tool in defined_tools}

    # Check if function exists
    tool = name2tool.get(func_name)
    if not tool:
        return None

    # Get parameters safely using getattr
    params = getattr(tool.function, "parameters", None)
    if not isinstance(params, dict):
        return None

    # Navigate to the type using dict.get() for safe access
    properties = params.get("properties")
    if not isinstance(properties, dict):
        return None

    arg_spec = properties.get(arg_key)
    if isinstance(arg_spec, dict):
        # Use the new type inference function for complex JSON Schema support
        return infer_type_from_json_schema(arg_spec)

    return None


def _convert_to_number(value: str) -> Any:
    """Convert string to appropriate number type (int or float).

    Args:
        value: String value to convert

    Returns:
        Converted number or original string if conversion fails
    """
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        else:
            return int(value)
    except (ValueError, AttributeError):
        return value


def parse_arguments(
    json_value: str, arg_type: Optional[str] = None
) -> Tuple[Any, bool]:
    """Parse argument value with multiple fallback strategies.

    Args:
        json_value: Raw string value to parse
        arg_type: Expected type hint ('string', 'number', 'object', etc.)

    Returns:
        Tuple of (parsed_value, is_valid_json)
    """
    # Strategy 1: Direct JSON parsing
    try:
        parsed_value = json.loads(json_value)

        # Type coercion for number type
        if arg_type == "number" and isinstance(parsed_value, str):
            parsed_value = _convert_to_number(parsed_value)

        return parsed_value, True
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Unescape and parse
    try:
        wrapped = json.loads('{"tmp": "' + json_value + '"}')
        parsed_value = json.loads(wrapped["tmp"])

        if arg_type == "number" and isinstance(parsed_value, str):
            parsed_value = _convert_to_number(parsed_value)

        return parsed_value, True
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Strategy 3: ast.literal_eval
    try:
        parsed_value = ast.literal_eval(json_value)
        return parsed_value, True
    except (ValueError, SyntaxError):
        pass

    # Strategy 4: Treat as string
    try:
        quoted_value = json.dumps(str(json_value))
        return json.loads(quoted_value), True
    except (json.JSONDecodeError, ValueError):
        return json_value, False


class Glm47MoeDetector(BaseFormatDetector):
    """
    Detector for GLM-4.7 and GLM-5 models.
    Assumes function call format:
      <tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value><arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call><tool_call>get_weather<arg_key>city</arg_key><arg_value>上海</arg_value><arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.func_call_regex = r"<tool_call>.*?</tool_call>"
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
        self._last_arguments = ""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self._streamed_raw_length = 0
        self._tool_call_completed = False  # Track if tool call has been completed
        self._sent_empty_object = (
            False  # Track if empty object has been sent for no-arg functions
        )
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Reset the streaming state machine for a new tool call."""
        self._stream_state = StreamState.INIT
        self._current_key = ""
        self._current_value = ""
        self._xml_tag_buffer = ""
        self._is_first_param = True
        self._value_started = False
        self._cached_value_type: Optional[str] = (
            None  # Cache the value type for consistency
        )
        self._tool_call_completed = False  # Reset tool call completion status
        self._sent_empty_object = False  # Reset empty object sent status

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a glm-4.5 / glm-4.6 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        # Extract all normal text (before, between, and after tool calls)
        normal_text_parts = []
        last_end = 0

        # Find all tool call matches
        for match in re.finditer(self.func_call_regex, text, re.DOTALL):
            # Add text before this tool call
            if match.start() > last_end:
                normal_text_parts.append(text[last_end : match.start()])
            last_end = match.end()

        # Add any remaining text after the last tool call
        if last_end < len(text):
            normal_text_parts.append(text[last_end:])

        # Combine all normal text parts
        normal_text = "".join(normal_text_parts).strip()

        # Parse tool calls
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = self.func_detail_regex.search(match_result)
                if func_detail is None:
                    continue
                func_name = func_detail.group(1) if func_detail.group(1) else ""
                func_args = func_detail.group(2) if func_detail.group(2) else ""
                arguments = {}
                if func_args:
                    pairs = self.func_arg_regex.findall(func_args)
                    # Parse arguments using shared method
                    arguments = self._parse_argument_pairs(pairs, func_name, tools)

                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": arguments}
                calls.extend(self.parse_base_json(match_result, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}", exc_info=True)
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def _get_value_type(self, func_name: str, key: str, tools: List[Tool]) -> str:
        """Get parameter type from tool definition, with fallback to auto-detection.

        Args:
            func_name: Name of the function
            key: Parameter name
            tools: List of available tools

        Returns:
            Type string: 'string', 'number', 'object', 'array', or 'boolean'
        """
        arg_type = get_argument_type(func_name, key, tools)
        if arg_type:
            return arg_type

        # Improved auto-detection type from value (best effort)
        value_content = self._current_value.strip() if self._current_value else ""

        if not value_content:
            return "string"

        # Try to parse as valid JSON first
        try:
            parsed = json.loads(value_content)
            if isinstance(parsed, dict):
                return "object"
            elif isinstance(parsed, list):
                return "array"
            elif isinstance(parsed, bool):
                return "boolean"
            elif isinstance(parsed, (int, float)):
                return "number"
            # For string values, check if they look like numbers
            elif isinstance(parsed, str):
                if parsed.isdigit() or (
                    parsed.startswith("-") and parsed[1:].isdigit()
                ):
                    return "number"
                return "string"
        except json.JSONDecodeError:
            # Not valid JSON, try heuristic detection
            first_char = value_content[0] if value_content else ""

            if first_char.isdigit() or first_char in ["-", "."]:
                return "number"
            elif first_char in ["{", "["]:
                return "object"
            elif first_char in ['"', "'"]:
                return "string"

        # Default to string (safest fallback)
        return "string"

    def _format_value_complete(self, value: str, value_type: str) -> str:
        """Format complete value based on type.

        Args:
            value: Raw value string
            value_type: Expected type ('string', 'number', 'object')

        Returns:
            Properly formatted JSON value string
        """
        if value_type == "string":
            # Ensure proper JSON string formatting with quotes
            return json.dumps(value, ensure_ascii=False)
        elif value_type == "number":
            try:
                num = _convert_to_number(value.strip() if value else "")
                return str(num)
            except (ValueError, AttributeError):
                # Fallback to string if not a valid number
                logger.warning(
                    f"Failed to parse '{value}' as number, treating as string"
                )
                return json.dumps(str(value) if value else "", ensure_ascii=False)
        else:
            # For object/array types, return as-is (should already be valid JSON)
            return value

    def _process_xml_to_json_streaming(
        self, raw_increment: str, func_name: str, tools: List[Tool]
    ) -> str:
        """Convert XML increment to JSON streaming output using state machine.

        This method processes XML fragments character by character and converts them
        to JSON format incrementally. It maintains state across calls to handle
        partial XML tags and values.

        Args:
            raw_increment: New XML content to process
            func_name: Name of the function being called
            tools: List of available tools for type inference

        Returns:
            JSON string increment to append to the output
        """
        json_output = ""

        for char in raw_increment:
            self._xml_tag_buffer += char

            if self._stream_state in [StreamState.INIT, StreamState.BETWEEN]:
                if self._xml_tag_buffer.endswith("<arg_key>"):
                    self._stream_state = StreamState.IN_KEY
                    self._current_key = ""
                    self._xml_tag_buffer = ""
                    json_output += "{" if self._is_first_param else ", "
                    self._is_first_param = False

            elif self._stream_state == StreamState.IN_KEY:
                if self._xml_tag_buffer.endswith("</arg_key>"):
                    self._current_key = self._xml_tag_buffer[:-10].strip()
                    self._xml_tag_buffer = ""
                    self._stream_state = StreamState.WAITING_VALUE
                    json_output += (
                        json.dumps(self._current_key, ensure_ascii=False) + ": "
                    )

            elif self._stream_state == StreamState.WAITING_VALUE:
                if self._xml_tag_buffer.endswith("<arg_value>"):
                    self._stream_state = StreamState.IN_VALUE
                    self._current_value = ""
                    self._xml_tag_buffer = ""
                    self._value_started = False
                    # Determine and cache the value type at the start
                    self._cached_value_type = self._get_value_type(
                        func_name, self._current_key, tools
                    )

            elif self._stream_state == StreamState.IN_VALUE:
                if self._xml_tag_buffer.endswith("</arg_value>"):
                    final_value = self._xml_tag_buffer[:-12]
                    self._current_value += final_value

                    # Use cached value type for consistency
                    value_type = self._cached_value_type or "string"

                    if self._value_started:
                        # Output any remaining content
                        if final_value:
                            if value_type == "string":
                                json_output += json.dumps(
                                    final_value, ensure_ascii=False
                                )[1:-1]
                            else:
                                json_output += final_value
                        # Always output closing quote for string type when value was started
                        if value_type == "string":
                            json_output += '"'
                    else:
                        # Value was never started (empty or complete in one chunk)
                        json_output += self._format_value_complete(
                            self._current_value, value_type
                        )

                    self._xml_tag_buffer = ""
                    self._stream_state = StreamState.BETWEEN
                    self._current_value = ""
                    self._value_started = False
                    self._cached_value_type = None  # Reset cached type
                else:
                    closing_tag = "</arg_value>"
                    is_potential_closing = len(self._xml_tag_buffer) <= len(
                        closing_tag
                    ) and closing_tag.startswith(self._xml_tag_buffer)

                    if not is_potential_closing:
                        content = self._xml_tag_buffer
                        # Use cached value type for consistency
                        value_type = self._cached_value_type or "string"

                        if value_type == "string":
                            if not self._value_started:
                                json_output += '"'
                                self._value_started = True
                            if content:
                                json_output += json.dumps(content, ensure_ascii=False)[
                                    1:-1
                                ]
                                self._current_value += content
                                self._xml_tag_buffer = ""
                        elif value_type == "number":
                            if content:
                                if not self._value_started:
                                    self._value_started = True
                                json_output += content
                                self._current_value += content
                                self._xml_tag_buffer = ""
                        else:
                            # For object/array types, output as-is
                            if content:
                                if not self._value_started:
                                    self._value_started = True
                                json_output += content
                                self._current_value += content
                                self._xml_tag_buffer = ""

        return json_output

    def _extract_match_groups(self, match: re.Match) -> tuple[str, str, str]:
        """Extract function name, arguments and end marker from regex match.

        Args:
            match: Regex match object

        Returns:
            (func_name, func_args_raw, is_tool_end)
        """
        func_name = match.group(1).strip()
        func_args_raw = match.group(2).strip() if match.group(2) else ""
        is_tool_end = match.group(3) or ""
        return func_name, func_args_raw, is_tool_end

    def _send_tool_name_if_needed(
        self, func_name: str, has_arg_key: bool, is_tool_end: str
    ) -> Optional[ToolCallItem]:
        """Send tool name if needed.

        Args:
            func_name: Function name
            has_arg_key: Whether current text contains <arg_key
            is_tool_end: Whether end marker is encountered

        Returns:
            Tool call item or None
        """
        if self.current_tool_name_sent:
            return None

        # Function name completeness check
        is_func_name_complete = has_arg_key or is_tool_end == self.eot_token

        if not is_func_name_complete:
            return None

        if not func_name:
            logger.warning("Empty function name detected, skipping tool call")
            return None

        # Send tool name
        self.current_tool_name_sent = True
        self._streamed_raw_length = 0
        self._reset_streaming_state()

        # Record tool info
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": func_name,
            "arguments": {},
        }

        return ToolCallItem(
            tool_index=self.current_tool_id,
            name=func_name,
            parameters="",
        )

    def _process_arguments_streaming(
        self, func_name: str, func_args_raw: str, tools: List[Tool]
    ) -> Optional[ToolCallItem]:
        """Process streaming arguments.

        Args:
            func_name: Function name
            func_args_raw: Raw argument string
            tools: List of available tools

        Returns:
            Tool call item with parameter updates or None
        """
        current_raw_length = len(func_args_raw)

        if current_raw_length <= self._streamed_raw_length:
            return None

        # Get new raw XML content
        raw_increment = func_args_raw[self._streamed_raw_length :]

        # Convert XML to JSON using state machine
        json_increment = self._process_xml_to_json_streaming(
            raw_increment, func_name, tools
        )

        # CRITICAL: Update streamed length BEFORE early return
        # Even if json_increment is empty, the input has been consumed by the state machine
        self._streamed_raw_length = current_raw_length

        if not json_increment:
            return None

        # Update state
        self._last_arguments += json_increment
        self.streamed_args_for_tool[self.current_tool_id] += json_increment

        return ToolCallItem(
            tool_index=self.current_tool_id,
            name=None,
            parameters=json_increment,
        )

    def _finalize_tool_call(
        self,
        func_name: str,
        func_args_raw: str,
        tools: List[Tool],
        match_end_pos: int,
        current_text: str,
    ) -> List[ToolCallItem]:
        """Complete tool call processing.

        Args:
            func_name: Function name
            func_args_raw: Raw argument string
            tools: List of available tools
            match_end_pos: Match end position
            current_text: Current text

        Returns:
            List of tool call items to add
        """
        calls = []

        # Handle no-arg function or need to close braces
        if self._is_first_param and not self._sent_empty_object:
            # No-arg function
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=None,
                    parameters="{}",
                )
            )
            self._last_arguments += "{}"
            self.streamed_args_for_tool[self.current_tool_id] += "{}"
            self._sent_empty_object = True
        elif not self._last_arguments.endswith("}") and not self._sent_empty_object:
            # Need to close brace
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=None,
                    parameters="}",
                )
            )
            self._last_arguments += "}"
            self.streamed_args_for_tool[self.current_tool_id] += "}"
            self._sent_empty_object = True

        # Parse final arguments
        if func_args_raw:
            try:
                pairs = self.func_arg_regex.findall(func_args_raw)
                if pairs:
                    arguments = self._parse_argument_pairs(pairs, func_name, tools)
                    self.prev_tool_call_arr[self.current_tool_id][
                        "arguments"
                    ] = arguments
            except Exception as e:
                logger.debug(f"Failed to parse arguments: {e}", exc_info=True)

        # Clean buffer
        self._buffer = current_text[match_end_pos:]

        # Reset state for next tool call
        self._tool_call_completed = True
        self.current_tool_id += 1
        self._last_arguments = ""
        self.current_tool_name_sent = False
        self._streamed_raw_length = 0
        self._reset_streaming_state()

        return calls

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for GLM-4.5 and GLM-4.6 format.
        Uses a state machine to convert XML to JSON incrementally for true character-by-character streaming.
        Outputs JSON increments immediately as XML data arrives.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call
        has_tool_call = self.bot_token in current_text

        if not has_tool_call:
            # Check if buffer could be the start of a tool call
            # Keep buffer if it could be a partial match of bot_token
            is_potential_start = any(
                self.bot_token.startswith(current_text[-i:])
                for i in range(1, min(len(current_text), len(self.bot_token)) + 1)
            )

            if not is_potential_start:
                # Not a potential tool call, return as normal text
                # Must return the entire buffer (current_text), not just new_text,
                # because buffer may contain previously accumulated characters like '<'
                # that turned out not to be part of a tool call
                output_text = current_text
                self._buffer = ""
                if self.eot_token in output_text:
                    output_text = output_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=output_text)
            else:
                # Could be start of tool call, keep buffering
                return StreamingParseResult(normal_text="", calls=[])

        # Extract any text before the first bot_token and return it as normal_text
        normal_text = ""
        first_bot_token_idx = current_text.find(self.bot_token)
        if first_bot_token_idx > 0:
            normal_text = current_text[:first_bot_token_idx]
            current_text = current_text[first_bot_token_idx:]
            # Update buffer to only include from the bot token onwards
            self._buffer = current_text

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            # Try to match a partial or complete tool call
            # Use a single flexible regex pattern that handles all cases
            partial_match = re.search(
                r"<tool_call>(.*?)(?:(<arg_key.*?))?(?:(</tool_call>)|$)",
                current_text,
                re.DOTALL,
            )

            if not partial_match:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            # Extract match groups using helper method
            func_name, func_args_raw, is_tool_end = self._extract_match_groups(
                partial_match
            )

            # Initialize tool call state if needed (keeping existing logic)
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = [""]
                self._streamed_raw_length = 0
                self.current_tool_name_sent = False  # Reset for new tool call
                self._reset_streaming_state()
            # Check if this is a continuation of an existing tool call or a new one
            elif not self.current_tool_name_sent:
                # Only increment tool_id if we're truly starting a NEW tool call
                # Don't increment if this is just the first time we're processing
                # a tool call that was received in the buffer
                # The key insight: only increment when we've COMPLETED a previous tool call
                # and now see another bot_token in new_text
                pass  # Remove the problematic auto-increment logic

            # Ensure tracking arrays are large enough (keeping existing logic)
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            # Determine if function name is complete by checking for <arg_key> in the full text
            # This is important for streaming scenarios where args come in later chunks
            has_arg_key = "<arg_key" in current_text

            # Send tool name if needed
            tool_name_item = self._send_tool_name_if_needed(
                func_name, has_arg_key, is_tool_end
            )
            if tool_name_item:
                calls.append(tool_name_item)

            # Process streaming arguments if tool name has been sent
            if self.current_tool_name_sent:
                arg_item = self._process_arguments_streaming(
                    func_name, func_args_raw, tools
                )
                if arg_item:
                    calls.append(arg_item)

                # Finalize tool call if end token is encountered
                if is_tool_end == self.eot_token and not self._tool_call_completed:
                    finalize_calls = self._finalize_tool_call(
                        func_name,
                        func_args_raw,
                        tools,
                        partial_match.end(),
                        current_text,
                    )
                    calls.extend(finalize_calls)
                    return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}", exc_info=True)
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _parse_argument_pairs(
        self, pairs: List[Tuple[str, str]], func_name: str, tools: List[Tool]
    ) -> Dict[str, Any]:
        """Parse argument key-value pairs with type coercion.

        Args:
            pairs: List of (key, value) tuples from regex matching
            func_name: Name of the function
            tools: List of available tools

        Returns:
            Dictionary of parsed arguments
        """
        arguments = {}
        for arg_key, arg_value in pairs:
            arg_key = arg_key.strip()
            arg_value = arg_value.strip()
            arg_type = get_argument_type(func_name, arg_key, tools)
            parsed_value, is_good_json = parse_arguments(arg_value, arg_type)

            if arg_type == "string":
                # Only convert to string if explicitly defined as string type
                if isinstance(parsed_value, str):
                    arguments[arg_key] = parsed_value
                elif isinstance(parsed_value, (dict, list)):
                    # If parsed as dict/list but schema says string, convert to JSON string
                    arguments[arg_key] = json.dumps(parsed_value, ensure_ascii=False)
                else:
                    arguments[arg_key] = str(parsed_value)
            elif arg_type is None:
                # If type is not defined, keep the parsed value as-is
                arguments[arg_key] = parsed_value if is_good_json else arg_value
            else:
                # For other types (number, object, array, etc.), use parsed value
                arguments[arg_key] = parsed_value if is_good_json else arg_value

        return arguments

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
