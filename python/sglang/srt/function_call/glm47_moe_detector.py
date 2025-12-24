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

    Args:
        func_name: Name of the function/tool
        arg_key: Name of the argument
        defined_tools: List of available tools

    Returns:
        The type string (e.g., 'string', 'number', 'object') or None if not found
    """
    name2tool = {tool.function.name: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    properties = (tool.function.parameters or {}).get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    if arg_key not in properties:
        return None
    return properties[arg_key].get("type", None)


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
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = self.func_detail_regex.search(match_result)
                func_name = func_detail.group(1)
                func_args = func_detail.group(2)
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
            Type string: 'string', 'number', or 'object'
        """
        arg_type = get_argument_type(func_name, key, tools)
        if arg_type:
            return arg_type

        # Auto-detect type from value (best effort)
        first_chars = self._current_value.strip()[:10] if self._current_value else ""
        if first_chars:
            first_char = first_chars[0]
            if first_char.isdigit() or first_char in ["-", "."]:
                return "number"
            elif first_char in ["{", "["]:
                return "object"

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
                num = _convert_to_number(value.strip())
                return str(num)
            except (ValueError, AttributeError):
                # Fallback to string if not a valid number
                logger.warning(
                    f"Failed to parse '{value}' as number, treating as string"
                )
                return json.dumps(str(value), ensure_ascii=False)
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

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            # Try to match a partial or complete tool call
            partial_match = re.search(
                pattern=r"<tool_call>(.*?)(<arg_key>.*?)?(</tool_call>|$)",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name = partial_match.group(1).strip()
                func_args_raw = partial_match.group(2).strip()
                is_tool_end = partial_match.group(3)

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]
                    self._streamed_raw_length = 0
                    self.current_tool_name_sent = False
                    self._reset_streaming_state()

                # Ensure we have enough entries in our tracking arrays
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                # Send tool name first if not sent yet
                if not self.current_tool_name_sent:
                    assert func_name, "func_name should not be empty"
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    self._streamed_raw_length = 0
                    self._reset_streaming_state()
                    # Store the tool call info
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                else:
                    # Process XML to JSON streaming
                    current_raw_length = len(func_args_raw)

                    if current_raw_length > self._streamed_raw_length:
                        # Get the new raw XML content
                        raw_increment = func_args_raw[self._streamed_raw_length :]

                        # Convert XML increment to JSON increment using state machine
                        json_increment = self._process_xml_to_json_streaming(
                            raw_increment, func_name, tools
                        )

                        if json_increment:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=json_increment,
                                )
                            )
                            self._last_arguments += json_increment
                            self.streamed_args_for_tool[
                                self.current_tool_id
                            ] += json_increment

                        # Update the streamed length
                        self._streamed_raw_length = current_raw_length

                    if is_tool_end == self.eot_token:
                        if self._is_first_param:
                            empty_object = "{}"
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=empty_object,
                                )
                            )
                            self._last_arguments += empty_object
                        elif not self._last_arguments.endswith("}"):
                            closing_brace = "}"
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=closing_brace,
                                )
                            )
                            self._last_arguments += closing_brace
                            self.streamed_args_for_tool[
                                self.current_tool_id
                            ] += closing_brace

                        try:
                            pairs = self.func_arg_regex.findall(func_args_raw)
                            if pairs:
                                arguments = self._parse_argument_pairs(
                                    pairs, func_name, tools
                                )
                                self.prev_tool_call_arr[self.current_tool_id][
                                    "arguments"
                                ] = arguments
                        except Exception as e:
                            logger.debug(
                                f"Failed to parse arguments: {e}", exc_info=True
                            )

                        # Remove the completed tool call from buffer
                        self._buffer = current_text[partial_match.end(3) :]

                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        self._streamed_raw_length = 0
                        self._reset_streaming_state()
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}", exc_info=True)
            return StreamingParseResult(normal_text=current_text)

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
