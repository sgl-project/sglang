import hashlib
import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    get_schema_properties,
    infer_type_from_json_schema,
    safe_literal_eval,
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
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    properties = get_schema_properties(tool.function.parameters)
    if arg_key not in properties:
        return None

    # Use new type inference function for complex JSON Schema support
    return infer_type_from_json_schema(properties[arg_key])


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

    # Strategy 2.5: string-typed values that are not valid JSON (S1/S2 failed) —
    # strip the wrapping quotes and keep the raw bytes, backslashes included.
    # Avoids ast.literal_eval so invalid escapes neither warn nor get reinterpreted.
    if arg_type == "string":
        if (
            len(json_value) >= 2
            and json_value[0] == json_value[-1]
            and json_value[0] in {'"', "'"}
        ):
            return json_value[1:-1], True
        return json_value, True

    # Strategy 3: ast.literal_eval
    try:
        parsed_value = safe_literal_eval(json_value)
        return parsed_value, True
    except (ValueError, SyntaxError):
        pass

    # Strategy 4: Treat as string
    try:
        quoted_value = json.dumps(str(json_value))
        return json.loads(quoted_value), True
    except (json.JSONDecodeError, ValueError):
        return json_value, False


class Glm4MoeDetector(BaseFormatDetector):
    """
    Detector for GLM-4.5 and GLM-4.6 models.
    Assumes function call format (with actual newlines):
      <tool_call>get_weather
      <arg_key>city</arg_key>
      <arg_value>北京</arg_value>
      <arg_key>date</arg_key>
      <arg_value>2024-06-27</arg_value>
      </tool_call>

    Or with literal \n characters (escaped as \\n in the output):
      <tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>北京</arg_value>\n</tool_call>

    Uses a streaming state machine to convert XML to JSON incrementally for maximum speed.
    """

    _STREAMING_PARTIAL_PATTERN = re.compile(
        r"<tool_call>(.*?)(?:\\n|\n)(.*?)(</tool_call>|$)", re.DOTALL
    )

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.func_call_regex = r"<tool_call>.*?</tool_call>"
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(?:\\n|\n)(.*)</tool_call>", re.DOTALL
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
                if func_detail is None:
                    continue
                func_name = func_detail.group(1) if func_detail.group(1) else ""
                func_args = func_detail.group(2) if func_detail.group(2) else ""
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
            partial_match = self._STREAMING_PARTIAL_PATTERN.search(current_text)
            if partial_match:
                func_name_raw = partial_match.group(1)
                func_args_raw = partial_match.group(2)
                is_tool_end = partial_match.group(3)

                # Only proceed if we have a non-empty function name
                if func_name_raw is None or not func_name_raw.strip():
                    # If we only have the start token without a function name,
                    # continue buffering until we get more content
                    return StreamingParseResult(normal_text="", calls=[])

                func_name = func_name_raw.strip()
                func_args_raw = func_args_raw.strip() if func_args_raw else ""

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

                # The name and final tool-call marker can arrive in the same
                # parse call, so continue into argument/finalization handling.
                if self.current_tool_name_sent:
                    # Process XML to JSON streaming
                    current_raw_length = len(func_args_raw)

                    if current_raw_length > self._streamed_raw_length:
                        # Get the new raw XML content
                        raw_increment = func_args_raw[self._streamed_raw_length :]

                        # Convert XML increment to JSON increment using state machine
                        json_increment = self._process_xml_to_json_streaming(
                            raw_increment, func_name, tools
                        )

                        # CRITICAL: Update streamed length BEFORE checking json_increment
                        # Even if json_increment is empty, the input has been consumed by the state machine
                        self._streamed_raw_length = current_raw_length

                        if json_increment:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=json_increment,
                                )
                            )
                            self._last_arguments += json_increment
                            self.streamed_args_for_tool[self.current_tool_id] += (
                                json_increment
                            )

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
                            self.streamed_args_for_tool[self.current_tool_id] += (
                                empty_object
                            )
                        else:
                            # The streamed outer `{` is only closed here; a
                            # trailing "}" may belong to a nested object value.
                            closing_brace = "}"
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=closing_brace,
                                )
                            )
                            self._last_arguments += closing_brace
                            self.streamed_args_for_tool[self.current_tool_id] += (
                                closing_brace
                            )

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


class _GlmTrieNode:
    """Trie node with Aho-Corasick failure link."""

    def __init__(self, node_id: int):
        self.id = node_id
        self.children: Dict[str, _GlmTrieNode] = {}
        self.is_end = False
        self.fail: _GlmTrieNode = None


def _glm_build_trie_with_failure_links(
    patterns: List[str],
) -> tuple["_GlmTrieNode", List["_GlmTrieNode"]]:
    """Build Trie and compute Aho-Corasick failure links."""
    root = _GlmTrieNode(0)
    all_nodes = [root]
    next_id = 1

    for pattern in patterns:
        node = root
        for char in pattern:
            if char not in node.children:
                new_node = _GlmTrieNode(next_id)
                next_id += 1
                all_nodes.append(new_node)
                node.children[char] = new_node
            node = node.children[char]
        node.is_end = True

    root.fail = root
    queue = deque()
    for child in root.children.values():
        child.fail = root
        queue.append(child)
    while queue:
        node = queue.popleft()
        for char, child in node.children.items():
            queue.append(child)
            fail_node = node.fail
            while fail_node != root and char not in fail_node.children:
                fail_node = fail_node.fail
            if char in fail_node.children and fail_node.children[char] != child:
                child.fail = fail_node.children[char]
            else:
                child.fail = root
            # A suffix match also completes a forbidden pattern.
            if child.fail.is_end:
                child.is_end = True

    return root, all_nodes


def _glm_get_transition(
    node: "_GlmTrieNode", char: str, root: "_GlmTrieNode"
) -> "_GlmTrieNode":
    """Follow Aho-Corasick failure links to the next state."""
    current = node
    while True:
        if char in current.children:
            return current.children[char]
        if current == root:
            return root
        current = current.fail


def _glm_escape_char_class(s: str) -> str:
    """Escape special characters for use in EBNF character class [...]."""
    result = []
    for c in s:
        if c in r"\]^-":
            result.append("\\" + c)
        elif c == "\n":
            result.append("\\n")
        elif c == "\t":
            result.append("\\t")
        elif c == "\r":
            result.append("\\r")
        elif ord(c) < 32 or ord(c) > 126:
            result.append(f"\\x{ord(c):02X}")
        else:
            result.append(c)
    return "".join(result)


def _glm_escape_string(c: str) -> str:
    """Escape a character for use in EBNF string literal "..."."""
    if c == '"':
        return '\\"'
    elif c == "\\":
        return "\\\\"
    elif c == "\n":
        return "\\n"
    elif c == "\t":
        return "\\t"
    elif c == "\r":
        return "\\r"
    elif ord(c) < 32 or ord(c) > 126:
        return f"\\x{ord(c):02X}"
    return c


def _glm_any_string_exclude(rule_name: str, negative_strings) -> List[str]:
    return list(_glm_cached_string_exclude(rule_name, tuple(negative_strings)))


@lru_cache(maxsize=32)
def _glm_cached_string_exclude(
    rule_name: str, negative_strings: tuple[str, ...]
) -> tuple[str, ...]:
    """Build EBNF that excludes forbidden substrings using Aho-Corasick states."""

    if not negative_strings:
        return (f"{rule_name} ::= [^]*",)
    sorted_strings = sorted(set(s for s in negative_strings if s))
    if not sorted_strings:
        return (f"{rule_name} ::= [^]*",)
    hash_input = "\x00".join(sorted_strings)
    hash_prefix = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
    root, all_nodes = _glm_build_trie_with_failure_links(sorted_strings)
    all_pattern_chars: Set[str] = set()
    for pattern in sorted_strings:
        all_pattern_chars.update(pattern)

    def state_name(node: "_GlmTrieNode") -> str:
        return f"s_{hash_prefix}_{node.id}"

    rules = []
    rules.append(f"{rule_name} ::= {state_name(root)}")

    for node in all_nodes:
        if node.is_end:
            continue

        excluded_chars: List[str] = []
        transitions_by_target: Dict[int, List[str]] = {}

        for char in all_pattern_chars:
            target = _glm_get_transition(node, char, root)
            if target.is_end:
                excluded_chars.append(char)
            else:
                if target.id not in transitions_by_target:
                    transitions_by_target[target.id] = []
                transitions_by_target[target.id].append(char)

        alternatives = []

        all_explicit_chars = set(excluded_chars)
        for chars in transitions_by_target.values():
            all_explicit_chars.update(chars)

        if all_explicit_chars:
            escaped = _glm_escape_char_class("".join(sorted(all_explicit_chars)))
            alternatives.append(f"[^{escaped}] {state_name(root)}")
        else:
            alternatives.append(f"[^] {state_name(root)}")

        for target_id in sorted(transitions_by_target.keys()):
            chars = transitions_by_target[target_id]
            target_node = next(n for n in all_nodes if n.id == target_id)
            for char in sorted(chars):
                alternatives.append(
                    f'"{_glm_escape_string(char)}" {state_name(target_node)}'
                )

        alternatives.append('""')

        rules.append(f"{state_name(node)} ::= {' | '.join(alternatives)}")

    return tuple(rules)


_GLM_XML_GRAMMAR_RULES = [
    'basic_string ::= (([\\"] basic_string_1 [\\"]))',
    'basic_string_1 ::= "" | [^"\\\\\\x00-\\x1F] basic_string_1 | "\\\\" escape basic_string_1',
    'escape ::= ["\\\\//bfnrt] | "u" [A-Fa-f0-9]{4}',
    'basic_integer ::= "-"? ("0" | [1-9] [0-9]*) ".0"?',
    'basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?',
    'basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"',
    'basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"',
    "ws ::= [ \\n\\t]*",
    "basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object",
    'basic_boolean ::= "true" | "false"',
    'basic_null ::= "null"',
]

_GLM_TYPE_MAPPING = {
    "string": "text_without_special_tokens",
    "number": "basic_number",
    "integer": "basic_number",
    "boolean": "basic_boolean",
    "null": "basic_null",
    "array": "basic_array",
    "object": "basic_object",
}


def _glm_hash_name(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def _glm_get_value_rule(prop: Any) -> str:
    if not isinstance(prop, dict):
        return "text_without_special_tokens"
    if "enum" in prop:
        return _glm_handle_enum(prop)
    if "type" in prop:
        return _glm_handle_type(prop)
    return "text_without_special_tokens"


def _glm_escape_ebnf_string(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)[1:-1]


def _glm_handle_enum(prop: dict) -> str:
    enum_values = prop["enum"]

    def format_enum_val(v: Any) -> str:
        value = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return f'"{_glm_escape_ebnf_string(value)}"'

    formatted_values = [format_enum_val(v) for v in enum_values]
    if not formatted_values:
        return "text_without_special_tokens"
    enum_rule = " | ".join(formatted_values)
    return f"({enum_rule})" if len(formatted_values) > 1 else enum_rule


def _glm_handle_type(prop: dict) -> str:
    prop_type = prop["type"]
    if isinstance(prop_type, list):
        type_rules = [
            _GLM_TYPE_MAPPING.get(t, "text_without_special_tokens") for t in prop_type
        ]
        return " | ".join(type_rules) if type_rules else "text_without_special_tokens"
    return _GLM_TYPE_MAPPING.get(prop_type, "text_without_special_tokens")


def _glm_build_tool_call_rules(
    non_terminal_name: str,
    functions: list[Any],
    special_tokens: "GlmSpecialTokenConfig",
    chat_template_version: Literal["glm45", "glm47"],
    required: bool = False,
    parallel_tool_calls: bool = True,
) -> list[str]:
    """Build non-strict XML tool-call rules with shallow value constraints."""
    if chat_template_version == "glm45":
        extra_seperator = '"\\n"'
    elif chat_template_version == "glm47":
        extra_seperator = ""
    else:
        raise NotImplementedError(
            f"Unsupported chat_template_version: {chat_template_version}"
        )

    repetition = (
        ("+" if required else "*") if parallel_tool_calls else ("" if required else "?")
    )
    rules = [
        f"{non_terminal_name} ::= ( {extra_seperator} tool_call_unit ){repetition}",
        f'tool_call_unit ::= "{special_tokens.begin_of_tool_call}" single_tool_call "{special_tokens.end_of_tool_call}"',
    ]

    # Include the index to distinguish duplicate function names.
    tool_alternatives = " | ".join(
        f"call_{_glm_hash_name(func.name + str(function_index))}"
        for function_index, func in enumerate(functions)
    )
    rules.append(f"single_tool_call ::= {tool_alternatives}")

    # Group alternatives so nullable values remain inside the argument tags.
    kv_template = f'"{special_tokens.begin_of_key}{{key}}{special_tokens.end_of_key}" {extra_seperator} "{special_tokens.begin_of_value}" ({{valrule}}) "{special_tokens.end_of_value}"'
    kv_separator = extra_seperator

    for function_index, func in enumerate(functions):
        tool_name = _glm_escape_ebnf_string(func.name)
        namehash = _glm_hash_name(func.name + str(function_index))
        params = func.parameters or {}
        properties = get_schema_properties(params)
        if isinstance(params, dict) and (
            "$ref" in params
            or "patternProperties" in params
            or (
                "properties" in params
                and any(keyword in params for keyword in ("allOf", "anyOf", "oneOf"))
            )
        ):
            properties = {}

        prop_kv_pairs = {}

        for prop_name, prop_schema in properties.items():
            # Composition branches can disagree on a property's value schema.
            value_rule = (
                "text_without_special_tokens"
                if any(keyword in params for keyword in ("allOf", "anyOf", "oneOf"))
                else _glm_get_value_rule(prop_schema)
            )
            pair = kv_template.format(
                key=_glm_escape_ebnf_string(prop_name), valrule=value_rule
            )
            prop_kv_pairs[prop_name] = pair

        # Non-strict arguments may be omitted, repeated, or emitted in any order.
        all_props = list(properties.keys())

        if all_props:
            all_choices = " | ".join(prop_kv_pairs[k] for k in all_props)
            arguments_rule = (
                f"( ( {all_choices} ) ( {kv_separator} ( {all_choices} ) )* )?"
            )
        else:
            arguments_rule = (
                f'( "{special_tokens.begin_of_key}" text_without_special_tokens '
                f'"{special_tokens.end_of_key}" {extra_seperator} '
                f'"{special_tokens.begin_of_value}" text_without_special_tokens '
                f'"{special_tokens.end_of_value}" {kv_separator} )*'
            )

        rules.append(
            f'call_{namehash} ::= "{tool_name}" {extra_seperator} ( arguments_{namehash} {extra_seperator} )?'
        )
        rules.append(f"arguments_{namehash} ::= {arguments_rule}")

    rules.extend(_GLM_XML_GRAMMAR_RULES)
    return rules


@dataclass
class GlmSpecialTokenConfig:
    begin_of_thinking: str = "<think>"
    end_of_thinking: str = "</think>"
    begin_of_tool_call: str = "<tool_call>"
    end_of_tool_call: str = "</tool_call>"
    begin_of_key: str = "<arg_key>"
    end_of_key: str = "</arg_key>"
    begin_of_value: str = "<arg_value>"
    end_of_value: str = "</arg_value>"
    assistant_token: str = "<|assistant|>"

    def all_special_tokens(self) -> list[str]:
        return vars(self).values()


def generate_glm_grammar(
    enable_thinking: bool,
    functions: list[Any] | None,
    special_tokens: GlmSpecialTokenConfig,
    chat_template_version: Literal["glm45", "glm47"],
    accommodate_chat_template: bool,
    allow_multiple_assistant_turns: bool,
    root_name: str = "root",
    required: bool = False,
    parallel_tool_calls: bool = True,
) -> str:
    ebnf_lines = [
        f'{root_name} ::= assistant_turn ( "{special_tokens.assistant_token}" assistant_turn )*'
        if allow_multiple_assistant_turns
        else f"{root_name} ::= assistant_turn",
        "assistant_turn ::= thinking_block text_block tool_call_blocks",
    ]

    thinking_exclusions = [
        special_tokens.begin_of_tool_call,
        special_tokens.end_of_tool_call,
        special_tokens.begin_of_key,
        special_tokens.end_of_key,
        special_tokens.begin_of_value,
        special_tokens.end_of_value,
        special_tokens.end_of_thinking,
    ]

    if chat_template_version == "glm45":
        extra_seperator = '"\\n"'
    elif chat_template_version == "glm47":
        extra_seperator = ""
    else:
        raise NotImplementedError(
            f"Unsupported chat_template_version: {chat_template_version}"
        )

    if chat_template_version == "glm45":
        if enable_thinking:
            ebnf_lines.append(
                rf'thinking_block ::= "\n{special_tokens.begin_of_thinking}" thinking_block_content "{special_tokens.end_of_thinking}"'
            )

            ebnf_lines.extend(
                _glm_any_string_exclude("thinking_block_content", thinking_exclusions)
            )
        else:
            if accommodate_chat_template:
                ebnf_lines.append('thinking_block ::= ""')
            else:
                ebnf_lines.append(
                    rf'thinking_block ::= "\n{special_tokens.begin_of_thinking}" "{special_tokens.end_of_thinking}"'
                )
    elif chat_template_version == "glm47":
        if enable_thinking:
            if accommodate_chat_template:
                ebnf_lines.append(
                    rf'thinking_block ::= thinking_block_content "{special_tokens.end_of_thinking}"'
                )
            else:
                ebnf_lines.append(
                    rf'thinking_block ::= "{special_tokens.begin_of_thinking}" thinking_block_content "{special_tokens.end_of_thinking}"'
                )
            ebnf_lines.extend(
                _glm_any_string_exclude("thinking_block_content", thinking_exclusions)
            )
        else:
            if accommodate_chat_template:
                ebnf_lines.append('thinking_block ::= ""')
            else:
                ebnf_lines.append(
                    rf'thinking_block ::= "{special_tokens.end_of_thinking}"'
                )
    else:
        raise NotImplementedError(
            f"Unsupported chat_template_version: {chat_template_version}"
        )

    ebnf_lines.extend(
        _glm_any_string_exclude(
            "text_without_special_tokens", special_tokens.all_special_tokens()
        )
    )

    ebnf_lines.append(
        f"text_block ::= ( {extra_seperator} text_without_special_tokens )?"
    )

    if functions:
        ebnf_lines.extend(
            _glm_build_tool_call_rules(
                non_terminal_name="tool_call_blocks",
                functions=functions,
                special_tokens=special_tokens,
                chat_template_version=chat_template_version,
                required=required,
                parallel_tool_calls=parallel_tool_calls,
            )
        )
    else:
        ebnf_lines.append('tool_call_blocks ::= ""')

    non_terminals = {}
    deduped_lines = []
    for line in ebnf_lines:
        assert "\n" not in line, "Each EBNF rule should be in a single line."
        lhs = line.split("::=")[0].strip()
        if lhs in non_terminals:
            if non_terminals[lhs] == line:
                continue
            raise ValueError(f"Duplicate non-terminal found: {lhs}")
        non_terminals[lhs] = line
        deduped_lines.append(line)

    return "\n".join(deduped_lines)
