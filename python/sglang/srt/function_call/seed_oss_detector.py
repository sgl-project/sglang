import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer
from sglang.srt.function_call.utils import _is_complete_json

logger = logging.getLogger(__name__)


class SeedOssDetector(BaseFormatDetector):
    """Detector for the Seed Open Source format using XML-like tags."""

    def __init__(self):
        super().__init__()
        self._buffer = ""
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = (
            []
        )  # map what has been streamed for each tool so far to a list

        # Format-specific tokens
        self.tool_call_start_token: str = "<seed:tool_call>"
        self.tool_call_end_token: str = "</seed:tool_call>"
        self.bot_token: str = self.tool_call_start_token  # For base class compatibility
        self.eot_token: str = self.tool_call_end_token  # For base class compatibility

        # Sentinel tokens for streaming mode
        self.function_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

        # Regular expressions for parsing
        self.tool_call_regex = re.compile(
            rf"{re.escape(self.tool_call_start_token)}(.*?){re.escape(self.tool_call_end_token)}|"
            rf"{re.escape(self.tool_call_start_token)}(.*?)$",
            re.DOTALL,
        )

        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )

        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )

        self._last_arguments = ""
        self.is_function = False

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Seed OSS format tool call."""
        return self.tool_call_start_token in text

    def _parse_xml_function_call(
        self, function_call_str: str, tools: List[Tool]
    ) -> ToolCallItem:
        """
        Parse a function call from the XML format.

        Args:
            function_call_str: The function call string in XML format
            tools: List of available tools

        Returns:
            A ToolCallItem representing the parsed function call
        """

        def get_arguments_config(func_name: str) -> dict:
            if not tools:
                return {}
            for i, tool in enumerate(tools):
                if tool.function and tool.function.name == func_name:
                    if not tool.function.parameters:
                        return {}
                    params = tool.function.parameters
                    if isinstance(params, dict) and "properties" in params:
                        return params["properties"]
                    elif isinstance(params, dict):
                        return params
                    else:
                        return {}
            logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
            return {}

        def convert_param_value(
            param_value: str, param_name: str, param_config: dict, func_name: str
        ) -> any:
            # Handle null value for any type
            if param_value.lower() == "null":
                return None

            if param_name not in param_config:
                if param_config != {}:
                    logger.warning(
                        f"Parsed parameter '{param_name}' is not defined in the tool "
                        f"parameters for tool '{func_name}', directly returning the string value."
                    )
                return param_value

            if (
                isinstance(param_config[param_name], dict)
                and "type" in param_config[param_name]
            ):
                param_type = str(param_config[param_name]["type"]).strip().lower()
            else:
                param_type = "string"

            if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
                return param_value
            elif (
                param_type.startswith("int")
                or param_type.startswith("uint")
                or param_type.startswith("long")
                or param_type.startswith("short")
                or param_type.startswith("unsigned")
            ):
                try:
                    param_value = int(param_value)
                except:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                        f"'{func_name}', degenerating to string."
                    )
                return param_value
            elif param_type.startswith("num") or param_type.startswith("float"):
                try:
                    float_param_value = float(param_value)
                    param_value = (
                        float_param_value
                        if float_param_value - int(float_param_value) != 0
                        else int(float_param_value)
                    )
                except:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                        f"'{func_name}', degenerating to string."
                    )
                return param_value
            elif param_type in ["boolean", "bool", "binary"]:
                param_value = param_value.lower()
                if param_value not in ["true", "false"]:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                    )
                return param_value == "true"
            else:
                if param_type == "object" or param_type.startswith("dict"):
                    try:
                        param_value = json.loads(param_value)
                        return param_value
                    except:
                        logger.warning(
                            f"Parsed value '{param_value}' of parameter '{param_name}' is not a valid JSON object in tool "
                            f"'{func_name}', will try other methods to parse it."
                        )
                try:
                    param_value = eval(param_value)
                except:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `eval()` in tool '{func_name}', degenerating to string."
                    )
                return param_value

        # Extract function name
        end_index = function_call_str.index(">")
        function_name = function_call_str[:end_index]
        param_config = get_arguments_config(function_name)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}

        for match in self.tool_call_parameter_regex.findall(parameters):
            match_text = match[0] if match[0] else match[1]
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = convert_param_value(
                param_value, param_name, param_config, function_name
            )

        return ToolCallItem(
            tool_index=-1,  # To be updated by the caller based on tools list
            name=function_name,
            parameters=json.dumps(param_dict, ensure_ascii=False),
        )

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        Args:
            text: The complete text to parse.
            tools: List of available tools.

        Returns:
            ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text, calls=[])

        try:
            # Extracting the content before tool call
            content_index = text.find(self.tool_call_start_token)
            content = text[:content_index] if content_index >= 0 else ""

            # Find all tool calls
            matched_ranges = self.tool_call_regex.findall(text)
            raw_tool_calls = [
                match[0] if match[0] else match[1] for match in matched_ranges
            ]

            # Back-off strategy if no tool_call tags found
            if len(raw_tool_calls) == 0:
                return StreamingParseResult(normal_text=text, calls=[])

            # Extract function calls from tool calls
            raw_function_calls = []
            for tool_call in raw_tool_calls:
                raw_function_calls.extend(
                    self.tool_call_function_regex.findall(tool_call)
                )

            function_calls = [
                match[0] if match[0] else match[1] for match in raw_function_calls
            ]

            if len(function_calls) == 0:
                return StreamingParseResult(normal_text=text, calls=[])

            # Parse each function call
            tool_calls = []
            for idx, function_call_str in enumerate(function_calls):
                tool_call = self._parse_xml_function_call(function_call_str, tools)
                if tool_call:
                    # Update tool index based on position in response
                    tool_call.tool_index = idx
                    tool_calls.append(tool_call)

                    # Store in prev_tool_call_arr for later use
                    if idx >= len(self.prev_tool_call_arr):
                        self.prev_tool_call_arr.append({})
                    self.prev_tool_call_arr[idx] = {
                        "name": tool_call.name,
                        "arguments": tool_call.parameters,
                    }

            return StreamingParseResult(normal_text=content, calls=tool_calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # Return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for the SeedOss format.

        Args:
            new_text: The new text increment to parse.
            tools: List of available tools.

        Returns:
            StreamingParseResult with parsed calls or normal text.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call
        has_tool_call = (
            self.tool_call_start_token in current_text
            or self.function_prefix in current_text
        )

        if not has_tool_call:
            self._buffer = ""
            # Clean up any end tokens in the normal text
            for e_token in [
                self.tool_call_end_token,
                self.function_end_token,
                self.parameter_end_token,
            ]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        # Initialize tool indices if not already done
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = {
                tool.function.name: i
                for i, tool in enumerate(tools)
                if tool.function and tool.function.name
            }

        calls = []
        try:
            # Check for function start
            if self.function_prefix in current_text and not self.current_tool_name_sent:
                # Extract function name
                func_start = current_text.find(self.function_prefix) + len(
                    self.function_prefix
                )
                func_end = current_text.find(">", func_start)

                if func_end != -1:
                    function_name = current_text[func_start:func_end]

                    # Initialize state if this is the first tool call
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.prev_tool_call_arr = []
                        self.streamed_args_for_tool = [""]

                    # Ensure we have enough entries in our tracking arrays
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=function_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    self.is_function = True

                    # Store the tool call info
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": function_name,
                        "arguments": {},
                    }

            # Check for parameter
            elif self.is_function and self.parameter_prefix in current_text:
                # Handle parameter
                param_matches = self.tool_call_parameter_regex.findall(current_text)
                if param_matches:
                    # Process each parameter match
                    for match in param_matches:
                        match_text = match[0] if match[0] else match[1]
                        if not match_text:
                            continue

                        idx = match_text.find(">")
                        if idx == -1:
                            continue

                        param_name = match_text[:idx]
                        param_value = str(match_text[idx + 1 :])

                        # Clean up parameter value
                        if param_value.startswith("\n"):
                            param_value = param_value[1:]
                        if param_value.endswith("\n"):
                            param_value = param_value[:-1]

                        # Check if parameter is complete
                        is_complete = self.parameter_end_token in current_text

                        # Extract the part we haven't sent yet
                        if param_value:
                            arguments_diff = json.dumps({param_name: param_value})
                            if self._last_arguments:
                                # Only send what's new
                                last_args_obj = json.loads(self._last_arguments)
                                if (
                                    param_name in last_args_obj
                                    and last_args_obj[param_name] == param_value
                                ):
                                    # No change, skip
                                    continue

                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=arguments_diff,
                                )
                            )
                            self._last_arguments = arguments_diff

                            # Update tracked parameters
                            if is_complete:
                                self.prev_tool_call_arr[self.current_tool_id][
                                    "arguments"
                                ] = json.loads(arguments_diff)

                                # Parameter is complete, clean up
                                self._last_arguments = ""

                                # Check if function is also complete
                                if self.function_end_token in current_text:
                                    self.is_function = False
                                    self.current_tool_name_sent = False
                                    self.current_tool_id += 1

                                    # Remove the processed function from buffer
                                    func_end_pos = current_text.find(
                                        self.function_end_token
                                    ) + len(self.function_end_token)
                                    self._buffer = current_text[func_end_pos:]

            # Check if a function has ended without us catching a specific parameter
            elif self.is_function and self.function_end_token in current_text:
                self.is_function = False
                self.current_tool_name_sent = False
                self.current_tool_id += 1

                # Remove the processed function from buffer
                func_end_pos = current_text.find(self.function_end_token) + len(
                    self.function_end_token
                )
                self._buffer = current_text[func_end_pos:]

            # Check if the entire tool call section has ended
            if self.tool_call_end_token in current_text:
                # Clean up buffer
                tool_end_pos = current_text.find(self.tool_call_end_token) + len(
                    self.tool_call_end_token
                )
                self._buffer = current_text[tool_end_pos:]

                # Reset state for next potential tool call
                if not self.is_function:
                    self.current_tool_id = -1

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text="")

    def structure_info(self) -> _GetInfoFunc:
        """Return metadata about the structure of SeedOss tool calls."""

        def _get_info() -> StructureInfo:
            return StructureInfo(
                start_phrase=self.tool_call_start_token,
                end_phrase=self.tool_call_end_token,
                likely_content_before_start=True,
                graceful_recovery_possible=True,
                can_generate_content_after_end=True,
                additional_escape_phrases=[
                    self.function_prefix,
                    self.function_end_token,
                    self.parameter_prefix,
                    self.parameter_end_token,
                ],
            )

        return _get_info

    def build_ebnf(self, tools: List[Tool]) -> str:
        """Build an EBNF grammar for the SeedOss format."""
        composer = EBNFComposer()

        # Define the overall structure
        composer.add_production(
            "tool_call",
            f'"{self.tool_call_start_token}" function "{self.tool_call_end_token}"',
        )

        # Define what a function looks like
        composer.add_production(
            "function",
            f'"{self.function_prefix}" function_name ">" parameters "{self.function_end_token}"',
        )

        # Function name is any of the available tool names
        function_names = " | ".join(
            [
                f'"{tool.function.name}"'
                for tool in tools
                if tool.function and tool.function.name
            ]
        )
        composer.add_production(
            "function_name", function_names if function_names else '"unknown_function"'
        )

        # Parameters are a sequence of parameter definitions
        composer.add_production("parameters", "parameter*")

        # Each parameter has a name and value
        composer.add_production(
            "parameter",
            f'"{self.parameter_prefix}" parameter_name ">" parameter_value "{self.parameter_end_token}"',
        )

        # Parameter name can be any key from the tools
        param_names = set()
        for tool in tools:
            if not (
                tool.function
                and tool.function.parameters
                and isinstance(tool.function.parameters, dict)
            ):
                continue

            properties = tool.function.parameters.get("properties", {})
            for param_name in properties.keys():
                param_names.add(param_name)

        param_name_production = (
            " | ".join([f'"{name}"' for name in param_names])
            if param_names
            else '"param"'
        )
        composer.add_production("parameter_name", param_name_production)

        # Parameter value can be any string (simplified)
        composer.add_production(
            "parameter_value",
            'string | number | "true" | "false" | "null" | object | array',
        )
        composer.add_production("string", r'"\""  [^"]* "\""')
        composer.add_production("number", r"[0-9]+ (\.[0-9]+)?")
        composer.add_production(
            "object", r'"{" (string ":" value ("," string ":" value)*)? "}"'
        )
        composer.add_production("array", r'"\[" (value ("," value)*)? "\]"')
        composer.add_production(
            "value", 'string | number | object | array | "true" | "false" | "null"'
        )

        return composer.compose()
