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
from sglang.srt.function_call.utils import _is_complete_json

logger = logging.getLogger(__name__)


class DeepSeekV32Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3.2 model function call format.

    The DeepSeek V3.2 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Examples:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜function_calls>` and `</｜DSML｜function_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V3.2 format specification
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜function_calls>"
        self.eot_token = "</｜DSML｜function_calls>"

        self.invoke_begin_regex = r'<｜DSML｜invoke\s+name="([^"]+)"\s*>'
        self.invoke_end_token = "</｜DSML｜invoke>"
        self.invoke_end_token_prefixes = [
            "<",
            "</",
            "</｜DSML｜",
            "</｜DSML｜inv",
            "</｜DSML｜invoke",
        ]

        self.xml_parameter_regex = r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>'
        self._last_arguments = ""
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call."""
        return self.bot_token in text or "<｜DSML｜invoke" in text

    def _parse_parameters_from_xml(self, invoke_content: str) -> dict:
        """
        Parse parameters from either XML-like format or JSON format to dict.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }
        """
        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = invoke_content.strip()

        if invoke_content_stripped.startswith("{") and invoke_content_stripped.endswith(
            "}"
        ):
            try:
                parameters = json.loads(invoke_content_stripped)
                if isinstance(parameters, dict):
                    return parameters
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, fall through to XML parsing
                pass

        # Fall back to XML parameter tag parsing (original format)
        parameters = {}
        param_matches = re.findall(self.xml_parameter_regex, invoke_content, re.DOTALL)
        for param_name, param_type, param_value in param_matches:
            # Convert value based on type
            if param_type == "true":  # string type
                parameters[param_name] = param_value.strip()
            else:
                # Try to parse as JSON for other types
                try:
                    parameters[param_name] = json.loads(param_value.strip())
                except (json.JSONDecodeError, ValueError):
                    parameters[param_name] = param_value.strip()
        return parameters

    def _parse_parameters_partially(self, invoke_content: str) -> str:
        invoke_content_stripped = invoke_content.strip()
        if not invoke_content_stripped:
            return "", "none"

        # 1. check json format
        if invoke_content_stripped.startswith("{"):
            return invoke_content_stripped, "json"

        # 2. check xml format
        xml_param_matches = re.findall(
            self.xml_parameter_regex, invoke_content, re.DOTALL
        )

        json_parts = []
        for param_name, param_type, param_value in xml_param_matches:
            # Convert value based on type
            if param_type == "true":  # string type
                json_parts.append(f'"{param_name}": "{param_value.strip()}"')
            else:
                # Try to parse as JSON for other types
                try:
                    json_parts.append(
                        f'"{param_name}": {json.dumps(json.loads(param_value.strip()), ensure_ascii=False)}'
                    )
                except (json.JSONDecodeError, ValueError):
                    json_parts.append(
                        f'"{param_name}": {json.dumps(param_value.strip(), ensure_ascii=False)}'
                    )
        return "{" + ", ".join(json_parts), "xml"

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

        calls = []
        try:
            # Extract content between function_calls tags
            function_calls_match = re.search(
                r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>",
                text,
                re.DOTALL,
            )
            if not function_calls_match:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            function_calls_content = function_calls_match.group(1)

            # Find all invoke blocks
            invoke_pattern = (
                r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>'
            )
            invoke_matches = re.findall(
                invoke_pattern, function_calls_content, re.DOTALL
            )

            for func_name, invoke_content in invoke_matches:
                # Parse parameters from XML format
                func_args = self._parse_parameters_from_xml(invoke_content)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result, tools))

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV32 format.
        Supports multiple consecutive invoke blocks.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if buffer contains any DSML markers or ends with potential tag prefix
        # This handles partial/streaming DSML content
        dsml_markers = ["｜DSML｜", "<｜", "</｜"]
        potentially_dsml = any(marker in current_text for marker in dsml_markers)

        # Also check if text ends with start of a tag (to handle "<" arriving separately)
        dsml_prefixes = ["<", "<｜", "</", "</｜"]
        ends_with_prefix = any(
            current_text.rstrip().endswith(prefix) for prefix in dsml_prefixes
        )

        if (
            not self.has_tool_call(current_text)
            and not potentially_dsml
            and not ends_with_prefix
        ):
            self._buffer = ""
            for e_token in [self.eot_token, self.invoke_end_token]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            while True:
                # Try to match an invoke block (may be partial)
                invoke_match = re.search(
                    pattern=r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)(</｜DSML｜invoke>|$)',
                    string=current_text,
                    flags=re.DOTALL,
                )
                if not invoke_match:
                    break

                func_name = invoke_match.group(1).strip()
                invoke_content = invoke_match.group(2)

                if any(
                    invoke_content.endswith(end)
                    for end in self.invoke_end_token_prefixes
                ):
                    break

                # group(3) is either "</｜DSML｜invoke>" (complete) or "" (incomplete, matched with $)
                is_tool_end = bool(invoke_match.group(3))

                print(f"\033[42m {current_text=} \033[0m")
                print(f"\033[42m {func_name=} {invoke_content=} {is_tool_end=} \033[0m")
                print("-" * 80)
                func_args_raw, format = self._parse_parameters_partially(invoke_content)
                # print(f"\033[42m {func_args_raw=} \033[0m")
                if is_tool_end and format == "xml":
                    func_args_raw += "}"

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    # send tool name first
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                else:
                    argument_diff = (
                        func_args_raw[len(self._last_arguments) :]
                        if func_args_raw.startswith(self._last_arguments)
                        else func_args_raw
                    )
                    if argument_diff:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=argument_diff,
                            )
                        )
                        self._last_arguments += argument_diff
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff

                    if _is_complete_json(func_args_raw) and is_tool_end:
                        # Update the stored arguments
                        try:
                            parsed_args = json.loads(func_args_raw)
                            self.prev_tool_call_arr[self.current_tool_id][
                                "arguments"
                            ] = parsed_args
                        except json.JSONDecodeError:
                            pass

                        # Remove the completed tool call from buffer, keep any remaining content
                        self._buffer = current_text[invoke_match.end(3) :]
                        current_text = self._buffer
                        print(f"\033[41m send over, clear buffer \033[0m")

                        # reset state
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False

                if not is_tool_end:
                    # if current tool is not end, it means the following content is definitely not a tool call, so break the loop
                    break

            # No more invoke blocks found
            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger=f"<｜DSML｜",
        )
