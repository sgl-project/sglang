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

logger = logging.getLogger(__name__)


class DeepSeekV32Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3.2 model function call format.

    The DeepSeek V3.2 format uses XML-like DSML tags to delimit function calls.
    Supports multiple parameter formats:

    Format 1 - XML Parameter Tags (with DSML):
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Format 2 - Direct JSON (with DSML):
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Format 3 - Without DSML tags (simplified):
    ```
    <function_calls>
        <invoke name="function_name">
        <parameter name="param_name" string="true">value</parameter>
        ...
        </invoke>
    </function_calls>
    ```

    Format 4 - Without DSML tags with JSON:
    ```
    <function_calls>
        <invoke name="function_name">
        {
            "param_name": "value"
        }
        </invoke>
    </function_calls>
    ```

    Examples:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

    <function_calls>
        <invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
        </invoke>
    </function_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜function_calls>` / `<function_calls>` and corresponding closing tags
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` / `<invoke name="...">` and corresponding closing tags
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V3.2 format specification
    """

    def __init__(self):
        super().__init__()
        # DSML format tokens
        self.bot_token = "<｜DSML｜function_calls>"
        self.eot_token = "</｜DSML｜function_calls>"
        self.invoke_begin_regex = r'<｜DSML｜invoke\s+name="([^"]+)"\s*>'
        self.invoke_end_token = "</｜DSML｜invoke>"
        self.parameter_regex = r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>'
        # Simplified format tokens (without DSML)
        self.bot_token_simple = "<function_calls>"
        self.eot_token_simple = "</function_calls>"
        self.invoke_begin_regex_simple = r'<invoke\s+name="([^"]+)"\s*>'
        self.invoke_end_token_simple = "</invoke>"
        self.parameter_regex_simple = (
            r'<parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</parameter>'
        )
        self._last_arguments = ""
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call."""
        return self.bot_token in text or self.bot_token_simple in text

    def _parse_parameters_from_xml(self, invoke_content: str) -> dict:
        """
        Parse parameters from either XML-like format or JSON format to dict.

        Supports two formats:
        1. XML parameter tags: Supports mixed DSML tags:
           - <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter> (both DSML)
           - <parameter name="..." string="...">value</parameter> (both simple)
           - <｜DSML｜parameter name="..." string="...">value</parameter> (start DSML, end simple)
           - <parameter name="..." string="...">value</｜DSML｜parameter> (start simple, end DSML)
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
        # Use flexible regex that supports mixed DSML tags
        # Matches: <(｜DSML｜)?parameter ...>(.*?)</(｜DSML｜)?parameter>
        param_regex = r'<(｜DSML｜)?parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</(｜DSML｜)?parameter>'
        param_matches = re.findall(param_regex, invoke_content, re.DOTALL)
        parameters = {}
        for start_dsml, param_name, param_type, param_value, end_dsml in param_matches:
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

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        # Find the start of function_calls tag (supports both DSML and simple format)
        # Use flexible pattern that matches both formats
        function_calls_start_pattern = r"<(｜DSML｜)?function_calls>"
        start_match = re.search(function_calls_start_pattern, text)

        if not start_match:
            # No function_calls tag found, return normal text
            return StreamingParseResult(normal_text=text.strip(), calls=[])

        idx = start_match.start()
        normal_text = text[:idx].strip()

        calls = []
        try:
            # Use flexible patterns that support mixed DSML tags
            # Match function_calls with optional DSML: <(｜DSML｜)?function_calls>(.*?)</(｜DSML｜)?function_calls>
            function_calls_pattern = (
                r"<(｜DSML｜)?function_calls>(.*?)</(｜DSML｜)?function_calls>"
            )
            # Match invoke with optional DSML: <(｜DSML｜)?invoke name="...">(.*?)</(｜DSML｜)?invoke>
            invoke_pattern = (
                r'<(｜DSML｜)?invoke\s+name="([^"]+)"\s*>(.*?)</(｜DSML｜)?invoke>'
            )

            # Extract content between function_calls tags
            function_calls_match = re.search(
                function_calls_pattern,
                text,
                re.DOTALL,
            )
            if not function_calls_match:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            function_calls_content = function_calls_match.group(
                2
            )  # Content is in group 2

            # Find all invoke blocks
            invoke_matches = re.findall(
                invoke_pattern, function_calls_content, re.DOTALL
            )

            for start_dsml, func_name, invoke_content, end_dsml in invoke_matches:
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
        Supports multiple consecutive invoke blocks, both with and without DSML tags.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (supports mixed DSML and simple format)
        has_tool_call = (
            self.bot_token in current_text
            or self.bot_token_simple in current_text
            or "<｜DSML｜invoke" in current_text
            or "<invoke" in current_text
        )

        # Check if buffer contains any relevant markers (supports mixed format)
        relevant_markers = [
            "｜DSML｜",
            "<｜",
            "</｜",
            "<function_calls",
            "<invoke",
            "</invoke",
            "</function_calls",
        ]
        potentially_has_tags = any(
            marker in current_text for marker in relevant_markers
        )

        # Check if text might be starting a tag or contains partial tags
        stripped_text = current_text.rstrip()
        ends_with_prefix = False

        # More robust check: look for any opening '<' that might start a relevant tag
        # Check if we have an incomplete tag at the end
        if stripped_text:
            # Check for patterns that suggest we're in the middle of a tag
            last_lt = stripped_text.rfind("<")
            if last_lt != -1:
                # We have a '<' somewhere
                after_lt = stripped_text[last_lt:]
                # Check if this could be the start of one of our tags
                tag_starts = [
                    "<function_calls",
                    "<invoke",
                    "</invoke",
                    "</function_calls",
                    "<｜DSML｜",
                    "</｜DSML｜",
                    "<｜",
                    "</｜",
                ]
                for tag_start in tag_starts:
                    # Check if what we have could be the beginning of this tag
                    if tag_start.startswith(after_lt) and len(after_lt) < len(
                        tag_start
                    ):
                        ends_with_prefix = True
                        break
                    # Or if the current text ends with a complete tag start
                    if after_lt == tag_start:
                        ends_with_prefix = True
                        break

        if not has_tool_call and not potentially_has_tags and not ends_with_prefix:
            self._buffer = ""
            for e_token in [
                self.eot_token,
                self.invoke_end_token,
                self.eot_token_simple,
                self.invoke_end_token_simple,
            ]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        all_calls: list[ToolCallItem] = []
        try:
            # Loop to handle multiple consecutive invoke blocks
            while True:
                # Use flexible pattern that supports mixed DSML tags
                # Match: <(｜DSML｜)?invoke name="...">(.*?)(</(｜DSML｜)?invoke>|$)
                invoke_pattern = r'<(｜DSML｜)?invoke\s+name="([^"]+)"\s*>(.*?)(</(｜DSML｜)?invoke>|$)'

                # Try to match an invoke block (may be partial)
                invoke_match = re.search(
                    pattern=invoke_pattern,
                    string=current_text,
                    flags=re.DOTALL,
                )

                if not invoke_match:
                    break

                start_dsml = invoke_match.group(1)
                func_name = invoke_match.group(2).strip()
                invoke_content = invoke_match.group(3)
                # group(4) is either closing tag (complete) or "" (incomplete, matched with $)
                # group(5) is DSML in end tag (if present)
                is_tool_end = bool(invoke_match.group(4))

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Parse current parameters from XML/JSON
                current_params = self._parse_parameters_from_xml(invoke_content)
                current_args_json = json.dumps(current_params, ensure_ascii=False)

                # Check if tool call is complete (has closing tag)
                if is_tool_end:
                    # Only emit the tool call when it's complete
                    calls_for_this_invoke: list[ToolCallItem] = []

                    # Check if invoke_content is empty or whitespace only
                    if not invoke_content.strip():
                        # Remove the incomplete tool call from buffer
                        self._buffer = current_text[invoke_match.end() :]
                        current_text = self._buffer
                        continue

                    # Send tool name
                    calls_for_this_invoke.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )

                    # Send parameters as complete JSON
                    calls_for_this_invoke.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=current_args_json,
                        )
                    )

                    # Ensure arrays are large enough for current tool
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    # Update the stored arguments
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": current_params,
                    }
                    self.streamed_args_for_tool[self.current_tool_id] = (
                        current_args_json
                    )

                    # Remove the completed tool call from buffer
                    self._buffer = current_text[invoke_match.end() :]
                    current_text = self._buffer

                    # Add calls for this invoke to all_calls
                    all_calls.extend(calls_for_this_invoke)

                    # Move to next tool call
                    self.current_tool_id += 1
                    self._last_arguments = ""
                    self.current_tool_name_sent = False

                    # Continue loop to check for more invoke blocks
                    continue
                else:
                    # Tool call not complete yet, don't return anything
                    break

            # No more invoke blocks found
            return StreamingParseResult(normal_text="", calls=all_calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger=f'<｜DSML｜invoke name="{name}">',
        )
