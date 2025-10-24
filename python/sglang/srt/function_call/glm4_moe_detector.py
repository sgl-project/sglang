import ast
import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer

logger = logging.getLogger(__name__)


def get_argument_type(func_name: str, arg_key: str, defined_tools: list):
    name2tool = {tool.function.name: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    if arg_key not in tool.function.parameters["properties"]:
        return None
    return tool.function.parameters["properties"][arg_key].get("type", None)


def parse_arguments(json_value):
    try:
        try:
            parsed_value = json.loads(json_value)
        except:
            parsed_value = ast.literal_eval(json_value)
        return parsed_value, True
    except:
        return json_value, False


class Glm4MoeDetector(BaseFormatDetector):
    """
    Detector for GLM-4.5 and GLM-4.6 models.
    Assumes function call format:
      <tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>北京</arg_value>\n<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n</tool_call>\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>上海</arg_value>\n<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n</tool_call>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.func_call_regex = r"<tool_call>.*?</tool_call>"
        self.func_detail_regex = r"<tool_call>([^\n]*)\n(.*)</tool_call>"
        self.func_arg_regex = r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>"

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
                func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
                func_name = func_detail.group(1)
                func_args = func_detail.group(2)
                pairs = re.findall(
                    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                    func_args,
                    re.DOTALL,
                )
                arguments = {}
                for arg_key, arg_value in pairs:
                    arg_key = arg_key.strip()
                    arg_value = arg_value.strip()
                    arg_type = get_argument_type(func_name, arg_key, tools)
                    if arg_type != "string":
                        arg_value, is_good_json = parse_arguments(arg_value)
                    arguments[arg_key] = arg_value
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": arguments}
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
        Streaming incremental parsing tool calls for GLM-4.5 and GLM-4.6 format.
        Now supports streaming tool names and arguments incrementally.
        """
        self._buffer += new_text
        current_text = self._buffer

        start = current_text.find(self.bot_token)
        if start == -1:
            self._buffer = ""
            if self.current_tool_id > 0:
                current_text = ""
            return StreamingParseResult(normal_text=current_text)

        # Extract normal text before tool calls
        normal_text = current_text[:start]

        # Try to parse partial tool call for streaming
        partial_result = self._parse_partial_tool_call(current_text[start:], tools)
        if partial_result:
            func_name, partial_args_str, is_complete = partial_result

            # Initialize state if this is the first tool call
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = [""]
                self.current_tool_name_sent = False

            # Ensure we have enough entries in our tracking arrays
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            tool_id = self.current_tool_id
            calls = []

            # Case 1: Send tool name if not sent yet
            if not self.current_tool_name_sent:
                self.current_tool_name_sent = True
                calls.append(
                    ToolCallItem(tool_index=tool_id, name=func_name, parameters="")
                )
            # Case 2: Stream arguments incrementally
            else:
                # Calculate diff between current and previously streamed arguments
                prev_args_str = self.streamed_args_for_tool[tool_id]

                # Always check if there's new content to stream
                if partial_args_str != prev_args_str:
                    # Try to parse both as JSON to compare properly
                    try:
                        prev_args = json.loads(prev_args_str) if prev_args_str else {}
                        current_args = json.loads(partial_args_str)

                        # Find new keys or changed values
                        new_content = {}
                        for key, value in current_args.items():
                            if key not in prev_args or prev_args[key] != value:
                                new_content[key] = value

                        if new_content:
                            argument_diff = json.dumps(new_content)
                        else:
                            argument_diff = ""
                    except:
                        # Fallback to string comparison
                        if partial_args_str.startswith(prev_args_str):
                            argument_diff = partial_args_str[len(prev_args_str) :]
                        else:
                            # If strings don't match, try to find common prefix
                            common_prefix = self._find_common_prefix(
                                prev_args_str, partial_args_str
                            )
                            if len(prev_args_str) < len(common_prefix):
                                argument_diff = partial_args_str[
                                    len(prev_args_str) : len(common_prefix)
                                ]
                            else:
                                argument_diff = ""
                else:
                    argument_diff = ""

                if argument_diff:
                    # Update streamed arguments
                    self.streamed_args_for_tool[tool_id] += argument_diff

                    calls.append(
                        ToolCallItem(
                            tool_index=tool_id, name=None, parameters=argument_diff
                        )
                    )

            # Update prev_tool_call_arr with current state
            try:
                parsed_args = json.loads(partial_args_str)
            except:
                parsed_args = {}

            self.prev_tool_call_arr[tool_id] = {
                "name": func_name,
                "arguments": parsed_args,
            }

            # If complete, advance to next tool
            if is_complete:
                # Remove processed portion from buffer
                end = current_text.find(self.eot_token)
                if end != -1:
                    self._buffer = current_text[end + len(self.eot_token) :]
                self.current_tool_name_sent = False
                self.current_tool_id += 1
            else:
                # Keep the buffer for partial tool call
                self._buffer = current_text[start:]

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        # No tool call found yet, return normal text before start token
        self._buffer = current_text[start:]
        return StreamingParseResult(normal_text=normal_text)

    def _parse_partial_tool_call(
        self, text: str, tools: List[Tool]
    ) -> tuple[str, str, bool] | None:
        """
        Parse partial tool call from buffer (for streaming)
        Returns (tool_name, partial_arguments_json, is_complete)
        """
        if not text.startswith(self.bot_token):
            return None

        after_start = text[len(self.bot_token) :]

        # Extract function name (until first newline)
        name_end = after_start.find("\n")
        if name_end == -1:
            name_end = len(after_start)
        func_name = after_start[:name_end].strip()

        if not func_name:
            return None

        # Check if we have complete tool call
        if self.eot_token in text:
            # Complete tool call
            end_pos = text.find(self.eot_token)
            args_text = after_start[name_end + 1 : end_pos - len(self.bot_token)]

            # Parse arguments using existing logic
            pairs = re.findall(
                r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                args_text,
                re.DOTALL,
            )
            arguments = {}
            for arg_key, arg_value in pairs:
                arg_key = arg_key.strip()
                arg_value = arg_value.strip()
                arg_type = get_argument_type(func_name, arg_key, tools)
                if arg_type != "string":
                    arg_value, is_good_json = parse_arguments(arg_value)
                arguments[arg_key] = arg_value

            arguments_str = json.dumps(arguments)
            return (func_name, arguments_str, True)
        else:
            # Partial tool call - try to parse partial arguments
            args_text = after_start[name_end + 1 :]
            partial_args = {}

            # Try to parse any complete key-value pairs
            pairs = re.findall(
                r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                args_text,
                re.DOTALL,
            )
            for arg_key, arg_value in pairs:
                arg_key = arg_key.strip()
                arg_value = arg_value.strip()

                if arg_key and arg_value:
                    arg_type = get_argument_type(func_name, arg_key, tools)
                    if arg_type != "string":
                        arg_value, is_good_json = parse_arguments(arg_value)
                    partial_args[arg_key] = arg_value

            partial_args_str = json.dumps(partial_args)
            return (func_name, partial_args_str, False)

    def _find_common_prefix(self, s1: str, s2: str) -> str:
        """Find the common prefix of two strings"""
        result = []
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                result.append(c1)
            else:
                break
        return "".join(result)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            individual_call_start_token=self.bot_token,
            individual_call_end_token=self.eot_token,
            tool_call_separator="\\n",
            function_format="xml",
            call_rule_fmt='"{name}" "\\n" ( {arguments_rule} "\\n" )?',
            key_value_rule_fmt='"<arg_key>{key}</arg_key>" "\\n" "<arg_value>" {valrule} "</arg_value>"',
            key_value_separator='"\\n"',
        )
