# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import ast
import html
import json
import logging
import re
from typing import Any, Dict, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult, _GetInfoFunc

logger = logging.getLogger(__name__)


def _get_param_type(func_name: str, param_name: str, tools: List[Tool]) -> str:
    """Get parameter type from tool schema."""
    for tool in tools:
        if tool.function.name == func_name:
            props = tool.function.parameters.get("properties", {})
            if param_name in props:
                return props[param_name].get("type", "string")
    return "string"


def _convert_param_value(
    param_value: str, param_name: str, func_name: str, tools: List[Tool]
) -> Any:
    """
    Convert parameter value based on its type in the schema.
    Adapted from vllm-project/vllm (vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py)
    """
    param_value = html.unescape(param_value)

    # Handle null value for any type
    if param_value.lower() == "null":
        return None

    param_type = _get_param_type(func_name, param_name, tools)

    if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
        return param_value
    elif (
        param_type.startswith("int")
        or param_type.startswith("integer")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        try:
            return int(param_value)
        except (ValueError, TypeError):
            logger.warning(
                "Parsed value '%s' of parameter '%s' is not an "
                "integer in tool '%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
            return param_value
    elif param_type.startswith("num") or param_type.startswith("float"):
        try:
            float_param_value = float(param_value)
            return (
                float_param_value
                if float_param_value - int(float_param_value) != 0
                else int(float_param_value)
            )
        except (ValueError, TypeError):
            logger.warning(
                "Parsed value '%s' of parameter '%s' is not a float "
                "in tool '%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
            return param_value
    elif param_type in ["boolean", "bool", "binary"]:
        param_value = param_value.lower()
        if param_value not in ["true", "false"]:
            logger.warning(
                "Parsed value '%s' of parameter '%s' is not a boolean "
                "(`true` or `false`) in tool '%s', degenerating to "
                "false.",
                param_value,
                param_name,
                func_name,
            )
        return param_value == "true"
    else:
        if (
            param_type in ["object", "array", "arr"]
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                param_value = json.loads(param_value)
                return param_value
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning(
                    "Parsed value '%s' of parameter '%s' cannot be "
                    "parsed with json.loads in tool '%s', will try "
                    "other methods to parse it.",
                    param_value,
                    param_name,
                    func_name,
                )
        try:
            param_value = ast.literal_eval(param_value)  # safer
        except (ValueError, SyntaxError, TypeError):
            logger.warning(
                "Parsed value '%s' of parameter '%s' cannot be "
                "converted via Python `ast.literal_eval()` in tool "
                "'%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
        return param_value


class MiMoDetector(BaseFormatDetector):
    """
    Detector for MiMo function call format.

    Format:
        <tool_call>
        <function=execute_bash>
        <parameter=command>pwd && ls</parameter>
        </function>
        </tool_call>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.func_regex = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
        self.param_regex = re.compile(
            r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse complete text for tool calls."""
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text = text[:idx]
        tool_indices = self._get_tool_indices(tools)

        calls = []
        last_end = idx

        for match in self.tool_call_regex.finditer(text):
            tool_call_body = match.group(1)

            parsed = self._parse_tool_call(tool_call_body, tools)

            if parsed:
                func_name = parsed.get("name")
                if func_name not in tool_indices:
                    # Unknown function
                    logger.warning(f"Unknown function: {func_name}")
                    if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                        # Return tool call block as normal text
                        normal_text += text[last_end : match.end()]
                        last_end = match.end()
                        continue
                calls.extend(self.parse_base_json(parsed, tools))

            last_end = match.end()

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming parsing: buffer until complete tool call block.
        """
        self._buffer += new_text
        current_text = self._buffer

        start = current_text.find(self.bot_token)
        if start == -1:
            if self.current_tool_id > 0:
                # Already processing tool calls, keep buffering
                # (more tool calls might come, don't discard text yet)
                return StreamingParseResult(normal_text="")
            else:
                # No tool calls seen yet, return as normal text
                self._buffer = ""
                return StreamingParseResult(normal_text=current_text)

        # Find end token AFTER the start token
        end = current_text.find(self.eot_token, start)
        if end == -1:
            # Incomplete tool call, return text before start and keep buffering
            normal_text = current_text[:start]
            self._buffer = current_text[start:]
            return StreamingParseResult(normal_text=normal_text)

        # Parse the complete tool call block
        result = self.detect_and_parse(current_text[: end + len(self.eot_token)], tools)

        if result.calls:
            # Valid tool call - initialize tracking if first one
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = [""]

            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            call = result.calls[0]
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": call.name,
                "arguments": json.loads(call.parameters) if call.parameters else {},
            }
            self.streamed_args_for_tool[self.current_tool_id] = call.parameters
            call.tool_index = self.current_tool_id
            self.current_tool_id += 1

        self._buffer = current_text[end + len(self.eot_token) :]
        return result

    def _parse_tool_call(
        self, tool_call_body: str, tools: List[Tool]
    ) -> Dict[str, Any]:
        """
        Parse content inside <tool_call>...</tool_call>.

        Structure:
            tool_call_body contains: <function=name>...params...</function>
        """
        # Match complete <function=name>body</function> block
        func_match = self.func_regex.search(tool_call_body)
        if not func_match:
            return None

        func_name = func_match.group(1).strip()
        func_body = func_match.group(2)

        params = {}
        for param_match in self.param_regex.finditer(func_body):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2)
            params[param_name] = _convert_param_value(
                param_value, param_name, func_name, tools
            )

        return {"name": func_name, "parameters": params}

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
