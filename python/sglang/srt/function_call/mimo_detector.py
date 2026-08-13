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

import html
import json
import logging
import re
from typing import Any, Dict, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import safe_literal_eval

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
            param_value = safe_literal_eval(param_value)
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

        # Streaming state. MiMo emits XML-like tool calls, while the OpenAI API
        # expects a JSON argument string. Keep completed parameters plus the
        # currently open string parameter so we can expose a stable JSON prefix
        # before the closing </tool_call> arrives.
        self.function_start_token = "<function="
        self.function_end_token = "</function>"
        self.parameter_start_token = "<parameter="
        self.parameter_end_token = "</parameter>"
        self._in_tool_call = False
        self._reject_current_tool = False
        self._raw_tool_prefix = ""
        self._streaming_tool_name: Optional[str] = None
        self._completed_args: Dict[str, Any] = {}
        self._streamed_json_len = 0

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
        """Incrementally parse MiMo tool calls into OpenAI JSON deltas."""
        self._buffer += new_text
        normal_parts: List[str] = []
        calls = []

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        while True:
            if not self._in_tool_call:
                start = self._buffer.find(self.bot_token)
                if start == -1:
                    partial_len = self._ends_with_partial_token(
                        self._buffer, self.bot_token
                    )
                    if partial_len:
                        normal_parts.append(self._buffer[:-partial_len])
                        self._buffer = self._buffer[-partial_len:]
                    else:
                        normal_parts.append(self._buffer)
                        self._buffer = ""
                    break

                normal_parts.append(self._buffer[:start])
                self._buffer = self._buffer[start + len(self.bot_token) :]
                self._in_tool_call = True
                self._raw_tool_prefix = self.bot_token

            if self._reject_current_tool:
                if not self._forward_rejected_tool(normal_parts):
                    break
                continue

            if self._streaming_tool_name is None:
                function_start = self._buffer.find(self.function_start_token)
                if function_start == -1:
                    # A closed block with no function is malformed. Preserve the
                    # legacy behavior and forward it as normal text.
                    if self.eot_token in self._buffer:
                        self._reject_current_tool = True
                        if not self._forward_rejected_tool(normal_parts):
                            break
                        continue
                    break

                name_end = self._buffer.find(
                    ">", function_start + len(self.function_start_token)
                )
                if name_end == -1:
                    break

                function_name = self._buffer[
                    function_start + len(self.function_start_token) : name_end
                ].strip()
                consumed_header = self._buffer[: name_end + 1]
                self._buffer = self._buffer[name_end + 1 :]

                if (
                    function_name not in self._tool_indices
                    and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
                ):
                    logger.warning("Unknown function: %s", function_name)
                    self._raw_tool_prefix += consumed_header
                    self._reject_current_tool = True
                    if not self._forward_rejected_tool(normal_parts):
                        break
                    continue

                self.current_tool_id += 1
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                self._streaming_tool_name = function_name
                self.current_tool_name_sent = True
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": function_name,
                    "arguments": {},
                }
                self._raw_tool_prefix = ""
                calls.append(
                    self._make_tool_call_item(
                        name=function_name,
                        parameters="",
                    )
                )

            calls.extend(self._stream_arguments(tools))
            if self._in_tool_call:
                break

        return StreamingParseResult(
            normal_text="".join(normal_parts),
            calls=calls,
        )

    def _make_tool_call_item(
        self, parameters: str, name: Optional[str] = None
    ) -> ToolCallItem:
        return ToolCallItem(
            tool_index=self.current_tool_id,
            name=name,
            parameters=parameters,
        )

    def _forward_rejected_tool(self, normal_parts: List[str]) -> bool:
        """Forward an unknown or malformed complete tool block as normal text."""
        end = self._buffer.find(self.eot_token)
        if end == -1:
            return False

        normal_parts.append(
            self._raw_tool_prefix + self._buffer[: end + len(self.eot_token)]
        )
        self._buffer = self._buffer[end + len(self.eot_token) :]
        self._reset_streaming_state()
        return True

    @staticmethod
    def _is_streamable_string_param(
        function_name: str, param_name: str, tools: List[Tool]
    ) -> bool:
        return _get_param_type(function_name, param_name, tools).lower() in {
            "string",
            "str",
            "text",
            "varchar",
            "char",
            "enum",
        }

    @staticmethod
    def _without_incomplete_html_entity(value: str) -> str:
        """Hold a trailing entity that html.unescape could change next chunk."""
        match = re.search(
            r"&(?:#[xX][0-9a-fA-F]*|#[0-9]*|[A-Za-z][A-Za-z0-9]*)?$", value
        )
        return value[: match.start()] if match else value

    def _stream_arguments(self, tools: List[Tool]) -> List[ToolCallItem]:
        """Emit the stable JSON prefix for the current MiMo tool call."""
        calls = []
        is_complete = self.eot_token in self._buffer
        end = self._buffer.find(self.eot_token) if is_complete else len(self._buffer)
        args_text = self._buffer[:end]

        function_end = args_text.find(self.function_end_token)
        if function_end != -1:
            args_text = args_text[:function_end]

        last_closed_end = 0
        for match in self.param_regex.finditer(args_text):
            param_name = match.group(1).strip()
            if param_name not in self._completed_args:
                self._completed_args[param_name] = _convert_param_value(
                    match.group(2),
                    param_name,
                    self._streaming_tool_name or "",
                    tools,
                )
            last_closed_end = match.end()

        partial_name: Optional[str] = None
        partial_value: Optional[str] = None
        tail = args_text[last_closed_end:]
        parameter_start = tail.find(self.parameter_start_token)
        if parameter_start != -1:
            name_end = tail.find(">", parameter_start + len(self.parameter_start_token))
            if name_end != -1:
                candidate_name = tail[
                    parameter_start + len(self.parameter_start_token) : name_end
                ].strip()
                raw_value = tail[name_end + 1 :]
                if self._is_streamable_string_param(
                    self._streaming_tool_name or "", candidate_name, tools
                ):
                    marker_hold = self._ends_with_partial_token(
                        raw_value, self.parameter_end_token
                    )
                    if marker_hold:
                        raw_value = raw_value[:-marker_hold]
                    raw_value = self._without_incomplete_html_entity(raw_value)
                    decoded_value = html.unescape(raw_value)

                    # _convert_param_value treats the exact literal "null" as
                    # JSON null even for string schemas. Do not commit to an
                    # opening JSON string until the value diverges from it.
                    if not "null".startswith(decoded_value.lower()):
                        partial_name = candidate_name
                        partial_value = decoded_value

        # Do not emit a lone opening brace before any argument content is
        # stable. This also lets the literal "null" remain undecided until its
        # parameter closes.
        if not is_complete and not self._completed_args and partial_value is None:
            return []

        snapshot_parts = [
            f"{json.dumps(key, ensure_ascii=False)}: "
            f"{json.dumps(value, ensure_ascii=False)}"
            for key, value in self._completed_args.items()
        ]
        if partial_name is not None and partial_value is not None:
            key_json = json.dumps(partial_name, ensure_ascii=False)
            escaped_value = json.dumps(partial_value, ensure_ascii=False)[1:-1]
            snapshot_parts.append(f'{key_json}: "{escaped_value}')

        snapshot = "{" + ", ".join(snapshot_parts) + "}"
        argument_diff: Optional[str] = None

        if is_complete:
            final_json = json.dumps(self._completed_args, ensure_ascii=False)
            streamed = self.streamed_args_for_tool[self.current_tool_id]
            if final_json.startswith(streamed):
                argument_diff = final_json[len(streamed) :]
            else:
                logger.warning(
                    "MiMo streamed arguments are not a prefix of final arguments "
                    "for tool '%s'",
                    self._streaming_tool_name,
                )

            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = dict(
                self._completed_args
            )
            self._buffer = self._buffer[end + len(self.eot_token) :]
        else:
            # The final brace is not stable until all parameters are known.
            stable_end = len(snapshot) - 1
            if stable_end > self._streamed_json_len:
                argument_diff = snapshot[self._streamed_json_len : stable_end]
                self._streamed_json_len = stable_end

        if argument_diff:
            self.streamed_args_for_tool[self.current_tool_id] += argument_diff
            calls.append(self._make_tool_call_item(parameters=argument_diff))
        if is_complete:
            self._reset_streaming_state()
        return calls

    def _reset_streaming_state(self):
        self._in_tool_call = False
        self._reject_current_tool = False
        self._raw_tool_prefix = ""
        self._streaming_tool_name = None
        self._completed_args = {}
        self._streamed_json_len = 0
        self.current_tool_name_sent = False

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
