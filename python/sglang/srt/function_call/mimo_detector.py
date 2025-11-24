import ast
import html
import json
import logging
import re
from typing import Any, Dict, List, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


def _convert_param_value(
    param_value: str, param_name: str, param_config: dict, func_name: str
) -> Any:
    """
    Convert parameter value based on its type in the schema.
    Adapted from vllm-project/vllm (vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py)
    """
    param_value = html.unescape(param_value)

    # Handle null value for any type
    if param_value.lower() == "null":
        return None

    if param_name not in param_config:
        if param_config != {}:
            logger.warning(
                "Parsed parameter '%s' is not defined in the tool "
                "parameters for tool '%s', directly returning the "
                "string value.",
                param_name,
                func_name,
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
    Detector for MiMo.
    Assumes function call format:
        <tool_call>
        <function=execute_bash>
        <parameter=command>
        pwd && ls
        </parameter>
        </function>
        </tool_call>
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )
        self._buf: str = ""

        # Streaming state variables
        self._current_function_name: str = ""
        self._current_parameters: Dict[str, Any] = {}
        self._streamed_parameters: Dict[str, str] = (
            {}
        )  # Track what parameter content we've streamed
        self._in_tool_call: bool = False
        self._function_name_sent: bool = False

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buf += new_text
        normal = ""
        calls: List[ToolCallItem] = []

        # Build tool indices for validation
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        # Build tool parameter configs for type conversion
        if not hasattr(self, "_tool_parameter_configs"):
            self._tool_parameter_configs = {
                tool.function.name: tool.function.parameters.get("properties", {})
                for tool in tools
                if tool.function.name
            }

        while True:
            # If we're not in a tool call and don't see a start token, return normal text
            if not self._in_tool_call and self.tool_call_start_token not in self._buf:
                normal += self._buf
                self._buf = ""
                break

            # Look for tool call start
            if not self._in_tool_call:
                s = self._buf.find(self.tool_call_start_token)
                if s == -1:
                    normal += self._buf
                    self._buf = ""
                    break

                normal += self._buf[:s]
                self._buf = self._buf[s:]

                self._in_tool_call = True
                self._function_name_sent = False
                self._current_function_name = ""
                self._current_parameters = {}
                self._streamed_parameters = {}

                # Remove the start token
                self._buf = self._buf[len(self.tool_call_start_token) :]
                continue

            # We're in a tool call, try to parse function name if not sent yet
            if not self._function_name_sent:
                # Look for function name pattern: <function=name>
                function_match = re.search(r"<function=([^>]+)>", self._buf)
                if function_match:
                    function_name = function_match.group(1).strip()

                    # Validate function name
                    if function_name in self._tool_indices:
                        self._current_function_name = function_name
                        self._function_name_sent = True

                        # Initialize tool call tracking
                        if self.current_tool_id == -1:
                            self.current_tool_id = 0

                        # Ensure tracking arrays are large enough
                        while len(self.prev_tool_call_arr) <= self.current_tool_id:
                            self.prev_tool_call_arr.append({})
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                        # Store tool call info
                        self.prev_tool_call_arr[self.current_tool_id] = {
                            "name": function_name,
                            "arguments": {},
                        }

                        # Send tool name with empty parameters
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        )

                        # Remove the processed function declaration
                        self._buf = self._buf[function_match.end() :]
                        continue
                    else:
                        # Invalid function name, reset state
                        logger.warning(f"Invalid function name: {function_name}")
                        self._reset_streaming_state()
                        normal += self._buf
                        self._buf = ""
                        break
                else:
                    # Function name not complete yet, wait for more text
                    break

            # Parse parameters incrementally
            if self._function_name_sent:
                # Process parameters and get any calls to emit
                parameter_calls = self._parse_and_stream_parameters(self._buf)
                calls.extend(parameter_calls)

                # Check if tool call is complete
                if self.tool_call_end_token in self._buf:
                    end_pos = self._buf.find(self.tool_call_end_token)

                    # Add closing brace to complete the JSON object
                    current_streamed = self.streamed_args_for_tool[self.current_tool_id]
                    if current_streamed:
                        # Count opening and closing braces to check if JSON is complete
                        open_braces = current_streamed.count("{")
                        close_braces = current_streamed.count("}")
                        if open_braces > close_braces:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters="}",
                                )
                            )
                            self.streamed_args_for_tool[self.current_tool_id] = (
                                current_streamed + "}"
                            )

                    # Complete the tool call
                    self._buf = self._buf[end_pos + len(self.tool_call_end_token) :]
                    self._reset_streaming_state()
                    self.current_tool_id += 1
                    continue
                else:
                    # Tool call not complete yet, wait for more text
                    break

        return StreamingParseResult(normal_text=normal, calls=calls)

    def _parse_and_stream_parameters(self, text_to_parse: str) -> List[ToolCallItem]:
        """
        Parse complete parameter blocks from text and return any tool call items to emit.

        This method:
        1. Finds all complete <parameter> blocks
        2. Parses them into a dictionary
        3. Compares with current parameters and generates diff if needed
        4. Updates internal state

        Args:
            text_to_parse: The text to search for parameter blocks

        Returns:
            List of ToolCallItem objects to emit (may be empty)
        """
        calls: List[ToolCallItem] = []

        # Find all complete parameter patterns
        param_matches = list(
            re.finditer(
                r"<parameter=([^>]+)>(.*?)</parameter>", text_to_parse, re.DOTALL
            )
        )

        # Build new parameters dictionary
        new_params = {}
        for match in param_matches:
            param_name = match.group(1).strip()
            param_value = match.group(2)
            new_params[param_name] = _convert_param_value(
                param_value,
                param_name,
                self._tool_parameter_configs.get(self._current_function_name, {}),
                self._current_function_name,
            )

        # Calculate parameter diff to stream with proper incremental JSON building
        if new_params != self._current_parameters:
            previous_args_json = self.streamed_args_for_tool[self.current_tool_id]

            # Build incremental JSON properly
            if not self._current_parameters:
                # First parameter(s) - start JSON object but don't close it yet
                items = []
                for key, value in new_params.items():
                    items.append(
                        f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}"
                    )
                json_fragment = "{" + ", ".join(items)

                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=None,
                        parameters=json_fragment,
                    )
                )
                self.streamed_args_for_tool[self.current_tool_id] = json_fragment

            else:
                # Additional parameters - add them incrementally
                new_keys = set(new_params.keys()) - set(self._current_parameters.keys())
                if new_keys:
                    # Build the continuation part (no closing brace yet)
                    continuation_parts = []
                    for key in new_keys:
                        value = new_params[key]
                        continuation_parts.append(
                            f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}"
                        )

                    json_fragment = ", " + ", ".join(continuation_parts)

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=json_fragment,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = (
                        previous_args_json + json_fragment
                    )

            # Update current state
            self._current_parameters = new_params
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = new_params

        return calls

    def _reset_streaming_state(self):
        """Reset streaming state for the next tool call"""
        self._in_tool_call = False
        self._function_name_sent = False
        self._current_function_name = ""
        self._current_parameters = {}
        self._streamed_parameters = {}
        self.current_tool_name_sent = False

    def _extract(self, text: str, tools: List[Tool]) -> Tuple[str, List[ToolCallItem]]:
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        cursor = 0
        while True:
            s = text.find(self.tool_call_start_token, cursor)
            if s == -1:
                normal_parts.append(text[cursor:])
                break
            normal_parts.append(text[cursor:s])
            e = text.find(self.tool_call_end_token, s)
            if e == -1:
                normal_parts.append(text[s:])
                break
            block = text[s : e + len(self.tool_call_end_token)]
            cursor = e + len(self.tool_call_end_token)
            calls.extend(self._parse_block(block, tools))
        return "".join(normal_parts), calls

    def _parse_block(self, block: str, tools: List[Tool]) -> List[ToolCallItem]:
        res: List[ToolCallItem] = []
        _tool_parameter_configs = {
            tool.function.name: tool.function.parameters.get("properties", {})
            for tool in tools
        }
        for m in self.tool_call_function_regex.findall(block):
            txt = m[0] if m[0] else m[1]
            if ">" not in txt:
                continue
            idx = txt.index(">")
            fname = txt[:idx].strip()
            body = txt[idx + 1 :]
            params: Dict[str, Any] = {}
            for pm in self.tool_call_parameter_regex.findall(body):
                ptxt = pm[0] if pm[0] else pm[1]
                if ">" not in ptxt:
                    continue
                pidx = ptxt.index(">")
                pname = ptxt[:pidx].strip()
                pval = ptxt[pidx + 1 :]
                params[pname] = _convert_param_value(
                    pval,
                    pname,
                    _tool_parameter_configs.get(fname, {}),
                    fname,
                )
            raw = {"name": fname, "arguments": params}
            try:
                # TODO: fix idx in function call, the index for a function
                # call will always be -1 in parse_base_json
                res.extend(self.parse_base_json(raw, tools))
            except Exception:
                logger.warning("invalid tool call for %s dropped", fname)
        return res

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
