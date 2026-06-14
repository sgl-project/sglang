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


class MinimaxM2Detector(BaseFormatDetector):
    """
    Detector for MiniMax M2 models.
    Assumes function call format:
        <minimax:tool_call>
        <invoke name="func1">
        <parameter name="param1">value1</parameter>
        <parameter name="param2">value2</parameter>
        </invoke>
        </minimax:tool_call>
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.tool_call_end_token: str = "</minimax:tool_call>"
        self.tool_call_prefix: str = '<invoke name="'
        self.tool_call_function_end_token: str = "</invoke>"
        self.tool_call_regex = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>|<minimax:tool_call>(.*?)$",
            re.DOTALL,
        )
        self.tool_call_function_regex = re.compile(
            r"<invoke name=\"(.*?)</invoke>|<invoke name=\"(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter name=\"(.*?)</parameter>|<parameter name=\"(.*?)$", re.DOTALL
        )
        self._buf: str = ""
        self._in_tool_call: bool = False
        self._function_name_sent: bool = False
        self._current_function_name: str = ""
        self._current_parameters: Dict[str, Any] = {}

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        """Convert parameter value to the correct type (legacy single-type version)."""
        return self._convert_param_value_with_types(value, [param_type])

    def _extract_types_from_schema(self, schema: Any) -> list[str]:
        """
        Extract all possible types from a JSON schema definition.
        Handles anyOf, oneOf, allOf, type arrays, and enum fields.

        Args:
            schema: The JSON schema definition for a parameter

        Returns:
            List of type strings (e.g., ["string", "integer", "null"])
        """
        if schema is None:
            return ["string"]

        if not isinstance(schema, dict):
            return ["string"]

        types: set[str] = set()

        # Handle direct "type" field
        if "type" in schema:
            type_value = schema["type"]
            if isinstance(type_value, str):
                types.add(type_value)
            elif isinstance(type_value, list):
                for t in type_value:
                    if isinstance(t, str):
                        types.add(t)

        # Handle enum - infer types from enum values
        if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
            for value in schema["enum"]:
                if value is None:
                    types.add("null")
                elif isinstance(value, bool):
                    types.add("boolean")
                elif isinstance(value, int):
                    types.add("integer")
                elif isinstance(value, float):
                    types.add("number")
                elif isinstance(value, str):
                    types.add("string")
                elif isinstance(value, list):
                    types.add("array")
                elif isinstance(value, dict):
                    types.add("object")

        # Handle anyOf, oneOf, allOf - recursively extract types
        for choice_field in ("anyOf", "oneOf", "allOf"):
            if choice_field in schema and isinstance(schema[choice_field], list):
                for choice in schema[choice_field]:
                    extracted = self._extract_types_from_schema(choice)
                    types.update(extracted)

        # If no types found, default to string
        if not types:
            return ["string"]

        return list(types)

    def _convert_param_value_with_types(
        self, value: str, param_types: list[str]
    ) -> Any:
        """
        Convert parameter value to the correct type based on a list of possible types.
        Tries each type in order until one succeeds.

        Args:
            value: The string value to convert
            param_types: List of possible type strings

        Returns:
            The converted value
        """
        if value.lower() == "null":
            return None

        # Normalize types
        normalized_types = [t.lower() for t in param_types]

        # Try null first if it's in the list
        if "null" in normalized_types or value.lower() in ("null", "none", "nil"):
            return None

        # Try each type in order of preference (most specific first, string as fallback)
        # Priority: integer > number > boolean > object > array > string
        type_priority = [
            "integer",
            "int",
            "number",
            "float",
            "boolean",
            "bool",
            "object",
            "array",
            "string",
            "str",
            "text",
        ]

        for param_type in type_priority:
            if param_type not in normalized_types:
                continue

            if param_type in ["string", "str", "text"]:
                return value
            elif param_type in ["integer", "int"]:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
            elif param_type in ["number", "float"]:
                try:
                    val = float(value)
                    return val if val != int(val) else int(val)
                except (ValueError, TypeError):
                    continue
            elif param_type in ["boolean", "bool"]:
                lower_val = value.lower().strip()
                if lower_val in ["true", "1", "yes", "on"]:
                    return True
                elif lower_val in ["false", "0", "no", "off"]:
                    return False
                continue
            elif param_type in ["object", "array"]:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    continue

        # Fallback: try JSON parse, then return as string
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _get_param_types_from_config(
        self, param_name: str, param_config: dict
    ) -> list[str]:
        """
        Get parameter types from parameter configuration.
        Handles anyOf, oneOf, allOf, and direct type definitions.

        Args:
            param_name: The name of the parameter
            param_config: The properties dict from the tool schema

        Returns:
            List of type strings
        """
        if param_name not in param_config:
            return ["string"]

        param_schema = param_config[param_name]
        if not isinstance(param_schema, dict):
            return ["string"]

        return self._extract_types_from_schema(param_schema)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buf += new_text
        normal = ""
        calls: List[ToolCallItem] = []

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        while True:
            # State 1: outside any wrapper, look for <minimax:tool_call>.
            if not self._in_tool_call:
                s = self._buf.find(self.tool_call_start_token)
                if s == -1:
                    # Hold any trailing prefix that might be the start of a
                    # split start token; flush the rest as normal text.
                    hold = self._ends_with_partial_token(
                        self._buf, self.tool_call_start_token
                    )
                    if hold:
                        normal += self._buf[:-hold]
                        self._buf = self._buf[-hold:]
                    else:
                        normal += self._buf
                        self._buf = ""
                    break
                normal += self._buf[:s]
                self._buf = self._buf[s + len(self.tool_call_start_token) :]
                self._in_tool_call = True
                self._function_name_sent = False
                self._current_function_name = ""
                self._current_parameters = {}
                continue

            # State 2: inside a wrapper, no current invoke — find next
            # <invoke name="..."> or the wrapper's closing tag.
            if not self._function_name_sent:
                wrapper_end = self._buf.find(self.tool_call_end_token)
                invoke_match = re.search(r'<invoke name="([^"]+)">', self._buf)
                if invoke_match is None or (
                    wrapper_end != -1 and wrapper_end < invoke_match.start()
                ):
                    if wrapper_end != -1:
                        self._buf = self._buf[
                            wrapper_end + len(self.tool_call_end_token) :
                        ]
                        self._in_tool_call = False
                        continue
                    break

                function_name = invoke_match.group(1).strip()
                if function_name not in self._tool_indices:
                    close = self._buf.find(
                        self.tool_call_function_end_token, invoke_match.end()
                    )
                    if close == -1:
                        break
                    logger.warning("invalid tool call for %s dropped", function_name)
                    self._buf = self._buf[
                        close + len(self.tool_call_function_end_token) :
                    ]
                    continue

                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": function_name,
                    "arguments": {},
                }
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=function_name,
                        parameters="",
                    )
                )
                self._current_function_name = function_name
                self._current_parameters = {}
                self._function_name_sent = True
                self._buf = self._buf[invoke_match.end() :]
                continue

            # State 3: inside an invoke, stream parameters incrementally.
            invoke_close_pos = self._buf.find(self.tool_call_function_end_token)
            param_match = re.search(
                r'<parameter name="([^"]+)">(.*?)</parameter>',
                self._buf,
                re.DOTALL,
            )

            # Only accept a <parameter> match that ends inside the current
            # invoke. Otherwise it belongs to the next invoke and would
            # cross-contaminate this call's arguments.
            if param_match is not None and (
                invoke_close_pos == -1 or param_match.end() <= invoke_close_pos
            ):
                pname = param_match.group(1).strip()
                pval = param_match.group(2).lstrip("\n").rstrip("\n")
                value = self._parse_parameter(
                    self._current_function_name, pname, pval, tools
                )
                if pname not in self._current_parameters:
                    if not self._current_parameters:
                        fragment = (
                            "{"
                            + json.dumps(pname, ensure_ascii=False)
                            + ": "
                            + json.dumps(value, ensure_ascii=False)
                        )
                    else:
                        fragment = (
                            ", "
                            + json.dumps(pname, ensure_ascii=False)
                            + ": "
                            + json.dumps(value, ensure_ascii=False)
                        )
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=fragment,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += fragment
                    self._current_parameters[pname] = value
                    self.prev_tool_call_arr[self.current_tool_id]["arguments"] = (
                        dict(self._current_parameters)
                    )
                self._buf = self._buf[param_match.end() :]
                continue

            if invoke_close_pos != -1:
                # Close the JSON object — opening "{" if no params arrived.
                current_streamed = self.streamed_args_for_tool[self.current_tool_id]
                if not current_streamed:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters="{}",
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = "{}"
                else:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters="}",
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += "}"
                self._buf = self._buf[
                    invoke_close_pos + len(self.tool_call_function_end_token) :
                ]
                self.current_tool_id += 1
                self._function_name_sent = False
                self._current_parameters = {}
                continue

            # No complete parameter, no </invoke> — wait for more text.
            break

        return StreamingParseResult(normal_text=normal, calls=calls)
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
        for m in self.tool_call_function_regex.findall(block):
            txt = m[0] if m[0] else m[1]
            if '">' not in txt:
                continue
            idx = txt.index('">')
            fname = txt[:idx].strip()
            body = txt[idx + 2 :]
            params: Dict[str, Any] = {}
            for pm in self.tool_call_parameter_regex.findall(body):
                ptxt = pm[0] if pm[0] else pm[1]
                if '">' not in ptxt:
                    continue
                pidx = ptxt.index('">')
                pname = ptxt[:pidx].strip()
                pval = ptxt[pidx + 2 :].lstrip("\n").rstrip("\n")
                params[pname] = self._parse_parameter(fname, pname, pval, tools)
            raw = {"name": fname, "arguments": params}
            try:
                # TODO: fix idx in function call, the index for a function
                # call will always be -1 in parse_base_json
                res.extend(self.parse_base_json(raw, tools))
            except Exception:
                logger.warning("invalid tool call for %s dropped", fname)
        return res

    def _parse_parameter(
        self, fname: str, pname: str, pval: str, tools: List[Tool]
    ) -> Any:
        param_config = {}
        for tool in tools:
            if tool.function.name == fname and tool.function.parameters is not None:
                parameters = tool.function.parameters
                if isinstance(parameters, dict) and "properties" in parameters:
                    param_config = parameters["properties"]
                    break

        param_type = self._get_param_types_from_config(pname, param_config)
        return self._convert_param_value_with_types(pval, param_type)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
