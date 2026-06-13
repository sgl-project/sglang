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

_PARAM_END_TAG = "</parameter>"
_PARAM_END_TAG_LEN = len(_PARAM_END_TAG)
# Hold back this many chars while streaming to avoid emitting a partial end tag
_STREAM_HOLD_BACK = _PARAM_END_TAG_LEN - 1  # 11


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

    String-typed parameters are streamed token-by-token.
    Non-string parameters (int/bool/object/array) are buffered until </parameter>.
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

        # Coarse streaming state
        self._current_function_name: str = ""
        self._current_parameters: Dict[str, Any] = {}
        self._streamed_parameters: Dict[str, str] = (
            {}
        )  # Track what parameter content we've streamed
        self._in_tool_call: bool = False
        self._function_name_sent: bool = False

        # Fine-grained per-parameter streaming state
        self._in_parameter: bool = False
        self._current_param_name: str = ""
        # How many raw chars of the current param value we have already JSON-encoded and sent.
        # Relative to the start of self._buf when the <parameter> tag was consumed.
        self._param_raw_sent_len: int = 0
        self._first_param_started: bool = False  # have we emitted the opening '{'
        self._current_param_is_string: bool = False  # stream char-by-char vs buffer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        while True:
            # ── Not yet in a tool call ──────────────────────────────────
            if not self._in_tool_call:
                s = self._buf.find(self.tool_call_start_token)
                if s == -1:
                    normal += self._buf
                    self._buf = ""
                    break
                normal += self._buf[:s]
                self._buf = self._buf[s + len(self.tool_call_start_token) :]
                self._in_tool_call = True
                self._function_name_sent = False
                self._current_function_name = ""
                self._current_parameters = {}
                self._streamed_parameters = {}
                self._in_parameter = False
                self._first_param_started = False
                continue

            # ── Parse function name ────────────────────────────────────
            if not self._function_name_sent:
                m = re.search(r'<invoke name="([^>]+)">', self._buf)
                if not m:
                    break  # wait for more text
                function_name = m.group(1).strip()
                if function_name not in self._tool_indices:
                    logger.warning("Invalid function name: %s", function_name)
                    self._reset_streaming_state()
                    normal += self._buf
                    self._buf = ""
                    break
                self._current_function_name = function_name
                self._function_name_sent = True
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
                self._buf = self._buf[m.end() :]
                continue

            # ── Inside function: stream parameters ────────────────────
            if self._in_parameter:
                new_calls, done = self._stream_current_param_value(tools)
                calls.extend(new_calls)
                if not done:
                    break  # wait for more text
                # parameter complete; loop to look for next parameter tag
                continue

            else:
                # Look for end of invoke first
                if self.tool_call_function_end_token in self._buf:
                    end_pos = self._buf.find(self.tool_call_function_end_token)
                    self._buf = self._buf[
                        end_pos + len(self.tool_call_function_end_token) :
                    ]
                    # Close the JSON object
                    current_streamed = self.streamed_args_for_tool[self.current_tool_id]
                    if not self._first_param_started:
                        # No parameters → emit empty object
                        fragment = "{}"
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=fragment,
                            )
                        )
                        self.streamed_args_for_tool[self.current_tool_id] = fragment
                    else:
                        open_braces = current_streamed.count(
                            "{"
                        ) - current_streamed.count("}")
                        if open_braces > 0:
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
                    self._reset_streaming_state(still_in_tool_call=True)
                    self.current_tool_id += 1
                    continue

                # Look for next parameter start tag
                pm = re.search(r'<parameter name="([^"]+)">', self._buf)
                if not pm:
                    break  # wait for more text

                param_name = pm.group(1).strip()
                self._current_param_name = param_name
                self._param_raw_sent_len = 0
                self._in_parameter = True

                param_types = self._get_param_types_for_current_tool(param_name, tools)
                self._current_param_is_string = self._is_string_streamable(param_types)

                # Emit the JSON key prefix (and opening string quote for string types)
                if not self._first_param_started:
                    key_prefix = '{"' + param_name + '": '
                    self._first_param_started = True
                else:
                    key_prefix = ', "' + param_name + '": '
                if self._current_param_is_string:
                    key_prefix += '"'  # open JSON string

                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=None,
                        parameters=key_prefix,
                    )
                )
                self.streamed_args_for_tool[self.current_tool_id] += key_prefix

                # Consume past the opening tag; buffer now starts at the value content
                self._buf = self._buf[pm.end() :]
                continue

        return StreamingParseResult(normal_text=normal, calls=calls)

    # ------------------------------------------------------------------
    # Per-parameter streaming helpers
    # ------------------------------------------------------------------

    def _stream_current_param_value(
        self, tools: List[Tool]
    ) -> Tuple[List[ToolCallItem], bool]:
        """
        Try to stream content for the parameter currently being parsed.

        Returns (calls, done):
            done=True  → </parameter> was found; self._in_parameter reset to False
            done=False → need more text
        """
        calls: List[ToolCallItem] = []
        end_pos = self._buf.find(_PARAM_END_TAG)

        if end_pos != -1:
            # Full parameter value is now available
            full_raw = self._buf[:end_pos]
            self._buf = self._buf[end_pos + _PARAM_END_TAG_LEN :]

            if self._current_param_is_string:
                unsent_raw = full_raw[self._param_raw_sent_len :]
                if unsent_raw:
                    fragment = _json_escape_string_content(unsent_raw)
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=fragment,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += fragment
                # Close the JSON string
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=None,
                        parameters='"',
                    )
                )
                self.streamed_args_for_tool[self.current_tool_id] += '"'
            else:
                # Non-string: emit properly typed value now
                converted = self._parse_parameter(
                    self._current_function_name,
                    self._current_param_name,
                    full_raw,
                    tools,
                )
                json_value = json.dumps(converted, ensure_ascii=False)
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=None,
                        parameters=json_value,
                    )
                )
                self.streamed_args_for_tool[self.current_tool_id] += json_value

            # Record parsed parameter in state
            converted_val = self._parse_parameter(
                self._current_function_name, self._current_param_name, full_raw, tools
            )
            self._current_parameters[self._current_param_name] = converted_val
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = dict(
                self._current_parameters
            )

            # Reset per-parameter state
            self._in_parameter = False
            self._current_param_name = ""
            self._param_raw_sent_len = 0
            self._current_param_is_string = False
            return calls, True

        # </parameter> not yet arrived
        if self._current_param_is_string and len(self._buf) > _STREAM_HOLD_BACK:
            # Stream what's safe (hold back last _STREAM_HOLD_BACK chars to avoid
            # splitting a partial </parameter> tag across two calls)
            streamable = self._buf[: len(self._buf) - _STREAM_HOLD_BACK]
            new_raw = streamable[self._param_raw_sent_len :]
            if new_raw:
                fragment = _json_escape_string_content(new_raw)
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=None,
                        parameters=fragment,
                    )
                )
                self.streamed_args_for_tool[self.current_tool_id] += fragment
                self._param_raw_sent_len = len(streamable)

        return calls, False

    # ------------------------------------------------------------------
    # Type helpers
    # ------------------------------------------------------------------

    def _get_param_types_for_current_tool(
        self, param_name: str, tools: List[Tool]
    ) -> list:
        for tool in tools:
            if (
                tool.function.name == self._current_function_name
                and tool.function.parameters is not None
            ):
                params = tool.function.parameters
                if isinstance(params, dict) and "properties" in params:
                    return self._get_param_types_from_config(
                        param_name, params["properties"]
                    )
        return ["string"]

    @staticmethod
    def _is_string_streamable(param_types: list) -> bool:
        """
        Return True if this parameter should be streamed as a JSON string value.
        Non-string types (int/number/bool/object/array) must be buffered until
        </parameter> so that type conversion can happen on the complete value.
        """
        non_string = {
            "integer",
            "int",
            "number",
            "float",
            "boolean",
            "bool",
            "object",
            "array",
        }
        normalized = [t.lower() for t in param_types]
        return not any(t in non_string for t in normalized)

    # ------------------------------------------------------------------
    # Existing helpers (type conversion, schema inspection)
    # ------------------------------------------------------------------

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        """Convert parameter value to the correct type (legacy single-type version)."""
        return self._convert_param_value_with_types(value, [param_type])

    def _extract_types_from_schema(self, schema: Any) -> list:
        if schema is None:
            return ["string"]
        if not isinstance(schema, dict):
            return ["string"]

        types: set = set()

        if "type" in schema:
            type_value = schema["type"]
            if isinstance(type_value, str):
                types.add(type_value)
            elif isinstance(type_value, list):
                for t in type_value:
                    if isinstance(t, str):
                        types.add(t)

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

        for choice_field in ("anyOf", "oneOf", "allOf"):
            if choice_field in schema and isinstance(schema[choice_field], list):
                for choice in schema[choice_field]:
                    extracted = self._extract_types_from_schema(choice)
                    types.update(extracted)

        if not types:
            return ["string"]
        return list(types)

    def _convert_param_value_with_types(self, value: str, param_types: list) -> Any:
        if value.lower() == "null":
            return None

        normalized_types = [t.lower() for t in param_types]

        if "null" in normalized_types or value.lower() in ("null", "none", "nil"):
            return None

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

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _get_param_types_from_config(self, param_name: str, param_config: dict) -> list:
        if param_name not in param_config:
            return ["string"]
        param_schema = param_config[param_name]
        if not isinstance(param_schema, dict):
            return ["string"]
        return self._extract_types_from_schema(param_schema)

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def _reset_streaming_state(self, still_in_tool_call: bool = False):
        """Reset streaming state, optionally keeping _in_tool_call=True for the next call."""
        self._in_tool_call = still_in_tool_call
        self._function_name_sent = False
        self._current_function_name = ""
        self._current_parameters = {}
        self._streamed_parameters = {}
        self.current_tool_name_sent = False
        # Per-parameter state
        self._in_parameter = False
        self._current_param_name = ""
        self._param_raw_sent_len = 0
        self._first_param_started = False
        self._current_param_is_string = False

    # ------------------------------------------------------------------
    # Non-streaming (batch) path
    # ------------------------------------------------------------------

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


def _json_escape_string_content(s: str) -> str:
    """Return the JSON-encoded content of a string, without surrounding quotes."""
    return json.dumps(s, ensure_ascii=False)[1:-1]
