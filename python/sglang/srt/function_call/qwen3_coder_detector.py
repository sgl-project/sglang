import ast
import html
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


def _safe_val(raw: str) -> Any:
    """
    Fallback value parsing when no schema is available.

    Tries JSON parsing, then ast.literal_eval, then returns raw string.
    """
    raw = html.unescape(raw.strip())

    # Handle explicit null
    if raw.lower() == "null":
        return None

    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw


class Qwen3CoderDetector(BaseFormatDetector):
    """
    Detector for Qwen 3 models.
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
        # More robust regex: handles missing </parameter> by using next <parameter= or </function> as delimiter
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
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

        # Cache for tool schema
        self._tools_schema_cache: Dict[str, Dict[str, Any]] = {}

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def _get_arguments_config(
        self, func_name: str, tools: Optional[List[Tool]]
    ) -> Dict[str, Any]:
        """
        Extract argument configuration (schema) for a function from tools.

        Args:
            func_name: The function name to look up
            tools: List of available tools

        Returns:
            Dictionary of parameter configurations with type info
        """
        if tools is None:
            return {}

        # Check cache first
        if func_name in self._tools_schema_cache:
            return self._tools_schema_cache[func_name]

        for tool in tools:
            if not hasattr(tool, "type") or not hasattr(tool, "function"):
                continue
            if tool.type == "function" and tool.function.name == func_name:
                if not hasattr(tool.function, "parameters"):
                    self._tools_schema_cache[func_name] = {}
                    return {}
                params = tool.function.parameters
                if isinstance(params, dict) and "properties" in params:
                    result = params["properties"]
                    self._tools_schema_cache[func_name] = result
                    return result
                elif isinstance(params, dict):
                    self._tools_schema_cache[func_name] = params
                    return params
                else:
                    self._tools_schema_cache[func_name] = {}
                    return {}

        logger.debug("Tool '%s' is not defined in the tools list.", func_name)
        self._tools_schema_cache[func_name] = {}
        return {}

    def _convert_param_value(
        self,
        param_value: str,
        param_name: str,
        param_config: Dict[str, Any],
        func_name: str,
    ) -> Any:
        """
        Convert parameter value based on its type in the schema.

        This method provides schema-aware type conversion similar to vLLM's implementation.

        Args:
            param_value: Raw string value from XML
            param_name: Name of the parameter
            param_config: Schema configuration for all parameters
            func_name: Name of the function (for logging)

        Returns:
            Converted value with appropriate Python type
        """
        # Strip and unescape HTML entities
        param_value = html.unescape(param_value.strip())

        # Get the type from schema first (needed for null handling)
        param_type = "string"  # default
        if param_name in param_config:
            param_schema = param_config[param_name]
            if isinstance(param_schema, dict) and "type" in param_schema:
                param_type = str(param_schema["type"]).strip().lower()

        # Handle null value - return string "null" for string types, None for others
        if param_value.lower() == "null":
            if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
                return "null"  # Keep as string literal for string types
            return None

        # If parameter not in schema, try generic parsing then fall back to string
        if param_name not in param_config:
            if param_config:
                logger.debug(
                    "Parameter '%s' not defined in schema for tool '%s', "
                    "attempting generic parsing.",
                    param_name,
                    func_name,
                )
            return _safe_val(param_value)

        # String types - return as-is
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value

        # Integer types
        if (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
            or param_type == "integer"
        ):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                logger.debug(
                    "Value '%s' for parameter '%s' is not an integer in tool '%s', "
                    "returning as string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value

        # Float/Number types
        if (
            param_type.startswith("num")
            or param_type.startswith("float")
            or param_type == "number"
            or param_type == "double"
        ):
            try:
                float_value = float(param_value)
                return float_value
            except (ValueError, TypeError):
                logger.debug(
                    "Value '%s' for parameter '%s' is not a number in tool '%s', "
                    "returning as string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value

        # Boolean types
        if param_type in ["boolean", "bool", "binary"]:
            lower_val = param_value.lower()
            if lower_val == "true":
                return True
            elif lower_val == "false":
                return False
            else:
                logger.debug(
                    "Value '%s' for parameter '%s' is not a boolean in tool '%s', "
                    "returning False.",
                    param_value,
                    param_name,
                    func_name,
                )
                return False

        # Object/Array types - try JSON parsing
        if (
            param_type in ["object", "array", "arr"]
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                return json.loads(param_value)
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.debug(
                    "Value '%s' for parameter '%s' cannot be parsed as JSON in tool '%s', "
                    "trying ast.literal_eval.",
                    param_value,
                    param_name,
                    func_name,
                )
                try:
                    return ast.literal_eval(param_value)
                except (ValueError, SyntaxError, TypeError):
                    logger.debug(
                        "Value '%s' for parameter '%s' cannot be parsed, "
                        "returning as string.",
                        param_value,
                        param_name,
                        func_name,
                    )
                    return param_value

        # Unknown type - try generic parsing
        return _safe_val(param_value)

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
                    is_valid = function_name in self._tool_indices
                    if not is_valid:
                        logger.warning(f"Invalid function name: {function_name}")
                        if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                            # Reset state and skip (default legacy behavior)
                            self._reset_streaming_state()
                            normal += self._buf
                            self._buf = ""
                            break

                    # Process tool call (valid or unknown with env=TRUE)
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
                    # Function name not complete yet, wait for more text
                    break

            # Parse parameters incrementally
            if self._function_name_sent:
                # Process parameters and get any calls to emit
                parameter_calls = self._parse_and_stream_parameters(self._buf, tools)
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

    def _parse_and_stream_parameters(
        self, text_to_parse: str, tools: List[Tool]
    ) -> List[ToolCallItem]:
        """
        Parse complete parameter blocks with schema-based type conversion.

        This method:
        1. Finds all complete <parameter> blocks
        2. Parses them into a dictionary with schema-aware type conversion
        3. Compares with current parameters and generates diff if needed
        4. Updates internal state

        Args:
            text_to_parse: The text to search for parameter blocks
            tools: List of available tools for schema lookup

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

        # Get schema for current function
        param_config = self._get_arguments_config(self._current_function_name, tools)

        # Build new parameters dictionary with type conversion
        new_params = {}
        for match in param_matches:
            param_name = match.group(1).strip()
            param_value = match.group(2)
            # Strip leading/trailing newlines from value
            param_value = param_value.lstrip("\n").rstrip("\n")

            # Use schema-based type conversion
            new_params[param_name] = self._convert_param_value(
                param_value, param_name, param_config, self._current_function_name
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
        for m in self.tool_call_function_regex.findall(block):
            txt = m[0] if m[0] else m[1]
            if ">" not in txt:
                continue
            idx = txt.index(">")
            fname = txt[:idx].strip()
            body = txt[idx + 1 :]

            # Get schema for this function
            param_config = self._get_arguments_config(fname, tools)

            params: Dict[str, Any] = {}
            for pm in self.tool_call_parameter_regex.findall(body):
                ptxt = pm if isinstance(pm, str) else (pm[0] if pm[0] else pm[1])
                if ">" not in ptxt:
                    continue
                pidx = ptxt.index(">")
                pname = ptxt[:pidx].strip()
                pval = ptxt[pidx + 1 :].lstrip("\n").rstrip("\n")

                # Use schema-based type conversion
                params[pname] = self._convert_param_value(
                    pval, pname, param_config, fname
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
