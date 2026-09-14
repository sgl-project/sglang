import json
import logging
import re

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _find_common_prefix, _partial_json_loads

logger = logging.getLogger(__name__)

# Keys the model sometimes emits as a spurious wrapper around the real
# parameter object, e.g. {"arguments": {...}} or {"input": "..."} instead of
# the tool's declared properties. See sgl-project/sglang#38924.
_WRAPPER_ARG_KEYS = ("arguments", "input")


def _tool_property_names(tool: Tool | None) -> set[str]:
    """Declared property names of a tool, or an empty set when unknown."""
    params = tool.function.parameters if tool is not None else None
    if isinstance(params, dict) and isinstance(params.get("properties"), dict):
        return set(params["properties"].keys())
    return set()


def _sole_tool_property(tool: Tool | None) -> str | None:
    """The single property to remap a bare scalar onto: the only declared
    property, else the only required one. None when the target is ambiguous."""
    params = tool.function.parameters if tool is not None else None
    if not isinstance(params, dict):
        return None
    props = params.get("properties")
    if isinstance(props, dict) and len(props) == 1:
        return next(iter(props))
    required = params.get("required")
    if (
        isinstance(required, list)
        and len(required) == 1
        and isinstance(required[0], str)
    ):
        return required[0]
    return None


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
        self.invoke_end_token = "</｜DSML｜invoke>"
        self.parameter_regex = r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>'
        self.partial_parameter_regex = (
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*)$'
        )
        self.function_calls_regex = (
            r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>"
        )
        # Long-form `<｜DSML｜invoke name="x">...</｜DSML｜invoke>` and the
        # self-closing `<｜DSML｜invoke name="x"/>` shape V4 emits for zero-arg
        # tools. The `end` group is empty when the closer hasn't streamed in.
        self.invoke_regex = (
            r'<｜DSML｜invoke\s+name="(?P<name>[^"]+)"\s*'
            r"(?:(?P<self_close>/>)"
            r"|>(?P<body>.*?)(?P<end>(?:</｜DSML｜invoke>|$)))"
        )
        self.prefix_parameter_end_call = ["</", "｜DSML｜", "parameter"]
        self.prefix_invoke_end_call = ["</", "｜DSML｜", "inv", "oke"]
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call."""
        return self.bot_token in text or "<｜DSML｜invoke" in text

    @staticmethod
    def _unpack_invoke_match(m: "re.Match[str]") -> tuple[str, str, bool]:
        """Returns (name, body, is_complete) for an invoke_regex match.

        Self-closing invokes have empty body and are always complete.
        Long-form bodies are always strings (possibly empty); they're
        incomplete when matched against `$` because the closing tag
        hasn't streamed in yet.
        """
        name = m.group("name").strip()
        if m.group("self_close"):
            return name, "", True
        return name, m.group("body"), bool(m.group("end"))

    def _parse_parameters_from_xml(
        self, invoke_content: str, allow_partial: bool = False
    ) -> str:
        """
        Parse parameters from either XML-like format or JSON format to str.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }
        """
        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = invoke_content.strip()
        if invoke_content_stripped.startswith("{"):
            if allow_partial:
                # Remove incomplete invoke end call prefix in case they are captured by param
                for token in reversed(self.prefix_invoke_end_call):
                    invoke_content_stripped = invoke_content_stripped.rstrip(token)
                return invoke_content_stripped
            elif invoke_content_stripped.endswith("}"):
                return invoke_content_stripped

        # Fall back to XML parameter tag parsing (original format)
        parameters = {}
        # Find all complete parameter matches
        param_matches = list(
            re.finditer(self.parameter_regex, invoke_content, re.DOTALL)
        )

        last_match_end = 0
        for match in param_matches:
            param_name = match.group(1)
            param_type = match.group(2)
            param_value = match.group(3)
            last_match_end = match.end()

            # Convert value based on type
            if param_type == "true":  # string type
                parameters[param_name] = param_value.strip()
            else:
                # Try to parse as JSON for other types
                try:
                    parameters[param_name] = json.loads(param_value.strip())
                except (json.JSONDecodeError, ValueError):
                    parameters[param_name] = param_value.strip()

        # If allowed, try to parse a partial parameter at the end
        if allow_partial:
            remaining_content = invoke_content[last_match_end:]

            # Remove incomplete parameter_end_call prefix in case they are captured by param
            for token in reversed(self.prefix_parameter_end_call):
                remaining_content = remaining_content.rstrip(token)

            # Match start of a parameter tag + value (potentially incomplete)
            # Regex: <tag name="..." string="...">VALUE... (no end tag)
            partial_match = re.search(
                self.partial_parameter_regex, remaining_content, re.DOTALL
            )

            if partial_match and (param_value := partial_match.group(3)):
                param_name = partial_match.group(1)
                if partial_match.group(2) == "true":
                    parameters[param_name] = param_value.strip()
                else:
                    try:
                        parameters[param_name] = _partial_json_loads(
                            param_value, Allow.ALL
                        )[0]
                    except (json.JSONDecodeError, MalformedJSON, ValueError):
                        parameters[param_name] = param_value.strip()

        return json.dumps(parameters, ensure_ascii=False)

    def _unwrap_wrapped_arguments(self, params: dict, tool: Tool | None) -> dict:
        """Undo a spurious ``{"arguments"/"input": ...}`` wrapper the model emits
        around the real parameters (sgl-project/sglang#38924).

        No-op unless the object is exactly one wrapper key that is not itself a
        declared property. A dict value is unwrapped directly, a JSON-object
        string is decoded first (the XML ``string="true"`` sub-case), and a bare
        scalar is remapped onto the tool's sole property when unambiguous.
        """
        if not isinstance(params, dict) or len(params) != 1:
            return params
        ((key, value),) = params.items()
        if key not in _WRAPPER_ARG_KEYS or key in _tool_property_names(tool):
            return params
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip().startswith("{"):
            try:
                decoded = json.loads(value.strip())
            except (json.JSONDecodeError, ValueError):
                decoded = None
            if isinstance(decoded, dict):
                return decoded
        target = _sole_tool_property(tool)
        if target is not None:
            return {target: value}
        return params

    def _looks_like_wrapped_arguments(
        self, invoke_content: str, tool: Tool | None
    ) -> bool:
        """Cheap pre-check on raw (possibly partial) invoke content, mirroring
        ``_unwrap_wrapped_arguments``. Lets streaming hold a wrapped call back
        until it is complete rather than emit the wrapped shape then rewrite it.
        """
        declared = _tool_property_names(tool)
        content = invoke_content.strip()
        if content.startswith("{"):
            # First key as emitted so far; group(2) is the closing quote once
            # the key name is complete. While it is still being written, hold
            # back if it is a live prefix of a wrapper key -- streaming cannot
            # yet tell "arguments" from a real "argument_list".
            m = re.match(r'\{\s*"([^"]*)("?)', content)
            if m is None:
                return True
            key, closed = m.group(1), m.group(2)
            if closed:
                return key in _WRAPPER_ARG_KEYS and key not in declared
            return any(
                w.startswith(key) and w not in declared for w in _WRAPPER_ARG_KEYS
            )
        names = re.findall(r'name="([^"]+)"\s+string=', content)
        return (
            len(names) == 1
            and names[0] in _WRAPPER_ARG_KEYS
            and names[0] not in declared
        )

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].removesuffix("\n\n") if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            sections = re.findall(self.function_calls_regex, text, re.DOTALL)
            if not sections:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            tools_by_name = {t.function.name: t for t in tools if t.function.name}

            # Find all invoke blocks
            for function_calls_content in sections:
                for invoke_match in re.finditer(
                    self.invoke_regex, function_calls_content, re.DOTALL
                ):
                    func_name, invoke_content, _ = self._unpack_invoke_match(
                        invoke_match
                    )
                    func_args = self._parse_parameters_from_xml(invoke_content)
                    # construct match_result for parse_base_json
                    match_result = {
                        "name": func_name,
                        "parameters": self._unwrap_wrapped_arguments(
                            json.loads(func_args), tools_by_name.get(func_name)
                        ),
                    }
                    calls.extend(self.parse_base_json(match_result, tools))

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV32 format.
        Supports multiple consecutive invoke blocks and argument streaming.
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
                if e_token in current_text:
                    current_text = current_text.replace(e_token, "")
            return StreamingParseResult(normal_text=current_text)

        all_calls: list[ToolCallItem] = []
        # Only recovered for the first call: the DSML guard above never releases a
        # buffer that still holds a marker, so later prose stays buffered.
        preamble = ""
        tools_by_name = {t.function.name: t for t in tools if t.function.name}
        try:
            # Loop to handle multiple consecutive invoke blocks
            while True:
                # Try to match an invoke block (may be partial)
                invoke_match = re.search(
                    pattern=self.invoke_regex,
                    string=current_text,
                    flags=re.DOTALL,
                )
                if not invoke_match:
                    break

                func_name, invoke_content, is_tool_end = self._unpack_invoke_match(
                    invoke_match
                )

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]
                    call_start = invoke_match.start()
                    bot_pos = current_text.rfind(self.bot_token, 0, call_start)
                    if bot_pos != -1:
                        call_start = bot_pos
                    # Same trailing-newline trim as detect_and_parse, so both agree.
                    preamble = current_text[:call_start].removesuffix("\n\n")

                # Ensure arrays are large enough for current tool
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                # 1. Send tool name if not sent yet
                if not self.current_tool_name_sent:
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True

                # 2. Parse current parameters (partial or complete)
                current_params = self._parse_parameters_from_xml(
                    invoke_content, allow_partial=not is_tool_end
                )

                # A spurious {"arguments"/"input": ...} wrapper (#38924) is only
                # rewritten once the invoke is complete; until then hold the
                # call's arguments back rather than stream a wrapped prefix we
                # would have to rewrite. Clients act on arguments only at
                # finish_reason: tool_calls, so a single late chunk is safe.
                wrapped = self._looks_like_wrapped_arguments(
                    invoke_content, tools_by_name.get(func_name)
                )
                if wrapped and is_tool_end:
                    try:
                        current_params = json.dumps(
                            self._unwrap_wrapped_arguments(
                                json.loads(current_params),
                                tools_by_name.get(func_name),
                            ),
                            ensure_ascii=False,
                        )
                    except (json.JSONDecodeError, ValueError):
                        pass

                # 3. Calculate and send incremental arguments
                sent_len = len(self.streamed_args_for_tool[self.current_tool_id])
                prev_params = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments"
                )

                argument_diff = None

                if wrapped and not is_tool_end:
                    argument_diff = None  # hold back until the wrapper resolves
                elif is_tool_end:
                    # If complete, send everything remaining
                    argument_diff = current_params[sent_len:]
                elif prev_params is not None:
                    # If partial, send stable prefix diff
                    if current_params != prev_params:
                        prefix = _find_common_prefix(current_params, prev_params)
                        if len(prefix) > sent_len:
                            argument_diff = prefix[sent_len:]

                if argument_diff:
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=argument_diff,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                # Update the stored arguments
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": func_name,
                    "arguments": current_params,
                }

                # Check if tool call is complete (has closing tag)
                if is_tool_end:
                    # Remove the completed tool call from buffer
                    self._buffer = current_text[invoke_match.end() :]
                    current_text = self._buffer  # Update for next iteration

                    # Move to next tool call
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False

                    # Continue loop to check for more invoke blocks
                    continue
                else:
                    # Tool call not complete yet, don't return anything
                    # Wait for more chunks until we see </｜DSML｜invoke>
                    break

            # No more invoke blocks found
            return StreamingParseResult(normal_text=preamble, calls=all_calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            # Re-emit verbatim rather than swallowing the turn; the preamble is
            # still inside current_text unless a completed call advanced past it.
            # Calls are dropped on purpose: the failure can land between a tool's
            # name and its arguments, and a half-formed call is worse than none.
            self._buffer = ""
            if not current_text.startswith(preamble):
                current_text = preamble + current_text
            return StreamingParseResult(normal_text=current_text)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger="<｜DSML｜invoke",
        )

    def get_structural_tag_name(self) -> str:
        return "deepseek_v3_2"
