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
from sglang.srt.function_call.utils import _partial_json_loads

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
        self.invoke_end_token = "</｜DSML｜invoke>"
        self.parameter_end_token = "</｜DSML｜parameter>"
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

    @staticmethod
    def _strip_partial_tag_suffix(text: str, tag: str) -> str:
        """Drop a trailing fragment of `text` that is the start of `tag`.

        `str.rstrip` cannot do this: it takes a character SET, so the previous
        `rstrip("oke")` also ate a value ending in "note", leaving "not".
        """
        for n in range(min(len(tag), len(text)), 0, -1):
            if text.endswith(tag[:n]):
                return text[:-n]
        return text

    def _direct_json_body(self, invoke_content: str, allow_partial: bool):
        """Format 2: the invoke body IS the arguments object, verbatim.
        Returns None when the body is not in that format.

        The partial closing tag is trimmed BEFORE whitespace is stripped. The
        other order is off by a byte: the tag's leading "<" shields the
        whitespace in front of it from strip(), so that whitespace is streamed
        and then vanishes once the tag completes.
        """
        text = invoke_content
        if allow_partial:
            text = self._strip_partial_tag_suffix(text, self.invoke_end_token)
        text = text.strip()
        if text.startswith("{") and (allow_partial or text.endswith("}")):
            return text
        return None

    def _complete_parameters(self, invoke_content: str) -> tuple[dict, int]:
        """Parameters whose closing tag has arrived, and the offset past the
        last one. These are settled: their serialised form cannot change."""
        parameters = {}
        last_match_end = 0
        for match in re.finditer(self.parameter_regex, invoke_content, re.DOTALL):
            name, param_type, value = match.group(1), match.group(2), match.group(3)
            value = value.strip()
            if param_type == "true":  # string type
                parameters[name] = value
            else:
                # Try to parse as JSON for other types
                try:
                    parameters[name] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    parameters[name] = value
            last_match_end = match.end()
        return parameters, last_match_end

    def _parse_parameters_from_xml(
        self, invoke_content: str, allow_partial: bool = False
    ) -> str:
        """
        Parse parameters from either XML-like format or JSON format to str.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }

        NOTE: the `allow_partial=True` result must not be streamed. A parameter
        still arriving can serialise as a structure on one chunk and as a
        quote-escaped string on the next, so consecutive results are not
        prefixes of each other. Use `_streamable_prefix` for that.
        """
        # First, try to parse as direct JSON (new format)
        body = self._direct_json_body(invoke_content, allow_partial)
        if body is not None:
            return body

        # Fall back to XML parameter tag parsing (original format)
        parameters, last_match_end = self._complete_parameters(invoke_content)

        # If allowed, try to parse a partial parameter at the end
        if allow_partial:
            # Remove an incomplete closing tag in case it was captured by param
            remaining_content = self._strip_partial_tag_suffix(
                invoke_content[last_match_end:], self.parameter_end_token
            )

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

    def _streamable_prefix(self, invoke_content: str) -> str:
        """The longest prefix of the final arguments JSON that cannot change.

        Every byte returned here is handed to the client and can never be
        recalled, so this is deliberately conservative:

        * a parameter whose closing tag has arrived is settled — include it;
        * a non-string parameter still arriving is not settled: whether its
          value ends up structural (`_partial_json_loads` succeeded) or a
          quote-escaped string (the fallback) is only decided once the whole
          value is in hand, and the two differ from the value's first byte.
          Stop before it;
        * a string parameter is safe — `json.dumps` escapes character by
          character, so its serialisation only grows by appending. Stream its
          content, minus the closing quote.
        """
        body = self._direct_json_body(invoke_content, allow_partial=True)
        if body is not None:
            return body

        parameters, last_match_end = self._complete_parameters(invoke_content)
        # Drop the closing brace: more parameters may still follow.
        prefix = json.dumps(parameters, ensure_ascii=False)[:-1]

        remaining_content = self._strip_partial_tag_suffix(
            invoke_content[last_match_end:], self.parameter_end_token
        )
        partial_match = re.search(
            self.partial_parameter_regex, remaining_content, re.DOTALL
        )
        if (
            partial_match
            and partial_match.group(2) == "true"
            and (param_value := partial_match.group(3))
        ):
            key = json.dumps(partial_match.group(1), ensure_ascii=False)
            value = json.dumps(param_value.strip(), ensure_ascii=False)
            # ", " and ": " are json.dumps' default separators; value[:-1] drops
            # the closing quote, because the content can still grow.
            prefix += f"{', ' if parameters else ''}{key}: {value[:-1]}"
        return prefix

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
                        "parameters": json.loads(func_args),
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

                # 2. Parse current parameters. While the invoke is still open,
                #    take only the part that is provably final. Diffing two
                #    partial serialisations is not sound: a parameter flips
                #    between structural and quote-escaped-string form as it
                #    arrives, so their common prefix need not be a prefix of the
                #    finished arguments.
                if is_tool_end:
                    current_params = self._parse_parameters_from_xml(invoke_content)
                else:
                    current_params = self._streamable_prefix(invoke_content)

                # 3. Calculate and send incremental arguments
                sent = self.streamed_args_for_tool[self.current_tool_id]

                argument_diff = None
                if current_params.startswith(sent):
                    argument_diff = current_params[len(sent) :] or None
                else:
                    # Bytes already streamed cannot be recalled; all that is
                    # left is to stop making it worse and be loud about it.
                    logger.warning(
                        "deepseekv32: streamed tool arguments diverged from the "
                        "parsed value (%d chars streamed, %d parsed); dropping "
                        "this delta rather than emitting invalid JSON",
                        len(sent),
                        len(current_params),
                    )

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
