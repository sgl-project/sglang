import json
import logging
import re

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


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
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call."""
        return self.bot_token in text or "<｜DSML｜invoke" in text

    @staticmethod
    def _text_before_dsml(text: str) -> str:
        """Prose preceding the first DSML tag, with the trailing blank line the
        chat template inserts before a tool call removed."""
        idx = text.find("｜DSML｜")
        if idx == -1:
            return text
        if idx >= 2 and text[idx - 2 : idx] == "</":
            idx -= 2
        elif idx >= 1 and text[idx - 1] == "<":
            idx -= 1
        return text[:idx].removesuffix("\n\n")

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

    def _parse_parameters_from_xml(self, invoke_content: str) -> str:
        """
        Parse parameters from either XML-like format or JSON format to str.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }
        """
        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = invoke_content.strip()
        if invoke_content_stripped.startswith("{"):
            parsed = json.loads(
                invoke_content_stripped, parse_constant=_reject_json_constant
            )
            if not isinstance(parsed, dict):
                raise ValueError("DeepSeek tool arguments must be a JSON object")
            return invoke_content_stripped

        # Fall back to XML parameter tag parsing (original format)
        parameters = {}
        # Find all complete parameter matches
        param_matches = list(
            re.finditer(self.parameter_regex, invoke_content, re.DOTALL)
        )

        # Stray prose around the parameter tags is ignored, as before. Leftover
        # text holding a DSML fragment is a parameter tag that did not match
        # (unclosed, self-closing, missing `string=`), and emitting the call
        # without that argument would be worse than emitting no call.
        last_match_end = 0
        leftover = []
        for match in param_matches:
            leftover.append(invoke_content[last_match_end : match.start()])
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
                    parameters[param_name] = json.loads(
                        param_value.strip(), parse_constant=_reject_json_constant
                    )
                except (json.JSONDecodeError, ValueError):
                    parameters[param_name] = param_value.strip()
        leftover.append(invoke_content[last_match_end:])

        leftover_text = "".join(leftover)
        if "｜DSML｜" in leftover_text or (not param_matches and leftover_text.strip()):
            raise ValueError("Malformed DeepSeek tool parameter")

        return json.dumps(parameters, ensure_ascii=False)

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
                    try:
                        func_args = self._parse_parameters_from_xml(invoke_content)
                    except ValueError as e:
                        logger.warning(f"Dropping malformed DeepSeek invoke: {e}")
                        continue
                    # construct match_result for parse_base_json
                    match_result = {
                        "name": func_name,
                        "parameters": json.loads(func_args),
                    }
                    calls.extend(self.parse_base_json(match_result, tools))

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # Fail closed: keep the prose, never surface DSML as content.
            return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV32 format.
        Supports multiple consecutive invoke blocks.
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
            # Trailing whitespace may be the blank line ahead of a DSML block;
            # hold it so it is trimmed with the preamble or dropped with the block.
            stripped = current_text.rstrip()
            if not stripped:
                return StreamingParseResult()
            self._buffer = current_text[len(stripped) :]
            current_text = stripped
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
                if not is_tool_end:
                    break

                # Initialize state on the first complete invoke, malformed or
                # not: the preamble is released once and the trailing DSML is
                # withheld by finish() either way.
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = []
                    call_start = invoke_match.start()
                    bot_pos = current_text.rfind(self.bot_token, 0, call_start)
                    if bot_pos != -1:
                        call_start = bot_pos
                    # Same trailing-newline trim as detect_and_parse, so both agree.
                    preamble = current_text[:call_start].removesuffix("\n\n")

                try:
                    current_params = self._parse_parameters_from_xml(invoke_content)
                except ValueError as e:
                    # Fail closed: drop this invoke, keep going for the next one.
                    logger.warning(f"Dropping malformed DeepSeek invoke: {e}")
                    self._buffer = current_text[invoke_match.end() :]
                    current_text = self._buffer
                    continue

                # Ensure arrays are large enough for current tool
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                all_calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=func_name,
                        parameters=current_params,
                    )
                )
                self.streamed_args_for_tool[self.current_tool_id] = current_params

                # Update the stored arguments
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": func_name,
                    "arguments": current_params,
                }

                # Remove the completed tool call from buffer and check for another.
                self._buffer = current_text[invoke_match.end() :]
                current_text = self._buffer
                self.current_tool_id += 1

            # No more invoke blocks found
            return StreamingParseResult(normal_text=preamble, calls=all_calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            # Fail closed: drop the buffer, keep any prose ahead of the DSML,
            # and never surface DSML as content.
            self._buffer = ""
            if self.current_tool_id == -1:
                preamble = self._text_before_dsml(current_text)
            return StreamingParseResult(normal_text=preamble, calls=all_calls)

    def finish(self, tools: list[Tool]) -> StreamingParseResult:
        if not self._buffer:
            return StreamingParseResult()

        buffered = self._buffer
        self._buffer = ""
        if self.current_tool_id != -1:
            return StreamingParseResult()

        return StreamingParseResult(normal_text=self._text_before_dsml(buffered))

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger="<｜DSML｜invoke",
        )

    def get_structural_tag_name(self) -> str:
        return "deepseek_v3_2"
