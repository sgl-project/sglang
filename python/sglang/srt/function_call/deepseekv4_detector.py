import json
import logging
import re
from html.parser import HTMLParser

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector

logger = logging.getLogger(__name__)

_HEREDOC_START = re.compile(
    r"""(?<!<)<<(?!<)(-?)[ \t]*(?:'([^'\n]+)'|"([^"\n]+)"|([A-Za-z_][A-Za-z0-9_]*))"""
)


def _mask_literals(text: str, *, heredocs: bool = False) -> str:
    """Keep offsets while excluding Markdown/code literals from protocol detection."""
    masked = list(text)
    fence = quote = ""
    ticks = 0
    escaped = False
    paired_quotes = {'"': '"', "'": "'", "“": "”", "‘": "’"}
    pending_heredocs: list[tuple[str, bool]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if pending_heredocs:
            delimiter, strip_tabs = pending_heredocs[0]
            end_line = line.rstrip("\r\n")
            if (end_line.lstrip("\t") if strip_tabs else end_line) == delimiter:
                pending_heredocs.pop(0)
            for index, char in enumerate(line):
                if char not in "\r\n":
                    masked[offset + index] = " "
            offset += len(line)
            continue
        boundary = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        protected = bool(fence)
        if boundary and not quote and not ticks:
            token, suffix = boundary.groups()
            if not fence:
                fence = token
            elif (
                token[0] == fence[0] and len(token) >= len(fence) and not suffix.strip()
            ):
                fence = ""
            protected = True
        if not quote and not ticks and line.lstrip().startswith(">"):
            protected = True
        if protected:
            for index, char in enumerate(line):
                if char not in "\r\n":
                    masked[offset + index] = " "
            offset += len(line)
            continue
        index = 0
        while index < len(line):
            char = line[index]
            if quote:
                if char not in "\r\n":
                    masked[offset + index] = " "
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = ""
            elif char == "`":
                end = index + 1
                while end < len(line) and line[end] == "`":
                    end += 1
                width = end - index
                if not ticks:
                    ticks = width
                elif ticks == width:
                    ticks = 0
                masked[offset + index : offset + end] = [" "] * width
                index = end
                continue
            elif ticks:
                if char not in "\r\n":
                    masked[offset + index] = " "
            elif (
                heredocs
                and char == "#"
                and (
                    index == 0 or line[index - 1].isspace() or line[index - 1] in ";|&("
                )
            ):
                for position in range(index, len(line)):
                    if line[position] not in "\r\n":
                        masked[offset + position] = " "
                break
            elif (
                heredocs
                and char == "<"
                and (opener := _HEREDOC_START.match(line, index))
            ):
                delimiter = next(
                    value for value in opener.groups()[1:] if value is not None
                )
                pending_heredocs.append((delimiter, opener[1] == "-"))
                width = opener.end() - index
                masked[offset + index : offset + opener.end()] = [" "] * width
                index = opener.end()
                continue
            elif char in paired_quotes and (
                char in "“‘"
                or offset + index == 0
                or not (
                    text[offset + index - 1].isalnum()
                    or text[offset + index - 1] == "_"
                )
            ):
                quote = paired_quotes[char]
                masked[offset + index] = " "
            index += 1
        offset += len(line)
    return "".join(masked)


def _validate_string_parameter(name: str, value: str, *, complete: bool) -> None:
    prefix = "</｜DSML｜parameter"
    if prefix not in value:
        return
    visible = _mask_literals(value, heredocs=True)
    for match in re.finditer(re.escape(prefix), visible):
        previous = match.start() - 1
        while previous >= 0 and value[previous] == "\\":
            previous -= 1
        if (match.start() - previous - 1) % 2:
            continue
        if match.end() == len(value):
            if not complete:
                continue
        elif (
            value[match.end()] == ">"
            or value[match.end()].isalnum()
            or value[match.end()] == "_"
        ):
            continue
        raise ValueError(
            f"Malformed DSML parameter terminator in string parameter {name!r}"
        )


class _BareControlValidator(HTMLParser):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.lines = text.splitlines()
        self.depth = {
            tag: 0 for tag in ("parameter", "invoke", "tool_calls", "function_calls")
        }

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.depth:
            self.depth[tag] += 1

    def handle_endtag(self, tag: str) -> None:
        if tag not in self.depth:
            return
        if self.depth[tag]:
            self.depth[tag] -= 1
            return
        line, _ = self.getpos()
        if re.fullmatch(
            r"(?:</?(?:parameter|invoke|tool_calls|function_calls)\b[^<>]*>\s*)+",
            self.lines[line - 1].strip(),
        ):
            raise ValueError(f"Orphan DeepSeek V4 protocol ending: </{tag}>")


class DeepSeekV4Detector(DeepSeekV32Detector):
    """
    Detector for DeepSeek V4 model function call format.

    The DeepSeek V4 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Examples:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V4 format specification
    """

    def __init__(self, strict_output: bool | None = None):
        super().__init__()
        self.bot_token = "<｜DSML｜tool_calls>"
        self.eot_token = "</｜DSML｜tool_calls>"
        self.function_calls_regex = r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>"
        self.strict_output = (
            envs.SGLANG_DSV4_STRICT_TOOL_OUTPUT.get()
            if strict_output is None
            else strict_output
        )
        self._quote_history: list[str] = []
        self._normal_chunks: list[str] = []
        self._active_invoke_start = 0

    def _find_invoke(self, text: str) -> re.Match | None:
        pattern = re.compile(self.invoke_regex, re.DOTALL)
        if self.current_tool_name_sent:
            return pattern.match(text, self._active_invoke_start)
        history = "".join(self._quote_history)
        masked = _mask_literals(history + text)[len(history) :]
        for marker in re.finditer(r"<｜DSML｜invoke\b", masked):
            match = pattern.match(text, marker.start())
            if match is not None:
                self._active_invoke_start = match.start()
                return match
        return None

    def _extract_preamble(self, text: str, invoke_start: int) -> str:
        history = "".join(self._quote_history)
        prefix = _mask_literals(history + text[:invoke_start])[len(history) :]
        marker = re.search(r"</?｜DSML｜", prefix)
        # A parsed invoke establishes the protocol boundary even when its outer
        # wrapper is damaged. Such wrapper bytes are not assistant prose.
        start = marker.start() if marker else invoke_start
        self._quote_history.clear()
        return text[:start].removesuffix("\n\n")

    def _raise_parse_error(self, error: Exception) -> None:
        if self.strict_output:
            raise ValueError(
                f"Failed to parse DeepSeek V4 tool output: {error}"
            ) from error

    def _parse_parameters_from_xml(
        self, invoke_content: str, allow_partial: bool = False
    ) -> str:
        if self.strict_output:
            if invoke_content.lstrip().startswith("{"):
                # Keep direct-JSON arguments private until their strings can be
                # validated; do not repair or reserialize partial JSON.
                if allow_partial:
                    return ""
                parameters = json.loads(invoke_content)
                for name, value in parameters.items():
                    if isinstance(value, str):
                        _validate_string_parameter(name, value, complete=True)
                return super()._parse_parameters_from_xml(invoke_content, False)
            last_match_end = 0
            for match in re.finditer(self.parameter_regex, invoke_content, re.DOTALL):
                last_match_end = match.end()
                if match.group(2) == "true":
                    _validate_string_parameter(
                        match.group(1), match.group(3), complete=True
                    )
                else:
                    try:
                        json.loads(match.group(3).strip())
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON in DSML non-string parameter {match.group(1)!r}"
                        ) from error
            if allow_partial:
                partial = re.search(
                    self.partial_parameter_regex,
                    invoke_content[last_match_end:],
                    re.DOTALL,
                )
                if partial and partial.group(2) == "true":
                    _validate_string_parameter(
                        partial.group(1), partial.group(3), complete=False
                    )
        return super()._parse_parameters_from_xml(invoke_content, allow_partial)

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        result = super().parse_streaming_increment(new_text, tools)
        if result.calls:
            self._quote_history.clear()
        elif result.normal_text:
            self._quote_history.append(result.normal_text)
        if self.strict_output and result.normal_text:
            self._normal_chunks.append(result.normal_text)
            result.normal_text = ""
        return result

    def finish(self, tools: list[Tool]) -> StreamingParseResult:
        super().finish(tools)
        if self.current_tool_name_sent:
            raise ValueError("Incomplete DSML invoke at end of stream")
        remaining = self._buffer
        if self.current_tool_id >= 0:
            while remaining.lstrip().startswith(self.eot_token):
                remaining = remaining.lstrip()[len(self.eot_token) :]
            if not remaining.strip():
                remaining = ""
        history = "".join(self._quote_history)
        masked = _mask_literals(history + remaining)[len(history) :]
        if re.search(r"</?｜DSML｜", masked):
            raise ValueError("Incomplete or malformed DSML tool block at end of stream")
        if self.strict_output:
            normal = "".join(self._normal_chunks) + remaining
            masked = _mask_literals(normal)
            if re.search(
                r"</(?:parameter|invoke|tool_calls|function_calls)>", masked
            ) and not re.fullmatch(
                r"\s*</(?:parameter|invoke|tool_calls|function_calls)>\s*", masked
            ):
                validator = _BareControlValidator(masked)
                try:
                    validator.feed(masked)
                    validator.close()
                except AssertionError as error:
                    raise ValueError("Invalid markup in DeepSeek V4 output") from error
            self._normal_chunks.clear()
        else:
            normal = remaining
        self._buffer = ""
        self._quote_history.clear()
        return StreamingParseResult(normal_text=normal)

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        detector = type(self)(strict_output=self.strict_output)
        parsed = detector.parse_streaming_increment(text, tools)
        tail = detector.finish(tools)
        accumulated: dict[int, ToolCallItem] = {}
        for item in parsed.calls + tail.calls:
            call = accumulated.setdefault(
                item.tool_index,
                ToolCallItem(tool_index=item.tool_index, parameters=""),
            )
            if item.name:
                call.name = item.name
            call.parameters += item.parameters
        calls = []
        for call in accumulated.values():
            calls.extend(
                self.parse_base_json(
                    {"name": call.name, "parameters": json.loads(call.parameters)},
                    tools,
                )
            )
        return StreamingParseResult(
            normal_text=parsed.normal_text + tail.normal_text, calls=calls
        )

    def get_structural_tag_name(self) -> str:
        return "deepseek_v4"
