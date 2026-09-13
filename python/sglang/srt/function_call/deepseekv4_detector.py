import logging

from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector

logger = logging.getLogger(__name__)


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

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜tool_calls>"
        self.eot_token = "</｜DSML｜tool_calls>"
        self.function_calls_regex = r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>"

    def _partial_marker_length(self, text: str) -> int:
        """Length of a trailing prefix of a V4 tool-call marker."""
        markers = (self.bot_token, "<｜DSML｜invoke")
        return max(
            (
                size
                for marker in markers
                for size in range(1, min(len(text), len(marker) - 1) + 1)
                if marker.startswith(text[-size:])
            ),
            default=0,
        )

    def has_tool_call(self, text: str) -> bool:
        """Treat a trailing partial marker as protocol at generation end."""
        return super().has_tool_call(text) or self._partial_marker_length(text) > 0

    def detect_and_parse(self, text, tools) -> StreamingParseResult:
        """Never return malformed or incomplete DSML as assistant text."""
        wrapped_start = text.find(self.bot_token)
        invoke_start = text.find("<｜DSML｜invoke")
        starts = [pos for pos in (wrapped_start, invoke_start) if pos >= 0]

        if not starts:
            partial = self._partial_marker_length(text)
            if not partial:
                return StreamingParseResult(normal_text=text, calls=[])
            return StreamingParseResult(
                normal_text=text[:-partial].removesuffix("\n\n"), calls=[]
            )

        start = min(starts)
        safe_normal = text[:start].removesuffix("\n\n")
        parseable = text
        if wrapped_start < 0 and invoke_start >= 0:
            parseable = self.bot_token + text[invoke_start:] + self.eot_token

        parsed = super().detect_and_parse(parseable, tools)
        if not parsed.calls:
            return StreamingParseResult(normal_text=safe_normal, calls=[])
        if wrapped_start < 0:
            parsed.normal_text = safe_normal
        return parsed

    def parse_streaming_increment(self, new_text, tools) -> StreamingParseResult:
        """Publish V4 tool calls only after their complete DSML envelope arrives."""
        self._buffer += new_text
        current = self._buffer
        wrapped_start = current.find(self.bot_token)
        invoke_start = current.find("<｜DSML｜invoke")
        starts = [pos for pos in (wrapped_start, invoke_start) if pos >= 0]

        if not starts:
            if current and current.isspace():
                return StreamingParseResult()
            partial = self._partial_marker_length(current)
            if partial:
                normal_text = current[:-partial]
                self._buffer = current[-partial:]
            else:
                normal_text = current
                self._buffer = ""
            return StreamingParseResult(normal_text=normal_text)

        start = min(starts)
        normal_text = current[:start].removesuffix("\n\n")
        is_wrapped = start == wrapped_start
        end_token = self.eot_token if is_wrapped else self.invoke_end_token
        end = current.find(end_token, start + 1)
        if end < 0:
            self._buffer = current[start:]
            return StreamingParseResult(normal_text=normal_text)

        section_end = end + len(end_token)
        section = current[start:section_end]
        self._buffer = current[section_end:]
        parsed = self.detect_and_parse(section, tools)

        first_index = self.current_tool_id + 1
        calls = [
            ToolCallItem(
                tool_index=first_index + offset,
                name=call.name,
                parameters=call.parameters,
            )
            for offset, call in enumerate(parsed.calls)
        ]
        if calls:
            self.current_tool_id += len(calls)

        tail = self.parse_streaming_increment("", tools) if self._buffer else None
        if tail is not None:
            normal_text += tail.normal_text
            calls.extend(tail.calls)
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def finish(self, tools) -> StreamingParseResult:
        """Drop an incomplete V4 protocol envelope instead of flushing it."""
        buffered = self._buffer
        self._buffer = ""
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        if not buffered or buffered.isspace():
            return StreamingParseResult()

        starts = [
            pos
            for pos in (
                buffered.find(self.bot_token),
                buffered.find("<｜DSML｜invoke"),
            )
            if pos >= 0
        ]
        if starts:
            return StreamingParseResult(
                normal_text=buffered[: min(starts)].removesuffix("\n\n")
            )
        partial = self._partial_marker_length(buffered)
        return StreamingParseResult(
            normal_text=(buffered[:-partial] if partial else buffered)
        )

    def get_structural_tag_name(self) -> str:
        return "deepseek_v4"
