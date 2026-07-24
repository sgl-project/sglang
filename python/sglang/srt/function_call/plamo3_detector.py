import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

# PLaMo3 special tokens
BEGIN_TOOL_REQUESTS = "<|plamo:begin_tool_requests:plamo|>"
END_TOOL_REQUESTS = "<|plamo:end_tool_requests:plamo|>"
BEGIN_TOOL_REQUEST = "<|plamo:begin_tool_request:plamo|>"
END_TOOL_REQUEST = "<|plamo:end_tool_request:plamo|>"
BEGIN_TOOL_NAME = "<|plamo:begin_tool_name:plamo|>"
END_TOOL_NAME = "<|plamo:end_tool_name:plamo|>"
BEGIN_TOOL_ARGUMENTS = "<|plamo:begin_tool_arguments:plamo|>"
END_TOOL_ARGUMENTS = "<|plamo:end_tool_arguments:plamo|>"
CONSTRAIN_JSON = "<|plamo:constrain|>json"
MSG = "<|plamo:msg|>"

_TOOL_MARKERS = (
    BEGIN_TOOL_REQUESTS,
    END_TOOL_REQUESTS,
    BEGIN_TOOL_REQUEST,
    END_TOOL_REQUEST,
    BEGIN_TOOL_NAME,
    END_TOOL_NAME,
    BEGIN_TOOL_ARGUMENTS,
    END_TOOL_ARGUMENTS,
)

# One non-streaming tool request block
_CALL_RE = re.compile(
    re.escape(BEGIN_TOOL_REQUEST)
    + r"\s*"
    + re.escape(BEGIN_TOOL_NAME)
    + r"(?P<name>.*?)"
    + re.escape(END_TOOL_NAME)
    + r"\s*"
    + re.escape(BEGIN_TOOL_ARGUMENTS)
    + r"(?P<args>.*?)"
    + re.escape(END_TOOL_ARGUMENTS)
    + r"\s*"
    + re.escape(END_TOOL_REQUEST),
    re.DOTALL,
)

# One streaming tool request block (no end-of-arguments or end-of-request)
_STREAM_CALL_RE = re.compile(
    re.escape(BEGIN_TOOL_REQUEST)
    + r"\s*"
    + re.escape(BEGIN_TOOL_NAME)
    + r"(?P<name>.*?)"
    + re.escape(END_TOOL_NAME)
    + r"\s*"
    + re.escape(BEGIN_TOOL_ARGUMENTS)
    + r"(?P<args>.*)",
    re.DOTALL,
)


def _arguments_json(args_block: str) -> str:
    """Return the JSON payload from an arguments block."""
    if MSG in args_block:
        # remove the prefix and keep only the JSON payload
        return args_block.split(MSG, 1)[1]
    return args_block


def _cut_at_first_marker(text: str) -> str:
    """Truncate text before the first tool marker."""
    first_marker_position = len(text)
    for marker in _TOOL_MARKERS:
        marker_position = text.find(marker)
        if marker_position != -1:
            first_marker_position = min(first_marker_position, marker_position)
    return text[:first_marker_position]


def _earliest_call_start(text: str) -> int | None:
    """Position where a tool call block could start, or None."""
    earliest_position = None
    for marker in (BEGIN_TOOL_REQUESTS, BEGIN_TOOL_REQUEST):
        marker_position = text.find(marker)
        if marker_position == -1:
            continue
        if earliest_position is None or marker_position < earliest_position:
            earliest_position = marker_position
    return earliest_position


def _strip_markers(text: str) -> str:
    """Remove complete tool markers from normal text."""
    for marker in _TOOL_MARKERS:
        text = text.replace(marker, "")
    return text


class Plamo3ToolDetector(BaseFormatDetector):  # type: ignore[misc]
    """Detector for the PLaMo3 tool-call format."""

    _buffer: str
    current_tool_name_sent: bool
    current_tool_id: int

    def __init__(self) -> None:
        super().__init__()
        self.bot_token = BEGIN_TOOL_REQUESTS
        self.eot_token = END_TOOL_REQUESTS
        self.tool_call_separator = ""
        self._streamed_raw_args = ""
        self._call_skipped = False

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text or BEGIN_TOOL_REQUEST in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing of a complete response."""
        start = _earliest_call_start(text)
        if start is None:
            return StreamingParseResult(normal_text=text)

        normal_text = text[:start]
        calls: List[ToolCallItem] = []
        try:
            for match in _CALL_RE.finditer(text, start):
                name = match.group("name").strip()
                raw_args = _cut_at_first_marker(_arguments_json(match.group("args"))).strip()

                try:
                    arguments = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError as e:
                    logger.warning(
                        "plamo3: dropping tool call %r with unparseable arguments: %s",
                        name,
                        e,
                    )
                    continue

                for call in self.parse_base_json(
                    {"name": name, "arguments": arguments}, tools
                ):
                    call.tool_index = len(calls)
                    calls.append(call)
            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error("plamo3: error in detect_and_parse: %s", e, exc_info=True)
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Parse one streaming chunk into normal text and tool-call deltas."""
        self._buffer += new_text
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []

        try:
            while True:
                buffer = self._buffer
                full_match = _CALL_RE.search(buffer)
                partial_match = None if full_match is not None else _STREAM_CALL_RE.search(buffer)
                call_match = full_match if full_match is not None else partial_match

                if call_match is not None and call_match.start() > 0:
                    normal_parts.append(_strip_markers(buffer[: call_match.start()]))
                    self._buffer = buffer[call_match.start() :]
                    continue

                if full_match is not None:
                    name = full_match.group("name").strip()
                    raw_args = _cut_at_first_marker(_arguments_json(full_match.group("args")))
                    calls.extend(self._stream_call(name, raw_args, final=True))
                    self._buffer = buffer[full_match.end() :]
                    self._finish_call()
                    continue

                if partial_match is not None:
                    args_block = partial_match.group("args")
                    if MSG in args_block:
                        args_text = _cut_at_first_marker(args_block.split(MSG, 1)[1])
                        held_marker_length = self._partial_marker_hold(args_text)
                        args_text = args_text[: len(args_text) - held_marker_length]
                        calls.extend(
                            self._stream_call(
                                partial_match.group("name").strip(), args_text, final=False
                            )
                        )
                    break

                keep_from = _earliest_call_start(buffer)
                if keep_from is None:
                    keep_from = len(buffer) - self._partial_marker_hold(buffer)
                if keep_from > 0:
                    normal_parts.append(_strip_markers(buffer[:keep_from]))
                    self._buffer = buffer[keep_from:]
                break

            return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

        except Exception as e:
            logger.error("plamo3: error in parse_streaming_increment: %s", e, exc_info=True)
            self._buffer = ""
            self._reset_inflight_call_state()
            return StreamingParseResult(
                normal_text="".join(normal_parts),
                calls=calls,
            )

    def _stream_call(self, name: str, args_text: str, *, final: bool) -> List[ToolCallItem]:
        """Emit the call name once, followed by newly received argument text."""
        if self._call_skipped:
            return []
        emitted: List[ToolCallItem] = []

        if not self.current_tool_name_sent:
            if name not in self._tool_indices:
                logger.warning("plamo3: model attempted to call undefined function %r", name)
                self._call_skipped = True
                return []

            if self.current_tool_id == -1:
                self.current_tool_id = 0

            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            self.prev_tool_call_arr[self.current_tool_id] = {"name": name, "arguments": {}}
            emitted.append(
                ToolCallItem(tool_index=self.current_tool_id, name=name, parameters="")
            )
            self.current_tool_name_sent = True

        if args_text.startswith(self._streamed_raw_args):
            argument_delta = args_text[len(self._streamed_raw_args) :]
        else:
            argument_delta = ""

        if argument_delta:
            emitted.append(
                ToolCallItem(
                    tool_index=self.current_tool_id, name=None, parameters=argument_delta
                )
            )
            self._streamed_raw_args += argument_delta
            self.streamed_args_for_tool[self.current_tool_id] += argument_delta

        if final:
            try:
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = (
                    json.loads(args_text) if args_text.strip() else {}
                )
            except json.JSONDecodeError:
                pass

        return emitted

    def _reset_inflight_call_state(self) -> None:
        """Reset state associated with the currently parsed call."""
        self.current_tool_name_sent = False
        self._streamed_raw_args = ""
        self._call_skipped = False

    def _finish_call(self) -> None:
        """Advance past a completed call and reset its state."""
        if not self._call_skipped and self.current_tool_name_sent:
            self.current_tool_id += 1
        self._reset_inflight_call_state()

    def _partial_marker_hold(self, text: str) -> int:
        """Length of a text suffix that could still grow into a tool marker."""
        return max(
            (self._ends_with_partial_token(text, marker) for marker in _TOOL_MARKERS),
            default=0,
        )

    def structure_info(self) -> _GetInfoFunc:
        """Constrained-generation wrapper matching the trained format exactly."""

        def get_info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=(
                    BEGIN_TOOL_REQUESTS
                    + BEGIN_TOOL_REQUEST
                    + BEGIN_TOOL_NAME
                    + name
                    + END_TOOL_NAME
                    + BEGIN_TOOL_ARGUMENTS
                    + CONSTRAIN_JSON
                    + MSG
                ),
                end=END_TOOL_ARGUMENTS + END_TOOL_REQUEST + END_TOOL_REQUESTS,
                trigger=BEGIN_TOOL_REQUESTS,
            )

        return get_info
