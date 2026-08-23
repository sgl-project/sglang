import json
import logging
import re
from typing import List, Literal, Optional, Union

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

try:
    from xgrammar.structural_tag import (
        ConstStringFormat,
        JSONSchemaFormat,
        OrFormat,
        PlusFormat,
        SequenceFormat,
        StructuralTag,
        TagFormat,
        TriggeredTagsFormat,
    )
except ImportError:  # pragma: no cover - xgrammar without rich structural-tag formats
    StructuralTag = None  # type: ignore[assignment,misc]
    ConstStringFormat = JSONSchemaFormat = None  # type: ignore[assignment,misc]
    OrFormat = PlusFormat = None  # type: ignore[assignment,misc]
    SequenceFormat = TagFormat = TriggeredTagsFormat = None  # type: ignore[assignment,misc]

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

# One request-body character without crossing into the next request
_REQUEST_CHAR = r"(?:(?!" + re.escape(BEGIN_TOOL_REQUEST) + r").)"

# One non-streaming tool request block
_CALL_RE = re.compile(
    re.escape(BEGIN_TOOL_REQUEST)
    + r"\s*"
    + re.escape(BEGIN_TOOL_NAME)
    + rf"(?P<name>{_REQUEST_CHAR}*?)"
    + re.escape(END_TOOL_NAME)
    + r"\s*"
    + re.escape(BEGIN_TOOL_ARGUMENTS)
    + rf"(?P<args>{_REQUEST_CHAR}*?)"
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
    + rf"(?P<name>{_REQUEST_CHAR}*?)"
    + re.escape(END_TOOL_NAME)
    + r"\s*"
    + re.escape(BEGIN_TOOL_ARGUMENTS)
    + rf"(?P<args>{_REQUEST_CHAR}*)",
    re.DOTALL,
)

# Streaming tool request header, complete as soon as the function name is delimited
_STREAM_NAME_RE = re.compile(
    re.escape(BEGIN_TOOL_REQUEST)
    + r"\s*"
    + re.escape(BEGIN_TOOL_NAME)
    + rf"(?P<name>{_REQUEST_CHAR}*?)"
    + re.escape(END_TOOL_NAME),
    re.DOTALL,
)


def _extract_argument_payload(args_block: str) -> str:
    """Return the argument payload from an arguments block."""
    if MSG in args_block:
        # remove the prefix and keep only the argument payload
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


def _tool_requests_start(text: str) -> Optional[int]:
    """Position where an outer tool-requests block starts, or None."""
    position = text.find(BEGIN_TOOL_REQUESTS)
    return position if position != -1 else None


def _tool_request_start(text: str) -> Optional[int]:
    """Position where an inner tool-request block starts, or None."""
    position = text.find(BEGIN_TOOL_REQUEST)
    return position if position != -1 else None


def _strip_markers(text: str) -> str:
    """Remove complete tool markers from normal text."""
    for marker in _TOOL_MARKERS:
        text = text.replace(marker, "")
    return text


def _tool_request_prefix(name: str) -> str:
    """Return the fixed prefix before a tool request's JSON arguments."""
    return (
        BEGIN_TOOL_REQUEST
        + BEGIN_TOOL_NAME
        + name
        + END_TOOL_NAME
        + BEGIN_TOOL_ARGUMENTS
        + CONSTRAIN_JSON
        + MSG
    )


def _should_emit_tool(name: str, tool_indices: dict) -> bool:
    """Return whether a known or configured unknown tool should be emitted."""
    if name in tool_indices:
        return True
    from sglang.srt.environ import envs

    logger.warning("plamo3: model attempted to call undefined function %r", name)
    return bool(envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get())


class Plamo3ToolDetector(BaseFormatDetector):
    """Detector for the PLaMo3 tool-call format.

    Trained format::

        <|plamo:begin_tool_requests:plamo|>
          <|plamo:begin_tool_request:plamo|>
            <|plamo:begin_tool_name:plamo|>{name}<|plamo:end_tool_name:plamo|>
            <|plamo:begin_tool_arguments:plamo|><|plamo:constrain|>json<|plamo:msg|>{json}<|plamo:end_tool_arguments:plamo|>
          <|plamo:end_tool_request:plamo|>
          ... more inner requests ...
        <|plamo:end_tool_requests:plamo|>

    The outer wrapper (``begin_tool_requests``/``end_tool_requests``) appears once
    and may contain one or more inner request blocks. ``get_structural_tag`` builds
    a constraint that keeps the outer wrapper open for all parallel calls, fixing
    the case where only the first inner call was guided.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bot_token = BEGIN_TOOL_REQUESTS
        self.eot_token = END_TOOL_REQUESTS
        self.tool_call_separator = ""
        self._streamed_raw_args = ""
        self._call_skipped = False
        self._tool_requests_started = False

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing of a complete response."""
        start = _tool_requests_start(text)
        if start is None:
            return StreamingParseResult(normal_text=text)

        normal_text = text[:start]
        calls: List[ToolCallItem] = []
        try:
            content_start = start + len(BEGIN_TOOL_REQUESTS)
            outer_end = text.find(END_TOOL_REQUESTS, content_start)
            content_end = outer_end if outer_end != -1 else len(text)
            last_end = content_start
            for match in _CALL_RE.finditer(text, content_start, content_end):
                # Surface any malformed text between calls as normal text so that
                # it stays observable (e.g. for RL training) instead of vanishing.
                if match.start() > last_end:
                    gap = _strip_markers(text[last_end : match.start()])
                    if gap.strip():
                        normal_text += gap
                name = match.group("name").strip()
                raw_args = _cut_at_first_marker(
                    _extract_argument_payload(match.group("args"))
                ).strip()
                # Conservative upstream policy: drop calls whose arguments are
                # not valid JSON rather than forwarding malformed payloads.
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "plamo3: dropping tool call %r with unparsable "
                        "arguments: %s",
                        name,
                        e,
                    )
                    last_end = match.end()
                    continue
                for call in self.parse_base_json(
                    {"name": name, "arguments": arguments}, tools
                ):
                    call.tool_index = len(calls)
                    calls.append(call)
                last_end = match.end()
            if last_end < content_end:
                gap = _strip_markers(text[last_end:content_end])
                if gap.strip():
                    normal_text += gap
            if outer_end != -1:
                normal_text += _strip_markers(
                    text[outer_end + len(END_TOOL_REQUESTS) :]
                )
            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error("plamo3: error in detect_and_parse: %s", e, exc_info=True)
            return StreamingParseResult(normal_text=text)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Parse one streaming chunk into normal text and tool-call deltas."""
        self._buffer += new_text
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []

        if not self._tool_requests_started:
            start = _tool_requests_start(self._buffer)
            if start is None:
                # Could still grow into the outer marker, so hold back the
                # matching suffix and emit the rest as normal text.
                held_marker_length = self._ends_with_partial_token(
                    self._buffer, BEGIN_TOOL_REQUESTS
                )
                emit_length = len(self._buffer) - held_marker_length
                normal_text = self._buffer[:emit_length]
                self._buffer = self._buffer[emit_length:]
                return StreamingParseResult(normal_text=normal_text)

            if start > 0:
                normal_parts.append(self._buffer[:start])
            # Consume the outer marker; the model will not emit it again.
            self._buffer = self._buffer[start + len(BEGIN_TOOL_REQUESTS) :]
            self._tool_requests_started = True

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        try:
            while True:
                buffer = self._buffer
                full_match = _CALL_RE.search(buffer)
                partial_match = (
                    None if full_match is not None else _STREAM_CALL_RE.search(buffer)
                )
                name_match = (
                    None
                    if (full_match or partial_match)
                    else _STREAM_NAME_RE.search(buffer)
                )
                call_match = full_match or partial_match or name_match

                if call_match is not None and call_match.start() > 0:
                    if self.current_tool_name_sent or self._call_skipped:
                        # Stop tracking the earlier incomplete or skipped request.
                        self._abandon_call()
                    else:
                        # Emit normal text, excluding any incomplete request
                        # that preceded the next recognizable one.
                        prefix = buffer[: call_match.start()]
                        incomplete_request_start = _tool_request_start(prefix)
                        normal_prefix = (
                            prefix
                            if incomplete_request_start is None
                            else prefix[:incomplete_request_start]
                        )
                        normal_parts.append(_strip_markers(normal_prefix))
                    # Move the buffer to the next recognizable request and
                    # parse it again.
                    self._buffer = buffer[call_match.start() :]
                    continue

                if full_match is not None:
                    name = full_match.group("name").strip()
                    raw_args = _cut_at_first_marker(
                        _extract_argument_payload(full_match.group("args"))
                    )
                    calls.extend(self._stream_call(name, raw_args, final=True))
                    self._buffer = buffer[full_match.end() :]
                    self._finish_call()
                    continue

                if partial_match is not None:
                    args_block = partial_match.group("args")
                    if MSG in args_block:
                        args_text = _cut_at_first_marker(
                            _extract_argument_payload(args_block)
                        )
                        held_marker_length = self._partial_marker_hold(args_text)
                        args_text = args_text[: len(args_text) - held_marker_length]
                        calls.extend(
                            self._stream_call(
                                partial_match.group("name").strip(),
                                args_text,
                                final=False,
                            )
                        )
                    break

                if name_match is not None:
                    if not self.current_tool_name_sent:
                        calls.extend(
                            self._stream_call(
                                name_match.group("name").strip(), "", final=False
                            )
                        )
                    break

                keep_from = _tool_request_start(buffer)
                if keep_from is None:
                    keep_from = len(buffer) - self._partial_marker_hold(buffer)
                if keep_from > 0:
                    normal_parts.append(_strip_markers(buffer[:keep_from]))
                    self._buffer = buffer[keep_from:]
                break

            return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

        except Exception as e:
            logger.error(
                "plamo3: error in parse_streaming_increment: %s", e, exc_info=True
            )
            self._buffer = ""
            self._abandon_call()
            return StreamingParseResult(
                normal_text="".join(normal_parts),
                calls=calls,
            )

    def _stream_call(
        self, name: str, args_text: str, *, final: bool
    ) -> List[ToolCallItem]:
        """Emit the call name once, followed by newly received argument text."""
        if self._call_skipped:
            return []
        args_text = args_text.strip()
        emitted: List[ToolCallItem] = []

        if not self.current_tool_name_sent:
            if not _should_emit_tool(name, self._tool_indices):
                self._call_skipped = True
                return []

            if self.current_tool_id == -1:
                self.current_tool_id = 0

            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": name,
                "arguments": {},
            }
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
                    tool_index=self.current_tool_id,
                    name=None,
                    parameters=argument_delta,
                )
            )
            self._streamed_raw_args += argument_delta
            self.streamed_args_for_tool[self.current_tool_id] += argument_delta

        if final:
            # Store the raw argument text so the serving layer can compute any
            # remaining (un-streamed) arguments by prefix-matching against
            # ``streamed_args_for_tool``. Validation/discard of malformed JSON is
            # handled in the non-streaming path; here we preserve what was
            # already streamed to the client.
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = args_text

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

    def _abandon_call(self) -> None:
        """Record partial arguments and reset an incomplete or skipped request."""
        if self.current_tool_name_sent:
            name = self.prev_tool_call_arr[self.current_tool_id].get("name")
            logger.warning("plamo3: tool request %r abandoned before completion", name)
            self.prev_tool_call_arr[self.current_tool_id][
                "arguments"
            ] = self._streamed_raw_args
        self._finish_call()

    def _partial_marker_hold(self, text: str) -> int:
        """Length of a text suffix that could still grow into a tool marker."""
        return max(
            (self._ends_with_partial_token(text, marker) for marker in _TOOL_MARKERS),
            default=0,
        )

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Flush text held for a possible marker at the end of the stream.

        Once a tool section has started, an incomplete request cannot be
        recovered into a valid tool call. Match the other native-format
        detectors by dropping that protocol fragment instead of exposing it as
        normal text or fabricating a completed call.
        """
        if self._tool_requests_started:
            if self.current_tool_name_sent:
                self._abandon_call()
            elif self._buffer:
                logger.warning(
                    "plamo3: tool requests ended with no complete tool call; "
                    "dropping %d buffered chars",
                    len(self._buffer),
                )
            self._buffer = ""
            self._reset_inflight_call_state()
            return StreamingParseResult()

        held_marker_length = self._partial_marker_hold(self._buffer)
        normal_text = (
            self._buffer[: len(self._buffer) - held_marker_length]
            if held_marker_length
            else self._buffer
        )
        self._buffer = ""
        return StreamingParseResult(normal_text=normal_text)

    # ------------------------------------------------------------------
    # Constrained generation
    # ------------------------------------------------------------------
    def structure_info(self) -> _GetInfoFunc:
        """Return a legacy constraint matching one complete outer wrapper."""

        def get_info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=BEGIN_TOOL_REQUESTS + _tool_request_prefix(name),
                end=END_TOOL_ARGUMENTS + END_TOOL_REQUEST + END_TOOL_REQUESTS,
                trigger=BEGIN_TOOL_REQUESTS,
            )

        return get_info

    def _build_structural_tag(
        self,
        tools: List[Tool],
        *,
        at_least_one: bool,
        parallel_tool_calls: bool,
    ) -> Optional[StructuralTag]:
        """Build a structural tag that keeps the outer wrapper open for all calls.

        The outer ``begin_tool_requests``/``end_tool_requests`` pair wraps a
        ``PlusFormat`` of per-tool inner blocks, so every parallel call (not just
        the first) is guided to a valid JSON argument payload.
        """
        if StructuralTag is None or not tools:
            return None

        inner_blocks = []
        for tool in tools:
            function = tool.function
            name = function.name
            if not name:
                continue
            schema = function.parameters if function.strict else {}
            inner_blocks.append(
                SequenceFormat(
                    elements=[
                        ConstStringFormat(value=_tool_request_prefix(name)),
                        JSONSchemaFormat(json_schema=schema or {}),
                        ConstStringFormat(value=END_TOOL_ARGUMENTS + END_TOOL_REQUEST),
                    ]
                )
            )
        if not inner_blocks:
            return None

        content = (
            inner_blocks[0]
            if len(inner_blocks) == 1
            else OrFormat(elements=inner_blocks)
        )
        outer = TagFormat(
            begin=BEGIN_TOOL_REQUESTS,
            content=PlusFormat(content=content) if parallel_tool_calls else content,
            end=END_TOOL_REQUESTS,
        )
        return StructuralTag(
            format=TriggeredTagsFormat(
                triggers=[BEGIN_TOOL_REQUESTS],
                tags=[outer],
                at_least_one=at_least_one,
                # PLaMo3 has exactly one outer wrapper. Parallelism is
                # represented by repeated inner requests inside that wrapper.
                stop_after_first=True,
            )
        )

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        """Model-native structural tag for required / strict-auto tool calls."""
        is_required = tool_choice == "required" or isinstance(tool_choice, ToolChoice)
        if isinstance(tool_choice, ToolChoice):
            selected_name = tool_choice.function.name
            tools = [
                tool for tool in tools or [] if tool.function.name == selected_name
            ]
        return self._build_structural_tag(
            tools or [],
            at_least_one=is_required,
            parallel_tool_calls=parallel_tool_calls,
        )

    def get_auto_tool_call_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        """Structural tag for ``tool_choice="auto"`` (non-strict) tool calls."""
        return self._build_structural_tag(
            tools or [],
            at_least_one=False,
            parallel_tool_calls=parallel_tool_calls,
        )
