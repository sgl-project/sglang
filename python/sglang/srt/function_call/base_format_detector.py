import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import orjson
from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

try:
    from xgrammar import StructuralTag, get_model_structural_tag
except ImportError:
    StructuralTag = Any
    get_model_structural_tag = None

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.environ import envs
from sglang.srt.function_call.compatibility import (
    CompatibilityContext,
    CompatibilityEvent,
    CompatibilityRecord,
    synthesize_json_close,
)
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    _closed_top_level_json_end,
    _find_common_prefix,
    _is_complete_json,
    _partial_json_loads,
)

logger = logging.getLogger(__name__)


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(self):
        # Streaming state management
        # Buffer for accumulating incomplete patterns that arrive across multiple streaming chunks
        self._buffer = ""
        # Stores complete tool call info (name and arguments) for each tool being parsed.
        # Used by serving layer for completion handling when streaming ends.
        # Format: [{"name": str, "arguments": dict}, ...]
        self.prev_tool_call_arr: List[Dict] = []
        # Index of currently streaming tool call. Starts at -1 (no active tool),
        # increments as each tool completes. Tracks which tool's arguments are streaming.
        self.current_tool_id: int = -1
        # Flag for whether current tool's name has been sent to client.
        # Tool names sent first with empty parameters, then arguments stream incrementally.
        self.current_tool_name_sent: bool = False
        # Tracks raw JSON string content streamed to client for each tool's arguments.
        # Critical for serving layer to calculate remaining content when streaming ends.
        # Each index corresponds to a tool_id. Example: ['{"location": "San Francisco"', '{"temp": 72']
        self.streamed_args_for_tool: List[str] = []
        # True after the detector has accepted the outer tool-call marker and
        # remains inside the model's multi-call sequence. This lets separator
        # continuations survive dropped malformed/unknown entries, including
        # when the separator and next call arrive in different chunks.
        self._in_tool_call_sequence: bool = False

        # Token configuration (override in subclasses)
        self.bot_token = ""
        self.eot_token = ""
        self.tool_call_separator = ", "

        # Compatibility-mode policy and audit trail.
        self.compatibility = CompatibilityContext()

    @property
    def compatibility_records(self) -> List[CompatibilityRecord]:
        """Tolerances applied to this request so far."""
        return list(self.compatibility.records)

    def fail_open_stream(
        self, error: Exception, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Recover after ``parse_streaming_increment`` raised exception.

        Called only by ``FunctionCallParser`` — the single recovery boundary —
        which then passes the rest of the stream through as normal text. The
        base behavior flushes ``_buffer`` (the accumulated text not yet
        surfaced; implementations append the chunk to it on entry, so it
        contains ``new_text``) as content and, when a tool call is mid-stream,
        closes its already-streamed JSON arguments. Detectors with richer
        internal state may override to also drain output parsed before the
        error.
        """
        # NOTE: when a call is mid-stream, ``_buffer`` still contains
        # argument text already emitted as tool-call deltas, so the client
        # may see it twice (closed tool-call JSON plus raw text). This is the
        # deliberate no-text-loss trade-off: the buffer cannot be split
        # reliably after an arbitrary failure.
        buffered = self._buffer
        self._buffer = ""
        return StreamingParseResult(
            normal_text=buffered if buffered else new_text,
            calls=self._close_mid_stream_call(),
        )

    def fail_open_nonstream(
        self, error: Exception, full_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Recover after ``detect_and_parse`` raised (compatibility scope 4).

        Called only by ``FunctionCallParser``. The base behavior surfaces the
        full original text as content; detectors that can salvage complete
        calls parsed before the error may override.
        """
        return StreamingParseResult(normal_text=full_text)

    def _close_mid_stream_call(self) -> List[ToolCallItem]:
        """Close the JSON of a mid-stream call so sent fragments stay valid.

        Appends a synthesized closing fragment (see
        ``compatibility.synthesize_json_close``) when the current tool call's
        streamed arguments are incomplete JSON, and records the closed
        arguments in ``prev_tool_call_arr`` so the serving layer's
        ``_check_for_unstreamed_tool_args`` does not double-send.
        """
        calls: List[ToolCallItem] = []
        tid = self.current_tool_id
        if not (0 <= tid < len(self.streamed_args_for_tool)):
            return calls
        streamed = self.streamed_args_for_tool[tid]
        if not streamed:
            return calls
        args = None
        try:
            args = json.loads(streamed)
        except json.JSONDecodeError:
            closing = synthesize_json_close(streamed)
            if closing is not None:
                if closing:
                    calls.append(ToolCallItem(tool_index=tid, parameters=closing))
                    self.streamed_args_for_tool[tid] = streamed + closing
                try:
                    args = json.loads(self.streamed_args_for_tool[tid])
                except json.JSONDecodeError:
                    args = None
        if tid < len(self.prev_tool_call_arr):
            self.prev_tool_call_arr[tid]["arguments"] = args if args is not None else {}
        return calls

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        """
        Get a mapping of tool names to their indices in the tools list.

        This utility method creates a dictionary mapping function names to their
        indices in the tools list, which is commonly needed for tool validation
        and ToolCallItem creation.

        Args:
            tools: List of available tools

        Returns:
            Dictionary mapping tool names to their indices
        """
        return {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }

    def _skip_unknown_tool(self, name: Optional[str]) -> bool:
        """Drop gate for a tool name absent from the request's tools.

        Returns True (and records ``UNKNOWN_TOOL_DROPPED``) when the call
        should be dropped; False when ``SGLANG_FORWARD_UNKNOWN_TOOLS`` says
        to forward it anyway.
        """
        logger.warning("Model attempted to call undefined function: %s", name)
        if envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
            return False
        self.compatibility.note(
            CompatibilityEvent.UNKNOWN_TOOL_DROPPED, detail=repr(name)
        )
        return True

    def _consume_leading_eot_token(self, text: str) -> str:
        if self.eot_token and text.startswith(self.eot_token):
            self._in_tool_call_sequence = False
            return text[len(self.eot_token) :]
        return text

    def _starts_with_tool_call_separator(self, text: str) -> bool:
        return bool(
            self.tool_call_separator and text.startswith(self.tool_call_separator)
        )

    def _has_tool_call_sequence_context(self) -> bool:
        """Whether a leading separator can belong to the active tool-call sequence."""
        return self._in_tool_call_sequence or self.current_tool_name_sent

    def _should_retry_buffer(self, previous_text: str) -> bool:
        return bool(self._buffer and self._buffer != previous_text)

    def _retry_streaming_tail(
        self,
        previous_text: str,
        calls: List[ToolCallItem],
        tools: List[Tool],
    ) -> StreamingParseResult:
        if not self._should_retry_buffer(previous_text):
            return StreamingParseResult(calls=calls)
        tail_result = self.parse_streaming_increment("", tools)
        return StreamingParseResult(
            normal_text=tail_result.normal_text,
            calls=[*calls, *tail_result.calls],
        )

    def _drop_completed_malformed_entry(
        self, current_text: str, start_idx: int, end_idx: int, detail: str
    ) -> bool:
        self.compatibility.note(
            CompatibilityEvent.MALFORMED_JSON_DROPPED,
            detail=detail[:80],
        )
        self._buffer = self._consume_leading_eot_token(
            current_text[start_idx + end_idx :]
        )
        self.current_tool_name_sent = False
        return self._should_retry_buffer(current_text)

    def _close_started_malformed_call(self) -> List[ToolCallItem]:
        """Finish a call whose name was already streamed before its body failed."""
        if not self.current_tool_name_sent:
            return []

        tid = self.current_tool_id
        self.current_tool_name_sent = False
        if tid < 0:
            return []

        while len(self.streamed_args_for_tool) <= tid:
            self.streamed_args_for_tool.append("")

        calls = self._close_mid_stream_call()
        if not self.streamed_args_for_tool[tid]:
            calls.append(ToolCallItem(tool_index=tid, parameters="{}"))
            self.streamed_args_for_tool[tid] = "{}"
            while len(self.prev_tool_call_arr) <= tid:
                self.prev_tool_call_arr.append({})
            self.prev_tool_call_arr[tid]["arguments"] = {}

        self.current_tool_id = tid + 1
        return calls

    def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = self._get_tool_indices(tools)
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            if not isinstance(act, dict):
                self.compatibility.note(
                    CompatibilityEvent.MALFORMED_JSON_DROPPED,
                    detail=repr(act)[:80],
                )
                continue

            name = act.get("name")
            if not name:
                self.compatibility.note(
                    CompatibilityEvent.MALFORMED_JSON_DROPPED,
                    detail=repr(act)[:80],
                )
                continue

            if not (name and name in tool_indices):
                if self._skip_unknown_tool(name):
                    continue

            results.append(
                ToolCallItem(
                    tool_index=tool_indices.get(name, -1),
                    name=name,
                    parameters=json.dumps(
                        act.get("parameters") or act.get("arguments", {}),
                        ensure_ascii=False,
                    ),
                )
            )

        return results

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        action = orjson.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def _ends_with_partial_token(self, buffer: str, bot_token: str) -> int:
        """
        Check if buffer ends with a partial bot_token.
        Return the length of the partial bot_token.

        For some format, the bot_token is not a token in model's vocabulary, such as
        `[TOOL_CALLS] [` in Mistral.
        """
        for i in range(1, min(len(buffer) + 1, len(bot_token))):
            if bot_token.startswith(buffer[-i:]):
                return i
        return 0

    def _hold_back_partial_tokens(
        self, text: str, tokens: Iterable[str]
    ) -> Tuple[str, str]:
        """Split ``text`` into ``(flushable, holdback)``.

        The holdback is the longest suffix that could still grow into one of
        ``tokens``. Custom streaming implementations use this before flushing
        buffered text as content, so a marker split across chunk boundaries
        is neither leaked to the client nor lost to the parser.
        """
        partial = max(self._ends_with_partial_token(text, token) for token in tokens)
        if partial:
            return text[:-partial], text[-partial:]
        return text, ""

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.

        This base implementation works best with formats where:
        1. bot_token is followed immediately by JSON (e.g., bot_token + JSON_array)
        2. JSON can be parsed incrementally using partial_json_loads
        3. Multiple tool calls are separated by "; " or ", "

        Examples of incompatible formats (need custom implementation, may reuse some logic from this class):
        - Each tool call is wrapped in a separate block: See Qwen25Detector
        - Multiple separate blocks: [TOOL_CALLS] [...] \n [TOOL_CALLS] [...]
        - Tool call is Pythonic style

        For incompatible formats, detectors should override this method with custom logic.
        """
        self._buffer += new_text
        emitted_calls: List[ToolCallItem] = []

        def empty_or_emitted() -> StreamingParseResult:
            return (
                StreamingParseResult(calls=emitted_calls)
                if emitted_calls
                else StreamingParseResult()
            )

        # A completed bad entry can be consumed without emitting output. Keep
        # reparsing so a same-chunk tail can still produce the next call:
        # '<tool_call>{"name":"bad","arguments":{}}, {"name":"get_weather",...}'
        # becomes ', {"name":"get_weather",...}' after the drop.
        while True:
            current_text = self._buffer
            if (
                self._in_tool_call_sequence
                and self.eot_token
                and current_text.startswith(self.eot_token)
            ):
                self._buffer = self._consume_leading_eot_token(current_text)
                if self._buffer:
                    continue
                return StreamingParseResult(calls=emitted_calls)

            # If the first entry was dropped, current_tool_id may still be -1
            # while the buffer starts with ', {"name":"get_weather",...}'.
            # The outer marker was already accepted, so the comma is sequence
            # syntax rather than normal text.
            starts_with_sequence_separator = (
                self._starts_with_tool_call_separator(current_text)
                and self._has_tool_call_sequence_context()
            )
            has_tool_call = self.has_tool_call(current_text)

            # Parse when a marker starts a sequence, or a leading separator
            # keeps consuming entries from the current sequence.
            if not (has_tool_call or starts_with_sequence_separator):
                normal_text, self._buffer = self._hold_back_partial_tokens(
                    current_text, (self.bot_token,)
                )
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                    self._in_tool_call_sequence = False
                if emitted_calls and normal_text:
                    # Preserve wire order: callers emit normal_text before calls,
                    # so text found after recovered calls must wait for the next tick.
                    self._buffer = normal_text + self._buffer
                    return StreamingParseResult(calls=emitted_calls)
                if emitted_calls:
                    return StreamingParseResult(calls=emitted_calls)
                return StreamingParseResult(normal_text=normal_text)

            if has_tool_call:
                self._in_tool_call_sequence = True

            # Build tool indices if not already built
            if not hasattr(self, "_tool_indices"):
                self._tool_indices = self._get_tool_indices(tools)

            flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

            try:
                # Priority check: if we're processing a subsequent tool, first
                # check if text starts with the tool separator. This is critical
                # for parallel tool calls because the bot_token (e.g., '[') can
                # also appear inside array parameters of the current tool.
                used_separator_branch = False
                if starts_with_sequence_separator:
                    start_idx = len(self.tool_call_separator)
                    used_separator_branch = True
                else:
                    tool_call_pos = current_text.find(self.bot_token)
                    if tool_call_pos != -1:
                        start_idx = tool_call_pos + len(self.bot_token)
                    else:
                        start_idx = 0

                if start_idx >= len(current_text):
                    return empty_or_emitted()

                try:
                    obj, end_idx = _partial_json_loads(current_text[start_idx:], flags)
                except (MalformedJSON, json.JSONDecodeError):
                    # Separator landed on non-JSON markup; fall back to
                    # bot_token which skips past all inter-object markup.
                    # e.g. Qwen25: separator "," matches between eot/bot tags.
                    if used_separator_branch and self.bot_token in current_text:
                        start_idx = current_text.find(self.bot_token) + len(
                            self.bot_token
                        )
                        if start_idx >= len(current_text):
                            return empty_or_emitted()
                        obj, end_idx = _partial_json_loads(
                            current_text[start_idx:], flags
                        )
                    else:
                        raise

                is_current_complete = _is_complete_json(
                    current_text[start_idx : start_idx + end_idx]
                )

                if not isinstance(obj, dict):
                    if not is_current_complete:
                        return empty_or_emitted()
                    if self._drop_completed_malformed_entry(
                        current_text,
                        start_idx,
                        end_idx,
                        repr(obj),
                    ):
                        continue
                    return empty_or_emitted()

                name = obj.get("name")
                if not name:
                    if not is_current_complete:
                        return empty_or_emitted()
                    if self._drop_completed_malformed_entry(
                        current_text,
                        start_idx,
                        end_idx,
                        repr(obj),
                    ):
                        continue
                    return empty_or_emitted()

                # Validate tool name if present (honoring
                # SGLANG_FORWARD_UNKNOWN_TOOLS, like the non-streaming path)
                if "name" in obj and obj["name"] not in self._tool_indices:
                    if not is_current_complete:
                        return empty_or_emitted()
                    if self._skip_unknown_tool(obj["name"]):
                        # Consume only this completed unknown call. Retry from
                        # the remaining buffer so following calls stay parseable
                        # even when they share the current chunk.
                        self._buffer = self._consume_leading_eot_token(
                            current_text[start_idx + end_idx :]
                        )
                        self.current_tool_name_sent = False
                        if self._should_retry_buffer(current_text):
                            continue
                        return empty_or_emitted()

                # Handle parameters/arguments consistency
                # NOTE: we assume here that the obj is always partial of a single tool call
                if "parameters" in obj:
                    assert (
                        "arguments" not in obj
                    ), "model generated both parameters and arguments"
                    obj["arguments"] = obj["parameters"]

                current_tool_call = obj

            except (MalformedJSON, json.JSONDecodeError):
                fragment = current_text[start_idx:]
                # A parser error alone does not tell us where the bad entry
                # ends. Wait until the top-level boundary appears; then
                # '{"arguments":{x}}, {"name":"get_weather",...}' can drop only
                # the bad entry and retry from the following comma.
                closed_end = _closed_top_level_json_end(fragment)
                if closed_end is not None:
                    self.compatibility.note(
                        CompatibilityEvent.MALFORMED_JSON_DROPPED,
                        detail=fragment[:closed_end][:80],
                    )
                    closing_calls = self._close_started_malformed_call()
                    self._buffer = self._consume_leading_eot_token(
                        fragment[closed_end:]
                    )
                    if closing_calls:
                        emitted_calls.extend(closing_calls)
                        if self._should_retry_buffer(current_text):
                            continue
                        return StreamingParseResult(calls=emitted_calls)
                    if self._should_retry_buffer(current_text):
                        continue
                return empty_or_emitted()

            if not current_tool_call:
                return empty_or_emitted()

            # Case 1: Handle tool name streaming
            # This happens when we encounter a tool but haven't sent its name yet
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")

                if function_name and (
                    function_name in self._tool_indices
                    or envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
                ):
                    # If this is a new tool (current_tool_id was -1), initialize it
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.streamed_args_for_tool.append("")
                    # If this is a subsequent tool, ensure streamed_args_for_tool is large enough
                    elif self.current_tool_id >= len(self.streamed_args_for_tool):
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                    # Send the tool name with empty parameters
                    res = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # Case 2: Handle streaming arguments
            # This happens when we've already sent the tool name and now need to stream arguments incrementally
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                if cur_arguments is not None:
                    # Calculate how much of the arguments we've already streamed
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_arguments = None
                    if self.current_tool_id < len(self.prev_tool_call_arr):
                        prev_arguments = self.prev_tool_call_arr[
                            self.current_tool_id
                        ].get("arguments")

                    argument_diff = None

                    # If the current tool's JSON is complete, send all remaining arguments
                    if is_current_complete:
                        argument_diff = cur_args_json[sent:]
                        completing_tool_id = (
                            self.current_tool_id
                        )  # Save the ID of the tool that's completing

                        # Only remove the processed portion, keep unprocessed content
                        self._buffer = self._consume_leading_eot_token(
                            current_text[start_idx + end_idx :]
                        )

                    # If the tool is still being parsed, send incremental changes
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                        if cur_args_json != prev_args_json:
                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    # Update prev_tool_call_arr with current state
                    if self.current_tool_id >= 0:
                        # Ensure prev_tool_call_arr is large enough
                        while len(self.prev_tool_call_arr) <= self.current_tool_id:
                            self.prev_tool_call_arr.append({})
                        self.prev_tool_call_arr[self.current_tool_id] = (
                            current_tool_call
                        )

                    # Advance to next tool if complete
                    if is_current_complete:
                        self.current_tool_name_sent = False
                        self.current_tool_id += 1

                    # Send the argument diff if there's something new
                    if argument_diff is not None:
                        # Use the correct tool_index: completing_tool_id for completed tools, current_tool_id for ongoing
                        tool_index_to_use = (
                            completing_tool_id
                            if is_current_complete
                            else self.current_tool_id
                        )
                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=tool_index_to_use,
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        self.streamed_args_for_tool[tool_index_to_use] += argument_diff

            if emitted_calls:
                emitted_calls.extend(res.calls)
                if res.calls and self._buffer:
                    continue
                return StreamingParseResult(
                    normal_text=res.normal_text, calls=emitted_calls
                )

            return res

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains function call markers specific to this format.
        """
        raise NotImplementedError()

    def supports_structural_tag(self) -> bool:
        """Return True if this detector supports structural tag format."""
        return True

    @abstractmethod
    def structure_info(self) -> _GetInfoFunc:
        """
        Return a function that creates StructureInfo for constrained generation.

        The returned function takes a tool name and returns a StructureInfo object
        containing the begin/end patterns and trigger tokens needed for constrained
        generation of function calls in this format.

        Returns:
            A function that takes a tool name (str) and returns StructureInfo
        """
        raise NotImplementedError()

    def get_structural_tag_name(self) -> Optional[str]:
        """Return the XGrammar model name for native structural tags, if supported."""
        return None

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
    ) -> Optional[StructuralTag]:
        """
        Return a model-native XGrammar structural tag when supported.

        Args:
            tools: List of available tools
            tool_choice: The tool choice setting from the request
            thinking_mode: Whether to include the model's reasoning prefix in
                the returned structural tag. Pass False when SGLang's
                ReasonerGrammarBackend will own the <think>...</think> prefix
                (the typical case when --reasoning-parser is configured) so
                only one layer constrains the reasoning section.

        Returns:
            StructuralTag if this detector supports model-native tags, otherwise None
        """
        structural_tag_name = self.get_structural_tag_name()
        if not structural_tag_name or get_model_structural_tag is None:
            return None

        converted_tools = [tool.model_dump() for tool in tools or []]
        converted_tool_choice = (
            tool_choice.model_dump()
            if isinstance(tool_choice, ToolChoice)
            else tool_choice
        )
        return get_model_structural_tag(
            model=structural_tag_name,
            tools=converted_tools,
            tool_choice=converted_tool_choice,
            reasoning=thinking_mode,
        )
