"""Multi-format tool-call detector that dispatches on a per-request tool_format.

Ported from vLLM's MultiFormatToolParser (v0.12.0-ifm_xllm-fix branch).

Dialects:
  - "default"   : delegate to HermesDetector
  - "qwen3"     : delegate to Qwen3CoderDetector (XML form)
  - "minimax"   : embedded XML extractor
  - "dsv32"     : embedded XML extractor with string-type flag
  - "glm"       : embedded <arg_key>/<arg_value> extractor
  - "gptoss"    : embedded "<tool_call>...to=functions.fn json\\n{...}\\n</tool_call>"
  - "python"    : embedded Python-AST literal extractor
  - "json"      : embedded IFM "<ifm|tool_call>{json}</ifm|tool_call>" extractor
  - "xml"       : embedded IFM "<ifm|tool_call>fn<ifm|arg_key>..." extractor
  - "xml_typed" : same as "xml" (the <ifm|arg_type> hint is always honored)

Streaming:
  - "default"/"qwen3" delegate to their sub-detectors (which stream natively).
  - The IFM dialects ("xml", "xml_typed", "json") — the K2-V3 format — DO stream:
      * xml / xml_typed: a character-by-character XML->JSON state machine
        (modeled on sglang's Glm4Moe/Glm47Moe detectors) emits the tool name
        first, then incremental JSON argument fragments.
      * json: streams at tool-call-block granularity (each <ifm|tool_call>
        block is emitted as soon as it completes).
    vLLM's K2V3ToolParser does NOT stream the IFM format; this is a deliberate
    SGLang enhancement and the reassembled deltas match the non-stream parse.
  - The remaining embedded dialects (minimax, dsv32, glm, gptoss, python) still
    buffer and emit nothing during streaming (parse runs on the final call).
"""

from __future__ import annotations

import ast
import json
import logging
import re
from enum import Enum
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class _IfmStreamState(str, Enum):
    """State machine states for the IFM XML->JSON streaming converter.

    Mirrors the GLM detectors' StreamState, with an extra IN_TYPE state for the
    optional ``<ifm|arg_type>`` hint that sits between key and value.
    """

    INIT = "INIT"
    BETWEEN = "BETWEEN"
    IN_KEY = "IN_KEY"
    AFTER_KEY = "AFTER_KEY"  # waiting for <ifm|arg_type> or <ifm|arg_value>
    IN_TYPE = "IN_TYPE"
    IN_VALUE = "IN_VALUE"

_EMBEDDED_DIALECTS = {
    "minimax",
    "dsv32",
    "glm",
    "gptoss",
    "python",
    "json",
    "xml",
    "xml_typed",
}
_DELEGATING_DIALECTS = {"default", "qwen3"}
_SUPPORTED_DIALECTS = _EMBEDDED_DIALECTS | _DELEGATING_DIALECTS


class MultiFormatDetector(BaseFormatDetector):
    """Dispatcher detector. Selects extraction strategy from tool_format."""

    def __init__(
        self,
        tool_format: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if tool_format is None and chat_template_kwargs:
            tool_format = chat_template_kwargs.get("tool_format")
        self.tool_format = tool_format or "default"

        self._delegate: Optional[BaseFormatDetector] = None

        if self.tool_format not in _SUPPORTED_DIALECTS:
            raise ValueError(
                f"Unsupported tool_format for multi_format parser: "
                f"{self.tool_format!r}. Supported formats: "
                f"{', '.join(sorted(_SUPPORTED_DIALECTS))}."
            )

        if self.tool_format == "default":
            from sglang.srt.function_call.hermes_detector import HermesDetector

            self._delegate = HermesDetector()
        elif self.tool_format == "qwen3":
            # Task 1 confirms this is the right SGLang detector for vLLM's qwen3 XML.
            from sglang.srt.function_call.qwen3_coder_detector import (
                Qwen3CoderDetector,
            )

            self._delegate = Qwen3CoderDetector()

        # Streaming state for the IFM dialects (xml/xml_typed/json). Unused by
        # the delegating dialects but harmless to initialize.
        self._last_arguments = ""
        self._streamed_raw_length = 0
        self._reset_ifm_stream_state()

    # BaseFormatDetector contract -------------------------------------

    def has_tool_call(self, text: str) -> bool:
        if self._delegate is not None:
            return self._delegate.has_tool_call(text)
        if self.tool_format in ("minimax", "dsv32"):
            return "<tool_calls>" in text
        if self.tool_format in ("json", "xml", "xml_typed"):
            return self._IFM_TOOL_CALL_START_TOKEN in text
        if self.tool_format == "glm":
            return "<tool_call>" in text or self._IFM_TOOL_CALL_START_TOKEN in text
        if self.tool_format == "python":
            return "<tool_call>" in text
        if self.tool_format == "gptoss":
            return "<tool_call>" in text and "to=functions." in text
        return False

    def detect_and_parse(
        self, text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if self._delegate is not None:
            return self._delegate.detect_and_parse(text, tools)

        try:
            if self.tool_format == "json":
                return self._extract_ifm(text, tools, self._ifm_json_calls)
            if self.tool_format in ("xml", "xml_typed"):
                return self._extract_ifm(text, tools, self._ifm_xml_calls)
            if self.tool_format == "minimax":
                return self._extract_minimax(text, tools, type_aware=False)
            if self.tool_format == "dsv32":
                return self._extract_minimax(text, tools, type_aware=True)
            if self.tool_format == "glm":
                # vLLM routes glm output containing IFM markers to the IFM
                # extractor before falling back to the <arg_key> form.
                if self._IFM_TOOL_CALL_START_TOKEN in text:
                    return self._extract_ifm(text, tools, self._ifm_xml_calls)
                return self._extract_glm(text, tools)
            if self.tool_format == "gptoss":
                return self._extract_gptoss(text, tools)
            if self.tool_format == "python":
                return self._extract_python(text, tools)
        except Exception:
            logger.exception(
                "MultiFormatDetector failed to extract for tool_format=%s",
                self.tool_format,
            )
        return StreamingParseResult(normal_text=text, calls=[])

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if self._delegate is not None:
            return self._delegate.parse_streaming_increment(new_text, tools)
        if self.tool_format in ("xml", "xml_typed"):
            return self._ifm_xml_streaming_increment(new_text, tools)
        if self.tool_format == "json":
            return self._ifm_json_streaming_increment(new_text, tools)
        # Remaining embedded dialects (minimax/dsv32/glm/gptoss/python) do not
        # stream: buffer the text and emit nothing. detect_and_parse runs on the
        # complete text passed by the serving layer, so this buffer is unused.
        self._buffer += new_text
        return StreamingParseResult()

    # IFM streaming (K2-V3 format) ------------------------------------
    #
    # vLLM's K2V3ToolParser does not stream the IFM dialects; the logic below is
    # an SGLang enhancement modeled on Glm47MoeDetector (whose
    # ``<tool_call>name<arg_key>..<arg_value>..`` format is structurally
    # identical to ``<ifm|tool_call>name<ifm|arg_key>..<ifm|arg_value>..``),
    # extended for the optional ``<ifm|arg_type>`` hint.

    # (Tool-call open/close token constants live with the other IFM tokens in
    # the "Class-level regex constants for IFM" block below.)
    _IFM_ARG_KEY_OPEN = "<ifm|arg_key>"
    _IFM_ARG_KEY_OPEN_PREFIX = "<ifm|arg_key"  # opening tag without the closing ">"
    _IFM_ARG_KEY_CLOSE = "</ifm|arg_key>"
    _IFM_ARG_TYPE_OPEN = "<ifm|arg_type>"
    _IFM_ARG_TYPE_CLOSE = "</ifm|arg_type>"
    _IFM_ARG_VALUE_OPEN = "<ifm|arg_value>"
    _IFM_ARG_VALUE_CLOSE = "</ifm|arg_value>"

    _IFM_STREAM_TOOL_CALL_REGEX = re.compile(
        r"<ifm\|tool_call>(.*?)(?:(<ifm\|arg_key.*?))?(?:(</ifm\|tool_call>)|$)",
        re.DOTALL,
    )

    def _reset_ifm_stream_state(self) -> None:
        """Reset the per-tool-call IFM streaming state machine."""
        self._ifm_state = _IfmStreamState.INIT
        self._ifm_current_key = ""
        self._ifm_current_value = ""
        self._ifm_tag_buffer = ""
        self._ifm_is_first_param = True
        self._ifm_value_started = False
        self._ifm_cached_value_type: Optional[str] = None
        self._ifm_inline_type: Optional[str] = None
        self._ifm_tool_completed = False
        self._ifm_sent_empty_object = False

    # Wrapper/closing tokens that frame tool calls and must never leak as
    # content. (The opening "<ifm|tool_call>" is handled by the main branch.)
    _IFM_SUPPRESS_TOKENS = (
        "<ifm|tool_calls>",
        "</ifm|tool_calls>",
        "</ifm|tool_call>",
    )
    # Proper prefixes of an opening "<ifm|" or closing "</ifm|" marker, longest
    # first — a trailing fragment matching one of these could still grow into a
    # structural token, so we keep buffering it.
    _IFM_MARKER_PARTIALS = ("</ifm", "<ifm", "</if", "<if", "</i", "<i", "</", "<")

    def _ifm_marker_cut(self, text: str) -> Optional[int]:
        """Earliest index at which an IFM marker (opening ``<ifm|`` or closing
        ``</ifm|``) begins, or a trailing partial that could grow into one.
        Returns None when ``text`` contains no IFM markup."""
        starts = [i for i in (text.find("<ifm|"), text.find("</ifm|")) if i != -1]
        if starts:
            return min(starts)
        for partial in self._IFM_MARKER_PARTIALS:
            if text.endswith(partial):
                return len(text) - len(partial)
        return None

    def _ifm_split_normal_text(self, buffer: str) -> tuple[str, str]:
        """Split a no-tool-call buffer into (emit, hold).

        Genuine content (including whitespace) is emitted; any IFM markup — a
        leading reasoning block, a tool-call wrapper/closing token, or a partial
        ``<ifm|``/``</ifm|`` token — is held until it resolves on a later chunk.
        This preserves character-by-character streaming of normal text while
        never leaking structural tokens or reasoning content."""
        text = buffer
        changed = True
        while changed:
            changed = False
            stripped = self._strip_ifm_reasoning_prefix(text)
            if stripped != text:
                text = stripped
                changed = True
            for token in self._IFM_SUPPRESS_TOKENS:
                if text.startswith(token):
                    text = text[len(token) :]
                    changed = True
        cut = self._ifm_marker_cut(text)
        if cut is None:
            return text, ""
        return text[:cut], text[cut:]

    def _ifm_prefix_normal_text(self, prefix: str) -> str:
        """Normal text from the segment before the first ``<ifm|tool_call>`` in
        the same chunk: reasoning/wrapper stripped, whitespace-only collapsed to
        "" (matching the non-stream ``_ifm_prefix`` semantics)."""
        emit, _ = self._ifm_split_normal_text(prefix)
        return emit if emit.strip() else ""

    def _ifm_stream_value_type(
        self, func_name: str, key: str, tools: List[Tool]
    ) -> str:
        """Resolve the streaming value type for an argument.

        Precedence matches the non-stream ``_coerce_argument_value``: schema type
        then the inline ``<ifm|arg_type>`` hint. Untyped values default to
        "string" (a streaming limitation shared with the GLM detectors, since the
        full value is not yet known when its type must be decided)."""
        target = self._schema_arg_type(func_name, key, tools) or self._ifm_inline_type
        if self._arg_type_is_string(target):
            return "string"
        if target in ("number", "integer", "float"):
            return "number"
        if target is not None:
            # object / array / boolean: stream the model's verbatim JSON.
            return "raw"
        return "string"

    def _process_ifm_xml_to_json_streaming(
        self, raw_increment: str, func_name: str, tools: List[Tool]
    ) -> str:
        """Convert an IFM XML increment to a JSON increment, char by char,
        preserving state across calls to handle tags/values split across chunks."""
        json_output = ""

        for char in raw_increment:
            self._ifm_tag_buffer += char

            if self._ifm_state in (_IfmStreamState.INIT, _IfmStreamState.BETWEEN):
                if self._ifm_tag_buffer.endswith(self._IFM_ARG_KEY_OPEN):
                    self._ifm_state = _IfmStreamState.IN_KEY
                    self._ifm_current_key = ""
                    self._ifm_tag_buffer = ""
                    json_output += "{" if self._ifm_is_first_param else ", "
                    self._ifm_is_first_param = False

            elif self._ifm_state == _IfmStreamState.IN_KEY:
                if self._ifm_tag_buffer.endswith(self._IFM_ARG_KEY_CLOSE):
                    self._ifm_current_key = self._ifm_tag_buffer[
                        : -len(self._IFM_ARG_KEY_CLOSE)
                    ].strip()
                    self._ifm_tag_buffer = ""
                    self._ifm_state = _IfmStreamState.AFTER_KEY
                    json_output += (
                        json.dumps(self._ifm_current_key, ensure_ascii=False) + ": "
                    )

            elif self._ifm_state == _IfmStreamState.AFTER_KEY:
                if self._ifm_tag_buffer.endswith(self._IFM_ARG_TYPE_OPEN):
                    self._ifm_state = _IfmStreamState.IN_TYPE
                    self._ifm_tag_buffer = ""
                elif self._ifm_tag_buffer.endswith(self._IFM_ARG_VALUE_OPEN):
                    self._ifm_state = _IfmStreamState.IN_VALUE
                    self._ifm_current_value = ""
                    self._ifm_tag_buffer = ""
                    self._ifm_value_started = False
                    self._ifm_cached_value_type = self._ifm_stream_value_type(
                        func_name, self._ifm_current_key, tools
                    )

            elif self._ifm_state == _IfmStreamState.IN_TYPE:
                if self._ifm_tag_buffer.endswith(self._IFM_ARG_TYPE_CLOSE):
                    self._ifm_inline_type = self._ifm_tag_buffer[
                        : -len(self._IFM_ARG_TYPE_CLOSE)
                    ].strip() or None
                    self._ifm_tag_buffer = ""
                    self._ifm_state = _IfmStreamState.AFTER_KEY

            elif self._ifm_state == _IfmStreamState.IN_VALUE:
                if self._ifm_tag_buffer.endswith(self._IFM_ARG_VALUE_CLOSE):
                    final_value = self._ifm_tag_buffer[
                        : -len(self._IFM_ARG_VALUE_CLOSE)
                    ]
                    self._ifm_current_value += final_value
                    value_type = self._ifm_cached_value_type or "string"

                    if self._ifm_value_started:
                        if final_value:
                            if value_type == "string":
                                json_output += json.dumps(
                                    final_value, ensure_ascii=False
                                )[1:-1]
                            else:
                                json_output += final_value
                        if value_type == "string":
                            json_output += '"'
                    else:
                        json_output += self._ifm_format_value_complete(
                            self._ifm_current_value, value_type
                        )

                    self._ifm_tag_buffer = ""
                    self._ifm_state = _IfmStreamState.BETWEEN
                    self._ifm_current_value = ""
                    self._ifm_value_started = False
                    self._ifm_cached_value_type = None
                    self._ifm_inline_type = None
                else:
                    closing = self._IFM_ARG_VALUE_CLOSE
                    is_potential_closing = len(self._ifm_tag_buffer) <= len(
                        closing
                    ) and closing.startswith(self._ifm_tag_buffer)
                    if not is_potential_closing:
                        content = self._ifm_tag_buffer
                        value_type = self._ifm_cached_value_type or "string"
                        if value_type == "string":
                            if not self._ifm_value_started:
                                json_output += '"'
                                self._ifm_value_started = True
                            if content:
                                json_output += json.dumps(content, ensure_ascii=False)[
                                    1:-1
                                ]
                                self._ifm_current_value += content
                                self._ifm_tag_buffer = ""
                        else:
                            if content:
                                if not self._ifm_value_started:
                                    self._ifm_value_started = True
                                json_output += content
                                self._ifm_current_value += content
                                self._ifm_tag_buffer = ""

        return json_output

    @staticmethod
    def _ifm_format_value_complete(value: str, value_type: str) -> str:
        """Format a value that arrived in a single chunk (state machine never
        emitted an opening quote for it)."""
        if value_type == "string":
            return json.dumps(value, ensure_ascii=False)
        if value_type == "number":
            stripped = value.strip()
            try:
                if "." in stripped or "e" in stripped.lower():
                    return str(float(stripped))
                return str(int(stripped))
            except (ValueError, AttributeError):
                return json.dumps(value, ensure_ascii=False)
        # object / array / boolean: already valid JSON. Guard an empty value
        # (malformed input) that would otherwise emit nothing -> invalid JSON.
        return value if value else '""'

    def _ifm_xml_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer
        start_token = self._IFM_TOOL_CALL_START_TOKEN

        if start_token not in current_text:
            emit, hold = self._ifm_split_normal_text(current_text)
            self._buffer = hold
            return StreamingParseResult(normal_text=emit)

        normal_text = ""
        first_idx = current_text.find(start_token)
        if first_idx > 0:
            normal_text = self._ifm_prefix_normal_text(current_text[:first_idx])
            current_text = current_text[first_idx:]
            self._buffer = current_text

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: List[ToolCallItem] = []
        try:
            partial_match = self._IFM_STREAM_TOOL_CALL_REGEX.search(current_text)
            if not partial_match:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            func_name = partial_match.group(1).strip()
            func_args_raw = (
                partial_match.group(2).strip() if partial_match.group(2) else ""
            )
            is_tool_end = partial_match.group(3) or ""

            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = [""]
                self._streamed_raw_length = 0
                self.current_tool_name_sent = False
                self._reset_ifm_stream_state()

            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            has_arg_key = self._IFM_ARG_KEY_OPEN_PREFIX in current_text

            name_item = self._ifm_send_tool_name(func_name, has_arg_key, is_tool_end)
            if name_item:
                calls.append(name_item)

            if self.current_tool_name_sent:
                arg_item = self._ifm_process_arguments(func_name, func_args_raw, tools)
                if arg_item:
                    calls.append(arg_item)

                if (
                    is_tool_end == self._IFM_TOOL_CALL_END_TOKEN
                    and not self._ifm_tool_completed
                ):
                    calls.extend(
                        self._finalize_ifm_xml_tool_call(
                            func_name, func_args_raw, tools
                        )
                    )
                    self._buffer = current_text[partial_match.end() :]
                    self.current_tool_id += 1
                    self._last_arguments = ""
                    self.current_tool_name_sent = False
                    self._streamed_raw_length = 0
                    self._reset_ifm_stream_state()
                    return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception:
            logger.exception(
                "MultiFormatDetector IFM xml streaming failed for tool_format=%s",
                self.tool_format,
            )
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _ifm_send_tool_name(
        self, func_name: str, has_arg_key: bool, is_tool_end: str
    ) -> Optional[ToolCallItem]:
        """Emit the tool name once it is known — an ``<ifm|arg_key>`` or the
        closing ``</ifm|tool_call>`` proves the name is complete. Returns the
        name item (with empty parameters), or None if the name is not yet ready
        or was already sent."""
        if self.current_tool_name_sent:
            return None
        name_complete = has_arg_key or is_tool_end == self._IFM_TOOL_CALL_END_TOKEN
        if not (name_complete and func_name):
            return None
        self.current_tool_name_sent = True
        self._streamed_raw_length = 0
        self._reset_ifm_stream_state()
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": func_name,
            "arguments": {},
        }
        return ToolCallItem(
            tool_index=self.current_tool_id, name=func_name, parameters=""
        )

    def _ifm_process_arguments(
        self, func_name: str, func_args_raw: str, tools: List[Tool]
    ) -> Optional[ToolCallItem]:
        """Feed newly-arrived raw argument text through the XML->JSON state
        machine and return the JSON increment as a ToolCallItem, or None when
        nothing new was produced this chunk."""
        current_raw_length = len(func_args_raw)
        if current_raw_length <= self._streamed_raw_length:
            return None
        raw_increment = func_args_raw[self._streamed_raw_length :]
        json_increment = self._process_ifm_xml_to_json_streaming(
            raw_increment, func_name, tools
        )
        # Advance even when no JSON was produced: the input has been consumed by
        # the state machine (it may be buffering a partial tag).
        self._streamed_raw_length = current_raw_length
        if not json_increment:
            return None
        self._last_arguments += json_increment
        self.streamed_args_for_tool[self.current_tool_id] += json_increment
        return ToolCallItem(
            tool_index=self.current_tool_id, name=None, parameters=json_increment
        )

    def _finalize_ifm_xml_tool_call(
        self, func_name: str, func_args_raw: str, tools: List[Tool]
    ) -> List[ToolCallItem]:
        """Close out the current tool call: emit the closing brace (or {} for a
        no-arg call) and record the fully-parsed arguments for end-of-stream
        flushing by the serving layer."""
        calls: List[ToolCallItem] = []

        if self._ifm_is_first_param and not self._ifm_sent_empty_object:
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id, name=None, parameters="{}"
                )
            )
            self._last_arguments += "{}"
            self.streamed_args_for_tool[self.current_tool_id] += "{}"
            self._ifm_sent_empty_object = True
        elif not self._last_arguments.endswith("}") and not self._ifm_sent_empty_object:
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id, name=None, parameters="}"
                )
            )
            self._last_arguments += "}"
            self.streamed_args_for_tool[self.current_tool_id] += "}"
            self._ifm_sent_empty_object = True

        # Record the final parsed arguments (reuse the non-stream extractor on the
        # reconstructed block) so the serving layer can flush any unstreamed
        # remainder consistently.
        try:
            block = func_name + func_args_raw
            for _, args in self._ifm_xml_calls(block, tools):
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = args
        except Exception:
            logger.debug("IFM xml finalize: argument re-parse failed", exc_info=True)

        self._ifm_tool_completed = True
        return calls

    def _ifm_json_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Stream the IFM ``json`` dialect at tool-call-block granularity: each
        ``<ifm|tool_call>{json}</ifm|tool_call>`` block is parsed and emitted as
        soon as it completes (name first, then the full argument JSON). A block
        may itself contain a JSON list of calls, each emitted in turn."""
        self._buffer += new_text
        current_text = self._buffer
        start_token = self._IFM_TOOL_CALL_START_TOKEN

        if start_token not in current_text:
            emit, hold = self._ifm_split_normal_text(current_text)
            self._buffer = hold
            return StreamingParseResult(normal_text=emit)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        normal_text = ""
        first_idx = current_text.find(start_token)
        if first_idx > 0:
            normal_text = self._ifm_prefix_normal_text(current_text[:first_idx])
            current_text = current_text[first_idx:]

        if self.current_tool_id == -1:
            self.current_tool_id = 0
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = [""]

        calls: List[ToolCallItem] = []
        consumed = 0
        for match in self._IFM_BLOCK_REGEX.finditer(current_text):
            try:
                for name, args in self._ifm_json_calls(match.group(1), tools):
                    args_json = json.dumps(args, ensure_ascii=False)
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": name,
                        "arguments": args,
                    }
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=name,
                            parameters="",
                        )
                    )
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=args_json,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = args_json
                    self.current_tool_id += 1
            except Exception:
                logger.exception(
                    "MultiFormatDetector IFM json streaming failed for a block"
                )
            consumed = match.end()

        self._buffer = current_text[consumed:] if consumed else current_text
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def supports_structural_tag(self) -> bool:
        if self._delegate is not None:
            return self._delegate.supports_structural_tag()
        return False

    def structure_info(self) -> _GetInfoFunc:
        if self._delegate is not None:
            return self._delegate.structure_info()
        raise NotImplementedError(
            f"structure_info is not implemented for tool_format={self.tool_format!r}"
        )

    # Embedded extractors (stubs filled in Tasks 5/7/8/9) -------------

    def _extract_minimax(
        self, text: str, tools: List[Tool], type_aware: bool
    ) -> StreamingParseResult:
        if self._MINIMAX_START not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for block in self._MINIMAX_BLOCK.findall(text):
            for func_name, body in self._MINIMAX_INVOKE.findall(block):
                args: dict[str, Any] = {}
                for pname, string_flag, pvalue in self._MINIMAX_PARAM.findall(body):
                    # string_flag is None (not "") when the attribute is absent
                    # in the source — Python's re returns None for non-participating
                    # optional capture groups. None == "true" is False, so the
                    # else branch is taken for both unset and string="false" cases.
                    if type_aware and string_flag == "true":
                        args[pname] = pvalue
                    else:
                        args[pname] = self._json_or_string(pvalue)
                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices.get(func_name, -1),
                        name=func_name,
                        parameters=json.dumps(args, ensure_ascii=False),
                    )
                )

        if not calls:
            return StreamingParseResult(normal_text=text, calls=[])

        prefix = text[: text.find(self._MINIMAX_START)]
        return StreamingParseResult(normal_text=prefix, calls=calls)

    def _extract_glm(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        matches = list(self._GLM_BLOCK.finditer(text))
        if not matches:
            return StreamingParseResult(normal_text=text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for match in matches:
            block = match.group(1)
            first_arg_idx = block.find("<arg_key>")
            if first_arg_idx == -1:
                func_name = block.strip()
                args: dict[str, Any] = {}
            else:
                func_name = block[:first_arg_idx].strip()
                args = {}
                arg_section = block[first_arg_idx:]
                for k, v in self._GLM_ARG.findall(arg_section):
                    key = k.strip()
                    raw = v.strip()
                    if self._glm_param_is_string(func_name, key, tools):
                        args[key] = raw
                    else:
                        args[key] = self._deserialize_glm_value(raw)
            if not func_name:
                continue
            calls.append(
                ToolCallItem(
                    tool_index=tool_indices.get(func_name, -1),
                    name=func_name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )

        if not calls:
            return StreamingParseResult(normal_text=text, calls=[])
        prefix = text[: matches[0].start()]
        return StreamingParseResult(normal_text=prefix, calls=calls)

    def _extract_gptoss(
        self, text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        matches = list(self._GPTOSS_BLOCK.finditer(text))
        if not matches:
            return StreamingParseResult(normal_text=text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for m in matches:
            func_name = m.group(1)
            args = json.loads(m.group(2).strip())
            calls.append(
                ToolCallItem(
                    tool_index=tool_indices.get(func_name, -1),
                    name=func_name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )
        if not calls:
            return StreamingParseResult(normal_text=text, calls=[])
        prefix = text[: matches[0].start()]
        return StreamingParseResult(normal_text=prefix, calls=calls)

    def _extract_python(
        self, text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        matches = list(self._PYTHON_BLOCK.finditer(text))
        if not matches:
            return StreamingParseResult(normal_text=text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for match in matches:
            block = match.group(1).strip()
            module = ast.parse(block)
            for stmt in module.body:
                if not isinstance(stmt, ast.Expr) or not isinstance(
                    stmt.value, ast.Call
                ):
                    raise ValueError(
                        "Expected Python function call(s) inside <tool_call>."
                    )
                call = stmt.value
                if not isinstance(call.func, ast.Name):
                    raise ValueError("Invalid tool-call name")
                func_name = call.func.id
                args: dict[str, Any] = {}
                for kw in call.keywords:
                    args[kw.arg] = self._python_literal(kw.value)
                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices.get(func_name, -1),
                        name=func_name,
                        parameters=json.dumps(args, ensure_ascii=False),
                    )
                )
        if not calls:
            return StreamingParseResult(normal_text=text, calls=[])
        prefix = text[: matches[0].start()]
        return StreamingParseResult(normal_text=prefix, calls=calls)

    # IFM extractors (BBQ 0518 K2-V3 format) --------------------------

    def _extract_ifm(self, text, tools, parse_block) -> StreamingParseResult:
        """Shared scaffold for the IFM dialects: split into <ifm|tool_call>
        blocks, turn each into (name, args) pairs via parse_block, and attach
        the reasoning-stripped prefix."""
        matches = list(self._IFM_BLOCK_REGEX.finditer(text))
        if not matches:
            return StreamingParseResult(normal_text=text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls = [
            ToolCallItem(
                tool_index=tool_indices.get(name, -1),
                name=name,
                parameters=json.dumps(args, ensure_ascii=False),
            )
            for match in matches
            for name, args in parse_block(match.group(1), tools)
        ]
        if not calls:
            return StreamingParseResult(normal_text=text, calls=[])
        return StreamingParseResult(
            normal_text=self._ifm_prefix(text, matches[0].start()), calls=calls
        )

    def _ifm_xml_calls(self, block: str, tools: List[Tool]):
        first = block.find("<ifm|arg_key>")
        name = (block if first == -1 else block[:first]).strip()
        if not name:
            return
        args = (
            {
                key.strip(): self._coerce_argument_value(
                    value.strip(),
                    name,
                    key.strip(),
                    tools,
                    arg_type=arg_type.strip() or None,
                    from_text=True,
                )
                for key, arg_type, value in self._IFM_ARG_REGEX.findall(block[first:])
            }
            if first != -1
            else {}
        )
        yield name, args

    def _ifm_json_calls(self, block: str, tools: List[Tool]):
        payload = json.loads(block.strip())
        for tool_call in payload if isinstance(payload, list) else [payload]:
            function = tool_call.get("function", tool_call)
            name = function.get("name")
            if not name:
                raise ValueError("Tool call JSON is missing a function name.")
            args = self._coerce_arguments(
                name, self._json_arguments_to_dict(function.get("arguments", {})), tools
            )
            yield name, args

    @classmethod
    def _ifm_prefix(cls, text: str, first_match_index: int) -> str:
        """Leading content before the tool calls, with IFM reasoning stripped.

        vLLM cuts the prefix at the <ifm|tool_calls> wrapper when present,
        else at the first <ifm|tool_call> block, then strips any leading
        reasoning-effort block. Returns "" when nothing remains (vLLM uses
        None for the same case).
        """
        group_index = text.find(cls._IFM_TOOL_CALLS_START_TOKEN)
        cut = group_index if group_index != -1 else first_match_index
        if cut <= 0:
            return ""
        content = cls._strip_ifm_reasoning_prefix(text[:cut])
        return content if content.strip() else ""

    @classmethod
    def _strip_ifm_reasoning_prefix(cls, content: str) -> str:
        while match := cls._IFM_REASONING_PREFIX_REGEX.match(content):
            content = content[match.end() :]
        return content

    @staticmethod
    def _json_arguments_to_dict(arguments: Any) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        if not isinstance(arguments, dict):
            raise ValueError("Tool call arguments must be a JSON object.")
        return arguments

    @staticmethod
    def _schema_arg_type(
        tool_name: str, arg_name: str, tools: List[Tool]
    ) -> Any:
        for tool in tools:
            if tool.function.name != tool_name or tool.function.parameters is None:
                continue
            properties = tool.function.parameters.get("properties", {})
            arg_spec = properties.get(arg_name, {})
            if not isinstance(arg_spec, dict):
                return None
            return arg_spec.get("type")
        return None

    @staticmethod
    def _arg_type_is_string(arg_type: Any) -> bool:
        if isinstance(arg_type, list):
            return "string" in arg_type
        return arg_type == "string"

    @staticmethod
    def _json_stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @classmethod
    def _coerce_argument_value(
        cls,
        value: Any,
        tool_name: str,
        arg_name: str,
        tools: List[Tool],
        *,
        arg_type: Optional[str] = None,
        from_text: bool = False,
    ) -> Any:
        target_type = cls._schema_arg_type(tool_name, arg_name, tools) or arg_type
        if cls._arg_type_is_string(target_type):
            return cls._json_stringify(value)
        if isinstance(value, str) and (from_text or target_type is not None):
            return cls._deserialize_glm_value(value)
        return value

    @classmethod
    def _coerce_arguments(
        cls, tool_name: str, arguments: dict[str, Any], tools: List[Tool]
    ) -> dict[str, Any]:
        return {
            arg_name: cls._coerce_argument_value(arg_value, tool_name, arg_name, tools)
            for arg_name, arg_value in arguments.items()
        }

    # Class-level regex constants for IFM (K2-V3) dialects --------------------

    _IFM_TOOL_CALLS_START_TOKEN = "<ifm|tool_calls>"
    _IFM_TOOL_CALL_START_TOKEN = "<ifm|tool_call>"
    _IFM_TOOL_CALLS_END_TOKEN = "</ifm|tool_calls>"
    _IFM_TOOL_CALL_END_TOKEN = "</ifm|tool_call>"
    _IFM_BLOCK_REGEX = re.compile(
        r"<ifm\|tool_call>(.*?)</ifm\|tool_call>",
        re.DOTALL,
    )
    _IFM_ARG_REGEX = re.compile(
        r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*"
        r"(?:<ifm\|arg_type>(.*?)</ifm\|arg_type>\s*)?"
        r"<ifm\|arg_value>(.*?)</ifm\|arg_value>",
        re.DOTALL,
    )
    _IFM_REASONING_PREFIX_REGEX = re.compile(
        r"\A\s*(?:"
        r"<ifm\|think>.*?</ifm\|think>|"
        r"<ifm\|think_fast>.*?</ifm\|think_fast>|"
        r"<ifm\|think_faster>.*?</ifm\|think_faster>"
        r")\s*",
        re.DOTALL,
    )

    # Class-level regex constants and helpers for minimax/dsv32 dialects ------

    _MINIMAX_START = "<tool_calls>"
    _MINIMAX_BLOCK = re.compile(r"<tool_calls>(.*?)</tool_calls>", re.DOTALL)
    _MINIMAX_INVOKE = re.compile(
        r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', re.DOTALL
    )
    _MINIMAX_PARAM = re.compile(
        r'<parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>(.*?)</parameter>',
        re.DOTALL,
    )

    @staticmethod
    def _json_or_string(value: str) -> Any:
        value = value.strip()
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    # Class-level regex constants and helpers for glm dialect ------------------

    _GLM_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _GLM_ARG = re.compile(
        r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
    )

    @staticmethod
    def _deserialize_glm_value(value: str) -> Any:
        value = value.strip()
        try:
            return json.loads(value)
        except Exception:
            pass
        try:
            return ast.literal_eval(value)
        except Exception:
            pass
        return value

    @staticmethod
    def _glm_param_is_string(
        tool_name: str, arg_name: str, tools: List[Tool]
    ) -> bool:
        for tool in tools:
            if tool.function.name != tool_name or tool.function.parameters is None:
                continue
            arg_type = (
                tool.function.parameters.get("properties", {})
                .get(arg_name, {})
                .get("type")
            )
            return arg_type == "string"
        return False

    # Class-level regex constants for gptoss dialect ---------------------------

    _GPTOSS_BLOCK = re.compile(
        r"<tool_call>\s*(?:assistant\s+)?to=functions\.(\S+?)"
        r"(?:\s+json)?\s*\n(.*?)\n?\s*</tool_call>",
        re.DOTALL,
    )

    # Class-level regex constants and helpers for python dialect ---------------

    _PYTHON_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    @staticmethod
    def _python_literal(node: ast.expr) -> Any:
        """Custom AST walker accepting only literal subtrees.

        This intentionally does NOT call ast.literal_eval — we walk the AST
        ourselves so the accepted subset is fully explicit, and any non-literal
        (BinOp, Call, Name except true/false/null, etc.) raises ValueError.
        """
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in {"true", "True"}:
                return True
            if node.id in {"false", "False"}:
                return False
            if node.id in {"null", "None"}:
                return None
        if isinstance(node, ast.Dict):
            if not all(isinstance(k, ast.Constant) for k in node.keys):
                raise ValueError("dict keys must be literals")
            return {
                k.value: MultiFormatDetector._python_literal(v)
                for k, v in zip(node.keys, node.values)
            }
        if isinstance(node, ast.List):
            return [MultiFormatDetector._python_literal(v) for v in node.elts]
        if isinstance(node, ast.Tuple):
            return [MultiFormatDetector._python_literal(v) for v in node.elts]
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            return (
                -node.operand.value
                if isinstance(node.op, ast.USub)
                else node.operand.value
            )
        raise ValueError("tool-call arguments must be literals")


class K2V3Detector(MultiFormatDetector):
    """K2-V3 tool parser for BBQ 0518 IFM tool-call and reasoning tokens.

    Ported from vLLM's K2V3ToolParser (a thin subclass of MultiFormatToolParser).
    Parses ``<ifm|tool_call>`` blocks and strips leading IFM reasoning-effort
    blocks (``<ifm|think>``/``<ifm|think_fast>``/``<ifm|think_faster>``) off the
    normal-text prefix. Defaults to the IFM ``xml`` dialect, overridable via the
    ``tool_call_format`` chat-template kwarg (matching vLLM's key).
    """

    def __init__(
        self,
        tool_format: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
    ):
        if tool_format is None and chat_template_kwargs:
            tool_format = chat_template_kwargs.get("tool_call_format")
        super().__init__(tool_format=tool_format or "xml")
