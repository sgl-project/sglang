# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from xml.parsers.expat import ParserCreate

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

_STRING_TYPES = frozenset(["string", "str", "text", "varchar", "char", "enum"])
_COMPLEX_TYPES = frozenset(["object", "array", "arr"])
_COMPLEX_PREFIXES = ("dict", "list")


def _get_param_type(func_name: str, param_name: str, tools: List[Tool]) -> str:
    """Get parameter type from tool schema."""
    for tool in tools:
        if tool.function.name == func_name:
            props = tool.function.parameters.get("properties", {})
            if param_name in props:
                return props[param_name].get("type", "string")
    return "string"


def _convert_param_value(
    param_value: str, param_name: str, func_name: str, tools: List[Tool]
) -> Any:
    """
    Convert parameter value based on its type in the schema.
    Adapted from vllm-project/vllm (vllm/tool_parsers/qwen3coder_tool_parser.py)
    """
    # NOTE: MiMo emits raw parameter text. We deliberately do NOT call
    # html.unescape() here: it would corrupt values such as "&current" into
    # "¤t".

    # Handle null value for any type
    if param_value.lower() == "null":
        return None

    param_type = _get_param_type(func_name, param_name, tools)

    if param_type in _STRING_TYPES:
        return param_value
    elif (
        param_type.startswith("int")
        or param_type.startswith("integer")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        try:
            return int(param_value)
        except (ValueError, TypeError):
            logger.warning(
                "Parsed value '%s' of parameter '%s' is not an "
                "integer in tool '%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
            return param_value
    elif param_type.startswith("num") or param_type.startswith("float"):
        try:
            float_param_value = float(param_value)
            return (
                float_param_value
                if float_param_value - int(float_param_value) != 0
                else int(float_param_value)
            )
        except (ValueError, TypeError):
            logger.warning(
                "Parsed value '%s' of parameter '%s' is not a float "
                "in tool '%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
            return param_value
    elif param_type in ["boolean", "bool", "binary"]:
        param_value = param_value.lower()
        if param_value not in ["true", "false"]:
            logger.warning(
                "Parsed value '%s' of parameter '%s' is not a boolean "
                "(`true` or `false`) in tool '%s', degenerating to "
                "false.",
                param_value,
                param_name,
                func_name,
            )
        return param_value == "true"
    else:
        if (
            param_type in _COMPLEX_TYPES
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                param_value = json.loads(param_value)
                return param_value
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning(
                    "Parsed value '%s' of parameter '%s' cannot be "
                    "parsed with json.loads in tool '%s', will try "
                    "other methods to parse it.",
                    param_value,
                    param_name,
                    func_name,
                )
        try:
            param_value = ast.literal_eval(param_value)  # safer
        except (ValueError, SyntaxError, TypeError):
            logger.warning(
                "Parsed value '%s' of parameter '%s' cannot be "
                "converted via Python `ast.literal_eval()` in tool "
                "'%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
        return param_value


def _is_complex_param_type(param_type: str) -> bool:
    if param_type in _COMPLEX_TYPES:
        return True
    return param_type.startswith(_COMPLEX_PREFIXES)


def _is_string_param_type(param_type: str) -> bool:
    return param_type in _STRING_TYPES


def _escape_xml_special_chars(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text


class _MiMoStreamingSAX:
    """
    Incremental SAX-based streaming parser for MiMo XML tool calls.
    Uses Python's xml.parsers.expat to parse the XML-ish format token-by-token.

    For string-type parameters: streams JSON-escaped content per-token.
    For complex-type parameters: buffers until </parameter>, then emits at once.

    Derived from vllm-project/vllm
    (vllm/tool_parsers/qwen3xml_tool_parser.py, ``StreamingXMLToolCallParser``),
    with the following intentional divergences:
      - Fail-closed on parse errors: a malformed element suppresses the
        current tool_call and the consumed raw bytes are surfaced as
        normal_text, instead of swallowing the expat exception and
        continuing with dirty parser state.
      - Unknown-function suppression in the streaming path, aligned with
        ``detect_and_parse`` / ``SGLANG_FORWARD_UNKNOWN_TOOLS``.
      - Multi tool_call orchestration is lifted to the detector layer; each
        SAX instance handles exactly one tool_call.
      - Simplified defer decision (every non-string param is deferred) and
        explicit JSON ``null`` for empty non-string params.
    """

    def __init__(self, tools: List[Tool]):
        self.tools = tools
        self._tool_indices: Dict[str, int] = {
            t.function.name: i for i, t in enumerate(tools) if t.function.name
        }
        self._init_parser()

        self._raw_buffer: str = ""
        self._last_processed_pos: int = 0

        self._current_function_name: Optional[str] = None
        self._current_param_name: Optional[str] = None
        self._current_param_value: str = ""
        self._current_param_json_sent: str = ""
        self._parameters: Dict[str, Any] = {}
        self._param_count: int = 0
        self._start_quote_emitted: bool = False
        self._trailing_newline_pending: bool = False

        self._defer_mode: bool = False
        self._defer_raw_value: str = ""

        self._pre_inside_parameter: bool = False
        self._pre_param_buffer: str = ""
        self._pre_param_name: Optional[str] = None
        self._pre_is_string_passthrough: bool = False

        self._pending_calls: List[ToolCallItem] = []
        self._tool_call_completed: bool = False
        self._in_tool_call: bool = False
        # Set when this tool_call must not surface to the client (unknown
        # function with forwarding disabled, or recovered from a parse
        # exception). All _emit calls become no-ops; the detector layer
        # converts the consumed raw bytes back to normal_text on completion.
        self._tool_call_suppressed: bool = False
        # True once anything (a function name or an argument fragment) has been
        # emitted to the client for this tool_call. A parse error *after* this
        # point can no longer be handled by re-surfacing raw bytes (that would
        # duplicate already-streamed output), so the detector must finalize the
        # partial call instead of suppressing it. See _process_complete_xml_elements.
        self._emitted_to_client: bool = False
        # True when this tool_call completed via a parse error *after* partial
        # emit (finalized rather than suppressed). The detector uses this to
        # discard the broken tool_call's leftover bytes instead of leaking them
        # as normal_text.
        self._tool_call_errored: bool = False

    def _init_parser(self):
        self._parser = ParserCreate()
        self._parser.StartElementHandler = self._start_element
        self._parser.EndElementHandler = self._end_element
        self._parser.CharacterDataHandler = self._char_data
        self._parser.Parse("<root>", False)

    def feed_chunk(self, chunk: str) -> List[ToolCallItem]:
        self._tool_call_completed = False
        self._pending_calls = []
        self._raw_buffer += chunk
        self._process_complete_xml_elements()
        result = self._pending_calls
        self._pending_calls = []
        return result

    @property
    def tool_call_completed(self) -> bool:
        return self._tool_call_completed

    def get_remaining_buffer(self) -> str:
        return self._raw_buffer[self._last_processed_pos :]

    def _emit(self, name: Optional[str] = None, parameters: str = ""):
        if self._tool_call_suppressed:
            return
        self._emitted_to_client = True
        self._pending_calls.append(
            ToolCallItem(tool_index=-1, name=name, parameters=parameters)
        )

    def _process_complete_xml_elements(self):
        while self._last_processed_pos < len(self._raw_buffer):
            if self._tool_call_completed:
                break
            element, end_pos = self._find_next_complete_element(
                self._last_processed_pos
            )
            if element is None:
                break
            try:
                preprocessed = self._preprocess_xml_chunk(element)
                if preprocessed:
                    self._parser.Parse(preprocessed, False)
            except Exception as e:
                if self._emitted_to_client:
                    # A function name and/or partial args have already been
                    # streamed to the client. Re-surfacing the raw bytes as
                    # normal_text would duplicate that output, and leaving the
                    # tool_index un-advanced would collide the next call onto
                    # this one. Instead, finalize the partial tool_call: mark
                    # completion (not suppression) so the detector closes it
                    # out and advances current_tool_id normally.
                    logger.warning(
                        "Error parsing XML element after partial emit: %s "
                        "(element=%r); finalizing the partial tool_call",
                        e,
                        element[:100],
                    )
                    self._tool_call_completed = True
                    self._tool_call_errored = True
                    self._last_processed_pos = end_pos
                    return
                logger.warning(
                    "Error parsing XML element: %s (element=%r); "
                    "fail-closing this tool_call -- detector will surface "
                    "the consumed raw bytes as normal text",
                    e,
                    element[:100],
                )
                # Nothing has been emitted yet, so fail-closed is safe: keep
                # _emit a no-op for the rest of the outstanding bytes and hand
                # control back to the detector by marking completion. The
                # detector layer will surface the consumed raw bytes as
                # normal_text and drop this SAX instance.
                self._tool_call_suppressed = True
                self._tool_call_completed = True
                self._last_processed_pos = end_pos
                return
            self._last_processed_pos = end_pos

    def _find_next_complete_element(self, start_pos: int) -> Tuple[Optional[str], int]:
        buf = self._raw_buffer[start_pos:]
        if not buf:
            return None, start_pos

        if buf.startswith("<"):
            next_lt = buf.find("<", 1)
            next_gt = buf.find(">", 1)
            if next_lt != -1 and next_gt != -1:
                if next_lt < next_gt:
                    return buf[:next_lt], start_pos + next_lt
                else:
                    return buf[: next_gt + 1], start_pos + next_gt + 1
            elif next_lt != -1:
                return buf[:next_lt], start_pos + next_lt
            elif next_gt != -1:
                return buf[: next_gt + 1], start_pos + next_gt + 1
            else:
                if self._in_tool_call or self._pre_inside_parameter:
                    return None, start_pos
                for prefix in (
                    "<tool_call>",
                    "</tool_call>",
                    "<function=",
                    "</function>",
                    "<parameter=",
                    "</parameter>",
                ):
                    if prefix.startswith(buf) or buf.startswith(prefix[: len(buf)]):
                        return None, start_pos
                return buf, start_pos + len(buf)
        else:
            next_lt = buf.find("<")
            if next_lt != -1:
                return buf[:next_lt], start_pos + next_lt
            else:
                return buf, start_pos + len(buf)

    def _preprocess_xml_chunk(self, chunk: str) -> str:
        if self._pre_inside_parameter:
            if chunk.startswith("</parameter>"):
                body_text = self._pre_param_buffer
                self._pre_inside_parameter = False
                self._pre_param_buffer = ""
                self._pre_is_string_passthrough = False

                if self._defer_mode:
                    self._defer_raw_value = body_text
                    safe_text = _escape_xml_special_chars(body_text)
                    self._pre_param_name = None
                    return f"{safe_text}</parameter>"
                else:
                    self._pre_param_name = None
                    return "</parameter>"

            if self._pre_param_buffer == "" and not self._defer_mode:
                param_type = (
                    _get_param_type(
                        self._current_function_name or "",
                        self._pre_param_name or "",
                        self.tools,
                    )
                    if self._pre_param_name
                    else "string"
                )
                if _is_string_param_type(param_type):
                    self._pre_is_string_passthrough = True
                    # Sentinel: keeps _pre_param_buffer non-empty so subsequent
                    # string content chunks short-circuit at the passthrough
                    # check below instead of re-running _get_param_type.
                    self._pre_param_buffer = " "
                    return _escape_xml_special_chars(chunk)
                else:
                    self._defer_mode = True
                    self._pre_param_buffer += chunk
                    return ""

            if self._pre_is_string_passthrough:
                return _escape_xml_special_chars(chunk)

            self._pre_param_buffer += chunk
            return ""

        # Rewrite MiMo's <function=name> / <parameter=name> into attribute form
        # so expat can parse them. Note: a parameter/function *name* containing
        # '>' or '"' would break these regexes (the name is matched with [^>]+
        # then re-emitted inside double quotes). MiMo tool/param names are
        # identifiers, so this is not a concern in practice; param *values* with
        # '>'/'"' are handled separately via _escape_xml_special_chars.
        processed = re.sub(r"<function=([^>]+)>", r'<function name="\1">', chunk)
        processed = re.sub(r"<parameter=([^>]+)>", r'<parameter name="\1">', processed)

        m = re.match(r'<parameter name="([^"]+)">', processed)
        if m:
            self._pre_param_name = m.group(1)
            self._pre_inside_parameter = True
            self._pre_param_buffer = ""
            self._defer_mode = False
            self._defer_raw_value = ""
            return processed

        is_tool_tag = any(
            processed.startswith(t)
            for t in (
                "<tool_call>",
                "</tool_call>",
                "<function",
                "</function>",
                "<parameter",
                "</parameter>",
            )
        )
        if not is_tool_tag and self._in_tool_call:
            return _escape_xml_special_chars(chunk)

        return processed

    def _start_element(self, name: str, attrs: Dict[str, str]):
        if name == "root":
            return
        if name == "tool_call":
            self._in_tool_call = True
            self._parameters = {}
            self._param_count = 0
        elif name == "function":
            func_name = attrs.get("name", "").strip()
            self._current_function_name = func_name
            if func_name:
                # Mirror detect_and_parse: unknown functions are dropped
                # unless SGLANG_FORWARD_UNKNOWN_TOOLS is set. Suppressing
                # here (before _emit) is the only place where streaming can
                # avoid leaking an unrecallable tool name to the client.
                if (
                    func_name not in self._tool_indices
                    and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
                ):
                    logger.warning(
                        "Unknown function in streaming tool_call: %s "
                        "(suppressing; raw text will be returned as normal text)",
                        func_name,
                    )
                    self._tool_call_suppressed = True
                    return
                self._emit(name=func_name, parameters="")
        elif name == "parameter":
            param_name = attrs.get("name", "").strip()
            self._current_param_name = param_name
            self._current_param_value = ""
            self._current_param_json_sent = ""
            self._start_quote_emitted = False
            self._trailing_newline_pending = False

            if param_name:
                # The streamed argument fragments are hand-assembled to match
                # json.dumps' DEFAULT separators (", " between items, ": " after
                # keys). The reconciliation on the client side concatenates
                # these fragments into one JSON string, so this must stay in
                # sync with the json.dumps calls below -- do not switch either
                # side to compact separators=(",", ":").
                if self._param_count == 0:
                    json_start = f'{{"{param_name}": '
                else:
                    json_start = f', "{param_name}": '
                self._emit(parameters=json_start)
                self._param_count += 1

    def _char_data(self, data: str):
        if not self._current_param_name:
            return

        if self._defer_mode:
            original_data = data
            if self._trailing_newline_pending:
                original_data = "\n" + original_data
                self._trailing_newline_pending = False
            if original_data.endswith("\n"):
                self._trailing_newline_pending = True
                original_data = original_data[:-1]
            self._current_param_value += original_data
            return

        param_type = _get_param_type(
            self._current_function_name or "",
            self._current_param_name,
            self.tools,
        )

        if _is_string_param_type(param_type) and not self._start_quote_emitted:
            self._emit(parameters='"')
            self._start_quote_emitted = True

        if not data:
            return

        original_data = data
        if self._trailing_newline_pending:
            original_data = "\n" + original_data
            self._trailing_newline_pending = False
        if original_data.endswith("\n"):
            self._trailing_newline_pending = True
            original_data = original_data[:-1]

        self._current_param_value += original_data

        if _is_string_param_type(param_type):
            # Escape only the newly-arrived characters and append, rather than
            # re-dumping the whole accumulated value each chunk (which would be
            # O(n^2) on long string params -- exactly this PR's target case).
            # JSON string escaping is per-character, so escaping the increment
            # in isolation is equivalent; the only cross-chunk concern, a
            # trailing newline, is already deferred via _trailing_newline_pending.
            diff = json.dumps(original_data, ensure_ascii=False)[1:-1]
            self._current_param_json_sent += diff
            if diff:
                self._emit(parameters=diff)
        else:
            # Unreachable in normal flow: non-string params are intercepted by
            # _preprocess_xml_chunk and routed into defer mode before any
            # content reaches expat. The defer branch above handles them.
            # Kept only as a safety net if preprocessing logic changes -- the
            # warning surfaces such a regression instead of failing silently.
            logger.warning(
                "Non-string param %r reached _char_data streaming path; "
                "expected it to be deferred by _preprocess_xml_chunk. "
                "Preprocessing logic may have regressed.",
                self._current_param_name,
            )
            converted = _convert_param_value(
                self._current_param_value,
                self._current_param_name,
                self._current_function_name or "",
                self.tools,
            )
            output = json.dumps(converted, ensure_ascii=False)
            diff = output[len(self._current_param_json_sent) :]
            self._current_param_json_sent = output
            if diff:
                self._emit(parameters=diff)

    def _end_element(self, name: str):
        if name == "root":
            return

        if name == "parameter" and self._current_param_name:
            param_name = self._current_param_name
            param_value = self._current_param_value

            if self._trailing_newline_pending:
                param_value += "\n"
                self._trailing_newline_pending = False

            if self._defer_mode:
                raw_text = (
                    self._defer_raw_value if self._defer_raw_value else param_value
                )
                converted = _convert_param_value(
                    raw_text,
                    param_name,
                    self._current_function_name or "",
                    self.tools,
                )
                output = json.dumps(converted, ensure_ascii=False)
                self._emit(parameters=output)
                self._parameters[param_name] = converted
            else:
                param_type = _get_param_type(
                    self._current_function_name or "",
                    param_name,
                    self.tools,
                )

                if _is_string_param_type(param_type):
                    if not param_value and not self._start_quote_emitted:
                        self._emit(parameters='""')
                    else:
                        remaining_json = json.dumps(param_value, ensure_ascii=False)[
                            1:-1
                        ]
                        diff = remaining_json[len(self._current_param_json_sent) :]
                        if diff:
                            self._emit(parameters=diff)
                        self._emit(parameters='"')
                    self._parameters[param_name] = param_value
                else:
                    # Reachable only when the parameter has no content at all
                    # (e.g. <parameter=n></parameter>): preprocessing never
                    # entered defer mode, so we land here with param_value="".
                    # For non-string types, emit JSON null rather than letting
                    # _convert_param_value fall back to the empty string "".
                    if param_value == "":
                        self._emit(parameters="null")
                        self._parameters[param_name] = None
                    else:
                        converted = _convert_param_value(
                            param_value,
                            param_name,
                            self._current_function_name or "",
                            self.tools,
                        )
                        output = json.dumps(converted, ensure_ascii=False)
                        diff = output[len(self._current_param_json_sent) :]
                        if diff:
                            self._emit(parameters=diff)
                        self._parameters[param_name] = converted

            self._current_param_name = None
            self._current_param_value = ""
            self._current_param_json_sent = ""
            self._start_quote_emitted = False
            self._defer_mode = False
            self._defer_raw_value = ""

        elif name == "function":
            if self._parameters:
                self._emit(parameters="}")
            else:
                self._emit(parameters="{}")
            self._current_function_name = None

        elif name == "tool_call":
            self._tool_call_completed = True
            self._in_tool_call = False


class MiMoDetector(BaseFormatDetector):
    """
    Detector for MiMo function call format.

    Format:
        <tool_call>
        <function=execute_bash>
        <parameter=command>pwd && ls</parameter>
        </function>
        </tool_call>

    Supports per-token streaming for string parameters via SAX-based
    incremental XML parsing. Complex-type parameters (object/array) are
    buffered until </parameter> closes, then emitted at once.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.func_regex = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
        self.param_regex = re.compile(
            r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL
        )
        self._sax: Optional[_MiMoStreamingSAX] = None

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse complete text for tool calls."""
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text = text[:idx]
        tool_indices = self._get_tool_indices(tools)

        calls = []
        last_end = idx

        for match in self.tool_call_regex.finditer(text):
            tool_call_body = match.group(1)

            parsed = self._parse_tool_call(tool_call_body, tools)

            if parsed:
                func_name = parsed.get("name")
                if func_name not in tool_indices:
                    # Unknown function
                    logger.warning(f"Unknown function: {func_name}")
                    if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                        # Return tool call block as normal text
                        normal_text += text[last_end : match.end()]
                        last_end = match.end()
                        continue
                calls.extend(self.parse_base_json(parsed, tools))

            last_end = match.end()

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Per-token streaming parsing using SAX-based incremental XML parser.

        String parameters are streamed as JSON fragments per-token.
        Complex-type parameters are buffered until </parameter> closes.

        Note: text appearing *between* tool calls is surfaced as normal_text
        here, whereas the one-shot ``detect_and_parse`` drops it (it only keeps
        text before the first tool call). This is intentional -- streaming
        cannot know a later tool call is coming, so it must forward interstitial
        text as it arrives rather than retroactively discard it.
        """
        self._buffer += new_text
        all_calls: List[ToolCallItem] = []
        all_normal_text = ""

        while True:
            if self._sax is None:
                start = self._buffer.find(self.bot_token)
                if start == -1:
                    if self._ends_with_partial_token(self._buffer, self.bot_token):
                        break
                    all_normal_text += self._buffer
                    self._buffer = ""
                    break

                all_normal_text += self._buffer[:start]
                self._buffer = self._buffer[start:]
                self._sax = _MiMoStreamingSAX(tools)
                if self.current_tool_id < 0:
                    self.current_tool_id = 0
                self._ensure_tracking_arrays()

            calls = self._sax.feed_chunk(self._buffer)
            self._buffer = ""
            all_calls.extend(self._process_sax_calls(calls))

            if self._sax.tool_call_completed:
                if self._sax._tool_call_suppressed:
                    # Suppression only happens before anything was emitted
                    # (unknown function, or a parse error before the first
                    # _emit). Nothing reached the client, so the consumed raw
                    # bytes flow back out as normal_text and current_tool_id is
                    # NOT bumped. A parse error *after* partial emit does not
                    # take this path -- the SAX finalizes the partial call
                    # instead (tool_call_completed without suppression), which
                    # falls through to the normal completion branch below.
                    all_normal_text += self._sax._raw_buffer[
                        : self._sax._last_processed_pos
                    ]
                    remaining = self._sax.get_remaining_buffer()
                    self._sax = None
                    if remaining.strip():
                        self._buffer = remaining
                        continue
                    break

                if self._sax._parameters and self.current_tool_id < len(
                    self.prev_tool_call_arr
                ):
                    self.prev_tool_call_arr[self.current_tool_id][
                        "arguments"
                    ] = self._sax._parameters.copy()
                self.current_tool_id += 1
                errored = self._sax._tool_call_errored
                remaining = self._sax.get_remaining_buffer()
                self._sax = None
                if errored:
                    # The partial call was finalized after a parse error. Its
                    # leftover bytes (the tail of the broken tool_call, e.g. a
                    # dangling </function>) are garbage: drop them up to the
                    # next <tool_call> rather than leaking them as normal_text.
                    next_call = remaining.find(self.bot_token)
                    if next_call != -1:
                        self._buffer = remaining[next_call:]
                        continue
                    break
                if remaining.strip():
                    self._buffer = remaining
                    continue
            break

        return StreamingParseResult(normal_text=all_normal_text, calls=all_calls)

    def _ensure_tracking_arrays(self):
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

    def _process_sax_calls(self, calls: List[ToolCallItem]) -> List[ToolCallItem]:
        result = []
        for call in calls:
            call.tool_index = self.current_tool_id
            if call.name is not None:
                self._ensure_tracking_arrays()
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": call.name,
                    "arguments": {},
                }
            if call.parameters:
                self._ensure_tracking_arrays()
                self.streamed_args_for_tool[self.current_tool_id] += call.parameters
            result.append(call)
        return result

    def _parse_tool_call(
        self, tool_call_body: str, tools: List[Tool]
    ) -> Dict[str, Any]:
        """
        Parse content inside <tool_call>...</tool_call>.

        Structure:
            tool_call_body contains: <function=name>...params...</function>
        """
        func_match = self.func_regex.search(tool_call_body)
        if not func_match:
            return None

        func_name = func_match.group(1).strip()
        func_body = func_match.group(2)

        params = {}
        for param_match in self.param_regex.finditer(func_body):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2)
            params[param_name] = _convert_param_value(
                param_value, param_name, func_name, tools
            )

        return {"name": func_name, "parameters": params}

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
