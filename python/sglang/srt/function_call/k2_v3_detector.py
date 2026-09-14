"""Tool-call parsing for the canonical K2 Horizon (K2-v3) IFM format."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    get_schema_properties,
    infer_type_from_json_schema,
)

logger = logging.getLogger(__name__)


class K2V3Detector(BaseFormatDetector):
    """Parse K2 Horizon's canonical ``<ifm|...>`` tool-call format.

    The wire content is self-describing: XML uses key/value tags, typed XML
    adds argument-type tags, and JSON begins with an object or array.

    Streaming keeps a partial block atomic and emits it as soon as its closing
    tag arrives. This deliberately favors correctness across arbitrarily split
    tags over emitting incomplete JSON argument fragments.
    """

    _TOOL_CALLS_START = "<ifm|tool_calls>"
    _TOOL_CALLS_END = "</ifm|tool_calls>"
    _TOOL_CALL_START = "<ifm|tool_call>"
    _TOOL_CALL_END = "</ifm|tool_call>"

    _ARG_KEY_START = "<ifm|arg_key>"

    _THINK_PAIRS = (
        ("<ifm|think>", "</ifm|think>"),
        ("<ifm|think_fast>", "</ifm|think_fast>"),
        ("<ifm|think_faster>", "</ifm|think_faster>"),
    )
    _GROUP_TOKENS = (_TOOL_CALLS_START, _TOOL_CALLS_END)
    _STREAM_MARKERS = (
        _TOOL_CALL_START,
        _TOOL_CALLS_START,
        _TOOL_CALLS_END,
        "<ifm|think>",
        "</ifm|think>",
        "<ifm|think_fast>",
        "</ifm|think_fast>",
        "<ifm|think_faster>",
        "</ifm|think_faster>",
    )

    _ARG_PATTERN = re.compile(
        r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*"
        r"(?:<ifm\|arg_type>(.*?)</ifm\|arg_type>\s*)?"
        r"<ifm\|arg_value>(.*?)</ifm\|arg_value>",
        re.DOTALL,
    )

    def __init__(self) -> None:
        super().__init__()
        self.bot_token = self._TOOL_CALL_START
        self.eot_token = self._TOOL_CALL_END
        self._next_tool_index = 0
        self._at_stream_start = True
        self._inside_tool_group = False

    def has_tool_call(self, text: str) -> bool:
        return self._TOOL_CALL_START in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text)

        # Reasoning extraction belongs to ReasoningParser. Keep a leading IFM
        # reasoning block out of tool parsing, but pass it through unchanged so
        # separate_reasoning=False returns the original assistant output.
        reasoning_prefix, working = self._split_leading_reasoning(text)

        # Canonical template output wraps all calls in one group. Whitespace
        # inside that wrapper is formatting, while the prefix immediately
        # before it (notably the newline after </ifm|think>) is user-visible.
        group_start = working.find(self._TOOL_CALLS_START)
        if group_start != -1:
            group_body_start = group_start + len(self._TOOL_CALLS_START)
            group_end = working.find(self._TOOL_CALLS_END, group_body_start)
            if group_end != -1:
                parsed_group = self._parse_complete_region(
                    working[group_body_start:group_end], tools
                )
                if parsed_group is None:
                    # The release template emits one canonical group. Avoid a
                    # partial recovery that silently drops its wrapper or only
                    # some malformed calls: surface the wire text intact.
                    return StreamingParseResult(normal_text=reasoning_prefix + working)
                normal_text = (
                    working[:group_start]
                    + working[group_end + len(self._TOOL_CALLS_END) :]
                )
                return StreamingParseResult(
                    normal_text=reasoning_prefix + normal_text,
                    calls=self._items_from_calls(parsed_group, tools, 0),
                )

        calls: list[ToolCallItem] = []
        normal_parts: list[str] = []
        cursor = 0

        while True:
            start = working.find(self._TOOL_CALL_START, cursor)
            if start == -1:
                normal_parts.append(working[cursor:])
                break

            normal_parts.append(working[cursor:start])
            body_start = start + len(self._TOOL_CALL_START)
            end = working.find(self._TOOL_CALL_END, body_start)
            if end == -1:
                # Preserve an unterminated block verbatim instead of silently
                # dropping the model's output.
                normal_parts.append(working[start:])
                break

            raw_block = working[start : end + len(self._TOOL_CALL_END)]
            parsed = self._parse_block(working[body_start:end], tools)
            if parsed is None:
                normal_parts.append(raw_block)
            else:
                calls.extend(self._items_from_calls(parsed, tools, len(calls)))
            cursor = end + len(self._TOOL_CALL_END)

        normal_text = reasoning_prefix + self._strip_group_tokens("".join(normal_parts))
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        return self._drain_stream(tools, flush=False)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        return self._drain_stream(tools, flush=True)

    def _drain_stream(self, tools: List[Tool], *, flush: bool) -> StreamingParseResult:
        normal_parts: list[str] = []
        calls: list[ToolCallItem] = []

        while self._buffer:
            if self._at_stream_start:
                state, reasoning_prefix = self._consume_stream_reasoning_prefix(
                    flush=flush
                )
                if state == "hold":
                    break
                if state == "consumed":
                    normal_parts.append(reasoning_prefix)
                    continue
                self._at_stream_start = False

            start = self._buffer.find(self._TOOL_CALL_START)
            if self._inside_tool_group:
                # The line breaks surrounding calls are part of IFM framing,
                # not assistant content. Hold/drop them with the wrapper.
                self._buffer = self._buffer.lstrip()
                if self._buffer.startswith(self._TOOL_CALLS_END):
                    self._buffer = self._buffer[len(self._TOOL_CALLS_END) :]
                    self._inside_tool_group = False
                    continue
                if (
                    self._TOOL_CALLS_END.startswith(self._buffer)
                    and self._buffer != self._TOOL_CALLS_END
                    and not flush
                ):
                    break
                start = self._buffer.find(self._TOOL_CALL_START)
            else:
                group_start = self._buffer.find(self._TOOL_CALLS_START)
                if group_start != -1 and (start == -1 or group_start < start):
                    normal_parts.append(self._buffer[:group_start])
                    self._buffer = self._buffer[
                        group_start + len(self._TOOL_CALLS_START) :
                    ]
                    self._inside_tool_group = True
                    continue

            if start == -1:
                visible, hold = self._split_visible_stream_text(
                    self._buffer, flush=flush
                )
                normal_parts.append(visible)
                self._buffer = hold
                break

            prefix = self._buffer[:start]
            normal_parts.append(self._strip_group_tokens(prefix))
            self._buffer = self._buffer[start:]

            body_start = len(self._TOOL_CALL_START)
            end = self._buffer.find(self._TOOL_CALL_END, body_start)
            if end == -1:
                if flush:
                    normal_parts.append(self._buffer)
                    self._buffer = ""
                break

            raw_block = self._buffer[: end + len(self._TOOL_CALL_END)]
            parsed = self._parse_block(self._buffer[body_start:end], tools)
            if parsed is None:
                normal_parts.append(raw_block)
            else:
                new_items = self._items_from_calls(parsed, tools, self._next_tool_index)
                calls.extend(new_items)
                self._next_tool_index += len(new_items)
            self._buffer = self._buffer[end + len(self._TOOL_CALL_END) :]

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def _parse_complete_region(
        self, text: str, tools: List[Tool]
    ) -> Optional[list[tuple[str, dict[str, Any]]]]:
        """Parse a canonical tool-call group, requiring full framing.

        Only whitespace may occur between calls. Any malformed or unterminated
        block fails the group atomically so its raw text can be surfaced.
        """

        parsed: list[tuple[str, dict[str, Any]]] = []
        cursor = 0
        while cursor < len(text):
            start = text.find(self._TOOL_CALL_START, cursor)
            if start == -1:
                return parsed if parsed and not text[cursor:].strip() else None
            if text[cursor:start].strip():
                return None
            body_start = start + len(self._TOOL_CALL_START)
            end = text.find(self._TOOL_CALL_END, body_start)
            if end == -1:
                return None
            block_calls = self._parse_block(text[body_start:end], tools)
            if block_calls is None:
                return None
            parsed.extend(block_calls)
            cursor = end + len(self._TOOL_CALL_END)
        return parsed or None

    def _consume_stream_reasoning_prefix(self, *, flush: bool) -> tuple[str, str]:
        """Separate a leading reasoning block from tool parsing without loss."""

        stripped = self._buffer.lstrip()
        whitespace_len = len(self._buffer) - len(stripped)
        for start_token, end_token in self._THINK_PAIRS:
            if (
                stripped
                and start_token.startswith(stripped)
                and stripped != start_token
            ):
                if not flush:
                    return "hold", ""
                passthrough = self._buffer
                self._buffer = ""
                return "consumed", passthrough
            if not stripped.startswith(start_token):
                continue
            end = stripped.find(end_token, len(start_token))
            if end == -1:
                if not flush:
                    return "hold", ""
                passthrough = self._buffer
                self._buffer = ""
                return "consumed", passthrough
            consumed_len = whitespace_len + end + len(end_token)
            passthrough = self._buffer[:consumed_len]
            self._buffer = self._buffer[consumed_len:]
            return "consumed", passthrough

        # Whitespace at the beginning may precede a reasoning token in the
        # next chunk, so keep it until that distinction is observable.
        if not stripped and whitespace_len and not flush:
            return "hold", ""
        return "none", ""

    def _split_visible_stream_text(self, text: str, *, flush: bool) -> tuple[str, str]:
        if flush:
            return self._strip_group_tokens(text), ""

        partial_len = max(
            (
                self._partial_marker_suffix_len(text, marker)
                for marker in self._STREAM_MARKERS
            ),
            default=0,
        )
        if partial_len:
            visible, hold = text[:-partial_len], text[-partial_len:]
        else:
            visible, hold = text, ""
        return self._strip_group_tokens(visible), hold

    @staticmethod
    def _partial_marker_suffix_len(text: str, marker: str) -> int:
        for size in range(min(len(text), len(marker) - 1), 0, -1):
            if marker.startswith(text[-size:]):
                return size
        return 0

    @classmethod
    def _strip_group_tokens(cls, text: str) -> str:
        for token in cls._GROUP_TOKENS:
            text = text.replace(token, "")
        return text

    @classmethod
    def _split_leading_reasoning(cls, text: str) -> tuple[str, str]:
        """Return an exact leading reasoning prefix and the parseable suffix."""

        cursor = 0
        while cursor < len(text):
            remainder = text[cursor:]
            stripped = remainder.lstrip()
            whitespace_len = len(remainder) - len(stripped)
            matched_pair = next(
                (
                    (start_token, end_token)
                    for start_token, end_token in cls._THINK_PAIRS
                    if stripped.startswith(start_token)
                ),
                None,
            )
            if matched_pair is None:
                # A truncated reasoning opener is still passthrough content;
                # never reinterpret tool-looking text following it.
                if stripped and any(
                    start_token.startswith(stripped)
                    for start_token, _ in cls._THINK_PAIRS
                ):
                    return text, ""
                return text[:cursor], text[cursor:]

            start_token, end_token = matched_pair
            end = stripped.find(end_token, len(start_token))
            if end == -1:
                return text, ""
            cursor += whitespace_len + end + len(end_token)

        return text, ""

    def _parse_block(
        self, block: str, tools: List[Tool]
    ) -> Optional[list[tuple[str, dict[str, Any]]]]:
        # The wire framing is self-describing: JSON begins with an object or
        # array, while XML and typed XML share the same tag parser.
        stripped = block.strip()
        looks_json = stripped.startswith(("{", "["))
        try:
            if looks_json:
                return self._parse_json_block(stripped, tools)
            return self._parse_xml_block(block, tools)
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Malformed K2-v3 tool-call block; forwarding it as text")
            return None

    def _parse_json_block(
        self, block: str, tools: List[Tool]
    ) -> list[tuple[str, dict[str, Any]]]:
        payload = json.loads(block)
        raw_calls = payload if isinstance(payload, list) else [payload]
        parsed: list[tuple[str, dict[str, Any]]] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, Mapping):
                raise ValueError("K2-v3 JSON tool call must be an object")
            function = raw_call.get("function", raw_call)
            if (
                not isinstance(function, Mapping)
                or not isinstance(function.get("name"), str)
                or not function["name"]
            ):
                raise ValueError("K2-v3 JSON tool call is missing a function name")
            name = function["name"]
            arguments = function.get("arguments", function.get("parameters", {}))
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments.strip() else {}
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, Mapping):
                raise ValueError("K2-v3 tool-call arguments must be an object")
            parsed.append(
                (
                    name,
                    self._coerce_arguments(
                        name, dict(arguments), tools, from_text=False
                    ),
                )
            )
        return parsed

    def _parse_xml_block(
        self, block: str, tools: List[Tool]
    ) -> list[tuple[str, dict[str, Any]]]:
        first_arg = block.find(self._ARG_KEY_START)
        name = (block if first_arg == -1 else block[:first_arg]).strip()
        if not name:
            raise ValueError("K2-v3 XML tool call is missing a function name")
        if first_arg == -1:
            return [(name, {})]

        argument_text = block[first_arg:]
        arguments: dict[str, Any] = {}
        cursor = 0
        for match in self._ARG_PATTERN.finditer(argument_text):
            if argument_text[cursor : match.start()].strip():
                raise ValueError("Malformed K2-v3 XML argument framing")
            key = match.group(1).strip()
            if not key:
                raise ValueError("K2-v3 XML argument is missing a key")
            inline_type = match.group(2).strip() if match.group(2) else None
            arguments[key] = self._coerce_value(
                match.group(3),
                name,
                key,
                tools,
                inline_type=inline_type,
                from_text=True,
            )
            cursor = match.end()
        if argument_text[cursor:].strip():
            raise ValueError("Malformed K2-v3 XML argument framing")
        return [(name, arguments)]

    def _items_from_calls(
        self,
        parsed: list[tuple[str, dict[str, Any]]],
        tools: List[Tool],
        start_index: int,
    ) -> list[ToolCallItem]:
        known_names = {tool.function.name for tool in tools}
        items: list[ToolCallItem] = []
        for name, arguments in parsed:
            if name not in known_names and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                logger.warning("Model attempted to call undefined function: %s", name)
                continue
            items.append(
                ToolCallItem(
                    tool_index=start_index + len(items),
                    name=name,
                    parameters=json.dumps(arguments, ensure_ascii=False),
                )
            )
        return items

    @staticmethod
    def _resolve_local_ref(schema: Any, root_schema: Any) -> Optional[dict[str, Any]]:
        """Resolve the local references emitted by the K2 tool template.

        The release templates inline ``#/$defs/...`` and ``#/definitions/...``
        before presenting a tool to the model. Parsing must inspect that same
        effective schema; otherwise a referenced string such as ``"00123"``
        is incorrectly coerced to an integer.
        """

        if not isinstance(schema, Mapping):
            return None
        resolved = dict(schema)
        seen: set[str] = set()
        while isinstance(resolved.get("$ref"), str):
            ref = resolved["$ref"]
            if ref in seen:
                return None
            seen.add(ref)
            if ref.startswith("#/$defs/"):
                definitions = (
                    root_schema.get("$defs", {})
                    if isinstance(root_schema, Mapping)
                    else {}
                )
                name = ref.removeprefix("#/$defs/")
            elif ref.startswith("#/definitions/"):
                definitions = (
                    root_schema.get("definitions", {})
                    if isinstance(root_schema, Mapping)
                    else {}
                )
                name = ref.removeprefix("#/definitions/")
            else:
                return resolved
            target = definitions.get(name) if isinstance(definitions, Mapping) else None
            if not isinstance(target, Mapping):
                return resolved
            # Match the template: annotations at the reference site override
            # fields from the referenced definition.
            resolved = {
                **target,
                **{key: value for key, value in resolved.items() if key != "$ref"},
            }
        return resolved

    @classmethod
    def _argument_type(
        cls, tool_name: str, argument_name: str, tools: List[Tool]
    ) -> Optional[str]:
        for tool in tools:
            if tool.function.name != tool_name:
                continue
            root_schema = tool.function.parameters
            schema = get_schema_properties(root_schema).get(argument_name)
            schema = cls._resolve_local_ref(schema, root_schema)
            if not isinstance(schema, dict):
                return None

            # For unions the canonical xml_typed template emits the actual
            # argument value type. Do not guess one branch and override that
            # wire-level type hint. Untyped XML can still infer from its value,
            # and JSON values have already been decoded with their true type.
            schema_type = schema.get("type")
            if "anyOf" in schema or "oneOf" in schema:
                return None
            if (
                isinstance(schema_type, list)
                and len([item for item in schema_type if item != "null"]) > 1
            ):
                return None
            return infer_type_from_json_schema(schema)
        return None

    @classmethod
    def _coerce_value(
        cls,
        value: Any,
        tool_name: str,
        argument_name: str,
        tools: List[Tool],
        *,
        inline_type: Optional[str] = None,
        from_text: bool,
    ) -> Any:
        target_type = cls._argument_type(tool_name, argument_name, tools) or inline_type
        if target_type == "any":
            return value
        if target_type == "string":
            return (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
        if not isinstance(value, str):
            return value
        if target_type is None:
            # JSON strings are already unambiguously strings after json.loads.
            # Only the XML dialect needs best-effort JSON decoding of an
            # untyped textual value.
            if not from_text:
                return value
            try:
                return json.loads(value.strip())
            except json.JSONDecodeError:
                return value
        try:
            return json.loads(value.strip())
        except json.JSONDecodeError:
            return value

    @classmethod
    def _coerce_arguments(
        cls,
        tool_name: str,
        arguments: dict[str, Any],
        tools: List[Tool],
        *,
        from_text: bool,
    ) -> dict[str, Any]:
        return {
            key: cls._coerce_value(value, tool_name, key, tools, from_text=from_text)
            for key, value in arguments.items()
        }

    def supports_structural_tag(self) -> bool:
        # XML arguments are not a JSON-schema body. Required/named tool choice
        # therefore uses SGLang's standard JSON constraint and JsonArrayParser.
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "K2-v3 native IFM does not use legacy structural tags"
        )
