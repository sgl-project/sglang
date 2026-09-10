import json
import re
from dataclasses import dataclass
from typing import Any

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import get_schema_properties

TOOL_CALL_BEGIN = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
ARG_KEY_BEGIN = "<arg_key>"
ARG_KEY_END = "</arg_key>"
ARG_VALUE_BEGIN = "<arg_value>"
ARG_VALUE_END = "</arg_value>"

ARG_PAIR_PATTERN = re.compile(
    rf"{re.escape(ARG_KEY_BEGIN)}(.*?){re.escape(ARG_KEY_END)}"
    rf"{re.escape(ARG_VALUE_BEGIN)}(.*?){re.escape(ARG_VALUE_END)}",
    re.DOTALL,
)


@dataclass(frozen=True)
class _Spark2_5ToolCall:
    name: str
    arguments: dict[str, Any]

    def arguments_json(self) -> str:
        return json.dumps(
            self.arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _get_param_type(tools: list[Tool], function_name: str, param_name: str) -> str:
    """Return a parameter's declared JSON Schema type, or ``string``."""
    for tool in tools:
        function = getattr(tool, "function", None)
        if function is None or function.name != function_name:
            continue
        parameters = getattr(function, "parameters", None)
        if not isinstance(parameters, dict):
            continue
        definition = get_schema_properties(parameters).get(param_name)
        if isinstance(definition, dict) and isinstance(definition.get("type"), str):
            return definition["type"]
    return "string"


def _convert_value(value: str, param_type: str) -> Any:
    """Convert Spark2_5 XML text according to the model's tool protocol."""
    if value.lower() == "null":
        return None

    normalized_type = param_type.lower()
    try:
        if normalized_type in {"string", "str", "text"}:
            return value
        if normalized_type in {"integer", "int"}:
            return int(value)
        if normalized_type in {"number", "float"}:
            number = float(value)
            return int(number) if number.is_integer() else number
        if normalized_type in {"boolean", "bool"}:
            normalized_value = value.strip().lower()
            if normalized_value not in {"true", "1", "false", "0"}:
                raise ValueError(f"invalid boolean: {value}")
            return normalized_value in {"true", "1"}
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value


def _parse_tool_call_xml(tool_xml: str, tools: list[Tool]) -> _Spark2_5ToolCall | None:
    if not tool_xml.startswith(TOOL_CALL_BEGIN) or not tool_xml.endswith(TOOL_CALL_END):
        return None

    body = tool_xml[len(TOOL_CALL_BEGIN) : -len(TOOL_CALL_END)]
    first_arg = body.find(ARG_KEY_BEGIN)
    function_name = (body if first_arg < 0 else body[:first_arg]).strip()
    if not function_name:
        return None

    arguments: dict[str, Any] = {}
    for match in ARG_PAIR_PATTERN.finditer(body):
        key, raw_value = match.group(1), match.group(2)
        if not key:
            continue
        arguments[key] = _convert_value(
            raw_value,
            _get_param_type(tools, function_name, key),
        )
    return _Spark2_5ToolCall(name=function_name, arguments=arguments)


def _partial_marker_suffix_length(text: str, marker: str) -> int:
    """Length of the suffix that may become ``marker`` in the next chunk."""
    for size in range(min(len(text), len(marker) - 1), 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


class Spark25Detector(BaseFormatDetector):
    """Detector for Spark2_5's XML-KV tool-call format.

    Wire format::

        <tool_call>function_name
        <arg_key>key</arg_key><arg_value>value</arg_value>
        </tool_call>

    Values are converted with the parameter's JSON Schema type. A complete
    block is emitted atomically in streaming mode so XML fragments are never
    exposed as JSON argument deltas.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOL_CALL_BEGIN
        self.eot_token = TOOL_CALL_END

    def has_tool_call(self, text: str) -> bool:
        return TOOL_CALL_BEGIN in text

    def _build_item(
        self,
        parsed: _Spark2_5ToolCall,
        tools: list[Tool],
        tool_index: int,
    ) -> ToolCallItem | None:
        validated = self.parse_base_json(
            {"name": parsed.name, "arguments": parsed.arguments}, tools
        )
        if not validated:
            return None
        return ToolCallItem(
            tool_index=tool_index,
            name=parsed.name,
            parameters=parsed.arguments_json(),
        )

    def _record_streamed_item(
        self, parsed: _Spark2_5ToolCall, item: ToolCallItem
    ) -> None:
        self.prev_tool_call_arr.append(
            {"name": parsed.name, "arguments": parsed.arguments}
        )
        self.streamed_args_for_tool.append(item.parameters)

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        calls: list[ToolCallItem] = []
        normal_parts: list[str] = []
        cursor = 0

        while cursor < len(text):
            start = text.find(TOOL_CALL_BEGIN, cursor)
            if start < 0:
                normal_parts.append(text[cursor:])
                break

            normal_parts.append(text[cursor:start])
            end = text.find(TOOL_CALL_END, start + len(TOOL_CALL_BEGIN))
            if end < 0:
                normal_parts.append(text[start:])
                break

            end += len(TOOL_CALL_END)
            raw_tool_call = text[start:end]
            parsed = _parse_tool_call_xml(raw_tool_call, tools)
            if parsed is None:
                normal_parts.append(raw_tool_call)
            else:
                item = self._build_item(parsed, tools, len(calls))
                if item is not None:
                    calls.append(item)
            cursor = end

        return StreamingParseResult(
            normal_text="".join(normal_parts),
            calls=calls,
        )

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        calls: list[ToolCallItem] = []
        normal_parts: list[str] = []

        while self._buffer:
            start = self._buffer.find(TOOL_CALL_BEGIN)
            if start < 0:
                keep = _partial_marker_suffix_length(self._buffer, TOOL_CALL_BEGIN)
                if keep:
                    normal_parts.append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                else:
                    normal_parts.append(self._buffer)
                    self._buffer = ""
                break

            if start > 0:
                normal_parts.append(self._buffer[:start])
                self._buffer = self._buffer[start:]

            end = self._buffer.find(TOOL_CALL_END, len(TOOL_CALL_BEGIN))
            if end < 0:
                break

            end += len(TOOL_CALL_END)
            raw_tool_call = self._buffer[:end]
            self._buffer = self._buffer[end:]
            parsed = _parse_tool_call_xml(raw_tool_call, tools)
            if parsed is None:
                normal_parts.append(raw_tool_call)
                continue

            item = self._build_item(parsed, tools, self.current_tool_id + 1)
            if item is not None:
                self.current_tool_id += 1
                self._record_streamed_item(parsed, item)
                calls.append(item)

        return StreamingParseResult(
            normal_text="".join(normal_parts),
            calls=calls,
        )

    def finish(self, tools: list[Tool]) -> StreamingParseResult:
        del tools
        pending = self._buffer
        self._buffer = ""
        if TOOL_CALL_BEGIN in pending:
            # A complete opening marker means this is a truncated protocol
            # block, not user-visible text. Partial marker prefixes are still
            # released because the stream has ended and they cannot become a
            # tool call anymore.
            pending = pending[: pending.find(TOOL_CALL_BEGIN)]
        return StreamingParseResult(normal_text=pending)

    def supports_structural_tag(self) -> bool:
        return False

    def parses_required_natively(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "Spark2_5 XML arguments cannot be represented by legacy structural tags"
        )
