import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


MINIMAX_NS_TOKEN = "]<]minimax[>["
STRING_TYPES = {"string", "str"}
INTEGER_TYPES = {"integer", "int"}
NUMBER_TYPES = {"number", "float"}
BOOLEAN_TYPES = {"boolean", "bool"}
SCALAR_TYPES = STRING_TYPES | INTEGER_TYPES | NUMBER_TYPES | BOOLEAN_TYPES | {"null"}
CONTAINER_TYPES = {"object", "array"}
# Exact-case only; "none"/"nil" are valid string enum values, never null-coerced.
NULL_STRINGS = {"null", "Null", "NULL"}


class MinimaxM3Detector(BaseFormatDetector):
    TOOL_CALL_START = MINIMAX_NS_TOKEN + "<tool_call>"
    TOOL_CALL_END = MINIMAX_NS_TOKEN + "</tool_call>"

    INVOKE_PREFIX = MINIMAX_NS_TOKEN + '<invoke name="'
    INVOKE_SUFFIX = MINIMAX_NS_TOKEN + "</invoke>"
    PARAM_START_PREFIX = MINIMAX_NS_TOKEN + "<"
    TAG_SPACING_CHARS = " \t\r\n"
    TAG_SPACING_RE = re.compile(re.escape(MINIMAX_NS_TOKEN) + r"[ \t\r\n]+(?=<)")

    def __init__(self):
        super().__init__()
        self._in_tool_call = False
        self._current_function_name: Optional[str] = None
        self._current_function_schema: Optional[Dict[str, Any]] = None
        self._current_param_name: Optional[str] = None
        self._current_param_schema: Optional[Dict[str, Any]] = None
        self._current_param_buffer = ""
        self._current_param_is_complex = False
        self._current_string_started = False
        self._is_first_param = True

    @classmethod
    def _normalize_tag_spacing(cls, text: str) -> str:
        return cls.TAG_SPACING_RE.sub(MINIMAX_NS_TOKEN, text)

    @classmethod
    def _flushable_prefix_length(cls, text: str, token: str) -> int:
        for length in range(min(len(token) - 1, len(text)), 0, -1):
            if token.startswith(text[-length:]):
                return len(text) - length

        namespace_start = text.rfind(MINIMAX_NS_TOKEN)
        if namespace_start == -1:
            return len(text)

        suffix = text[namespace_start + len(MINIMAX_NS_TOKEN) :]
        if suffix and all(ch in cls.TAG_SPACING_CHARS for ch in suffix):
            return namespace_start
        return len(text)

    def has_tool_call(self, text: str) -> bool:
        return self.TOOL_CALL_START in self._normalize_tag_spacing(text)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        original_text = text
        try:
            text = self._normalize_tag_spacing(text)
            normal_text, calls = self._extract(text, tools)
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as exc:
            logger.warning(
                "invalid MiniMax M3 tool call returned as content: %s",
                exc,
                exc_info=True,
            )
            return StreamingParseResult(normal_text=original_text, calls=[])

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def _extract(self, text: str, tools: List[Tool]) -> Tuple[str, List[ToolCallItem]]:
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        cursor = 0
        while True:
            start = text.find(self.TOOL_CALL_START, cursor)
            if start == -1:
                normal_parts.append(text[cursor:])
                break

            normal_parts.append(text[cursor:start])

            end = text.find(self.TOOL_CALL_END, start)
            if end == -1:
                normal_parts.append(text[start:])
                break

            block = text[start + len(self.TOOL_CALL_START) : end]
            cursor = end + len(self.TOOL_CALL_END)
            calls.extend(self._parse_block(block, tools))

        return "".join(normal_parts), calls

    def _parse_block(self, block: str, tools: List[Tool]) -> List[ToolCallItem]:
        results: List[ToolCallItem] = []
        cursor = 0
        while True:
            start = block.find(self.INVOKE_PREFIX, cursor)
            end = block.find(self.INVOKE_SUFFIX, start)
            if start == -1 or end == -1:
                break

            invoke_str = block[start:end]
            cursor = end + len(self.INVOKE_SUFFIX)

            name_end = invoke_str.find('">', len(self.INVOKE_PREFIX))
            if name_end == -1:
                continue

            func_name = invoke_str[len(self.INVOKE_PREFIX) : name_end]
            body = invoke_str[name_end + len('">') :]
            params = self._parse_parameter(
                body, self._get_function_parameters_schema(func_name, tools)
            )
            action = {"name": func_name, "arguments": params}
            try:
                parsed_calls = self.parse_base_json(action, tools)
                for call in parsed_calls:
                    call.tool_index = len(results)
                    results.append(call)
            except Exception:
                logger.warning("invalid tool call for %s dropped", func_name)
        return results

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        self._buffer = self._normalize_tag_spacing(self._buffer)
        normal_text = ""
        calls: List[ToolCallItem] = []

        while True:
            if not self._in_tool_call:
                start = self._buffer.find(self.TOOL_CALL_START)
                if start == -1:
                    flush_length = self._flushable_prefix_length(
                        self._buffer, self.TOOL_CALL_START
                    )
                    normal_text += self._buffer[:flush_length]
                    self._buffer = self._buffer[flush_length:]
                    break

                normal_text += self._buffer[:start]
                self._buffer = self._buffer[start + len(self.TOOL_CALL_START) :]
                self._in_tool_call = True
                self._current_function_name = None
                continue

            if self._current_function_name is None:
                if self._consume_tool_call_end():
                    continue
                if not self._consume_invoke_start(tools, calls):
                    break
                continue

            if self._current_param_name is None:
                if self._consume_invoke_end(calls):
                    continue
                if not self._consume_param_start(calls):
                    break
                continue

            if self._current_param_is_complex:
                if not self._consume_complex_param(calls):
                    break
            elif not self._consume_scalar_param(calls):
                break

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _consume_tool_call_end(self) -> bool:
        start = self._buffer.find(self.TOOL_CALL_END)
        invoke_start = self._buffer.find(self.INVOKE_PREFIX)
        if start == -1 or (invoke_start != -1 and invoke_start < start):
            return False

        self._buffer = self._buffer[start + len(self.TOOL_CALL_END) :]
        self._in_tool_call = False
        return True

    def _consume_invoke_start(
        self, tools: List[Tool], calls: List[ToolCallItem]
    ) -> bool:
        start = self._buffer.find(self.INVOKE_PREFIX)
        if start == -1:
            return False

        name_start = start + len(self.INVOKE_PREFIX)
        name_end = self._buffer.find('">', name_start)
        if name_end == -1:
            return False

        function_name = self._buffer[name_start:name_end]
        self._buffer = self._buffer[name_end + len('">') :]
        self._current_function_name = function_name
        self._current_function_schema = self._get_function_parameters_schema(
            function_name, tools
        )
        self._is_first_param = True

        if self.current_tool_id == -1:
            self.current_tool_id = 0
        self._append_stream_call(calls, "", name=function_name)
        self._append_stream_call(calls, "{")
        return True

    def _consume_invoke_end(self, calls: List[ToolCallItem]) -> bool:
        start = self._buffer.find(self.INVOKE_SUFFIX)
        param_start = self._buffer.find(self.PARAM_START_PREFIX)
        if start == -1 or (param_start != -1 and param_start < start):
            return False

        self._buffer = self._buffer[start + len(self.INVOKE_SUFFIX) :]
        self._append_stream_call(calls, "}")
        self.current_tool_id += 1
        self._current_function_name = None
        self._current_function_schema = None
        return True

    def _consume_param_start(self, calls: List[ToolCallItem]) -> bool:
        start = self._buffer.find(self.PARAM_START_PREFIX)
        if start == -1:
            return False

        gt = self._buffer.find(">", start + len(self.PARAM_START_PREFIX))
        if gt == -1:
            return False

        tag = self._buffer[start + len(self.PARAM_START_PREFIX) : gt].strip()
        self._buffer = self._buffer[gt + 1 :]
        self._current_param_name = tag
        self._current_param_schema = self._get_child_schema(
            self._current_function_schema, tag
        )
        self._current_param_buffer = ""
        self._current_param_is_complex = self._schema_has_type(
            self._current_param_schema, ("object", "array")
        ) or self._buffer.startswith(self.PARAM_START_PREFIX)
        self._current_string_started = False

        prefix = "{}{}: ".format(
            "" if self._is_first_param else ", ",
            json.dumps(tag, ensure_ascii=False),
        )
        self._append_stream_call(calls, prefix)
        self._is_first_param = False
        return True

    def _consume_complex_param(self, calls: List[ToolCallItem]) -> bool:
        end_token = self._parameter_end_token(self._current_param_name)
        end = self._buffer.find(end_token)
        if end == -1:
            flush_length = self._flushable_prefix_length(self._buffer, end_token)
            self._current_param_buffer += self._buffer[:flush_length]
            self._buffer = self._buffer[flush_length:]
            return False

        self._current_param_buffer += self._buffer[:end]
        value = self._parse_parameter(
            self._current_param_buffer, self._current_param_schema
        )
        self._append_stream_call(calls, json.dumps(value, ensure_ascii=False))
        self._buffer = self._buffer[end + len(end_token) :]
        self._clear_current_param()
        return True

    def _consume_scalar_param(self, calls: List[ToolCallItem]) -> bool:
        end_token = self._parameter_end_token(self._current_param_name)
        end = self._buffer.find(end_token)
        if end == -1:
            flush_length = self._flushable_prefix_length(self._buffer, end_token)
            text = self._buffer[:flush_length]
            self._buffer = self._buffer[flush_length:]
            self._stream_scalar_text(text, calls)
            return False

        self._stream_scalar_text(self._buffer[:end], calls)
        if self._schema_has_type(self._current_param_schema, tuple(STRING_TYPES)):
            if not self._current_string_started:
                self._append_stream_call(calls, '"')
            self._append_stream_call(calls, '"')
        else:
            value = self._convert_leaf_value(
                self._current_param_buffer, self._current_param_schema
            )
            self._append_stream_call(calls, json.dumps(value, ensure_ascii=False))

        self._buffer = self._buffer[end + len(end_token) :]
        self._clear_current_param()
        return True

    def _stream_scalar_text(self, text: str, calls: List[ToolCallItem]) -> None:
        if not text:
            return

        if self._schema_has_type(self._current_param_schema, tuple(STRING_TYPES)):
            escaped = json.dumps(text, ensure_ascii=False)[1:-1]
            if self._current_string_started:
                self._append_stream_call(calls, escaped)
            else:
                self._append_stream_call(calls, '"' + escaped)
                self._current_string_started = True
        else:
            self._current_param_buffer += text

    def _clear_current_param(self) -> None:
        self._current_param_name = None
        self._current_param_schema = None
        self._current_param_buffer = ""
        self._current_param_is_complex = False
        self._current_string_started = False

    def _append_stream_call(
        self, calls: List[ToolCallItem], parameters: str, *, name: Optional[str] = None
    ) -> None:
        if (
            name is None
            and calls
            and calls[-1].tool_index == self.current_tool_id
            and calls[-1].name is None
        ):
            calls[-1].parameters += parameters
        else:
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=name,
                    parameters=parameters,
                )
            )

    def _parameter_end_token(self, tag: Optional[str]) -> str:
        return MINIMAX_NS_TOKEN + f"</{tag}>"

    def _get_function_parameters_schema(
        self, function_name: str, tools: List[Tool]
    ) -> Optional[Dict[str, Any]]:
        for tool in tools:
            if tool.function.name == function_name:
                parameters = tool.function.parameters
                if isinstance(parameters, dict):
                    return parameters
                break
        return None

    def _get_child_schema(
        self, parent_schema: Any, child_tag: str, parent_value: Any = None
    ) -> Optional[Dict]:
        if not isinstance(parent_schema, dict):
            return None

        if self._schema_has_type(parent_schema, ("array",)) and child_tag == "item":
            return self._get_array_item_schema(parent_schema, parent_value)

        properties = parent_schema.get("properties")
        if isinstance(properties, dict) and child_tag in properties:
            child_schema = properties[child_tag]
            return child_schema if isinstance(child_schema, dict) else None

        additional_properties = parent_schema.get("additionalProperties")
        if isinstance(additional_properties, dict):
            return additional_properties

        return None

    def _get_array_item_schema(
        self, array_schema: Dict[str, Any], array_value: Any
    ) -> Optional[Dict]:
        item_index = len(array_value) if isinstance(array_value, list) else 0

        prefix_items = array_schema.get("prefixItems")
        if isinstance(prefix_items, list) and item_index < len(prefix_items):
            item_schema = prefix_items[item_index]
            return item_schema if isinstance(item_schema, dict) else None

        additional_items = array_schema.get("additionalItems")
        if isinstance(additional_items, dict):
            return additional_items

        items = array_schema.get("items")
        if isinstance(items, dict):
            return items

        if isinstance(prefix_items, list) and prefix_items:
            item_schema = prefix_items[-1]
            return item_schema if isinstance(item_schema, dict) else None

        return None

    def _schema_types(self, schema: Any) -> List[str]:
        if not isinstance(schema, dict):
            return []

        schema_type = schema.get("type")
        if isinstance(schema_type, str):
            return [schema_type.lower()]
        if isinstance(schema_type, list):
            return [t.lower() for t in schema_type if isinstance(t, str)]
        return []

    def _schema_has_type(self, schema: Any, schema_types: Tuple[str, ...]) -> bool:
        return any(t in self._schema_types(schema) for t in schema_types)

    def _is_scalar_schema(self, schema: Any) -> bool:
        schema_types = set(self._schema_types(schema))
        return bool(schema_types & SCALAR_TYPES) and not schema_types & CONTAINER_TYPES

    def _new_container_for_schema(self, schema: Any) -> Any:
        if self._schema_has_type(schema, ("array",)):
            return []
        return {}

    def _convert_leaf_value(self, value: str, schema: Any) -> Any:
        schema_types = set(self._schema_types(schema))
        null_permitted = "null" in schema_types

        # Return verbatim; streaming emits strings literally, so null-coercing
        # here would diverge the non-streaming path from streaming.
        if schema_types & STRING_TYPES and not null_permitted:
            return value

        if null_permitted:
            if value in NULL_STRINGS:
                return None
            if value == "":
                return None

        lower_value = value.lower().strip()

        if schema_types & INTEGER_TYPES:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

        if schema_types & NUMBER_TYPES:
            try:
                parsed = float(value)
                # int(parsed) raises OverflowError for inf and ValueError for
                # nan, so a non-finite literal (e.g. "1e999") falls through to
                # the verbatim return instead of crashing the stream.
                return parsed if parsed != int(parsed) else int(parsed)
            except (TypeError, ValueError, OverflowError):
                pass

        if schema_types & BOOLEAN_TYPES:
            if lower_value in ("true", "1", "yes", "on"):
                return True
            if lower_value in ("false", "0", "no", "off"):
                return False

        return value

    def _assign_child(
        self, parent_value: Any, child_tag: str, child_value: Any
    ) -> None:
        if isinstance(parent_value, list):
            parent_value.append(child_value)
            return

        if child_tag in parent_value:
            existing_value = parent_value[child_tag]
            if isinstance(existing_value, list):
                existing_value.append(child_value)
            else:
                parent_value[child_tag] = [existing_value, child_value]
            return

        parent_value[child_tag] = child_value

    def _parse_parameter(self, body: str, parameters_schema: Optional[Dict]) -> dict:
        if self._schema_has_type(
            parameters_schema, ("array",)
        ) and self._body_starts_with_item(body):
            root: Any = []
        else:
            root = {}
        stack: List[Dict[str, Any]] = [
            {"tag": "", "schema": parameters_schema, "value": root}
        ]

        for chunk in body.split(MINIMAX_NS_TOKEN):
            chunk = chunk.strip()
            if not chunk:
                continue

            if chunk.startswith("</"):
                gt = chunk.find(">", 2)
                tag = chunk[2:gt].strip() if gt != -1 else chunk[2:].strip()
                if len(stack) == 1:
                    raise ValueError(f"unexpected closing tag: {tag}")
                if stack[-1]["tag"] != tag:
                    raise ValueError(
                        f"mismatched closing tag: expected {stack[-1]['tag']}, got {tag}"
                    )

                frame = stack.pop()
                self._assign_child(stack[-1]["value"], frame["tag"], frame["value"])
                continue

            if chunk.startswith("<"):
                gt = chunk.index(">")
                tag = chunk[1:gt].strip()
                text = chunk[gt + 1 :]
                parent_frame = stack[-1]
                child_schema = self._get_child_schema(
                    parent_frame["schema"], tag, parent_frame["value"]
                )

                if text or self._is_scalar_schema(child_schema):
                    value = self._convert_leaf_value(text, child_schema)
                else:
                    value = self._new_container_for_schema(child_schema)
                stack.append({"tag": tag, "schema": child_schema, "value": value})

        return root

    def _body_starts_with_item(self, body: str) -> bool:
        for chunk in body.split(MINIMAX_NS_TOKEN):
            chunk = chunk.strip()
            if chunk:
                return chunk.startswith("<item>")
        return False
