import json
import logging
import re
from typing import Any

try:
    import json_repair
except ImportError:
    json_repair = None

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    _is_complete_json,
    get_schema_properties,
)

logger = logging.getLogger(__name__)


class DotsToolDetector(BaseFormatDetector):
    """Detector for the dots function-call format.

    The canonical format contains one or more XML ``invoke`` elements inside a
    ``dots_function_call`` block::

        <dots_function_call>
        <invoke name="search">
        <parameter name="query">weather in Shanghai</parameter>
        </invoke>
        </dots_function_call>

    A JSON object with ``name`` and ``arguments`` is accepted as a fallback.
    Multiple wrapper blocks and multiple invokes in one block are supported.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<dots_function_call>"
        self.eot_token = "</dots_function_call>"
        self.func_call_regex = re.compile(
            rf"{re.escape(self.bot_token)}\s*(.*?)\s*{re.escape(self.eot_token)}",
            re.DOTALL,
        )
        self.invoke_regex = re.compile(
            r"<invoke\s+name\s*=\s*(?P<name>[^>]+)>(?P<body>.*?)</invoke>",
            re.DOTALL,
        )
        self.parameter_regex = re.compile(
            r"<parameter\s+name\s*=\s*(?P<name>[^>]+)>(?P<value>.*?)</parameter>",
            re.DOTALL,
        )

    @staticmethod
    def _extract_name(value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            return value[1:-1]
        return value

    @staticmethod
    def _load_json(value: str) -> Any:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            if json_repair is None:
                raise
            return json_repair.loads(value)

    @classmethod
    def _convert_param_value(cls, value: str, param_type: Any) -> Any:
        if value.lower() == "null":
            return None

        if isinstance(param_type, list):
            param_type = next((item for item in param_type if item != "null"), "string")
        if not isinstance(param_type, str):
            param_type = str(param_type)
        param_type = param_type.lower()

        if param_type in {"string", "str", "text"}:
            return value
        if param_type in {"integer", "int"}:
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        if param_type in {"number", "float"}:
            try:
                number = float(value)
                return int(number) if number.is_integer() else number
            except (TypeError, ValueError):
                return value
        if param_type in {"boolean", "bool"}:
            return value.lower() in {"true", "1"}

        try:
            return cls._load_json(value)
        except (json.JSONDecodeError, ValueError, TypeError):
            return value

    def _resolve_param_type(
        self, schema: Any, defs: dict[str, Any], depth: int = 0
    ) -> Any | None:
        """Resolve a parameter type through local refs and schema compositions."""
        if not isinstance(schema, dict) or depth > 10:
            return None
        if "type" in schema:
            return schema["type"]

        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            return self._resolve_param_type(
                defs.get(ref.rsplit("/", 1)[-1]), defs, depth + 1
            )

        for keyword in ("anyOf", "oneOf", "allOf"):
            alternatives = schema.get(keyword)
            if not isinstance(alternatives, list):
                continue
            for alternative in alternatives:
                if isinstance(alternative, dict) and alternative.get("type") == "null":
                    continue
                resolved = self._resolve_param_type(alternative, defs, depth + 1)
                if resolved is not None:
                    return resolved
        return None

    @staticmethod
    def _tool_schema(name: str, tools: list[Tool]) -> tuple[dict, dict]:
        for tool in tools:
            if tool.function.name != name:
                continue
            schema = tool.function.parameters
            if not isinstance(schema, dict):
                break
            defs = schema.get("$defs", {})
            return (
                get_schema_properties(schema),
                defs if isinstance(defs, dict) else {},
            )
        return {}, {}

    def _parse_xml_invoke(self, match: re.Match, tools: list[Tool]) -> dict[str, Any]:
        name = self._extract_name(match.group("name"))
        properties, defs = self._tool_schema(name, tools)
        arguments: dict[str, Any] = {}

        for parameter in self.parameter_regex.finditer(match.group("body")):
            param_name = self._extract_name(parameter.group("name"))
            value = parameter.group("value").strip()
            param_type: Any = "string"
            if param_name in properties:
                param_type = (
                    self._resolve_param_type(properties[param_name], defs) or "string"
                )
            arguments[param_name] = self._convert_param_value(value, param_type)

        return {"name": name, "arguments": arguments}

    def _parse_block(self, content: str, tools: list[Tool]) -> list[dict[str, Any]]:
        content = content.strip()
        if content.startswith("<invoke"):
            return [
                self._parse_xml_invoke(match, tools)
                for match in self.invoke_regex.finditer(content)
            ]

        parsed = self._load_json(content)
        if not isinstance(parsed, dict):
            raise TypeError("dots JSON tool call must be an object")
        return [parsed]

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        marker_index = text.find(self.bot_token)
        if marker_index == -1:
            return StreamingParseResult(normal_text=text)

        calls: list[ToolCallItem] = []
        for block in self.func_call_regex.finditer(text):
            try:
                for parsed in self._parse_block(block.group(1), tools):
                    calls.extend(self.parse_base_json(parsed, tools))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                logger.warning("Failed to parse dots tool call: %s", exc)

        return StreamingParseResult(
            normal_text=text[:marker_index].strip(), calls=calls
        )

    def _append_stream_call(
        self, parsed: dict[str, Any], item: ToolCallItem
    ) -> ToolCallItem:
        self.current_tool_id += 1
        arguments = parsed.get("arguments", parsed.get("parameters", {})) or {}
        serialized = json.dumps(arguments, ensure_ascii=False)
        self.prev_tool_call_arr.append(
            {"name": parsed.get("name"), "arguments": arguments}
        )
        self.streamed_args_for_tool.append(serialized)
        item.tool_index = self.current_tool_id
        item.parameters = serialized
        return item

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        """Buffer incomplete XML and emit every complete call in the new data."""
        self._buffer += new_text
        normal_parts: list[str] = []
        calls: list[ToolCallItem] = []

        while self._buffer:
            marker_index = self._buffer.find(self.bot_token)
            if marker_index == -1:
                partial_len = self._ends_with_partial_token(
                    self._buffer, self.bot_token
                )
                if partial_len:
                    normal_parts.append(self._buffer[:-partial_len])
                    self._buffer = self._buffer[-partial_len:]
                else:
                    normal_parts.append(self._buffer)
                    self._buffer = ""
                normal_parts = [
                    part.replace(self.eot_token, "") for part in normal_parts
                ]
                break

            if marker_index > 0:
                normal_parts.append(self._buffer[:marker_index])
                self._buffer = self._buffer[marker_index:]

            end_index = self._buffer.find(self.eot_token, len(self.bot_token))
            if end_index == -1:
                self._stream_complete_json_body(tools, calls)
                break

            content = self._buffer[len(self.bot_token) : end_index]
            self._buffer = self._buffer[end_index + len(self.eot_token) :]
            try:
                parsed_calls = self._parse_block(content, tools)
                if not parsed_calls:
                    raise ValueError("dots tool-call block contains no invoke")
                block_calls: list[ToolCallItem] = []
                for index, parsed in enumerate(parsed_calls):
                    validated = self.parse_base_json(parsed, tools)
                    if index == 0 and self.current_tool_name_sent and validated:
                        item = validated[0]
                        arguments = item.parameters or ""
                        streamed = self.streamed_args_for_tool[self.current_tool_id]
                        remaining = arguments.removeprefix(streamed)
                        if remaining:
                            block_calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=remaining,
                                )
                            )
                        self.prev_tool_call_arr[self.current_tool_id] = parsed
                        self.streamed_args_for_tool[self.current_tool_id] = arguments
                    else:
                        block_calls.extend(
                            self._append_stream_call(parsed, item) for item in validated
                        )
                if block_calls:
                    calls.extend(block_calls)
                elif not self.current_tool_name_sent:
                    normal_parts.append(content.strip())
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                logger.warning("Failed to parse streamed dots tool call: %s", exc)
                normal_parts.append(content.strip())

            self.current_tool_name_sent = False

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def _stream_complete_json_body(
        self, tools: list[Tool], calls: list[ToolCallItem]
    ) -> None:
        """Emit a complete JSON body while its closing XML tag is pending."""
        content = self._buffer[len(self.bot_token) :].strip()
        if not content or not _is_complete_json(content):
            return

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(parsed, dict):
            return

        validated = self.parse_base_json(parsed, tools)
        if not validated:
            return

        item = validated[0]
        arguments = item.parameters or ""
        if not self.current_tool_name_sent:
            self.current_tool_id += 1
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=item.name,
                    parameters="",
                )
            )
            self.prev_tool_call_arr.append(
                {"name": item.name, "arguments": parsed.get("arguments", {})}
            )
            self.streamed_args_for_tool.append("")
            self.current_tool_name_sent = True

        streamed = self.streamed_args_for_tool[self.current_tool_id]
        argument_diff = arguments.removeprefix(streamed)
        if argument_diff:
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=None,
                    parameters=argument_diff,
                )
            )
            self.streamed_args_for_tool[self.current_tool_id] += argument_diff

    def flush_pending_normal_text(self) -> str:
        """Flush a partial opening marker as plain text at end of stream."""
        if not self._buffer or self.bot_token in self._buffer:
            return ""

        normal_text = self._buffer.replace(self.eot_token, "")
        self._buffer = ""
        return normal_text

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        # Kept for the detector interface. It is not used while structural tags
        # are disabled for dots' mixed XML/JSON format.
        return lambda name: StructureInfo(
            begin=f'{self.bot_token}{{"name": "{name}", "arguments": ',
            end=f"}}{self.eot_token}",
            trigger=self.bot_token,
        )
