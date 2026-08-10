"""Tool call detector for TeleChat4 models.

Supports two tool-call formats inside ``<tool_call>...</tool_call>`` blocks:

1. **JSON** – ``<tool_call>{"name": "func", "arguments": {...}}</tool_call>``
2. **Tag-based** – ``<tool_call>func<param_key>k1</param_key><param_value>v1
   </param_value>...</tool_call>``

Usage: ``--enable-auto-tool-choice --tool-call-parser telechat4``
"""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Mapping, Sequence

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

TOOL_START_TOKEN = "<tool_call>"
TOOL_END_TOKEN = "</tool_call>"
PARAM_KEY_START_TOKEN = "<param_key>"
PARAM_KEY_END_TOKEN = "</param_key>"
PARAM_VALUE_START_TOKEN = "<param_value>"
PARAM_VALUE_END_TOKEN = "</param_value>"

TOOL_CALL_REGEX = re.compile(
    rf"{re.escape(TOOL_START_TOKEN)}(.*?){re.escape(TOOL_END_TOKEN)}",
    re.DOTALL,
)
PARAM_REGEX = re.compile(
    rf"{re.escape(PARAM_KEY_START_TOKEN)}(.*?){re.escape(PARAM_KEY_END_TOKEN)}"
    rf"\s*{re.escape(PARAM_VALUE_START_TOKEN)}(.*?){re.escape(PARAM_VALUE_END_TOKEN)}",
    re.DOTALL,
)


def _tool_parameters(tool: Tool) -> Mapping[str, Any]:
    params = tool.function.parameters
    if isinstance(params, Mapping):
        return params
    return {}


def _iter_tool_names(tools: Sequence[Tool]) -> List[str]:
    names = [t.function.name for t in tools if t.function.name]
    return sorted(names, key=len, reverse=True)


def _is_string_type(tool_name: str, arg_name: str, tools: Sequence[Tool]) -> bool:
    for tool in tools:
        if tool.function.name != tool_name:
            continue
        properties = _tool_parameters(tool).get("properties", {})
        if not isinstance(properties, Mapping):
            return False
        arg_schema = properties.get(arg_name, {})
        if not isinstance(arg_schema, Mapping):
            return False
        arg_type = arg_schema.get("type")
        if isinstance(arg_type, str):
            return arg_type == "string"
        if isinstance(arg_type, Sequence):
            return "string" in arg_type
        return False
    return False


def _deserialize(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    return value


def _json_arguments(value: str) -> Dict[str, Any]:
    parsed = _deserialize(value)
    if not isinstance(parsed, Mapping):
        return {}
    arguments = parsed.get("arguments", parsed.get("parameters", parsed))
    if isinstance(arguments, Mapping):
        return dict(arguments)
    return {}


def _split_payload(payload: str, tools: Sequence[Tool]) -> tuple[str, str, str]:
    """Split payload into (tool_name, params_text, json_text).

    Handles three formats:
    1. Pure JSON: ``{"name": "func", "arguments": {...}}``
    2. Tag-based: ``func<param_key>k1</param_key>...``
    3. Name + JSON: ``func{...}``
    """
    payload = payload.strip()
    param_pos = payload.find(PARAM_KEY_START_TOKEN)
    if param_pos != -1:
        return payload[:param_pos].strip(), payload[param_pos:], ""

    # Pure JSON: extract tool name from the "name" field.
    if payload.startswith("{"):
        return "", "", payload

    for tool_name in _iter_tool_names(tools):
        if payload == tool_name:
            return tool_name, "", ""
        if payload.startswith(tool_name):
            rest = payload[len(tool_name) :].strip()
            if rest.startswith("{"):
                return tool_name, "", rest

    return payload, "", ""


def _parse_payload(payload: str, tools: Sequence[Tool]) -> tuple[str, Dict[str, Any]]:
    tool_name, params_text, json_text = _split_payload(payload, tools)
    arguments = _json_arguments(json_text) if json_text else {}

    # Pure JSON format: extract tool name from the JSON object.
    if not tool_name and json_text:
        parsed = _deserialize(json_text)
        if isinstance(parsed, Mapping):
            tool_name = parsed.get("name", "") or ""

    for key, value in PARAM_REGEX.findall(params_text):
        arg_key = key.strip()
        arg_val = value.strip()
        if not _is_string_type(tool_name, arg_key, tools):
            arg_val = _deserialize(arg_val)
        arguments[arg_key] = arg_val

    return tool_name, arguments


def _partial_suffix_len(text: str, token: str) -> int:
    max_len = min(len(text), len(token) - 1)
    for size in range(max_len, 0, -1):
        if token.startswith(text[-size:]):
            return size
    return 0


class TeleChat4Detector(BaseFormatDetector):
    """Detector for TeleChat4 tool call format.

    Format::

        <tool_call>{"name": "func", "arguments": {...}}</tool_call>

    or::

        <tool_call>func<param_key>k1</param_key><param_value>v1</param_value>
        ...</tool_call>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOL_START_TOKEN
        self.eot_token = TOOL_END_TOKEN
        self._buffer = ""

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []

        try:
            for match in TOOL_CALL_REGEX.finditer(text):
                tool_name, arguments = _parse_payload(match.group(1), tools)
                if not tool_name or tool_name not in tool_indices:
                    logger.warning("TeleChat4Detector: unknown tool '%s'", tool_name)
                    continue
                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices[tool_name],
                        name=tool_name,
                        parameters=json.dumps(arguments, ensure_ascii=False),
                    )
                )
        except Exception:
            logger.exception("TeleChat4Detector: failed to parse tool calls")
            return StreamingParseResult(normal_text=text)

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        content = ""
        calls: List[ToolCallItem] = []
        tool_indices = self._get_tool_indices(tools)

        while True:
            start_idx = self._buffer.find(self.bot_token)
            if start_idx == -1:
                partial_len = _partial_suffix_len(self._buffer, self.bot_token)
                if partial_len:
                    content += self._buffer[:-partial_len]
                    self._buffer = self._buffer[-partial_len:]
                else:
                    content += self._buffer
                    self._buffer = ""
                return StreamingParseResult(normal_text=content, calls=calls)

            content += self._buffer[:start_idx]
            self._buffer = self._buffer[start_idx:]
            end_idx = self._buffer.find(self.eot_token)
            if end_idx == -1:
                return StreamingParseResult(normal_text=content, calls=calls)

            end_pos = end_idx + len(self.eot_token)
            tool_text = self._buffer[:end_pos]
            extracted = self.detect_and_parse(tool_text, tools)

            if not extracted.calls:
                content += tool_text
            else:
                calls.extend(extracted.calls)

            self._buffer = self._buffer[end_pos:]

    def structure_info(self) -> _GetInfoFunc:
        """Return function that creates StructureInfo for guided generation."""

        def get_info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=f'<tool_call>{{"name": "{name}", "arguments":',
                end="}</tool_call>",
                trigger="<tool_call>",
            )

        return get_info
