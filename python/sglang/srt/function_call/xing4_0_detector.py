"""Tool call detector for Xing4_0 models.

Supports two tool-call formats inside special tool blocks:

1. **JSON** - a JSON object with "name" and "arguments" fields.
2. **Tag-based** - func name followed by param_key/param_value tags.

Usage: ``--enable-auto-tool-choice --tool-call-parser xing4_0``
"""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Mapping, Sequence

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
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


def _mapping_arguments(parsed: Mapping) -> Dict[str, Any]:
    arguments = parsed.get("arguments", parsed.get("parameters", parsed))
    if isinstance(arguments, Mapping):
        return dict(arguments)
    return {}


def _split_payload(payload: str, tools: Sequence[Tool]) -> tuple[str, str, Any]:
    """Split payload into (tool_name, params_text, parsed_json).

    JSON payloads are deserialized once here and returned as the third
    element (``None`` when the payload carries no JSON), so callers never
    re-parse the same text.

    Handles three formats:
    1. Pure JSON: ``{"name": "func", "arguments": {...}}``
    2. Tag-based: ``func<param_key>k1</param_key>...``
    3. Name + JSON: ``func{...}``
    """
    payload = payload.strip()
    param_pos = payload.find(PARAM_KEY_START_TOKEN)
    if param_pos != -1:
        return payload[:param_pos].strip(), payload[param_pos:], None

    # Pure JSON: extract tool name from the "name" field.
    if payload.startswith("{"):
        return "", "", _deserialize(payload)

    for tool_name in _iter_tool_names(tools):
        if payload == tool_name:
            return tool_name, "", None
        if payload.startswith(tool_name):
            rest = payload[len(tool_name) :].strip()
            if rest.startswith("{"):
                return tool_name, "", _deserialize(rest)

    # Name + JSON for tools not declared in `tools` (forwarded as-is).
    brace_pos = payload.find("{")
    if brace_pos > 0:
        candidate_name = payload[:brace_pos].strip()
        parsed = _deserialize(payload[brace_pos:])
        if candidate_name and isinstance(parsed, Mapping):
            return candidate_name, "", parsed

    return payload, "", None


def _parse_payload(payload: str, tools: Sequence[Tool]) -> tuple[str, Dict[str, Any]]:
    tool_name, params_text, parsed = _split_payload(payload, tools)
    arguments = _mapping_arguments(parsed) if isinstance(parsed, Mapping) else {}

    # Pure JSON format: extract tool name from the JSON object.
    if not tool_name and isinstance(parsed, Mapping):
        raw_name = parsed.get("name")
        tool_name = raw_name.strip() if isinstance(raw_name, str) else ""

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


class Xing4_0Detector(BaseFormatDetector):
    """Detector for Xing4_0 tool call format.

    Format::

        A JSON object with "name" and "arguments" fields enclosed in tool tokens

    or::

        func name followed by param_key/param_value tags enclosed in tool tokens
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOL_START_TOKEN
        self.eot_token = TOOL_END_TOKEN
        self._buffer = ""
        self._tool_indices = None

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        if self._tool_indices is None:
            self._tool_indices = self._get_tool_indices(tools)
        tool_indices = self._tool_indices
        calls: List[ToolCallItem] = []

        try:
            for match in TOOL_CALL_REGEX.finditer(text):
                tool_name, arguments = _parse_payload(match.group(1), tools)
                if not tool_name:
                    continue
                if tool_name not in tool_indices:
                    logger.warning(
                        "Model attempted to call undefined function: %s", tool_name
                    )
                    if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                        continue
                calls.append(
                    ToolCallItem(
                        tool_index=len(calls),
                        name=tool_name,
                        parameters=json.dumps(arguments, ensure_ascii=False),
                    )
                )
        except Exception:
            logger.exception("Xing4_0Detector: failed to parse tool calls")
            return StreamingParseResult(normal_text=text)

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        content = ""
        calls: List[ToolCallItem] = []

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

            for call in extracted.calls:
                self.current_tool_id += 1
                call.tool_index = self.current_tool_id
                calls.append(call)

            self._buffer = self._buffer[end_pos:]

    def structure_info(self) -> _GetInfoFunc:
        """Return function that creates StructureInfo for guided generation."""

        def get_info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=f'{TOOL_START_TOKEN}{{"name": "{name}", "arguments":',
                end=f"}}{TOOL_END_TOKEN}",
                trigger=TOOL_START_TOKEN,
            )

        return get_info
