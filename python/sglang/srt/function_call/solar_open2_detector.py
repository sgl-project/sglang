# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

import json
import logging
from typing import Any, List, Literal, Optional, Union

import regex as re
from jsonschema import Draft202012Validator
from referencing import Registry
from xgrammar import StructuralTag

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.solar_open2_structural_tag import (
    get_solar_open2_structural_tag,
)

logger = logging.getLogger(__name__)


def _coerce_union(value: str, schema: dict) -> Any:
    """Prefer valid non-string JSON; Solar strings otherwise remain raw text."""

    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = item
        return result

    try:
        schema_size = len(json.dumps(schema))
        # Text sizes conservatively bound schema/value nodes and their product.
        if max(len(value), schema_size) > 65536 or len(value) * schema_size > 1000000:
            return value
        pending = [schema]
        while pending:
            node = pending.pop()
            if not isinstance(node, dict):
                continue
            if (
                node.keys()
                & {
                    "$ref",
                    "$dynamicRef",
                    "$recursiveRef",
                    "$schema",
                    "pattern",
                    "patternProperties",
                    "unevaluatedProperties",
                    "unevaluatedItems",
                    "dependentSchemas",
                }
                or node.get("uniqueItems") is True
            ):
                return value
            for key, child in node.items():
                if key in ("properties", "$defs", "definitions") and isinstance(
                    child, dict
                ):
                    pending.extend(child.values())
                elif key not in ("enum", "const", "default", "examples"):
                    if isinstance(child, dict):
                        pending.append(child)
                    elif isinstance(child, list):
                        pending.extend(child)
        parsed = json.loads(value, object_pairs_hook=unique_object)
        if isinstance(parsed, str):
            return value
        # Reject non-finite numbers and lone surrogates before response encoding.
        json.dumps(parsed, ensure_ascii=False, allow_nan=False).encode("utf-8")
        # The request path already checks the schema. Do not retrieve references.
        if Draft202012Validator(schema, registry=Registry()).is_valid(parsed):
            return parsed
    except Exception:
        # Malformed values or unsupported schemas must preserve inspectable text.
        pass
    return value


class SolarOpen2Detector(BaseFormatDetector):
    """Parse Solar's named, raw-text arguments into JSON tool calls."""

    TOOL_CALL_START = "<|tool_call:start|>"
    TOOL_CALL_END = "<|tool_call:end|>"
    TOOL_ARG_START = "<|tool_arg:start|>"
    TOOL_ARG_VALUE = "<|tool_arg:value|>"
    TOOL_ARG_END = "<|tool_arg:end|>"
    _STRING_TYPES = ("string", "str", "text", "varchar", "char", "enum")

    def __init__(self, tokenizer=None):
        super().__init__()
        self._tokenizer = tokenizer
        self.bot_token = self.TOOL_CALL_START
        self.eot_token = self.TOOL_CALL_END
        tc_start, tc_end, ta_start, ta_value, ta_end = map(
            re.escape,
            (
                self.TOOL_CALL_START,
                self.TOOL_CALL_END,
                self.TOOL_ARG_START,
                self.TOOL_ARG_VALUE,
                self.TOOL_ARG_END,
            ),
        )
        self.tool_call_pattern = re.compile(
            rf"{tc_start}(.+?)\n((?:{ta_start}.*?{ta_end}\n?)*){tc_end}",
            re.DOTALL,
        )
        self.tool_arg_pattern = re.compile(
            rf"{ta_start}(.*?){ta_value}(.*?){ta_end}", re.DOTALL
        )
        self._state = "waiting"
        self._pending_ws = ""
        self._content_emitted = False
        self._candidate = ""
        self._func_name = ""
        self._arg_name = ""
        self._arg_type: Union[str, dict] = "string"
        self._first_arg = True
        self._string_opened = False
        self.incomplete_tool_call_indices: set[int] = set()

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def _get_param_type(
        self, func_name: str, param_name: str, tools: List[Tool]
    ) -> Union[str, dict]:
        """Return the legacy type name, or the complete untyped union schema."""
        for tool in tools or []:
            if tool.type != "function" or tool.function.name != func_name:
                continue
            params = tool.function.parameters
            props = params.get("properties") if isinstance(params, dict) else None
            spec = props.get(param_name) if isinstance(props, dict) else None
            if isinstance(spec, dict):
                if "type" not in spec and ("anyOf" in spec or "oneOf" in spec):
                    # A schema keeps union values on the buffered streaming path.
                    return spec
                param_type = spec.get("type", "string")
                if isinstance(param_type, list):
                    param_type = next((t for t in param_type if t != "null"), "string")
                return str(param_type).strip().lower()
        return "string"

    @staticmethod
    def _coerce(value: str, param_type: Union[str, dict]) -> Any:
        if isinstance(param_type, dict):
            return _coerce_union(value, param_type)
        if value.strip().lower() == "null":
            return None
        try:
            if param_type in ("integer", "int"):
                return int(value)
            if param_type in ("number", "float", "double"):
                number = float(value)
                return int(number) if number - int(number) == 0 else number
            if param_type in ("boolean", "bool"):
                normalized = value.strip().lower()
                if normalized in ("true", "1", "yes"):
                    return True
                if normalized in ("false", "0", "no"):
                    return False
            if param_type in ("array", "list", "object", "dict"):
                return json.loads(value)
            if param_type in ("null", "none"):
                return None
        except (ValueError, TypeError, OverflowError):
            logger.warning("Solar Open2: cannot convert %r to %s", value, param_type)
        return value

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text)
        calls = []
        normal_text_parts = []
        last_end = 0
        try:
            for match in self.tool_call_pattern.finditer(text, timeout=1):
                name = match.group(1).strip()
                if not name:
                    return StreamingParseResult(normal_text=text)
                arguments = {}
                for arg in self.tool_arg_pattern.finditer(match.group(2), timeout=1):
                    key, value = arg.groups()
                    arguments[key] = self._coerce(
                        value, self._get_param_type(name, key, tools)
                    )
                calls.append(
                    ToolCallItem(
                        tool_index=len(calls),
                        name=name,
                        parameters=json.dumps(arguments, ensure_ascii=False),
                    )
                )
                normal_text_parts.append(text[last_end : match.start()])
                last_end = match.end()
        except TimeoutError:
            logger.warning("Solar Open2 tool parsing timed out")
            return StreamingParseResult(normal_text=text)
        if not calls:
            return StreamingParseResult(normal_text=text)
        normal_text_parts.append(text[last_end:])
        normal_text = "".join(normal_text_parts)
        return StreamingParseResult(
            normal_text=normal_text if normal_text.strip() else "",
            calls=calls,
        )

    def _emit(self, fragment: str, calls: List[ToolCallItem]) -> None:
        if not fragment:
            return
        idx = self.current_tool_id
        self.streamed_args_for_tool[idx] += fragment
        self.prev_tool_call_arr[idx]["arguments"] = self.streamed_args_for_tool[idx]
        if calls and calls[-1].tool_index == idx:
            calls[-1].parameters += fragment
        else:
            calls.append(ToolCallItem(tool_index=idx, parameters=fragment))

    @staticmethod
    def _string_body(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)[1:-1]

    def _stream_string(self, calls: List[ToolCallItem]) -> None:
        holdback = self._ends_with_partial_token(self._buffer, self.TOOL_ARG_END)
        end = len(self._buffer) - holdback
        raw = self._buffer[:end]
        if not raw:
            return
        stripped = raw.strip().lower()
        # Even a string-schema value may become null; wait until it cannot.
        if not self._string_opened and (
            not stripped or stripped == "null" or "null".startswith(stripped)
        ):
            return
        if not self._string_opened:
            prefix = "{" if self._first_arg else ", "
            self._emit(
                f'{prefix}{json.dumps(self._arg_name, ensure_ascii=False)}: "', calls
            )
            self._first_arg = False
            self._string_opened = True
        self._emit(self._string_body(raw), calls)
        self._buffer = self._buffer[end:]

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        content = ""
        calls = []
        while True:
            if self._state == "waiting":
                idx = self._buffer.find(self.bot_token)
                if idx == -1:
                    holdback = self._ends_with_partial_token(
                        self._buffer, self.bot_token
                    )
                    end = len(self._buffer) - holdback
                    if end:
                        chunk = self._pending_ws + self._buffer[:end]
                        self._buffer = self._buffer[end:]
                        body = chunk.rstrip()
                        self._pending_ws = chunk[len(body) :]
                        if body:
                            content += body
                            self._content_emitted = True
                    break
                prefix = self._pending_ws + self._buffer[:idx]
                self._pending_ws = ""
                suppressed = ""
                if prefix.strip() or (
                    prefix and self._content_emitted and self.current_tool_id == -1
                ):
                    content += prefix
                    self._content_emitted = True
                else:
                    suppressed = prefix
                self._candidate = suppressed + self.bot_token
                self._buffer = self._buffer[idx + len(self.bot_token) :]
                self._state = "name"

            elif self._state == "name":
                idx = self._buffer.find("\n")
                if idx == -1:
                    break
                raw_name = self._buffer[:idx]
                self._func_name = raw_name.strip()
                self._buffer = self._buffer[idx + 1 :]
                if not self._func_name:
                    content += self._candidate + raw_name + "\n"
                    self._content_emitted = True
                    self._candidate = ""
                    self._state = "waiting"
                    continue
                self._candidate = ""
                self.current_tool_id += 1
                self.incomplete_tool_call_indices.add(self.current_tool_id)
                self.current_tool_name_sent = True
                self.prev_tool_call_arr.append(
                    {"name": self._func_name, "arguments": ""}
                )
                self.streamed_args_for_tool.append("")
                self._first_arg = True
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=self._func_name,
                        parameters="",
                    )
                )
                self._state = "call"

            elif self._state == "call":
                arg_idx = self._buffer.find(self.TOOL_ARG_START)
                end_idx = self._buffer.find(self.eot_token)
                if arg_idx == -1 and end_idx == -1:
                    break
                if arg_idx != -1 and (end_idx == -1 or arg_idx < end_idx):
                    self._buffer = self._buffer[arg_idx + len(self.TOOL_ARG_START) :]
                    self._state = "arg_name"
                else:
                    self._buffer = self._buffer[end_idx + len(self.eot_token) :]
                    self._emit("{}" if self._first_arg else "}", calls)
                    self.incomplete_tool_call_indices.discard(self.current_tool_id)
                    self._state = "waiting"

            elif self._state == "arg_name":
                idx = self._buffer.find(self.TOOL_ARG_VALUE)
                if idx == -1:
                    break
                self._arg_name = self._buffer[:idx]
                self._buffer = self._buffer[idx + len(self.TOOL_ARG_VALUE) :]
                self._arg_type = self._get_param_type(
                    self._func_name, self._arg_name, tools
                )
                self._string_opened = False
                self._state = "arg_value"

            elif self._state == "arg_value":
                idx = self._buffer.find(self.TOOL_ARG_END)
                if idx == -1:
                    if self._arg_type in self._STRING_TYPES:
                        self._stream_string(calls)
                    break
                value = self._buffer[:idx]
                self._buffer = self._buffer[idx + len(self.TOOL_ARG_END) :]
                if self._string_opened:
                    self._emit(self._string_body(value) + '"', calls)
                else:
                    prefix = "{" if self._first_arg else ", "
                    key = json.dumps(self._arg_name, ensure_ascii=False)
                    converted = json.dumps(
                        self._coerce(value, self._arg_type), ensure_ascii=False
                    )
                    self._emit(f"{prefix}{key}: {converted}", calls)
                    self._first_arg = False
                self._string_opened = False
                self._state = "call"
        return StreamingParseResult(normal_text=content, calls=calls)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        text = ""
        if self._state == "waiting":
            if self.current_tool_id == -1 or self._pending_ws.strip() or self._buffer:
                text = self._pending_ws + self._buffer
        elif self._state == "name":
            text = self._candidate + self._buffer
        self._pending_ws = self._buffer = self._candidate = ""
        return StreamingParseResult(normal_text=text)

    def supports_structural_tag(self) -> bool:
        return True

    def parses_required_natively(self) -> bool:
        return False

    def get_structural_tag(
        self,
        tools: Optional[List[Tool]] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        return get_solar_open2_structural_tag(
            tools or [],
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            vocab=self._tokenizer.get_vocab() if self._tokenizer is not None else None,
            parallel_tool_calls=parallel_tool_calls,
        )

    def structure_info(self):
        raise NotImplementedError("Solar Open2 uses native tagged arguments")
