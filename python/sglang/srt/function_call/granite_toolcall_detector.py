"""Tool-call detector for Granite 4.2 (``--tool-call-parser granite_toolcall_parser``).

Granite 4.2 emits tool calls in an XML layout::

    <tool_call>
    <function=func_name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

This detector is self-contained (it does not inherit from the Qwen detectors).
It only leaves ordinary content untouched: stripping the newline that the chat
template puts after ``</think>`` is a reasoning-parser concern, so it works the
same standalone, or with ``granite_thinking_parser`` / ``nemotron_3``.
"""

import json
import logging
import re
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    get_schema_properties,
    infer_type_from_json_schema,
    safe_literal_eval,
)

logger = logging.getLogger(__name__)

_STRING_TYPES = ("string", "str", "text", "varchar", "char", "enum")
_INT_PREFIXES = ("int", "uint", "long", "short", "unsigned")
_FLOAT_PREFIXES = ("num", "float")
_BOOL_TYPES = ("boolean", "bool", "binary")
_CONTAINER_TYPES = ("object", "array", "arr")
_CONTAINER_PREFIXES = ("dict", "list")


class GraniteToolCallDetector(BaseFormatDetector):
    """Detector for the Granite 4.2 XML tool-call format."""

    def __init__(self):
        super().__init__()

        self.call_open = "<tool_call>"
        self.call_close = "</tool_call>"
        self.func_open = "<function="
        self.func_close = "</function>"
        self.param_open = "<parameter="
        self.param_close = "</parameter>"

        # Non-streaming patterns.
        self._call_block_re = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self._func_block_re = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self._param_block_re = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        # Streaming state. ``_buffer`` comes from the base class.
        self._cursor: int = 0
        self._params_emitted: int = 0
        self._args_open: bool = False
        self._in_call: bool = False
        self._active_func: Optional[str] = None

    def has_tool_call(self, text: str) -> bool:
        return self.call_open in text

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def _lookup_param_schema(self, func_name: str, tools: Optional[List[Tool]]) -> dict:
        """Return the ``{param: schema}`` mapping declared for ``func_name``."""
        if tools is None:
            return {}
        for tool in tools:
            try:
                tool_type = tool.type
                fn = tool.function
                fn_name = fn.name
            except AttributeError:
                continue

            if tool_type == "function" and fn_name == func_name:
                try:
                    schema = fn.parameters
                except AttributeError:
                    return {}
                if not isinstance(schema, dict):
                    return {}
                props = get_schema_properties(schema)
                if props or "properties" in schema:
                    return props
                return schema
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    @staticmethod
    def _declared_type(schema: Any) -> str:
        inferred = infer_type_from_json_schema(schema)
        if inferred is None:
            return "string"
        return str(inferred).strip().lower()

    def _coerce(self, raw: str, name: str, schema_map: dict, func_name: str) -> Any:
        """Convert the raw text of a parameter using the tool's JSON schema."""
        if raw.lower() == "null":
            return None

        if name not in schema_map:
            if schema_map != {}:
                logger.warning(
                    f"Parsed parameter '{name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return raw

        kind = self._declared_type(schema_map[name])

        if kind in _STRING_TYPES:
            return raw

        if kind.startswith(_INT_PREFIXES):
            try:
                return int(raw)
            except Exception:
                logger.warning(
                    f"Parsed value '{raw}' of parameter '{name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string."
                )
                return raw

        if kind.startswith(_FLOAT_PREFIXES):
            try:
                keep_int = "." not in raw and "e" not in raw.lower()
                number = float(raw)
                if keep_int and number.is_integer():
                    return int(number)
                return number
            except Exception:
                logger.warning(
                    f"Parsed value '{raw}' of parameter '{name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
                return raw

        if kind in _BOOL_TYPES:
            lowered = raw.lower()
            if lowered not in ("true", "false"):
                logger.warning(
                    f"Parsed value '{lowered}' of parameter '{name}' is not a boolean (`true` or `false`) in tool '{func_name}', degenerating to false."
                )
            return lowered == "true"

        if kind in _CONTAINER_TYPES or kind.startswith(_CONTAINER_PREFIXES):
            try:
                return json.loads(raw)
            except Exception:
                logger.warning(
                    f"Parsed value '{raw}' of parameter '{name}' cannot be parsed with json.loads in tool "
                    f"'{func_name}', will try other methods to parse it."
                )
        try:
            return safe_literal_eval(raw)
        except Exception:
            logger.warning(
                f"Parsed value '{raw}' of parameter '{name}' cannot be converted via Python `ast.literal_eval()` in tool '{func_name}', degenerating to string."
            )
            return raw

    @staticmethod
    def _trim_value(value: str) -> str:
        """Drop one leading and one trailing newline around a parameter value."""
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        return value

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.call_open not in text:
            return StreamingParseResult(normal_text=text)

        calls: List[ToolCallItem] = []
        try:
            blocks = self._call_block_re.findall(text)
            if not blocks and self.func_open in text:
                # Closing tag missing or tags stripped: treat the text as one block.
                blocks = [text]

            index = 0
            for block in blocks:
                for match in self._func_block_re.findall(block):
                    body = match[0] or match[1]
                    if ">" not in body:
                        continue

                    split = body.index(">")
                    func_name = body[:split]
                    schema_map = self._lookup_param_schema(func_name, tools)

                    args = {}
                    for chunk in self._param_block_re.findall(body[split + 1 :]):
                        if ">" not in chunk:
                            continue
                        gt = chunk.index(">")
                        pname = chunk[:gt]
                        args[pname] = self._coerce(
                            self._trim_value(chunk[gt + 1 :]),
                            pname,
                            schema_map,
                            func_name,
                        )

                    calls.append(
                        ToolCallItem(
                            tool_index=index,
                            name=func_name,
                            parameters=json.dumps(args, ensure_ascii=False),
                        )
                    )
                    index += 1

            first = text.find(self.call_open)
            if first == -1:
                first = text.find(self.func_open)
            prefix = text[:first] if first > 0 else ""
            return StreamingParseResult(normal_text=prefix, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        if not self._buffer:
            return StreamingParseResult()

        calls: List[ToolCallItem] = []
        content: List[str] = []
        known_tags = (
            self.call_open,
            self.call_close,
            self.func_open,
            self.func_close,
            self.param_open,
            self.param_close,
        )

        while True:
            rest = self._buffer[self._cursor :]
            if not rest:
                break

            # <tool_call>
            if rest.startswith(self.call_open):
                self._cursor += len(self.call_open)
                self._in_call = True
                continue

            # <function=NAME>
            if rest.startswith(self.func_open):
                gt = rest.find(">")
                if gt == -1:
                    break
                func_name = rest[len(self.func_open) : gt]

                self.current_tool_id += 1
                self.current_tool_name_sent = True
                self._params_emitted = 0
                self._args_open = False
                self._active_func = func_name

                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id, name=func_name, parameters=""
                    )
                )
                self._cursor += gt + 1
                continue

            # <parameter=NAME>VALUE
            if rest.startswith(self.param_open):
                gt = rest.find(">")
                if gt == -1:
                    break
                tail = rest[gt + 1 :]

                # A value ends at </parameter>, or (abnormally) at the next
                # <parameter= or at </function>; whichever comes first.
                stops = []
                for marker, consume in (
                    (self.param_close, len(self.param_close)),
                    (self.param_open, 0),
                    (self.func_close, 0),
                ):
                    pos = tail.find(marker)
                    if pos != -1:
                        stops.append((pos, consume))
                if not stops:
                    break

                stop_pos, consume = min(stops, key=lambda s: s[0])
                pname = rest[len(self.param_open) : gt]
                value = self._trim_value(tail[:stop_pos])

                if not self._args_open:
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="{")
                    )
                    self._args_open = True

                schema_map = self._lookup_param_schema(self._active_func, tools)
                typed = self._coerce(value, pname, schema_map, self._active_func)
                piece = f"{json.dumps(pname)}: {json.dumps(typed, ensure_ascii=False)}"
                if self._params_emitted > 0:
                    piece = f", {piece}"

                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters=piece)
                )
                self._params_emitted += 1
                self._cursor += gt + 1 + stop_pos + consume
                continue

            # </function>
            if rest.startswith(self.func_close):
                if not self._args_open:
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="{")
                    )
                    self._args_open = True
                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters="}")
                )
                self._cursor += len(self.func_close)
                self._active_func = None
                continue

            # </tool_call>
            if rest.startswith(self.call_close):
                self._cursor += len(self.call_close)
                self._in_call = False
                continue

            # Anything else: plain text, whitespace between tags, or a partial tag.
            lt = rest.find("<")
            if lt == -1:
                if not self._in_call:
                    content.append(rest)
                self._cursor += len(rest)
            elif lt == 0:
                if any(tag.startswith(rest) for tag in known_tags):
                    break  # possibly a split tag, wait for more input
                if not self._in_call:
                    content.append("<")
                self._cursor += 1
            else:
                if not self._in_call:
                    content.append(rest[:lt])
                self._cursor += lt

        if self._cursor > 0:
            self._buffer = self._buffer[self._cursor :]
            self._cursor = 0

        return StreamingParseResult(calls=calls, normal_text="".join(content))

    # ------------------------------------------------------------------
    # Structural tag (xgrammar ships the same XML format under this name)
    # ------------------------------------------------------------------
    def supports_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def get_structural_tag_name(self) -> str:
        return "qwen_3_coder"
