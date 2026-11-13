# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
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
import json
import logging
import re
from json.encoder import encode_basestring
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    ConstStringFormat,
    JSONSchemaFormat,
    SequenceFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import get_schema_properties

logger = logging.getLogger(__name__)


def _lookup_json_pointer(root: Any, ref: str) -> Any:
    """Resolve a local JSON pointer such as "#/$defs/X" against root."""
    node = root
    for part in ref[1:].split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def resolve_local_refs(schema: Any, root: Any, seen: tuple = ()) -> Any:
    """Inline local $ref at the levels extract_types_from_schema reads,
    the node itself and its anyOf/oneOf/allOf branches; cycles stop at the ref."""
    if not isinstance(schema, dict):
        return schema
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#") and ref not in seen:
        target = _lookup_json_pointer(root, ref)
        if isinstance(target, dict):
            siblings = {k: v for k, v in schema.items() if k != "$ref"}
            return resolve_local_refs({**target, **siblings}, root, seen + (ref,))
    resolved = dict(schema)
    for keyword in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(keyword), list):
            resolved[keyword] = [
                resolve_local_refs(b, root, seen) for b in schema[keyword]
            ]
    return resolved


# A string parameter whose whole value is one of these converts to null.
NULL_WORDS = ("null", "none", "nil")
STRING_TYPES = {"string", "str", "text"}


def extract_types_from_schema(schema: Any) -> list[str]:
    if schema is None:
        return ["string"]

    if not isinstance(schema, dict):
        return ["string"]

    types: set[str] = set()

    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            types.add(type_value)
        elif isinstance(type_value, list):
            for t in type_value:
                if isinstance(t, str):
                    types.add(t)

    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        for value in schema["enum"]:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")

    for choice_field in ("anyOf", "oneOf", "allOf"):
        if choice_field in schema and isinstance(schema[choice_field], list):
            for choice in schema[choice_field]:
                extracted = extract_types_from_schema(choice)
                types.update(extracted)

    if not types:
        return ["string"]

    return list(types)


def convert_param_value_with_types(value: str, param_types: list[str]) -> Any:
    normalized_types = [t.lower() for t in param_types]

    # A non-nullable string keeps its value verbatim: "null"/"none"/"nil"
    # are ordinary string content, not JSON null. Coercing them would also
    # diverge this path from the streamed one (cf. minimax_m3).
    if "null" not in normalized_types and any(
        t in STRING_TYPES for t in normalized_types
    ):
        return value
    if value.lower() in NULL_WORDS:
        return None

    type_priority = [
        "integer",
        "int",
        "number",
        "float",
        "boolean",
        "bool",
        "object",
        "array",
        "string",
        "str",
        "text",
    ]

    for param_type in type_priority:
        if param_type not in normalized_types:
            continue

        if param_type in ["string", "str", "text"]:
            return value
        elif param_type in ["integer", "int"]:
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        elif param_type in ["number", "float"]:
            try:
                val = float(value)
                return val if val != int(val) else int(val)
            # int() of an infinity ("inf", "1e999") raises OverflowError.
            except (ValueError, TypeError, OverflowError):
                continue
        elif param_type in ["boolean", "bool"]:
            lower_val = value.lower().strip()
            if lower_val in ["true", "1", "yes", "on"]:
                return True
            elif lower_val in ["false", "0", "no", "off"]:
                return False
            continue
        elif param_type in ["object", "array"]:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                continue

    # Fallback: try JSON parse, then return as string
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


_INVOKE_END = "</invoke>\n"
_EMPTY_PARAMETERS = {"type": "object", "properties": {}}

_INVOKE_OPEN = re.compile(r'<invoke name="([^>]+)">')
_PARAM_OPEN = re.compile(r'<parameter name="([^>]+)">')
_PARAM_END = "</parameter>"
_FUNCTION_END = "</invoke>"
_MAX_NULL_WORD = max(map(len, NULL_WORDS))
_NULLABLE_STRING_TYPES = STRING_TYPES | {"null"}


def _json_string_body(text: str) -> str:
    """JSON-escape text without the quotes, as json.dumps(ensure_ascii=False) does;
    the escaping is per character, so escaped pieces concatenate."""
    return encode_basestring(text)[1:-1]


class Ernie5_5Detector(BaseFormatDetector):
    """ERNIE 5.5 Flash tool-call format:

    <ernie:tool_call>
    <invoke name="func1">
    <parameter name="param1">value1</parameter>
    <parameter name="param2">value2</parameter>
    </invoke>
    </ernie:tool_call>
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<ernie:tool_call>"
        self.tool_call_end_token: str = "</ernie:tool_call>"
        self.tool_call_function_regex = re.compile(
            r"<invoke name=\"(.*?)</invoke>|<invoke name=\"(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter name=\"(.*?)</parameter>|<parameter name=\"(.*?)$", re.DOTALL
        )
        # A parameter value ends at whichever comes first,
        # tolerating a missing </parameter> or </invoke> like the non-streaming parse.
        self._value_ends = (_PARAM_END, _FUNCTION_END, self.tool_call_end_token)
        self._max_end_prefix = max(map(len, self._value_ends)) - 1

        self._buf: str = ""
        self._tool_indices: Optional[Dict[str, int]] = None
        self._in_tool_call = False
        self._in_invoke = False
        self._invoke_name = ""
        # Aliased by prev_tool_call_arr, so it holds exactly what was streamed;
        # serving_chat completes a cut-off call from it.
        self._args: Dict[str, Any] = {}
        self._param: Optional[str] = None
        self._param_raw = ""
        self._param_is_string = False
        # Value chars already sent; None until a string value is quoted.
        self._param_sent: Optional[int] = None

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """The arguments JSON is closed when the call ends, never by counting braces,
        which string values may contain. A string-only parameter streams as
        generated, other types once complete. Undefined tools are dropped and a
        missing </parameter> or </invoke> is tolerated, as in the non-streaming parse.
        """
        if self._tool_indices is None:
            self._tool_indices = self._get_tool_indices(tools)
        self._buf += new_text
        normal = ""
        calls: List[ToolCallItem] = []
        try:
            while self._buf:
                if not self._in_tool_call:
                    start = self._buf.find(self.tool_call_start_token)
                    if start == -1:
                        # Hold back a tail that may be the start tag split over chunks.
                        keep = self._ends_with_partial_token(
                            self._buf, self.tool_call_start_token
                        )
                        cut = len(self._buf) - keep
                        normal, self._buf = normal + self._buf[:cut], self._buf[cut:]
                        break
                    normal += self._buf[:start]
                    self._buf = self._buf[start + len(self.tool_call_start_token) :]
                    self._in_tool_call = True
                elif self._param is not None:
                    if not self._advance_param(tools, calls):
                        break
                elif self._in_invoke:
                    if not self._advance_invoke(tools, calls):
                        break
                elif not self._advance_block(calls):
                    break
        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
        return StreamingParseResult(normal_text=normal, calls=calls)

    def _advance_block(self, calls: List[ToolCallItem]) -> bool:
        """Between calls: open the next one or close the block; False to wait."""
        invoke = _INVOKE_OPEN.search(self._buf)
        end = self._buf.find(self.tool_call_end_token)
        if end != -1 and (invoke is None or end < invoke.start()):
            # Text after the block is content again.
            self._buf = self._buf[end + len(self.tool_call_end_token) :]
            self._in_tool_call = False
            return True
        if invoke is None:
            return False
        name = invoke.group(1).strip()
        if name in self._tool_indices or envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
            self._buf = self._buf[invoke.end() :]
            self._open_invoke(name, calls)
            return True
        stop = self._first_of((_FUNCTION_END, self.tool_call_end_token), invoke.end())
        if stop is None:
            return False
        logger.warning(f"Model attempted to call undefined function: {name}")
        pos, tag = stop
        self._buf = self._buf[pos + len(tag) if tag == _FUNCTION_END else pos :]
        return True

    def _open_invoke(self, name: str, calls: List[ToolCallItem]) -> None:
        self._in_invoke, self._invoke_name, self._args = True, name, {}
        self.current_tool_id = len(self.prev_tool_call_arr)
        self.prev_tool_call_arr.append({"name": name, "arguments": self._args})
        self.streamed_args_for_tool.append("")
        calls.append(
            ToolCallItem(tool_index=self.current_tool_id, name=name, parameters="")
        )

    def _advance_invoke(self, tools: List[Tool], calls: List[ToolCallItem]) -> bool:
        """Open the next parameter or close the call; False to wait for text."""
        param = _PARAM_OPEN.search(self._buf)
        stop = self._first_of((_FUNCTION_END, self.tool_call_end_token))
        if stop is not None and (param is None or stop[0] < param.start()):
            # The first key lands in _args as it is sent,
            # so _args tells whether the arguments JSON was opened.
            self._emit(calls, "}" if self._args else "{}")
            self._in_invoke = False
            pos, tag = stop
            # A missing </invoke> leaves the block end to _advance_block.
            self._buf = self._buf[pos + len(tag) if tag == _FUNCTION_END else pos :]
            return True
        if param is None:
            return False
        self._param = param.group(1).strip()
        self._param_raw, self._param_sent = "", None
        types = {
            t.lower() for t in self._param_types(self._invoke_name, self._param, tools)
        }
        # convert_param_value_with_types returns raw text (or None for a null word)
        # for exactly these types, so the value can be sent as it grows.
        self._param_is_string = bool(types & STRING_TYPES) and (
            types <= _NULLABLE_STRING_TYPES
        )
        self._buf = self._buf[param.end() :]
        return True

    def _advance_param(self, tools: List[Tool], calls: List[ToolCallItem]) -> bool:
        """Consume text of the open parameter; True once its value has ended."""
        # Only new text, plus an end tag split across chunks, needs searching.
        start = max(0, len(self._param_raw) - self._max_end_prefix)
        if self._param_raw:
            self._param_raw += self._buf
        else:
            # The non-streaming parse strips leading newlines from the value.
            self._param_raw = self._buf.lstrip("\n")
        self._buf = ""
        raw = self._param_raw
        ends = [(i, t) for t in self._value_ends if (i := raw.find(t, start)) != -1]
        if not ends:
            if self._param_is_string:
                self._stream_string_value(calls)
            return False
        end, tag = min(ends)
        # Only </parameter> belongs to the value; the others close more.
        self._buf = raw[end + len(tag) :] if tag == _PARAM_END else raw[end:]
        value = raw[:end].rstrip("\n")
        if self._param_sent is None:
            converted = self._parse_parameter(
                self._invoke_name, self._param, value, tools
            )
            self._emit(
                calls,
                self._open_key(self._param) + json.dumps(converted, ensure_ascii=False),
            )
        else:
            # Quoted only once the value could no longer be a null word,
            # so it converts to itself.
            converted = value
            self._emit(calls, _json_string_body(value[self._param_sent :]) + '"')
        self._args[self._param] = converted
        self._param, self._param_raw = None, ""
        return True

    def _stream_string_value(self, calls: List[ToolCallItem]) -> None:
        raw = self._param_raw
        end = len(raw)
        # Hold a trailing "<", "</par", ... that may start an end tag;
        # the tags have "<" only at their start, so only the last "<" can begin one.
        lt = raw.rfind("<", max(0, end - self._max_end_prefix))
        if lt != -1 and any(t.startswith(raw[lt:]) for t in self._value_ends):
            end = lt
        # Hold trailing newlines too: the non-streaming parse strips them.
        while end > 0 and raw[end - 1] == "\n":
            end -= 1
        sent = self._param_sent
        if sent is None:
            # The opening quote rules out null,
            # so wait while the value may still be a null word.
            if end <= _MAX_NULL_WORD and any(
                word.startswith(raw[:end].lower()) for word in NULL_WORDS
            ):
                return
            prefix, sent = self._open_key(self._param) + '"', 0
        elif end > sent:
            prefix = ""
        else:
            return
        self._emit(calls, prefix + _json_string_body(raw[sent:end]))
        self._param_sent = end
        self._args[self._param] = raw[:end]

    def _first_of(
        self, tags: Tuple[str, ...], pos: int = 0
    ) -> Optional[Tuple[int, str]]:
        """The earliest of tags in the buffer from pos, as (index, tag)."""
        found = [(i, t) for t in tags if (i := self._buf.find(t, pos)) != -1]
        return min(found) if found else None

    def _open_key(self, name: str) -> str:
        """'{"name": ' for the first argument, ', "name": ' after it."""
        separator = ", " if self._args else "{"
        return f"{separator}{json.dumps(name, ensure_ascii=False)}: "

    def _emit(self, calls: List[ToolCallItem], fragment: str) -> None:
        self.streamed_args_for_tool[self.current_tool_id] += fragment
        calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id, name=None, parameters=fragment
            )
        )

    def _extract(self, text: str, tools: List[Tool]) -> Tuple[str, List[ToolCallItem]]:
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        cursor = 0
        while True:
            s = text.find(self.tool_call_start_token, cursor)
            if s == -1:
                normal_parts.append(text[cursor:])
                break
            normal_parts.append(text[cursor:s])
            e = text.find(self.tool_call_end_token, s)
            body_start = s + len(self.tool_call_start_token)
            if e == -1:
                # A generation cut off (max_tokens, or the end tag as a stop string)
                # still parses its complete calls, as the streaming path does.
                calls.extend(self._parse_block(text[body_start:], tools))
                break
            block = text[body_start:e]
            cursor = e + len(self.tool_call_end_token)
            calls.extend(self._parse_block(block, tools))
        return "".join(normal_parts), calls

    def _parse_block(self, block: str, tools: List[Tool]) -> List[ToolCallItem]:
        res: List[ToolCallItem] = []
        for m in self.tool_call_function_regex.findall(block):
            txt = m[0] if m[0] else m[1]
            if '">' not in txt:
                continue
            idx = txt.index('">')
            fname = txt[:idx].strip()
            body = txt[idx + 2 :]
            params: Dict[str, Any] = {}
            for pm in self.tool_call_parameter_regex.findall(body):
                ptxt = pm[0] if pm[0] else pm[1]
                if '">' not in ptxt:
                    continue
                pidx = ptxt.index('">')
                pname = ptxt[:pidx].strip()
                pval = ptxt[pidx + 2 :].lstrip("\n").rstrip("\n")
                params[pname] = self._parse_parameter(fname, pname, pval, tools)
            raw = {"name": fname, "arguments": params}
            try:
                res.extend(self.parse_base_json(raw, tools))
            except Exception:
                logger.warning("invalid tool call for %s dropped", fname)
        return res

    def supports_structural_tag(self) -> bool:
        # required / named tool_choice is constrained by the native tag below,
        # so the output must be parsed natively rather than as a JSON array.
        return True

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        """Constrain tool calls to the native <ernie:tool_call> XML format.

        Each tool's parameters schema is compiled as its own root, so local
        $ref such as "#/$defs/X" or "#/properties/x" resolve as written;
        the generic JSON-array fallback nests the schema and breaks them.

        ``thinking_mode`` is ignored; the think section is left to
        ReasonerGrammarBackend (--reasoning-parser).
        """
        tools = tools or []
        if isinstance(tool_choice, ToolChoice):
            tools = [t for t in tools if t.function.name == tool_choice.function.name]
        if not tools:
            return None

        invokes = TagsWithSeparatorFormat(
            tags=[self._invoke_tag(tool) for tool in tools],
            # Each invoke already ends with "\n", matching the model's output.
            separator="",
            at_least_one=True,
            stop_after_first=not parallel_tool_calls,
        )
        if tool_choice == "auto":
            # Reached for a strict tool or SGLANG_TOOL_STRICT_LEVEL >= FUNCTION;
            # text is free until the tool-call token, then the call is constrained.
            body = TriggeredTagsFormat(
                triggers=[self.tool_call_start_token],
                tags=[
                    TagFormat(
                        begin=self.tool_call_start_token + "\n",
                        content=invokes,
                        end=self.tool_call_end_token,
                    )
                ],
            )
        else:
            body = SequenceFormat(
                elements=[
                    ConstStringFormat(value=self.tool_call_start_token + "\n"),
                    invokes,
                    ConstStringFormat(value=self.tool_call_end_token),
                ]
            )
        return StructuralTag(format=body)

    @staticmethod
    def _invoke_tag(tool: Tool) -> TagFormat:
        return TagFormat(
            begin=f'<invoke name="{tool.function.name}">',
            content=JSONSchemaFormat(
                json_schema=tool.function.parameters or _EMPTY_PARAMETERS,
                # xgrammar's name for this <parameter name=...> XML layout.
                style="minimax_xml",
            ),
            end=_INVOKE_END,
        )

    def _parse_parameter(
        self, fname: str, pname: str, pval: str, tools: List[Tool]
    ) -> Any:
        return convert_param_value_with_types(
            pval, self._param_types(fname, pname, tools)
        )

    def _param_types(self, fname: str, pname: str, tools: List[Tool]) -> List[str]:
        # A parameter declared only via a local $ref (e.g. "#/properties/x")
        # has no "type" of its own, so resolve it first.
        root = next(
            (t.function.parameters for t in tools if t.function.name == fname), None
        )
        schema = resolve_local_refs(get_schema_properties(root).get(pname), root)
        return extract_types_from_schema(schema)

    def structure_info(self) -> _GetInfoFunc:
        # get_structural_tag builds the constraint natively.
        raise NotImplementedError
