import copy
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Set, Union
from urllib.parse import unquote

from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    Format,
    OptionalFormat,
    OrFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TriggeredTagsFormat,
)

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import get_schema_properties

logger = logging.getLogger(__name__)


# Bare (suffix-less) Hunyuan special tokens. The shipping Hy3 tokenizer appends
# a shared suffix to each (e.g. ``<tool_calls:opensource>``); resolve the real
# token string from the vocab at runtime and fall back to these literals.
_HUNYUAN_TOKEN_NAMES = (
    "tool_calls",
    "tool_call",
    "tool_sep",
    "arg_key",
    "arg_value",
    "think",
)

_HUNYUAN_TOKEN_RE = re.compile(
    r"^<(?P<name>" + "|".join(_HUNYUAN_TOKEN_NAMES) + r")(?::[^>]+)?>$"
)


def resolve_hunyuan_tokens(tokenizer) -> Dict[str, str]:
    """Map bare token names to their real (possibly suffixed) strings in vocab.

    Returns ``{name: token_str}`` for each name found. A bare literal is used
    when the tokenizer carries no suffixed form, so the same detector serves
    both the preview (suffix-less) and shipping (suffixed) Hy3 tokenizers.
    """
    tokens: Dict[str, str] = {}
    vocab = None
    if tokenizer is not None:
        try:
            vocab = tokenizer.get_vocab()
        except Exception as e:
            logger.warning("Failed to read Hunyuan tokenizer vocab: %s", e)
            vocab = None
    if isinstance(vocab, dict):
        for tok in vocab:
            if not isinstance(tok, str):
                continue
            m = _HUNYUAN_TOKEN_RE.match(tok)
            if m:
                tokens[m.group("name")] = tok
    for name in _HUNYUAN_TOKEN_NAMES:
        tokens.setdefault(name, f"<{name}>")
    return tokens


_SCHEMA_TYPE_KEYS = ("type", "anyOf", "oneOf")
_MAX_REF_HOPS = 3
_MAX_ANCHOR_DEPTH = 8


def _is_ref_type(schema: Any) -> bool:
    return (
        isinstance(schema, dict)
        and isinstance(schema.get("$ref"), str)
        and not any(schema.get(key) for key in _SCHEMA_TYPE_KEYS)
    )


def _resolve_json_pointer(root_schema: Any, pointer: str) -> Any:
    if pointer == "":
        return root_schema
    if not pointer.startswith("/"):
        return None
    target = root_schema
    for raw_token in pointer[1:].split("/"):
        token = unquote(raw_token).replace("~1", "/").replace("~0", "~")
        if isinstance(target, dict):
            if token not in target:
                return None
            target = target[token]
        elif isinstance(target, list):
            if not token.isdigit() or int(token) >= len(target):
                return None
            target = target[int(token)]
        else:
            return None
    return target


def _find_schema_anchor(root_schema: Any, anchor: str) -> Optional[dict]:
    stack = [(root_schema, 0)]
    visited = set()
    while stack:
        node, depth = stack.pop()
        if depth > _MAX_ANCHOR_DEPTH or id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, dict):
            if node.get("$anchor") == anchor:
                return node
            children = list(node.values())
        elif isinstance(node, list):
            children = node
        else:
            continue
        stack.extend((child, depth + 1) for child in children)
    return None


def _resolve_local_schema_ref(root_schema: dict, ref: str) -> Optional[dict]:
    base, separator, fragment = ref.partition("#")
    # Schema resolution must never fetch an external document during parsing.
    if not separator or base:
        return None
    if fragment == "" or fragment.startswith("/"):
        target = _resolve_json_pointer(root_schema, fragment)
    else:
        target = _find_schema_anchor(root_schema, unquote(fragment))
    return target if isinstance(target, dict) else None


def _resolve_ref_type(
    schema: dict,
    root_schema: dict,
    seen_refs: frozenset = frozenset(),
    ref_hops: int = 0,
) -> dict:
    ref = schema.get("$ref")
    if not isinstance(ref, str):
        return {}
    if ref in seen_refs or ref_hops >= _MAX_REF_HOPS:
        logger.warning(
            "Could not resolve JSON Schema $ref %r; leaving it untyped.", ref
        )
        return {}
    target = _resolve_local_schema_ref(root_schema, ref)
    if target is None:
        logger.warning("Unknown local JSON Schema $ref %r; leaving it untyped.", ref)
        return {}
    type_schema = {
        key: copy.deepcopy(target[key]) for key in _SCHEMA_TYPE_KEYS if target.get(key)
    }
    if type_schema:
        return (
            _resolve_union_ref_types(
                type_schema, root_schema, seen_refs | {ref}, ref_hops + 1
            )
            or type_schema
        )
    if _is_ref_type(target):
        return _resolve_ref_type(target, root_schema, seen_refs | {ref}, ref_hops + 1)
    return {}


def _resolve_union_ref_types(
    schema: dict,
    root_schema: dict,
    seen_refs: frozenset = frozenset(),
    ref_hops: int = 0,
) -> Optional[dict]:
    if ref_hops >= _MAX_REF_HOPS:
        return None
    resolved_schema = None
    for union_key in ("anyOf", "oneOf"):
        branches = schema.get(union_key)
        if not isinstance(branches, list):
            continue
        for index, original_branch in enumerate(branches):
            if _is_ref_type(original_branch):
                resolved_branch = _resolve_ref_type(
                    original_branch, root_schema, seen_refs, ref_hops
                )
            elif isinstance(original_branch, dict) and any(
                key in original_branch for key in ("anyOf", "oneOf")
            ):
                resolved_branch = _resolve_union_ref_types(
                    original_branch, root_schema, seen_refs, ref_hops + 1
                )
            else:
                continue
            if resolved_branch:
                if resolved_schema is None:
                    resolved_schema = copy.deepcopy(schema)
                resolved_schema[union_key][index] = resolved_branch
    return resolved_schema


class HunyuanDetector(BaseFormatDetector):
    """
    Detector for Hunyuan (HYV3) tool call format.

    Format:
        <tool_calls>
        <tool_call>function_name<tool_sep>
        <arg_key>key1</arg_key>
        <arg_value>value1</arg_value>
        </tool_call>
        </tool_calls>

    Streaming behavior:
      * Phase 1 emits the tool name once <tool_sep> is seen.
      * Phase 2 streams argument JSON incrementally. Closed <arg_value>
        pairs are parsed with schema-aware type coercion; pure-string
        args may be streamed char-by-char (with JSON escaping). The
        closing "}" is withheld until </tool_call> arrives.
    """

    _TYPE_ALIASES: Dict[str, str] = {
        "str": "string",
        "text": "string",
        "varchar": "string",
        "char": "string",
        "enum": "string",
        "bool": "boolean",
        "binary": "boolean",
        "int": "integer",
        "float": "number",
        "double": "number",
        "list": "array",
        "dict": "object",
        "map": "object",
    }

    _INTEGER_PREFIXES = ("int", "uint", "long", "short", "unsigned")
    _NUMBER_PREFIXES = ("num", "float")

    def __init__(self, tokenizer=None):
        super().__init__()

        # Hy4 dropped <tool_sep> from its vocab entirely; Hy3 carries it (bare
        # or suffixed). Detect the dialect from the vocab, not from the
        # resolved literal (which falls back to "<tool_sep>" for both).
        self._has_tool_sep = True
        if tokenizer is not None:
            try:
                self._has_tool_sep = any(
                    re.fullmatch(r"<tool_sep(?::[^>]+)?>", tok)
                    for tok in tokenizer.get_vocab()
                    if isinstance(tok, str)
                )
            except Exception:
                pass

        t = resolve_hunyuan_tokens(tokenizer)
        tool_calls = t["tool_calls"]
        tool_call = t["tool_call"]
        tool_sep = t["tool_sep"]
        arg_key = t["arg_key"]
        arg_value = t["arg_value"]

        def _close(open_tok: str) -> str:
            return "</" + open_tok[1:] if open_tok.startswith("<") else open_tok

        self.bot_token = tool_calls
        self.eot_token = _close(tool_calls)
        self.tool_call_start_token = tool_call
        self.tool_call_end_token = _close(tool_call)
        self.tool_sep_token = tool_sep
        self.arg_key_start_token = arg_key
        self.arg_key_end_token = _close(arg_key)
        self.arg_value_start_token = arg_value
        self.arg_value_end_token = _close(arg_value)
        self.think_start_token = t["think"]
        self.think_end_token = _close(t["think"])
        # The tool and reasoning parsers split output at these literals, so free
        # text in the grammar must never contain them.
        self._structural_tokens = [
            self.think_start_token,
            self.think_end_token,
            self.bot_token,
            self.eot_token,
            self.tool_call_start_token,
            self.tool_call_end_token,
            self.arg_key_start_token,
            self.arg_key_end_token,
            self.arg_value_start_token,
            self.arg_value_end_token,
        ]
        if self._has_tool_sep:
            self._structural_tokens.append(self.tool_sep_token)

        tc_end = _close(tool_call)
        ak_end = _close(arg_key)
        av_end = _close(arg_value)
        self.tool_call_regex = re.compile(
            re.escape(tool_call)
            + r"(.*?)"
            + r"(?:"
            + re.escape(tool_sep)
            + r"|(?="
            + re.escape(arg_key)
            + r"|"
            + re.escape(tc_end)
            + r"))"
            + r"(.*?)"
            + re.escape(tc_end),
            re.DOTALL,
        )
        self.func_args_regex = re.compile(
            re.escape(arg_key)
            + r"(.*?)"
            + re.escape(ak_end)
            + r"\s*"
            + re.escape(arg_value)
            + r"(.*?)"
            + re.escape(av_end),
            re.DOTALL,
        )

        # Streaming state
        self._in_tool_calls: bool = False
        self._bare_tool_call: bool = False
        self._streaming_tool_name: Optional[str] = None
        self._completed_args: Dict[str, Any] = {}
        self._completed_arg_pairs: List[tuple[str, Any]] = []
        self._streamed_json_len: int = 0

    # ------------------------------------------------------------------
    # Type-normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_type(raw_type: str) -> str:
        exact = HunyuanDetector._TYPE_ALIASES.get(raw_type)
        if exact is not None:
            return exact
        lower = raw_type.lower()
        if any(lower.startswith(p) for p in HunyuanDetector._INTEGER_PREFIXES):
            return "integer"
        if any(lower.startswith(p) for p in HunyuanDetector._NUMBER_PREFIXES):
            return "number"
        return raw_type

    @staticmethod
    def _get_arg_schema(
        function_name: str, arg_key: str, tools: Optional[List[Tool]]
    ) -> dict:
        if not tools:
            return {}
        for tool in tools:
            if tool.function.name != function_name:
                continue
            parameters = tool.function.parameters
            if not isinstance(parameters, dict):
                return {}
            arg_schema = get_schema_properties(parameters).get(arg_key, {})
            if not isinstance(arg_schema, dict):
                return {}
            if _is_ref_type(arg_schema):
                return _resolve_ref_type(arg_schema, parameters) or arg_schema
            return _resolve_union_ref_types(arg_schema, parameters) or arg_schema
        return {}

    @staticmethod
    def _get_schema_options(arg_schema: dict) -> List[dict]:
        """Priority: single ``type`` > ``anyOf`` > ``oneOf``; else default string."""
        if isinstance(arg_schema.get("type"), list):
            return [{"type": type_value} for type_value in arg_schema["type"]]
        if "type" in arg_schema:
            return [arg_schema]
        for union_key in ("anyOf", "oneOf"):
            if union_key in arg_schema:
                return [
                    option
                    for branch in arg_schema[union_key]
                    for option in HunyuanDetector._get_schema_options(branch)
                ]
        return [{"type": "string"}]

    @staticmethod
    def _get_types(arg_schema: dict) -> Set[str]:
        schemas = HunyuanDetector._get_schema_options(arg_schema)
        return {
            HunyuanDetector._normalize_type(s.get("type", "string")) for s in schemas
        } - {"null"}

    @staticmethod
    def _allows_null(arg_schema: dict) -> bool:
        return any(
            HunyuanDetector._normalize_type(schema.get("type", "string")) == "null"
            for schema in HunyuanDetector._get_schema_options(arg_schema)
        )

    @staticmethod
    def _is_only_string_type(
        function_name: str, arg_key: str, tools: Optional[List[Tool]]
    ) -> bool:
        """Partial values stream only for non-nullable pure strings."""
        arg_schema = HunyuanDetector._get_arg_schema(function_name, arg_key, tools)
        return HunyuanDetector._get_types(arg_schema) == {
            "string"
        } and not HunyuanDetector._allows_null(arg_schema)

    @staticmethod
    def _try_parse_bool(value: str) -> Optional[bool]:
        lower = value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        return None

    @staticmethod
    def _try_parse_int(value: str) -> Optional[int]:
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _try_parse_number(value: str):
        """int if no '.'/'e'/'E', else float."""
        try:
            if "." in value or "e" in value or "E" in value:
                return float(value)
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _deserialize(value: str) -> Any:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

    @staticmethod
    def _parse_value(
        value: str,
        function_name: str,
        arg_key: str,
        tools: Optional[List[Tool]],
    ) -> Any:
        """Unified value parser: null → bool → int → number → json → string."""
        arg_schema = HunyuanDetector._get_arg_schema(function_name, arg_key, tools)
        types = HunyuanDetector._get_types(arg_schema)
        if value == "null" and HunyuanDetector._allows_null(arg_schema):
            return None

        if "boolean" in types:
            r = HunyuanDetector._try_parse_bool(value)
            if r is not None:
                return r

        if "integer" in types:
            r = HunyuanDetector._try_parse_int(value)
            if r is not None:
                return r

        if "number" in types:
            r = HunyuanDetector._try_parse_number(value)
            if r is not None:
                return r

        if types - {"string", "boolean", "integer", "number"}:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass

        if "string" in types:
            return value

        return HunyuanDetector._deserialize(value)

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text or self.tool_call_start_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = min(
            (
                pos
                for pos in (
                    text.find(self.bot_token),
                    text.find(self.tool_call_start_token),
                )
                if pos != -1
            ),
            default=-1,
        )
        if idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text = text[:idx].strip() if idx > 0 else ""

        tool_indices = self._get_tool_indices(tools)
        forward_unknown = envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()

        calls: List[ToolCallItem] = []
        try:
            for function_name, function_args in self.tool_call_regex.findall(text):
                function_name = function_name.strip()
                if function_name not in tool_indices and not forward_unknown:
                    logger.warning(
                        "Model attempted to call undefined function: %s", function_name
                    )
                    continue

                arg_dict: Dict[str, Any] = {}
                for key, value in self.func_args_regex.findall(function_args):
                    key = key.strip()
                    arg_dict[key] = self._parse_value(value, function_name, key, tools)

                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices.get(function_name, -1),
                        name=function_name,
                        parameters=json.dumps(arg_dict, ensure_ascii=False),
                    )
                )
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}", exc_info=True)
            return StreamingParseResult(normal_text=text)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _reset_streaming_tool_state(self):
        self._streaming_tool_name = None
        self._completed_args = {}
        self._completed_arg_pairs = []
        self._streamed_json_len = 0

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        try:
            return self._parse_streaming_increment_impl(new_text, tools)
        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}", exc_info=True)
            return StreamingParseResult()

    def _parse_streaming_increment_impl(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        # Not yet inside <tool_calls>: emit normal text or buffer partial markers.
        if not self._in_tool_calls:
            combined = self._buffer + new_text
            candidates = [
                (pos, bare)
                for pos, bare in (
                    (combined.find(self.bot_token), False),
                    (combined.find(self.tool_call_start_token), True),
                )
                if pos != -1
            ]
            if candidates:
                marker_pos, self._bare_tool_call = min(
                    candidates, key=lambda candidate: candidate[0]
                )
                normal_text = combined[:marker_pos]
                self._buffer = (
                    combined[marker_pos:]
                    if self._bare_tool_call
                    else combined[marker_pos + len(self.bot_token) :]
                )
                self._in_tool_calls = True
                return self._continue_streaming(tools, leading_normal=normal_text)

            partial_len = max(
                self._ends_with_partial_token(combined, self.bot_token),
                self._ends_with_partial_token(combined, self.tool_call_start_token),
            )
            if partial_len:
                self._buffer = combined[-partial_len:]
                return StreamingParseResult(normal_text=combined[:-partial_len])
            self._buffer = ""
            return StreamingParseResult(normal_text=combined)

        self._buffer += new_text
        return self._continue_streaming(tools)

    def _continue_streaming(
        self, tools: List[Tool], leading_normal: str = ""
    ) -> StreamingParseResult:
        """Drive the state machine after <tool_calls> is open."""
        calls: List[ToolCallItem] = []

        while True:
            if self._streaming_tool_name is None:
                # Phase 1: wait for <tool_call>..<tool_sep>.
                tc_start = self._buffer.find(self.tool_call_start_token)
                if self._bare_tool_call:
                    outer_pos = self._buffer.find(self.bot_token)
                    eot_pos = self._buffer.find(self.eot_token)
                    if eot_pos != -1 and (
                        (outer_pos == -1 or eot_pos < outer_pos)
                        and (tc_start == -1 or eot_pos < tc_start)
                    ):
                        trailing = self._buffer[eot_pos + len(self.eot_token) :]
                        next_positions = [
                            pos
                            for pos in (
                                trailing.find(self.bot_token),
                                trailing.find(self.tool_call_start_token),
                            )
                            if pos != -1
                        ]
                        if next_positions:
                            next_pos = min(next_positions)
                            leading_normal += trailing[:next_pos]
                            next_is_outer = trailing[next_pos:].startswith(
                                self.bot_token
                            )
                            if next_is_outer:
                                next_pos += len(self.bot_token)
                            self._buffer = trailing[next_pos:]
                            self._in_tool_calls = True
                            self._bare_tool_call = not next_is_outer
                            continue
                        partial_len = max(
                            self._ends_with_partial_token(trailing, self.bot_token),
                            self._ends_with_partial_token(
                                trailing, self.tool_call_start_token
                            ),
                        )
                        if partial_len:
                            safe_end = len(trailing) - partial_len
                            leading_normal += trailing[:safe_end]
                            self._buffer = trailing[safe_end:]
                        else:
                            leading_normal += trailing
                            self._buffer = ""
                        self._in_tool_calls = False
                        self._bare_tool_call = False
                        break
                    if outer_pos != -1 and (tc_start == -1 or outer_pos < tc_start):
                        leading_normal += self._buffer[:outer_pos]
                        self._buffer = self._buffer[outer_pos + len(self.bot_token) :]
                        self._bare_tool_call = False
                        continue
                if tc_start == -1:
                    if self._bare_tool_call:
                        partial_len = max(
                            self._ends_with_partial_token(self._buffer, self.bot_token),
                            self._ends_with_partial_token(self._buffer, self.eot_token),
                            self._ends_with_partial_token(
                                self._buffer, self.tool_call_start_token
                            ),
                        )
                        if partial_len:
                            safe_end = len(self._buffer) - partial_len
                            leading_normal += self._buffer[:safe_end]
                            self._buffer = self._buffer[safe_end:]
                        elif self._buffer:
                            leading_normal += self._buffer
                            self._buffer = ""
                        break
                    if self.eot_token in self._buffer:
                        eot_pos = self._buffer.find(self.eot_token)
                        self._buffer = self._buffer[eot_pos + len(self.eot_token) :]
                        self._in_tool_calls = False
                    break

                name_start = tc_start + len(self.tool_call_start_token)
                boundaries = [
                    (self._buffer.find(token, name_start), token)
                    for token in (
                        self.tool_sep_token,
                        self.arg_key_start_token,
                        self.tool_call_end_token,
                    )
                ]
                boundaries = [(pos, token) for pos, token in boundaries if pos != -1]
                if not boundaries:
                    self._buffer = self._buffer[tc_start:]
                    break
                boundary_pos, boundary_token = min(boundaries, key=lambda item: item[0])

                tool_name = self._buffer[name_start:boundary_pos].strip()

                if (
                    tool_name not in self._tool_indices
                    and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
                ):
                    logger.warning(
                        "Model attempted to call undefined function: %s", tool_name
                    )

                self._streaming_tool_name = tool_name
                self.current_tool_id += 1
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=tool_name,
                        parameters="",
                    )
                )

                if boundary_token == self.tool_sep_token:
                    boundary_pos += len(boundary_token)
                self._buffer = self._buffer[boundary_pos:]

            # Phase 2: stream argument JSON of the current tool.
            before_name = self._streaming_tool_name
            calls.extend(self._stream_args(tools))
            if self._streaming_tool_name is not None:
                break  # current tool still open; need more data.
            if self._streaming_tool_name == before_name:
                break  # safety: avoid infinite loop if state didn't advance.

        return StreamingParseResult(normal_text=leading_normal, calls=calls)

    def _stream_args(self, tools: List[Tool]) -> List[ToolCallItem]:
        """Emit argument-JSON deltas for the currently-open tool call."""
        is_complete = self.tool_call_end_token in self._buffer

        if is_complete:
            end_idx = self._buffer.find(self.tool_call_end_token)
            args_text = self._buffer[:end_idx]
        else:
            args_text = self._buffer

        # 1. Absorb closed <arg_key>..<arg_value> pairs, including duplicates.
        matches = list(self.func_args_regex.finditer(args_text))
        for m in matches[len(self._completed_arg_pairs) :]:
            key, value = m.groups()
            key = key.strip()
            parsed_value = self._parse_value(
                value, self._streaming_tool_name or "", key, tools
            )
            self._completed_arg_pairs.append((key, parsed_value))
            self._completed_args[key] = parsed_value
        last_closed_end = matches[-1].end() if matches else 0

        # 2. Detect a partial (unclosed) kv pair at the tail.
        tail = args_text[last_closed_end:]
        partial_key: Optional[str] = None
        partial_value: Optional[str] = None

        ak_start = tail.find(self.arg_key_start_token)
        if ak_start != -1:
            ak_end = tail.find(
                self.arg_key_end_token, ak_start + len(self.arg_key_start_token)
            )
            if ak_end != -1:
                partial_key = tail[
                    ak_start + len(self.arg_key_start_token) : ak_end
                ].strip()
                av_start = tail.find(self.arg_value_start_token, ak_end)
                if av_start != -1 and self._is_only_string_type(
                    self._streaming_tool_name or "", partial_key, tools
                ):
                    partial_value = tail[av_start + len(self.arg_value_start_token) :]

        # Avoid emitting a lone "{" before any arg content is knowable.
        if not is_complete and not self._completed_args and partial_value is None:
            return []

        # 3. Build the JSON snapshot manually to control streaming boundaries.
        snapshot_parts: List[str] = []
        for k, v in self._completed_arg_pairs:
            k_json = json.dumps(k, ensure_ascii=False)
            v_json = json.dumps(v, ensure_ascii=False)
            snapshot_parts.append(f"{k_json}: {v_json}")

        if partial_key is not None and partial_value is not None:
            # Hold back suffixes that can become either closing marker, so a
            # split tag never leaks into the streamed JSON string value.
            hold = max(
                self._ends_with_partial_token(partial_value, self.arg_value_end_token),
                self._ends_with_partial_token(partial_value, self.tool_call_end_token),
            )
            safe_value = partial_value[:-hold] if hold else partial_value
            k_json = json.dumps(partial_key, ensure_ascii=False)
            escaped = (
                safe_value.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )
            # No closing `"` here — it's appended when the value closes.
            snapshot_parts.append(f'{k_json}: "{escaped}')

        snapshot = "{" + ", ".join(snapshot_parts) + "}"

        argument_diff: Optional[str] = None

        if is_complete:
            if partial_key is not None and partial_value is not None:
                self._completed_arg_pairs.append((partial_key, safe_value))
                self._completed_args[partial_key] = safe_value
            final_parts = [
                f"{json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
                for k, v in self._completed_arg_pairs
            ]
            final_json = "{" + ", ".join(final_parts) + "}"
            if self._streamed_json_len < len(final_json):
                argument_diff = final_json[self._streamed_json_len :]
            self._streamed_json_len = len(final_json)

            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": self._streaming_tool_name,
                "arguments": dict(self._completed_args),
            }

            end_idx = self._buffer.find(self.tool_call_end_token)
            self._buffer = self._buffer[end_idx + len(self.tool_call_end_token) :]
            self._reset_streaming_tool_state()
        else:
            # Withhold the trailing "}" while the tool call is still open.
            end = len(snapshot) - 1
            if end > self._streamed_json_len:
                argument_diff = snapshot[self._streamed_json_len : end]
                self._streamed_json_len = end

        if argument_diff:
            self.streamed_args_for_tool[self.current_tool_id] += argument_diff
            return [
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    parameters=argument_diff,
                )
            ]
        return []

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        result = self._continue_streaming(tools)
        if self._buffer and self._streaming_tool_name is None:
            normal_text = self._buffer
            self._buffer = ""
            self._in_tool_calls = False
            self._bare_tool_call = False
            result.normal_text += normal_text
        return result

    def structure_info(self) -> _GetInfoFunc:
        # Hy4 has no <tool_sep> and no separators between adjacent
        # <tool_call> blocks, so the Hy3 begin/end literals never match.
        if not self._has_tool_sep:
            return lambda name: StructureInfo(
                begin=f"{self.bot_token}{self.tool_call_start_token}{name}",
                end=f"{self.tool_call_end_token}{self.eot_token}",
                trigger=self.bot_token,
            )
        return lambda name: StructureInfo(
            begin=f"{self.bot_token}\n{self.tool_call_start_token}{name}{self.tool_sep_token}",
            end=f"{self.tool_call_end_token}\n{self.eot_token}",
            trigger=self.bot_token,
        )

    def supports_structural_tag(self) -> bool:
        return False

    def parses_required_natively(self) -> bool:
        # get_structural_tag constrains forced choices in the native dialect; a
        # json_schema constraint would force plain JSON the model never emits.
        return True

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        is_named = isinstance(tool_choice, ToolChoice)
        if is_named:
            tools = [
                tool
                for tool in tools or []
                if tool.function.name == tool_choice.function.name
            ]
        if not tools:
            return None

        calls = self._tool_calls_format(
            tools, single_call=is_named or not parallel_tool_calls
        )
        if tool_choice == "auto":
            block = TagFormat(begin=self.bot_token, content=calls, end=self.eot_token)
            suffix = TriggeredTagsFormat(
                triggers=[self.bot_token],
                tags=[block],
                stop_after_first=not parallel_tool_calls,
            )
        else:
            suffix = self._sequence(
                self._newline(),
                ConstStringFormat(value=self.bot_token),
                calls,
                ConstStringFormat(value=self.eot_token),
            )
        if not thinking_mode:
            return StructuralTag(format=suffix)
        reasoning = self._free_text(open_token=None, close_token=self.think_end_token)
        return StructuralTag(format=SequenceFormat(elements=[reasoning, suffix]))

    def _newline(self) -> Optional[Format]:
        # Hy3 may break lines between its tags; Hy4 emits them back to back.
        if not self._has_tool_sep:
            return None
        return OptionalFormat(content=ConstStringFormat(value="\n"))

    def _sequence(self, *elements: Optional[Format]) -> Format:
        present = [element for element in elements if element is not None]
        return present[0] if len(present) == 1 else SequenceFormat(elements=present)

    def _tool_calls_format(self, tools: List[Tool], single_call: bool) -> Format:
        tags = [self._tool_call_tag(tool) for tool in tools]
        call = tags[0] if len(tags) == 1 else OrFormat(elements=tags)
        if single_call:
            return self._sequence(self._newline(), call, self._newline())
        more = StarFormat(content=self._sequence(self._newline(), call))
        return self._sequence(self._newline(), call, more, self._newline())

    def _tool_call_tag(self, tool: Tool) -> TagFormat:
        name = tool.function.name
        begin = self.tool_call_start_token + name
        if self._has_tool_sep:
            begin += self.tool_sep_token
        parameters = tool.function.parameters or {}
        properties = parameters.get("properties") or {}
        required = parameters.get("required") or []
        if properties or required:
            # Values stay free text: the dialect writes them unquoted, so only
            # the argument keys and their order can be constrained.
            required_keys = [key for key in properties if key in required]
            required_keys += [key for key in required if key not in properties]
            pairs = [self._arg_pair(key) for key in required_keys]
            optional = [
                self._arg_pair(key) for key in properties if key not in required
            ]
            if optional:
                choice = (
                    optional[0] if len(optional) == 1 else OrFormat(elements=optional)
                )
                pairs.append(StarFormat(content=choice))
        else:
            pairs = [StarFormat(content=self._arg_pair(None))]
        return TagFormat(
            begin=begin,
            content=self._sequence(self._newline(), *pairs),
            end=self.tool_call_end_token,
        )

    def _arg_pair(self, key: Optional[str]) -> Format:
        if key is None:
            key_format = self._free_text(
                open_token=self.arg_key_start_token,
                close_token=self.arg_key_end_token,
            )
        else:
            key_format = ConstStringFormat(
                value=self.arg_key_start_token + key + self.arg_key_end_token
            )
        value_format = self._free_text(
            open_token=self.arg_value_start_token,
            close_token=self.arg_value_end_token,
        )
        return self._sequence(
            key_format, self._newline(), value_format, self._newline()
        )

    def _free_text(self, open_token: Optional[str], close_token: str) -> Format:
        return self._sequence(
            None if open_token is None else ConstStringFormat(value=open_token),
            AnyTextFormat(excludes=self._structural_tokens),
            ConstStringFormat(value=close_token),
        )
