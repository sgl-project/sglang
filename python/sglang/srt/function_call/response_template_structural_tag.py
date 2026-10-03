# Copyright 2026 SGLang Team
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
"""Structural tags for tool calls, derived from a response template."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Callable

import regex as re

from sglang.srt.parser.chat_parsing.content_parsers import parse_content
from sglang.srt.parser.chat_parsing.response_parser import _schema_types
from sglang.srt.parser.chat_parsing.response_templates import (
    ResponseTemplate,
    ResponseTemplateField,
)

_REGEX_META = re.compile(r"[.^$*+?{}\[\]\\|()]")
_LITERAL_ESCAPES = {"n": "\n", "r": "\r", "t": "\t"}
# Stands for `^` while expanding a delimiter pattern.
_START = "\0"
# Arbitrary bound on the strings expanded from one delimiter pattern.
_MAX_LITERALS = 16
_WHITESPACE = {"type": "regex", "pattern": r"\s*"}
_ANY_TEXT = {"type": "any_text"}
# Keywords that describe a schema without constraining its values.
_ANNOTATIONS = frozenset(
    {"title", "description", "default", "examples", "deprecated", "$comment"}
)
_OBJECT_KEYWORDS = frozenset(
    {"type", "properties", "required", "additionalProperties", "$defs", "definitions"}
)
# Characters JSON writes outside strings.
_JSON_SYNTAX = frozenset(' \t\n\r{}[],:"-+.0123456789eEtrufalsn')
# Pieces of the text of a call for `_spells_closer`. Text and JSON hold no
# closer, and JSON ends with a character it writes outside strings.
_TEXT_PIECE = (r"[\s\S]*", r"[\s\S]+", False)
_JSON_PIECE = (
    r"[\s\S]*",
    r"[\s\S]*[" + "".join(map(re.escape, sorted(_JSON_SYNTAX))) + "]",
    False,
)


def _split_group(pattern: str, name: str) -> tuple[str, str, str] | None:
    """Split a named regex group into its prefix, body and suffix."""
    opener = f"(?P<{name}>"
    start = pattern.find(opener)
    if start < 0:
        return None
    depth = 0
    in_class = False
    index = start
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            index += 2
            continue
        if in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                body = pattern[start + len(opener) : index]
                return pattern[:start], body, pattern[index + 1 :]
        index += 1
    return None


def _literals(pattern: str, *, message_start: bool = False) -> list[str] | None:
    """Expand literal alternatives, allowing `^` only at the message start."""
    literals, index = _alternatives(pattern, 0)
    if literals is None or index != len(pattern):
        return None
    if not message_start:
        return None if any(_START in literal for literal in literals) else literals
    return [
        literal.removeprefix(_START)
        for literal in literals
        if _START not in literal[1:] and literal != _START
    ]


def _alternatives(pattern: str, index: int) -> tuple[list[str] | None, int]:
    literals = []
    while True:
        branch, index = _concatenation(pattern, index)
        if branch is None:
            return None, index
        literals.extend(branch)
        if not pattern.startswith("|", index):
            break
        index += 1
    literals = list(dict.fromkeys(literals))
    return (literals if len(literals) <= _MAX_LITERALS else None), index


def _concatenation(pattern: str, index: int) -> tuple[list[str] | None, int]:
    literals = [""]
    while index < len(pattern) and pattern[index] not in "|)":
        if pattern.startswith("(?:", index):
            atom, index = _alternatives(pattern, index + 3)
            if atom is None or not pattern.startswith(")", index):
                return None, index
            index += 1
        elif pattern[index] == "\\" and index + 1 < len(pattern):
            escaped = pattern[index + 1]
            if escaped.isalnum() and escaped not in _LITERAL_ESCAPES:
                return None, index
            atom = [_LITERAL_ESCAPES.get(escaped, escaped)]
            index += 2
        elif pattern[index] == "^":
            atom = [_START]
            index += 1
        elif _REGEX_META.fullmatch(pattern[index]):
            return None, index
        else:
            atom = [pattern[index]]
            index += 1
        if pattern.startswith("?", index):
            atom = ["", *atom]
            index += 1
        literals = [prefix + suffix for prefix in literals for suffix in atom]
        if len(literals) > _MAX_LITERALS:
            return None, index
    return literals, index


def _openers(
    field: ResponseTemplateField, message_start: list[str] | None = None
) -> list[str] | None:
    """Derive field openers, including continuations of message-start anchors."""
    if field.open_literals or field.open_re is None:
        openers = field.open_literals
    else:
        openers = _literals(
            field.open_re.pattern, message_start=message_start is not None
        )
    return _continuing(openers, message_start, field.open_re)


def _continuing(
    literals: list[str] | None, anchors: list[str] | None, opener: re.Pattern
):
    """Keep anchor continuations only where the parser finds the opener."""
    if not literals or not anchors:
        return literals
    rests = [
        literal[len(anchor) :]
        for literal in literals
        for anchor in anchors
        if len(literal) > len(anchor) and literal.startswith(anchor)
    ]
    rests = [
        rest
        for rest in rests
        if any(opener.fullmatch(rest, index) for index in range(len(rest)))
    ]
    return list(dict.fromkeys([*literals, *rests]))


def _escape(text: str) -> str:
    return _REGEX_META.sub(lambda match: "\\" + match.group(), text)


def _call_opener(
    opener: tuple[str, str, str], name: str, message_start: list[str] | None
) -> str:
    before, _, after = opener
    if message_start is not None and before:
        literals = _literals(before, message_start=True)
        # Text before a name inside a group is no pattern of its own.
        if literals:
            literals = _continuing(literals, message_start, re.compile(before))
            before = "(?:" + "|".join(map(_escape, literals)) + ")"
    return before + name + after


def _thinking_format(
    thinking: ResponseTemplateField, message_start: list[str] | None
) -> dict[str, Any] | None:
    openers, closers = _openers(thinking, message_start), _closers(thinking)
    if not openers or not closers:
        return None
    return _one_of([_tag(opener, _ANY_TEXT, closers) for opener in openers])


def message_start_anchors(template: ResponseTemplate) -> list[str]:
    """The strings the assistant message starts after, if literal."""
    if template.start_anchor_literals:
        return template.start_anchor_literals
    return _literals(template.start_anchor_re.pattern) or []


def message_start_openers(
    field: ResponseTemplateField, anchors: list[str]
) -> list[str]:
    """The literal openers of `field` right after one of the `anchors`."""
    return _openers(field, anchors) or []


def typed_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Resolve annotated aliases and infer types needed for argument decoding."""
    # Imported here, as it imports xgrammar.
    from sglang.srt.function_call import kimik3_structural_tag as kimik3

    def typed(schema: Any, seen: frozenset[str] = frozenset()) -> Any:
        if not isinstance(schema, dict):
            return schema
        aliased = {"$ref", "allOf"} & set(schema)
        rest = {key: value for key, value in schema.items() if key not in aliased}
        target = None
        if len(aliased) == 1 and not set(rest) - _ANNOTATIONS - {"nullable"}:
            ref, branches = schema.get("$ref"), schema.get("allOf")
            if isinstance(ref, str) and ref not in seen:
                target, seen = kimik3._resolve_local_ref(ref, parameters), seen | {ref}
            elif isinstance(branches, list) and len(branches) == 1:
                target = branches[0]
        if isinstance(target, dict):
            return typed({**target, **rest}, seen)
        for keyword in ("anyOf", "oneOf"):
            if isinstance(schema.get(keyword), list):
                items = [typed(item, seen) for item in schema[keyword]]
                schema = {**schema, keyword: items}
        if not {"type", "anyOf", "oneOf"} & set(schema):
            types = kimik3._schema_types(schema, parameters)
            if types and len(types) < len(kimik3._JSON_TYPES):
                schema = {**schema, "type": types[0] if len(types) == 1 else types}
        return schema

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return parameters
    typed_properties = {key: typed(schema) for key, schema in properties.items()}
    return {**parameters, "properties": typed_properties}


def _closers(field: ResponseTemplateField) -> list[str] | None:
    """Return literal closers, an empty list if absent, or `None` if underivable."""
    if field.close_re is None:
        return []
    return field.close_literals or _literals(field.close_re.pattern)


# Every request derives the tags for its tools anew.
@lru_cache(maxsize=512)
def _endings(literals: tuple[str, ...]) -> str:
    return "|".join(
        re.escape(literal[index:])
        for literal in literals
        for index in range(len(literal))
    )


def _literal_piece(literals: list[str], before: str = "") -> tuple[str, str, bool]:
    either = before + "(?:" + "|".join(map(re.escape, literals)) + ")"
    return either, f"{either}|{_endings(tuple(literals))}", True


@lru_cache(maxsize=512)
def _spells_closer(
    closers: tuple[str, ...],
    pieces: tuple[tuple[str, str, bool], ...],
    *,
    repeats: bool = False,
) -> bool:
    """Check whether argument framing can spell a call closer across pieces."""
    texts = [text for text, _, _ in pieces]
    loop = f"(?:{''.join(texts)})*" if repeats else ""
    # Another call or an answer may follow the call's own closer.
    closed = "(?:" + "|".join(map(re.escape, closers)) + r")[\s\S]*"
    for index, (_, endings, holds) in enumerate(pieces):
        ends = re.compile(endings)
        rest = re.compile("".join(texts[index + 1 :]) + loop + closed, re.DOTALL)
        for closer in closers:
            if holds and ends.fullmatch(closer, partial=True):
                return True
            # A closer that starts in this piece and ends in the text after it.
            for split in range(1, len(closer)):
                if ends.fullmatch(closer[:split]) and rest.fullmatch(
                    closer[split:], partial=True
                ):
                    return True
    return False


def _const_strings(values: list[str]) -> dict[str, Any]:
    return _one_of([{"type": "const_string", "value": value} for value in values])


def _sequence(*elements: dict[str, Any]) -> dict[str, Any]:
    return {"type": "sequence", "elements": list(elements)}


def _one_of(formats: list[dict[str, Any]]) -> dict[str, Any]:
    return formats[0] if len(formats) == 1 else {"type": "or", "elements": formats}


def _optional(content: dict[str, Any]) -> dict[str, Any]:
    return {"type": "optional", "content": content}


def _tag(begin: str, content: dict[str, Any], end: str | list[str]) -> dict[str, Any]:
    return {"type": "tag", "begin": begin, "content": content, "end": end}


def _json_value_schema(schema: Any, parameters: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    if schema.get("nullable"):
        # XGrammar ignores `nullable`.
        schema = {"anyOf": [schema, {"type": "null"}]}
    definitions = {
        key: parameters[key] for key in ("$defs", "definitions") if key in parameters
    }
    return {**definitions, **schema}


def _string_choices(schema: Any) -> list[str] | None:
    if not isinstance(schema, dict):
        return None
    choices = [schema["const"]] if "const" in schema else schema.get("enum")
    if not isinstance(choices, list) or not choices:
        return None
    return choices if all(isinstance(choice, str) for choice in choices) else None


def _typed_by_parser(schema: Any) -> bool:
    """Whether the parser reads schema values with their JSON type."""
    if schema is True or _schema_types(schema):
        return True
    return isinstance(schema, dict) and (
        not set(schema) - _ANNOTATIONS or _string_choices(schema) is not None
    )


def _text_format(schema: Any) -> dict[str, Any] | None:
    """Constrain raw text without changing its parsed value."""
    if schema is True:
        return _ANY_TEXT
    if not isinstance(schema, dict) or set(schema) - _ANNOTATIONS - {
        "type",
        "nullable",
        "enum",
        "const",
        "anyOf",
        "oneOf",
    }:
        return None
    branches = [*(schema.get("anyOf") or []), *(schema.get("oneOf") or [])]
    if "enum" in schema or "const" in schema:
        choices = _string_choices(schema)
        if choices is None or branches or set(_schema_types(schema)) - {"string"}:
            return None
        return _const_strings(choices)
    if any(_text_format(branch) is not _ANY_TEXT for branch in branches):
        return None
    return _ANY_TEXT


def _value_format(
    schema: Any, parameters: dict[str, Any], json_values: bool, closers: list[str]
) -> dict[str, Any] | None:
    """Use JSON for decoded values and raw text for string values."""
    if schema is False:
        return None
    types = _schema_types(schema)
    if json_values or (types and "string" not in types):
        return _json_format(_json_value_schema(schema, parameters), closers)
    return _text_format(schema)


def _json_format(schema: Any, closers: list[str]) -> dict[str, Any] | None:
    """Use JSON unless the closer is ambiguous with its syntax or the schema is unsupported."""
    if closers and (
        any(set(closer) <= _JSON_SYNTAX for closer in closers)
        or not _json_compiles(json.dumps(schema))
    ):
        return None
    return {"type": "json_schema", "json_schema": schema}


# Every request derives the tags for its tools anew.
@lru_cache(maxsize=512)
def _json_compiles(schema: str) -> bool:
    from xgrammar import Grammar

    try:
        Grammar.from_json_schema(schema)
    except RuntimeError:
        return False
    return True


def _xml_arguments_format(
    content_args: dict[str, Any],
    parameters: dict[str, Any],
    *,
    strict: bool,
    closers: list[str],
) -> dict[str, Any] | None:
    if set(content_args) - {"tag_pattern", "value_parser", "merge_duplicates"}:
        return None
    value_parser = content_args.get("value_parser") or {"name": "text"}
    if value_parser.get("name") not in ("text", "json") or (
        value_parser.get("name") == "json" and value_parser.get("args")
    ):
        return None
    key_split = _split_group(content_args["tag_pattern"], "key")
    value_split = key_split and _split_group(key_split[2], "value")
    if not key_split or not value_split:
        return None
    key_pattern = key_split[1]
    begins, middles, ends = map(
        _literals, (key_split[0], value_split[0], value_split[2])
    )
    if not begins or not middles or not ends:
        return None
    properties = parameters.get("properties")
    json_values = value_parser.get("name") == "json"
    typed = properties is None or (
        isinstance(properties, dict)
        and (json_values or all(map(_typed_by_parser, properties.values())))
    )
    # Other schema shapes cannot be typed by the XML parser.
    if set(parameters) - _ANNOTATIONS - _OBJECT_KEYWORDS or not typed:
        return None
    required = parameters.get("required") or []
    names = ()
    if strict and properties is not None:
        # Required entries that are no names leave the tool to the JSON schema.
        if not isinstance(required, list) or not all(
            isinstance(key, str) for key in required
        ):
            return None
        names = (*properties, *(key for key in required if key not in properties))
    pieces = (
        _literal_piece(begins, before=r"\s*"),
        (f"(?:{key_pattern})", _endings(names), False),
        _literal_piece(middles),
        # Values of every format are some text.
        _TEXT_PIECE,
        _literal_piece(ends),
    )
    if _spells_closer(tuple(closers), pieces, repeats=True):
        return None

    # Raw text has no escaping for a call closer.
    text = {"type": "any_text", "excludes": closers} if closers else _ANY_TEXT

    def any_parameters() -> dict[str, Any]:
        name = {"type": "regex", "pattern": key_pattern}
        content = _sequence(name, _const_strings(middles), text)
        tags = [_tag(begin, content, ends) for begin in begins]
        return {"type": "star", "content": _sequence(_WHITESPACE, _one_of(tags))}

    if not strict:
        return any_parameters()
    if properties is None:
        extra = parameters.get("additionalProperties", True)
        if extra is False:
            return {"type": "const_string", "value": ""}
        return any_parameters() if extra is True or extra == {} else None
    # A declared key repeated as an extra one would bypass its schema.
    if parameters.get("additionalProperties", False) is not False:
        return None

    def read_back(written: str) -> dict[str, Any] | None:
        """The arguments the parser reads from `written`, unless a closer ends
        the call in it."""
        if any(closer in written for closer in closers):
            return None
        try:
            return parse_content(written, "xml-inline", content_args)
        except ValueError:
            return None

    def parameter(
        key: str, value: dict[str, Any], choices: list[str] | None = None
    ) -> dict[str, Any] | None:
        if not re.fullmatch(key_pattern, key) or any(
            list(read_back(b + key + m + "0" + e) or ()) != [key]
            for b in begins
            for m in middles
            for e in ends
        ):
            raise ValueError(f"parameter name {key!r} cannot be encoded in tool calls")
        if choices is None:
            tags = [_tag(b + key + m, value, ends) for b in begins for m in middles]
            return _sequence(_WHITESPACE, _one_of(tags))
        # Optional framing whitespace must not change a text choice.
        tags, framed = [], set()
        for b in begins:
            for m in middles:
                by_ends: dict[tuple[str, ...], list[str]] = {}
                for choice in choices:
                    fits = tuple(
                        e
                        for e in ends
                        if read_back(b + key + m + choice + e) == {key: choice}
                    )
                    by_ends.setdefault(fits, []).append(choice)
                for fits, group in by_ends.items():
                    if fits:
                        tags.append(
                            _tag(b + key + m, _const_strings(group), list(fits))
                        )
                        framed.update(group)
        if framed != set(choices):
            return None
        return _sequence(_WHITESPACE, _one_of(tags))

    elements = []
    for key, schema in properties.items():
        value = _value_format(schema, parameters, json_values, closers + ends)
        if value is None:
            return None
        choices = None
        if value["type"] in ("const_string", "or"):
            choices = _string_choices(schema)
        element = parameter(key, text if value == _ANY_TEXT else value, choices)
        if element is None:
            return None
        elements.append(element if key in required else _optional(element))
    elements.extend(parameter(key, text) for key in required if key not in properties)
    return _sequence(*elements) if elements else {"type": "const_string", "value": ""}


def _arguments_format(
    field: ResponseTemplateField, parameters: dict[str, Any], *, strict: bool
) -> dict[str, Any] | None:
    closers = _closers(field) or []
    if field.content == "json" and not field.content_args:
        if _spells_closer(tuple(closers), (_JSON_PIECE,)):
            return None
        schema = parameters if strict else {"type": "object"}
        return _json_format(schema, closers)
    if field.content == "xml-inline":
        return _xml_arguments_format(
            field.content_args, parameters, strict=strict, closers=closers
        )
    return None


def _names_function_by_opener(field: ResponseTemplateField) -> bool:
    function = field.transform.get("function") if field.transform else None
    return (
        field.open_re is not None
        and "name" in field.open_re.groupindex
        and not field.transform_each
        and isinstance(function, dict)
        and function.get("name") == "{name}"
        and function.get("arguments") == "{content}"
    )


def _reply_format(
    answer: ResponseTemplateField, message_start: list[str] | None
) -> dict[str, Any] | None:
    """Free text in the `answer` field, or `None` if it cannot be derived."""
    openers, closers = _openers(answer, message_start), _closers(answer)
    if not openers or closers is None:
        return None
    # Text cannot close the answer early, so no unconstrained call can follow
    # it. Like a call, the answer may end at a stop token instead of a closer.
    elements = [_const_strings(openers), {"type": "any_text", "excludes": closers}]
    if closers:
        elements.append(_optional(_const_strings(closers)))
    return _sequence(*elements)


def supports_tool_call_structural_tag(
    field: ResponseTemplateField, tools: list[dict[str, Any]] | None = None
) -> bool:
    """Whether these tools can use the template format for forced calls."""
    probe = [{"function": {"name": "f", "parameters": {"properties": {}}}}]
    try:
        tag = tool_call_structural_tag(field, tools or probe, parallel_tool_calls=True)
    except ValueError:
        # Derivable, only a name does not fit the template.
        return True
    return tag is not None


def tool_call_structural_tag(
    field: ResponseTemplateField,
    tools: list[dict[str, Any]],
    *,
    parallel_tool_calls: bool,
    thinking: ResponseTemplateField | None = None,
    answer: ResponseTemplateField | None = None,
    message_start: list[str] | None = None,
) -> dict[str, Any] | None:
    """Derive native calls, respecting strict arguments, count and optional reasoning.

    Raises `ValueError` for names the template cannot encode."""
    if not tools or not _names_function_by_opener(field):
        return None
    opener = _split_group(field.open_re.pattern, "name")
    closer_literals = _closers(field)
    if opener is None or closer_literals is None:
        return None
    arguments = []
    for tool in tools:
        function = tool["function"]
        # Like the generic JSON schema, a tool without parameters takes none.
        parameters = typed_parameters(
            function.get("parameters") or {"type": "object", "properties": {}}
        )
        strict = bool(function.get("strict"))
        arguments.append(_arguments_format(field, parameters, strict=strict))
        if arguments[-1] is None:
            return None
    names = []
    for tool in tools:
        name = tool["function"]["name"]
        if not re.fullmatch(opener[1], name):
            raise ValueError(f"tool name {name!r} cannot be encoded in tool calls")
        names.append(_escape(name))
    closers = _const_strings(closer_literals) if closer_literals else None

    def call(start: list[str] | None) -> dict[str, Any]:
        return _one_of(
            [
                _sequence(
                    {"type": "regex", "pattern": _call_opener(opener, name, start)},
                    tool_arguments,
                )
                for name, tool_arguments in zip(names, arguments)
            ]
        )

    def output(start: list[str] | None) -> list[dict[str, Any]] | None:
        elements = [call(start)]
        if parallel_tool_calls and field.repeats:
            repeated = _sequence(closers, call(None)) if closers else call(None)
            elements.append({"type": "star", "content": repeated})
        if closers:
            # EOS can terminate a call without writing its closer.
            elements.append(_optional(closers))
        if answer is not None:
            reply = _reply_format(answer, start)
            if reply is None:
                return None
            elements = [_one_of([_sequence(*elements), reply])]
        return elements

    return _structural_tag(output, thinking, message_start)


def json_answer_structural_tag(
    answer: ResponseTemplateField,
    json_schema: Any,
    tool_field: ResponseTemplateField | None = None,
    tool_names: list[str] | None = None,
    *,
    thinking: ResponseTemplateField | None = None,
    message_start: list[str] | None = None,
) -> dict[str, Any] | None:
    """Derive a JSON answer with optional calls to declared tools before it.

    Raises `ValueError` for names the template cannot encode."""
    closers = _closers(answer)
    if closers is None:
        return None
    opener = call_closers = None
    if tool_field is not None and tool_names and _names_function_by_opener(tool_field):
        opener = _split_group(tool_field.open_re.pattern, "name")
        call_closers = _closers(tool_field)
        # The arguments are any text, so their end must not spell a closer.
        if call_closers and _spells_closer(tuple(call_closers), (_TEXT_PIECE,)):
            return None
    if opener is not None:
        for name in tool_names:
            if not re.fullmatch(opener[1], name):
                raise ValueError(f"tool name {name!r} cannot be encoded in tool calls")

    def json_reply(start: list[str] | None) -> dict[str, Any] | None:
        openers = _openers(answer, start)
        if not openers:
            return None
        # The parser serves all text after the answer opener as JSON.
        return _sequence(
            _const_strings(openers), {"type": "json_schema", "json_schema": json_schema}
        )

    def call(start: list[str] | None) -> dict[str, Any]:
        # The parser reads back only calls to the declared tools.
        names = "|".join(map(_escape, tool_names))
        pattern = _call_opener(opener, f"(?:{names})", start)
        # The arguments may hold the answer opener, but not the call closer.
        arguments = {"type": "any_text", "excludes": call_closers}
        return _sequence({"type": "regex", "pattern": pattern}, arguments)

    def output(start: list[str] | None) -> list[dict[str, Any]] | None:
        reply = json_reply(start)
        if reply is None:
            return None
        if opener is None or call_closers is None:
            return [reply]
        calls = [call(start)]
        if call_closers:
            closer = _const_strings(call_closers)
            if tool_field.repeats:
                calls.append({"type": "star", "content": _sequence(closer, call(None))})
            after = json_reply(None)
            ending = _sequence(closer, _optional(after)) if after else closer
            calls.append(_optional(ending))
        return [_one_of([reply, _sequence(*calls)])]

    if message_start is None:
        thinking = None
    return _structural_tag(output, thinking, message_start)


def _structural_tag(
    output: Callable[[list[str] | None], list[dict[str, Any]] | None],
    thinking: ResponseTemplateField | None,
    message_start: list[str] | None,
) -> dict[str, Any] | None:
    """Frame output at the message start or after optional reasoning."""
    rest = output(None)
    if rest is None:
        return None
    think = None
    if thinking is not None:
        think = _thinking_format(thinking, message_start)
        if think is None:
            return None
    if message_start is None:
        if think is not None:
            rest.insert(0, _optional(think))
        return {"type": "structural_tag", "format": _sequence(*rest)}
    start = output(message_start)
    if start is None:
        return None
    format = _sequence(*start)
    if think is not None:
        format = _one_of([_sequence(think, *rest), format])
    return {"type": "structural_tag", "format": format}
