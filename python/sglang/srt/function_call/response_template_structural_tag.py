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

import msgspec
import regex as re

try:
    from re import _parser as _sre_parse
except ImportError:  # Python 3.10
    import sre_parse as _sre_parse

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
# Precedes the text of a final lookahead while expanding a delimiter pattern.
_AHEAD = "\1"
# Arbitrary bound on the strings expanded from one delimiter pattern.
_MAX_LITERALS = 16
# Templates write at most a newline between parameters.
_SEPARATOR = {"type": "regex", "pattern": r"\n?"}
_ANY_TEXT = {"type": "any_text"}
# Character classes in XGrammar regexes take whitespace only as escapes.
_BLANKS = {" ": " ", "\t": r"\t", "\n": r"\n", "\r": r"\r"}
# Keywords that describe a schema without constraining its values.
_ANNOTATIONS = frozenset(
    {
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "$comment",
        "$schema",
        "$id",
    }
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
_REPEATS = tuple(
    getattr(_sre_parse, name)
    for name in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT")
    if hasattr(_sre_parse, name)
)
_CATEGORIES = {
    _sre_parse.CATEGORY_DIGIT: r"\d",
    _sre_parse.CATEGORY_NOT_DIGIT: r"\D",
    _sre_parse.CATEGORY_SPACE: r"\s",
    _sre_parse.CATEGORY_NOT_SPACE: r"\S",
    _sre_parse.CATEGORY_WORD: r"\w",
    _sre_parse.CATEGORY_NOT_WORD: r"\W",
}
_HOST_LABEL = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
_DATE = r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
_TIME = (
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?"
    r"(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)
_OCTET = r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]?[0-9])"
_ATOM = r"[A-Za-z0-9_!#$%&'*+/=?^`{|}~-]+"
_URI_CHARS = r"A-Za-z0-9_.~!$&'()*+,;="
_ESCAPED = r"%[0-9A-Fa-f]{2}"
_PATH = rf"(?:[{_URI_CHARS}:@-]|{_ESCAPED})"
_QUERY = rf"(?:[{_URI_CHARS}:@/?-]|{_ESCAPED})*"
# Raw string forms of the formats XGrammar holds in JSON strings. Emails leave
# out quoted local parts, whose quotes XGrammar writes as JSON escapes.
_STRING_FORMATS = {
    "date": _DATE,
    "time": _TIME,
    "date-time": f"{_DATE}T{_TIME}",
    "email": (
        rf"{_ATOM}(?:\.{_ATOM})*@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
        r"(?:\.[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9])*"
    ),
    "hostname": rf"{_HOST_LABEL}(?:\.{_HOST_LABEL})*",
    "ipv4": rf"(?:{_OCTET}\.){{3}}{_OCTET}",
    "uuid": r"[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}",
    "uri": (
        rf"[A-Za-z][A-Za-z+.-]*:(?://(?:(?:[{_URI_CHARS}:-]|{_ESCAPED})*@)?"
        rf"(?:[{_URI_CHARS}-]|{_ESCAPED})*(?::[0-9]*)?(?:/{_PATH}*)*"
        rf"|/?(?:{_PATH}+(?:/{_PATH}*)*)?)(?:\?{_QUERY})?(?:#{_QUERY})?"
    ),
}


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


def _literals(
    pattern: str, *, message_start: bool = False, lookahead: bool = False
) -> list[str] | None:
    """Expand literal alternatives, allowing `^` only at the message start and
    a final lookahead only with `lookahead`."""
    literals, index = _alternatives(pattern, 0)
    if literals is None or index != len(pattern):
        return None
    if any(literal.count(_AHEAD) > lookahead for literal in literals):
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
        if pattern.startswith(("(?:", "(?="), index):
            ahead = pattern[index + 2] == "="
            atom, index = _alternatives(pattern, index + 3)
            if atom is None or not pattern.startswith(")", index):
                return None, index
            index += 1
            if ahead:
                atom = [_AHEAD + literal for literal in atom]
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
        # A lookahead reads as a literal only at the end of the pattern.
        if any(_AHEAD in prefix for prefix in literals):
            return None, index
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
    thinking: ResponseTemplateField,
    message_start: list[str] | None,
    nonempty: bool = False,
) -> dict[str, Any] | None:
    """The reasoning, which holds at least a character if `nonempty`."""
    openers, closers = _openers(thinking, message_start), _closers(thinking)
    if not openers or not closers:
        return None
    text = _sequence(_nonempty_start(closers), _ANY_TEXT) if nonempty else _ANY_TEXT
    return _one_of([_tag(opener, text, closers) for opener in openers])


def _nonempty_start(closers: list[str]) -> dict[str, Any]:
    """A first character that starts none of the `closers`."""
    starts = {closer[0] for closer in closers if closer}
    excluded = "".join(map(_class_escape, sorted(starts)))
    return {"type": "regex", "pattern": f"[^{excluded}]"}


def _class_escape(char: str) -> str:
    return _BLANKS.get(char, "\\" + char if char in "\\]^-[" else char)


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


def _text_format(schema: Any, excludes: list[str]) -> dict[str, Any] | None:
    """Constrain raw text, which the `excludes` end, without changing its
    parsed value."""
    if schema is True:
        return _ANY_TEXT
    if isinstance(schema, dict) and {"format", "pattern"} & set(schema):
        return _string_regex(schema, excludes)
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
    if any(_text_format(branch, excludes) is not _ANY_TEXT for branch in branches):
        return None
    return _ANY_TEXT


def _string_regex(schema: dict[str, Any], excludes: list[str]) -> dict[str, Any] | None:
    """Hold the `format` or `pattern` of a raw string. Like XGrammar, a format
    overrides a pattern, either one drops length bounds, and the pattern
    matches the whole string."""
    if set(schema) - _ANNOTATIONS - {
        "type",
        "format",
        "pattern",
        "minLength",
        "maxLength",
    } or _schema_types(schema) != ("string",):
        return None
    pattern = (
        _STRING_FORMATS.get(schema["format"])
        if "format" in schema
        else schema["pattern"]
    )
    if (
        not isinstance(pattern, str)
        or _may_write(pattern, {text[0] for text in excludes if text})
        or not _regex_compiles(pattern)
    ):
        return None
    return {"type": "regex", "pattern": pattern}


def _may_write(pattern: str, chars: set[str]) -> bool:
    """Whether a match of `pattern` may hold one of `chars`, as any pattern may
    that Python cannot parse."""
    try:
        parsed = _sre_parse.parse(pattern)
    except (_sre_parse.error, OverflowError, RecursionError):
        return True
    if parsed.state.flags & _sre_parse.SRE_FLAG_IGNORECASE:
        return True

    def may(items: Any) -> bool:
        for op, arg in items:
            if op is _sre_parse.LITERAL:
                written = {chr(arg)}
            elif op is _sre_parse.IN:
                written = {char for char in chars if _in_class(arg, char)}
            elif op in _REPEATS:
                written = chars if arg[1] and may(arg[2]) else set()
            elif op is _sre_parse.SUBPATTERN:
                if arg[1] & _sre_parse.SRE_FLAG_IGNORECASE:
                    return True
                written = chars if may(arg[3]) else set()
            elif op is _sre_parse.BRANCH:
                written = chars if any(map(may, arg[1])) else set()
            elif op is _sre_parse.AT:
                continue
            else:
                # Such as `.`, which XGrammar lets match line breaks.
                return True
            if written & chars:
                return True
        return False

    return may(parsed)


def _in_class(items: list[tuple[Any, Any]], char: str) -> bool:
    negated = bool(items) and items[0][0] is _sre_parse.NEGATE
    matched = False
    for op, arg in items[negated:]:
        if op is _sre_parse.LITERAL:
            matched |= ord(char) == arg
        elif op is _sre_parse.RANGE:
            matched |= arg[0] <= ord(char) <= arg[1]
        elif op is _sre_parse.CATEGORY and arg in _CATEGORIES:
            matched |= bool(re.fullmatch(_CATEGORIES[arg], char))
        else:
            return True
    return matched != negated


# Every request derives the tags for its tools anew.
@lru_cache(maxsize=512)
def _regex_compiles(pattern: str) -> bool:
    from xgrammar import Grammar

    try:
        Grammar.from_regex(pattern)
    except RuntimeError:
        return False
    return True


def _value_format(
    schema: Any, parameters: dict[str, Any], json_values: bool, closers: list[str]
) -> dict[str, Any] | None:
    """Use JSON for decoded values and raw text for string values."""
    if schema is False:
        return None
    types = _schema_types(schema)
    if json_values or (types and "string" not in types):
        return _json_format(_json_value_schema(schema, parameters), closers)
    return _text_format(schema, closers)


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


class _XmlFraming(msgspec.Struct, frozen=True):
    """The literal pieces of an `xml-inline` parameter tag, and the raw text a
    value may hold between them."""

    content_args: dict[str, Any]
    json_values: bool
    key_pattern: str
    begins: list[str]
    middles: list[str]
    ends: list[str]
    # Ends that are the next parameter's opener, looked ahead for.
    follows: list[str]
    closers: list[str]
    text: dict[str, Any]

    @property
    def starts(self) -> list[str]:
        # Each value but the last then ends by opening the next parameter,
        # whose tag starts after the opener.
        return [""] if self.follows else self.begins

    def tag(
        self, begin: str, value: dict[str, Any], fits: list[str], last: bool
    ) -> dict[str, Any]:
        if self.follows and not last:
            fits = [
                e + sep + self.begins[0] for e in fits for sep in ("", "\n")
            ] + self.follows
        return _tag(begin, value, fits)

    def separated(self, tags: list[dict[str, Any]]) -> dict[str, Any]:
        return _one_of(tags) if self.follows else _sequence(_SEPARATOR, _one_of(tags))

    def arguments(self, body: dict[str, Any], optional: bool) -> dict[str, Any]:
        """`body` after the opener of its first parameter."""
        if not self.follows:
            return body
        body = _sequence(_SEPARATOR, _const_strings(self.begins), body)
        return _optional(body) if optional else body

    def any_parameter(self, last: bool) -> dict[str, Any]:
        name = {"type": "regex", "pattern": self.key_pattern}
        content = _sequence(name, _const_strings(self.middles), self.text)
        return self.separated(
            [self.tag(start, content, self.ends, last) for start in self.starts]
        )

    def any_parameters(self) -> dict[str, Any]:
        more = {"type": "star", "content": self.any_parameter(False)}
        if not self.follows:
            return more
        return self.arguments(_sequence(more, self.any_parameter(True)), True)

    def read_back(self, written: str) -> dict[str, Any] | None:
        """The arguments the parser reads from `written`, unless a closer ends
        the call in it."""
        if any(closer in written for closer in self.closers):
            return None
        try:
            return parse_content(written, "xml-inline", self.content_args)
        except ValueError:
            return None

    def parameter(
        self, key: str, value: dict[str, Any], choices: list[str] | None, last: bool
    ) -> dict[str, Any] | None:
        begins, middles, ends = self.begins, self.middles, self.ends
        if not re.fullmatch(self.key_pattern, key) or any(
            list(self.read_back(b + key + m + "0" + e) or ()) != [key]
            for b in begins
            for m in middles
            for e in ends
        ):
            raise ValueError(f"parameter name {key!r} cannot be encoded in tool calls")
        if choices is None:
            return self.separated(
                [
                    self.tag(s + key + m, value, ends, last)
                    for s in self.starts
                    for m in middles
                ]
            )
        # Optional framing whitespace must not change a text choice.
        tags, framed = [], set()
        for b, s in zip(begins, self.starts):
            for m in middles:
                by_ends: dict[tuple[str, ...], list[str]] = {}
                for choice in choices:
                    fits = tuple(
                        e
                        for e in ends
                        if self.read_back(b + key + m + choice + e) == {key: choice}
                    )
                    by_ends.setdefault(fits, []).append(choice)
                for fits, group in by_ends.items():
                    if fits:
                        tags.append(
                            self.tag(
                                s + key + m, _const_strings(group), list(fits), last
                            )
                        )
                        framed.update(group)
        if framed != set(choices):
            return None
        return self.separated(tags)


def _xml_framing(
    content_args: dict[str, Any], closers: list[str]
) -> _XmlFraming | None:
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
    begins, middles = _literals(key_split[0]), _literals(value_split[0])
    ends = _literals(value_split[2], lookahead=True)
    if not begins or not middles or not ends:
        return None
    aheads = [end.split(_AHEAD) for end in ends if _AHEAD in end]
    ends = [end for end in ends if _AHEAD not in end]
    # A value may also end where the next parameter opens, if the template
    # looks ahead for its only opener.
    if not ends or (
        aheads and (len(begins) > 1 or {a for _, a in aheads} != {*begins})
    ):
        return None
    follows = [before + ahead for before, ahead in aheads]
    # Raw text has no escaping for a call closer, nor, where a value may end
    # before the next opener, for any value end.
    excludes = closers + follows + ends if follows else closers
    return _XmlFraming(
        content_args=content_args,
        json_values=value_parser.get("name") == "json",
        key_pattern=key_split[1],
        begins=begins,
        middles=middles,
        ends=ends,
        follows=follows,
        closers=closers,
        text={"type": "any_text", "excludes": excludes} if excludes else _ANY_TEXT,
    )


def _xml_arguments_format(
    content_args: dict[str, Any],
    parameters: dict[str, Any],
    *,
    strict: bool,
    closers: list[str],
    sorted_arguments: bool,
) -> dict[str, Any] | None:
    framing = _xml_framing(content_args, closers)
    if framing is None:
        return None
    properties = parameters.get("properties")
    typed = properties is None or (
        isinstance(properties, dict)
        and (framing.json_values or all(map(_typed_by_parser, properties.values())))
    )
    # Other schema shapes cannot be typed by the XML parser.
    if strict and (set(parameters) - _ANNOTATIONS - _OBJECT_KEYWORDS or not typed):
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
        _literal_piece(framing.begins, before=_SEPARATOR["pattern"]),
        (f"(?:{framing.key_pattern})", _endings(names), False),
        _literal_piece(framing.middles),
        # Values of every format are some text.
        _TEXT_PIECE,
        _literal_piece(framing.ends),
    )
    if _spells_closer(tuple(closers), pieces, repeats=True):
        return None
    if not strict:
        return framing.any_parameters()
    if properties is None:
        extra = parameters.get("additionalProperties", True)
        if extra is False:
            return {"type": "const_string", "value": ""}
        return framing.any_parameters() if extra is True or extra == {} else None
    # A declared key repeated as an extra one would bypass its schema.
    if parameters.get("additionalProperties", False) is not False:
        return None
    return _strict_xml_arguments(
        framing, parameters, required, sorted_arguments=sorted_arguments
    )


def _strict_xml_arguments(
    framing: _XmlFraming,
    parameters: dict[str, Any],
    required: list[str],
    *,
    sorted_arguments: bool,
) -> dict[str, Any] | None:
    """Declared parameters, each held to its schema, required ones present."""
    properties = parameters["properties"]
    excludes = framing.closers + framing.follows + framing.ends
    values = {}
    for key, schema in properties.items():
        value = _value_format(schema, parameters, framing.json_values, excludes)
        if value is None:
            return None
        choices = None
        if value["type"] in ("const_string", "or"):
            choices = _string_choices(schema)
        values[key] = (framing.text if value == _ANY_TEXT else value, choices)
    values.update(
        (key, (framing.text, None)) for key in required if key not in properties
    )
    if not values:
        return {"type": "const_string", "value": ""}
    more, final = {}, {}
    for key, (value, choices) in values.items():
        more[key] = framing.parameter(key, value, choices, last=False)
        final[key] = framing.parameter(key, value, choices, last=True)
        if more[key] is None or final[key] is None:
            return None

    def ordered(keys: tuple[str, ...]) -> dict[str, Any]:
        before = [more[k] if k in required else _optional(more[k]) for k in keys]
        if not framing.follows:
            return _sequence(*before)

        def last(start: int, stop: int) -> dict[str, Any]:
            """Parameters of `keys[start:stop]` that end at one of them."""
            if stop - start == 1:
                return final[keys[start]]
            # Halving keeps the tag O(n log n) in size and O(log n) deep.
            middle = (start + stop) // 2
            return _one_of(
                [
                    last(start, middle),
                    _sequence(*before[start:middle], last(middle, stop)),
                ]
            )

        # The last parameter follows every required one.
        first = max((i for i, k in enumerate(keys) if k in required), default=0)
        return _sequence(*before[:first], last(first, len(keys)))

    orders = [tuple(values)]
    if sorted_arguments and sorted(values) != list(values):
        orders.append(tuple(sorted(values)))
    return framing.arguments(_one_of([ordered(keys) for keys in orders]), not required)


def _arguments_format(
    field: ResponseTemplateField,
    parameters: dict[str, Any],
    *,
    strict: bool,
    sorted_arguments: bool,
) -> dict[str, Any] | None:
    closers = _closers(field) or []
    if field.content == "json" and not field.content_args:
        if _spells_closer(tuple(closers), (_JSON_PIECE,)):
            return None
        schema = parameters if strict else {"type": "object"}
        return _json_format(schema, closers)
    if field.content == "xml-inline":
        return _xml_arguments_format(
            field.content_args,
            parameters,
            strict=strict,
            closers=closers,
            sorted_arguments=sorted_arguments,
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
    answer: ResponseTemplateField,
    message_start: list[str] | None,
    content: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """The `answer` field holding `content`, or free text if not given, or
    `None` if it cannot be derived."""
    openers, closers = _openers(answer, message_start), _closers(answer)
    if not openers or closers is None:
        return None
    if content is not None:
        # The parser serves all text after the answer opener as the content.
        return _sequence(_const_strings(openers), content)
    return _sequence(_const_strings(openers), *_answer_text(closers))


def _answer_text(closers: list[str]) -> list[dict[str, Any]]:
    # Text cannot close the answer early, so no unconstrained call can follow
    # it. Like a call, the answer may end at a stop token instead of a closer.
    elements = [{"type": "any_text", "excludes": closers}]
    if closers:
        elements.append(_optional(_const_strings(closers)))
    return elements


def answer_rest_structural_tag(answer: ResponseTemplateField) -> dict[str, Any] | None:
    """The rest of an `answer` that is already open, or `None` if underivable."""
    closers = _closers(answer)
    if closers is None:
        return None
    return {"type": "structural_tag", "format": _sequence(*_answer_text(closers))}


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


def holds_arguments(field: ResponseTemplateField, parameters: dict[str, Any]) -> bool:
    """Whether `strict_arguments` holds `xml-inline` arguments to typed
    `parameters`, as only that parser reads values as JSON."""
    if field.content != "xml-inline":
        return False
    return _holds_xml_arguments(
        json.dumps(field.content_args, sort_keys=True),
        tuple(_closers(field) or ()),
        json.dumps(parameters, sort_keys=True),
    )


# Every request parses the calls of its tools anew.
@lru_cache(maxsize=512)
def _holds_xml_arguments(
    content_args: str, closers: tuple[str, ...], parameters: str
) -> bool:
    try:
        format = _xml_arguments_format(
            json.loads(content_args),
            json.loads(parameters),
            strict=True,
            closers=list(closers),
            sorted_arguments=False,
        )
    except ValueError:
        return False
    return format is not None


def tool_call_structural_tag(
    field: ResponseTemplateField | None,
    tools: list[dict[str, Any]],
    *,
    parallel_tool_calls: bool,
    thinking: ResponseTemplateField | None = None,
    answer: ResponseTemplateField | None = None,
    answer_format: dict[str, Any] | None = None,
    message_start: list[str] | None = None,
    grammar_requires_reasoning: bool = False,
    sorted_arguments: bool = False,
    strict_arguments: bool = False,
) -> dict[str, Any] | None:
    """Derive native calls, respecting strict arguments, count and reasoning.

    An `answer` field lets the model answer instead, in `answer_format` if
    given; without tools, only the answer is derived. `strict_arguments`
    holds the arguments of every tool as strict where they can be. Strict
    `xml-inline` arguments come in declared order, or also alphabetical with
    `sorted_arguments`. Raises `ValueError` for names the template cannot
    encode."""
    if not tools and answer is not None:

        def reply(start: list[str] | None) -> list[dict[str, Any]] | None:
            format = _reply_format(answer, start, answer_format)
            return None if format is None else [format]

        return _structural_tag(
            reply, thinking, message_start, grammar_requires_reasoning
        )
    if not tools or field is None or not _names_function_by_opener(field):
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
        tool_arguments = None
        if strict or strict_arguments:
            try:
                tool_arguments = _arguments_format(
                    field, parameters, strict=True, sorted_arguments=sorted_arguments
                )
            except ValueError:
                if strict:
                    raise
        if tool_arguments is None and not strict:
            tool_arguments = _arguments_format(
                field, parameters, strict=False, sorted_arguments=sorted_arguments
            )
        if tool_arguments is None:
            return None
        arguments.append(tool_arguments)
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
            reply = _reply_format(answer, start, answer_format)
            if reply is None:
                return None
            elements = [_one_of([_sequence(*elements), reply])]
        return elements

    return _structural_tag(output, thinking, message_start, grammar_requires_reasoning)


def _structural_tag(
    output: Callable[[list[str] | None], list[dict[str, Any]] | None],
    thinking: ResponseTemplateField | None,
    message_start: list[str] | None,
    grammar_requires_reasoning: bool = False,
) -> dict[str, Any] | None:
    """Frame output at the message start or after reasoning, which is optional
    unless `grammar_requires_reasoning` makes the message open with non-empty
    reasoning."""
    rest = output(None)
    if rest is None:
        return None
    think = None
    if thinking is not None:
        think = _thinking_format(thinking, message_start, grammar_requires_reasoning)
        if think is None:
            return None
    if message_start is None:
        if think is not None:
            rest.insert(0, _optional(think))
        return {"type": "structural_tag", "format": _sequence(*rest)}
    if think is not None and grammar_requires_reasoning:
        return {"type": "structural_tag", "format": _sequence(think, *rest)}
    start = output(message_start)
    if start is None:
        return None
    format = _sequence(*start)
    if think is not None:
        format = _one_of([_sequence(think, *rest), format])
    return {"type": "structural_tag", "format": format}
