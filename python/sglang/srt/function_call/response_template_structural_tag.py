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

from typing import Any

import regex as re

from sglang.srt.parser.chat_parsing.response_parser import _schema_types
from sglang.srt.parser.chat_parsing.response_templates import ResponseTemplateField

_REGEX_META = re.compile(r"[.^$*+?{}\[\]\\|()]")
_LITERAL_ESCAPES = {"n": "\n", "r": "\r", "t": "\t"}
_WHITESPACE = {"type": "regex", "pattern": r"\s*"}
_ANY_TEXT = {"type": "any_text"}


def _split_group(pattern: str, name: str) -> tuple[str, str, str] | None:
    """Split `pattern` into the text before the named group `name`, the group
    body, and the text after it."""
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


def _literal(pattern: str) -> str | None:
    """The only string `pattern` matches, or `None` if it is not a literal."""
    chars = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\" and index + 1 < len(pattern):
            escaped = pattern[index + 1]
            if escaped in _LITERAL_ESCAPES:
                chars.append(_LITERAL_ESCAPES[escaped])
            elif escaped.isalnum():
                return None
            else:
                chars.append(escaped)
            index += 2
        elif _REGEX_META.fullmatch(char):
            return None
        else:
            chars.append(char)
            index += 1
    return "".join(chars)


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


def _value_format(
    schema: Any, parameters: dict[str, Any], json_values: bool
) -> dict[str, Any]:
    """The format of one argument value, as the parser reads it back: JSON when
    it decodes the value, otherwise raw text."""
    types = _schema_types(schema)
    if json_values or (types and "string" not in types):
        return {
            "type": "json_schema",
            "json_schema": _json_value_schema(schema, parameters),
        }
    if types in ((), ("string",)) and (choices := _string_choices(schema)):
        return _one_of([{"type": "const_string", "value": value} for value in choices])
    return _ANY_TEXT


def _xml_arguments_format(
    content_args: dict[str, Any], parameters: dict[str, Any]
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
    begin, key_pattern, _ = key_split
    begin, middle, end = map(_literal, (begin, value_split[0], value_split[2]))
    if begin is None or middle is None or end is None:
        return None

    name = {"type": "regex", "pattern": key_pattern}
    any_parameters = {
        "type": "star",
        "content": _sequence(
            _WHITESPACE,
            _tag(
                begin,
                _sequence(name, {"type": "const_string", "value": middle}, _ANY_TEXT),
                end,
            ),
        ),
    }
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return any_parameters

    def parameter(key: str, value: dict[str, Any]) -> dict[str, Any]:
        if not re.fullmatch(key_pattern, key):
            raise ValueError(f"parameter name {key!r} cannot be encoded in tool calls")
        return _sequence(_WHITESPACE, _tag(begin + key + middle, value, end))

    required = parameters.get("required") or []
    json_values = value_parser.get("name") == "json"
    elements = []
    for key, schema in properties.items():
        element = parameter(key, _value_format(schema, parameters, json_values))
        elements.append(element if key in required else _optional(element))
    elements.extend(
        parameter(key, _ANY_TEXT) for key in required if key not in properties
    )
    # Extra keys could repeat a declared one and bypass its schema, so they
    # need an explicit opt-in rather than the JSON Schema default.
    if parameters.get("additionalProperties", False) is not False:
        elements.append(any_parameters)
    return _sequence(*elements) if elements else {"type": "const_string", "value": ""}


def _arguments_format(
    field: ResponseTemplateField, parameters: dict[str, Any]
) -> dict[str, Any] | None:
    if field.content == "json" and not field.content_args:
        return {"type": "json_schema", "json_schema": parameters or {"type": "object"}}
    if field.content == "xml-inline":
        return _xml_arguments_format(field.content_args, parameters)
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


def _reply_format(answer: ResponseTemplateField) -> dict[str, Any] | None:
    """Free text in the `answer` field, or `None` if it cannot be derived."""
    if not answer.open_literals or (
        answer.close_re is not None and not answer.close_literals
    ):
        return None
    close_literals = answer.close_literals or []
    openers = [{"type": "const_string", "value": v} for v in answer.open_literals]
    closers = [{"type": "const_string", "value": v} for v in close_literals]
    # Text cannot close the answer early, so no unconstrained call can follow
    # it. Like a call, the answer may end at a stop token instead of a closer.
    elements = [_one_of(openers), {"type": "any_text", "excludes": close_literals}]
    if closers:
        elements.append(_optional(_one_of(closers)))
    return _sequence(*elements)


def supports_tool_call_structural_tag(field: ResponseTemplateField) -> bool:
    """Whether `tool_call_structural_tag` derives tags for `field`."""
    probe = {"function": {"name": "f", "parameters": {"properties": {}}}}
    try:
        tag = tool_call_structural_tag(field, [probe], parallel_tool_calls=True)
    except ValueError:
        # Derivable, only the probe's name does not fit the template.
        return True
    return tag is not None


def tool_call_structural_tag(
    field: ResponseTemplateField,
    tools: list[dict[str, Any]],
    *,
    parallel_tool_calls: bool,
    thinking: ResponseTemplateField | None = None,
    answer: ResponseTemplateField | None = None,
) -> dict[str, Any] | None:
    """An xgrammar structural tag for calls to `tools` in the format of the tool
    call `field`, or `None` if it cannot be derived.

    `tools` are OpenAI function tool dicts. The tag forces at least one call,
    unless an `answer` field lets the model reply instead. With a `thinking`
    field, it also accepts an optional thinking block first. Raises
    `ValueError` for a tool or parameter name the format cannot encode.
    """
    if not tools or not _names_function_by_opener(field):
        return None
    opener = _split_group(field.open_re.pattern, "name")
    if opener is None or (field.close_re is not None and not field.close_literals):
        return None
    calls = []
    for tool in tools:
        function = tool["function"]
        arguments = _arguments_format(field, function.get("parameters") or {})
        if arguments is None:
            return None
        if not re.fullmatch(opener[1], function["name"]):
            raise ValueError(
                f"tool name {function['name']!r} cannot be encoded in tool calls"
            )
        name = _REGEX_META.sub(lambda match: "\\" + match.group(), function["name"])
        pattern = opener[0] + name + opener[2]
        calls.append(_sequence({"type": "regex", "pattern": pattern}, arguments))
    call = _one_of(calls)
    closers = [
        {"type": "const_string", "value": literal}
        for literal in field.close_literals or ()
    ]

    elements = [call]
    if parallel_tool_calls and field.repeats:
        repeated = _sequence(_one_of(closers), call) if closers else call
        elements.append({"type": "star", "content": repeated})
    if closers:
        # The final closer may also be a stop token, which the grammar cannot
        # match as text, so generation may end right after the arguments.
        elements.append(_optional(_one_of(closers)))
    if answer is not None:
        reply = _reply_format(answer)
        if reply is None:
            return None
        elements = [_one_of([_sequence(*elements), reply])]
    if thinking is not None:
        if not thinking.open_literals or not thinking.close_literals:
            return None
        tags = [
            _tag(literal, _ANY_TEXT, thinking.close_literals)
            for literal in thinking.open_literals
        ]
        elements.insert(0, _optional(_one_of(tags)))
    return {"type": "structural_tag", "format": _sequence(*elements)}
