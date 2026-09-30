# Copyright 2026 The HuggingFace Team. All rights reserved.
# ruff: noqa
# fmt: off
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
"""This file contains the parsers used by chat response parsing. Each parser takes a chunk of captured text
and parses it into a single key in the output message dictionary. Functions are generally boilerplate."""

from __future__ import annotations

import json
from typing import Any

import regex as re


def _text(text: str, args: dict) -> str:
    return text.strip() if args.get("strip", True) else text


def _int(text: str, args: dict) -> int:
    return int(_text(text, args))


def _float(text: str, args: dict) -> float:
    return float(_text(text, args))


def _bool(text: str, args: dict) -> bool:
    return _text(text, args).lower() in ("true", "1")


# Sentinel characters for lax-JSON string pre-extraction — ASCII control chars
# that should never appear in real LLM output.
_LAX_OPEN, _LAX_CLOSE = "\x01", "\x02"


def _json(text: str, args: dict) -> Any:
    """JSON parser with optional dialect knobs for LLM-emitted quirks.

    `args`:
      - `unquoted_keys` (bool): quote bare-identifier keys before parsing.
      - `string_delims` ([[open, close], ...]): strings delimited by these
        custom markers are pre-extracted, then restored as standard JSON strings.
      - `allow_non_json` (bool): return stripped text if parsing fails.
    """
    string_delims = args.get("string_delims", [])
    unquoted_keys = args.get("unquoted_keys", False)

    if string_delims and (_LAX_OPEN in text or _LAX_CLOSE in text):
        raise ValueError("json: input contains reserved sentinel characters (\\x01/\\x02); cannot parse safely.")

    working = text
    captured: list[str] = []
    for open_d, close_d in string_delims:
        pattern = re.escape(open_d) + r"(.*?)" + re.escape(close_d)

        def _capture(m: Any) -> str:
            captured.append(m.group(1))
            return f"{_LAX_OPEN}{len(captured) - 1}{_LAX_CLOSE}"

        working = re.sub(pattern, _capture, working, flags=re.DOTALL)

    if unquoted_keys:
        working = re.sub(r"(?<=[{,])(\w+):", r'"\1":', working)

    for i, s in enumerate(captured):
        working = working.replace(f"{_LAX_OPEN}{i}{_LAX_CLOSE}", json.dumps(s))

    try:
        return json.loads(working)
    except (json.JSONDecodeError, RecursionError) as e:
        if args.get("allow_non_json"):
            return _text(text, args)
        if working == text:
            raise ValueError(f"json parser could not parse region as JSON.\nContent: {text!r}\nError: {e}") from e
        raise ValueError(
            f"json: could not parse after dialect transforms.\n"
            f"Original: {text!r}\nTransformed: {working!r}\nError: {e}"
        ) from e


def _sub_parse(raw: str, value_parser: dict | None) -> Any:
    if value_parser is None:
        return raw
    return parse_content(raw, value_parser.get("name", "text"), value_parser.get("args", {}))


# A whole JSON string, and the rest of one after its opening quote.
_JSON_STRING = re.compile(r'"(?:[^"\\]++|\\.)*+"', re.DOTALL)
_JSON_STRING_REST = re.compile(r'(?:[^"\\]++|\\.)*+', re.DOTALL)


_JSON_MARKER = re.compile(r'["{}\[\]]')
_JSON_LITERAL = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|true|false|null')


def _json_advance(text: str, scan: dict, end: int) -> bool:
    """Advance the current JSON value, holding an escape cut off at `end`."""
    pos = scan["pos"]
    if scan.get("json_done"):
        return True
    if "json_start" not in scan:
        while pos < end and text[pos] in " \t\n\r":
            pos += 1
        if pos == end:
            scan["pos"] = pos
            return False
        scan.update(json_start=pos, depth=0)
    while pos < end:
        if scan["string"]:
            pos = _JSON_STRING_REST.match(text, pos, end).end()
            if pos == end or text[pos] != '"':
                break
            scan["string"] = False
            if scan["depth"] == 0:
                scan["json_done"] = True
        else:
            marker = _JSON_MARKER.search(text, pos, end)
            if marker is None:
                pos = end
                break
            pos = marker.start()
            if text[pos] == '"':
                scan["string"] = True
            elif text[pos] in "{[":
                scan["depth"] += 1
            else:
                scan["depth"] -= 1
                if scan["depth"] == 0:
                    scan["json_done"] = True
        pos += 1
        if scan.get("json_done"):
            break
    scan["pos"] = pos
    return scan.get("json_done", False) or (
        not scan["string"] and scan["depth"] == 0
        and _JSON_LITERAL.fullmatch(text[scan["json_start"]:end].strip()) is not None
    )


def _json_tag(tag: re.Pattern, text: str, head: int, value: int, end: int):
    """Match the tag at `head` whose JSON value, from `value`, ends no earlier than that JSON."""
    m = tag.match(text, head, end)
    scan = {"pos": value, "string": False}
    if m is None or _json_advance(text, scan, m.end("value")):
        return m
    # Only a string or container can hold a tag end. Mask no further than its end: a quote in the tag end
    # could pair with a later one.
    if not _json_advance(text, scan, end) or text[scan["json_start"]] not in '"[{':
        return m
    masked = _JSON_STRING.sub(lambda s: '"' + "\0" * (len(s.group()) - 2) + '"', text[value : scan["pos"]])
    return tag.match(text[:value] + masked + text[scan["pos"] : end], head)


def _xml_tags(text: str, tag_pattern: str, json_keys: frozenset = frozenset()):
    """Find parameter tags while shielding JSON strings for `json_keys`."""
    tag = re.compile(tag_pattern, re.DOTALL)
    pos = 0
    while (m := tag.search(text, pos)) is not None:
        if m.groupdict().get("key") in json_keys:
            m = _json_tag(tag, text, m.start(), m.start("value"), len(text)) or m
            yield m.group("key"), text[m.start("value") : m.end("value")]
        else:
            yield m.groupdict().get("key"), m.groupdict().get("value", "")
        pos = m.end()


def _xml_inline(text: str, args: dict, json_keys: frozenset = frozenset()) -> dict:
    """Parse shallow XML-ish tags into a dict. `tag_pattern` regex must have named
    groups `key` and `value`. Optional `value_parser` recurses; `merge_duplicates`
    collects duplicate keys into a list."""
    tag_pattern = args.get("tag_pattern")
    if tag_pattern is None:
        raise ValueError("xml-inline: 'tag_pattern' content_arg is required")
    value_parser = args.get("value_parser")
    merge = args.get("merge_duplicates", False)

    out: dict[str, Any] = {}
    for key, raw in _xml_tags(text, tag_pattern, json_keys):
        if key is None:
            raise ValueError(f"xml-inline: tag_pattern must have a named group 'key'. Pattern: {tag_pattern}")
        value = _sub_parse(raw, value_parser)
        if key in out and merge:
            if not isinstance(out[key], list):
                out[key] = [out[key]]
            out[key].append(value)
        else:
            out[key] = value
    return out


def _kv_lines(text: str, args: dict) -> dict:
    """Parse line-delimited `key<sep>value` pairs into a dict."""
    line_sep = args.get("line_sep", "\n")
    kv_sep = args.get("kv_sep", ":")
    value_parser = args.get("value_parser")

    out: dict[str, Any] = {}
    for line in text.split(line_sep):
        line = _text(line, args)
        if not line or kv_sep not in line:
            continue
        k, v = line.split(kv_sep, 1)
        k, v = _text(k, args), _text(v, args)
        out[k] = _sub_parse(v, value_parser)
    return out


CONTENT_PARSERS = {
    "text": _text,
    "int": _int,
    "float": _float,
    "bool": _bool,
    "json": _json,
    "xml-inline": _xml_inline,
    "kv-lines": _kv_lines,
}

# Parsers whose output is the verbatim body text (modulo whitespace) — chunks
# from these fields stream with `dirty=False` because each chunk is part of
# the final value. Structured parsers (`json`, `xml-inline`, `kv-lines`) only
# produce a meaningful value on close, so their chunks stream raw bytes
# flagged `dirty=True` while the parsed value is delivered in `region_close`.
STREAMABLE_PARSERS = frozenset({"text", "int", "float", "bool"})


def parse_content(text: str, name: str, args: dict, json_keys: frozenset = frozenset()) -> Any:
    if json_keys:
        return _xml_inline(text, args, json_keys)
    return CONTENT_PARSERS[name](text, args)


_PLACEHOLDER = re.compile(r"\{(\w+(?:\.\w+)*)\}")


def _apply_transform(transform: Any, scope: dict) -> Any:
    """Recursively walk a transform template, which is used to restructure
    parsed output into the actual shape we want. A dotted placeholder like
    `{content.args}` descends into keys of the looked-up value."""
    if isinstance(transform, dict):
        return {k: _apply_transform(v, scope) for k, v in transform.items()}
    if isinstance(transform, list):
        return [_apply_transform(v, scope) for v in transform]
    if not isinstance(transform, str):
        return transform
    whole = _PLACEHOLDER.fullmatch(transform)
    if not whole:
        return transform
    path = whole.group(1)
    root, *keys = path.split(".")
    if root not in scope:
        raise KeyError(f"transform placeholder '{{{path}}}' is not defined. Available: {sorted(scope)}")
    value = scope[root]
    for key in keys:
        if not isinstance(value, dict):
            raise ValueError(f"transform placeholder '{{{path}}}' cannot index into {type(value).__name__} at '{key}'")
        if key not in value:
            raise ValueError(f"transform placeholder '{{{path}}}' is missing key '{key}'. Available: {sorted(value)}")
        value = value[key]
    return value


def validate_transform_strings(scope: str, transform: Any) -> None:
    """Walk a transform template and reject any string that mixes a `{name}`
    placeholder with literal text. Only whole-string placeholders (e.g.
    `"{content}"`) and plain literals are supported. Called from the template
    loader so authors get a clear error at load time, not at parse time."""
    if isinstance(transform, dict):
        for v in transform.values():
            validate_transform_strings(scope, v)
        return
    if isinstance(transform, list):
        for v in transform:
            validate_transform_strings(scope, v)
        return
    if not isinstance(transform, str):
        return
    if _PLACEHOLDER.search(transform) and not _PLACEHOLDER.fullmatch(transform):
        raise ValueError(
            f"{scope}: transform string {transform!r} mixes a {{placeholder}} with literal text. "
            'Use either a whole-string placeholder (e.g. "{content}") or a plain literal; '
            "string interpolation is not supported."
        )


def process_field(body: str, field, captures: dict, json_keys: frozenset = frozenset()) -> Any:
    """Run `body` through the field's content parser, then optionally apply the
    transform template. When `transform_each` is set, the parsed content must
    be a list and the template is applied to each element (with the element's
    keys unpacked into the template scope, alongside any regex captures).

    `field` is a `spec.Field`; typed via duck-typing to avoid a cyclic import."""
    value = parse_content(body, field.content, field.content_args, json_keys)
    if field.transform is None:
        return value
    if field.transform_each:
        if not isinstance(value, list):
            raise ValueError(
                f"Field '{field.name}': transform_each requires the parsed content to be a list, "
                f"got {type(value).__name__}."
            )
        out = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Field '{field.name}': transform_each requires each list element to be a dict, "
                    f"got {type(item).__name__}."
                )
            out.append(_apply_transform(field.transform, {**captures, **item}))
        return out
    return _apply_transform(field.transform, {**captures, "content": value})


__all__ = [
    "CONTENT_PARSERS",
    "STREAMABLE_PARSERS",
    "parse_content",
    "process_field",
    "validate_transform_strings",
]
# fmt: on
