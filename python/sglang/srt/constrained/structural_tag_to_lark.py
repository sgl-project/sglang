# Copyright 2023-2024 SGLang Team
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
"""Translate an xgrammar-native structural tag into an llguidance Lark grammar.

Model-native tool-call constraints (e.g. Kimi-K3 in
``function_call/kimik3_structural_tag.py``) are xgrammar ``Format`` trees that
reach the grammar backend as ``{"type": "structural_tag", "format": {...}}``.
llguidance cannot read that format, so this module walks the serialized tree
(plain dicts) and emits an equivalent Lark grammar.

Special tokens: llguidance never matches a special token through a quoted
literal or a regex, so every substring of a literal that spells one of the
tokenizer's special tokens is emitted as a token reference (``<[id]>``).

Free text (``any_text`` and the text between ``triggered_tags`` tags) is a run
of ordinary bytes interleaved with special tokens. Excluded strings, triggers
and the end of the enclosing tag are kept out of it at special-token
boundaries, so each of them must start with a special token. A free-text run
must likewise be followed by a special token (or the end of the grammar):
llguidance lexes greedily and cannot end a byte run in front of a plain-text
literal. Shapes outside this subset raise ``ValueError``.

JSON-schema values are embedded as ``%json{...}``. A few keywords llguidance
cannot compile are relaxed first so the schema compiles the way it does under
xgrammar; see ``_relax_json_schema``.
"""

import copy
import json
import logging
import re
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union

import msgspec

logger = logging.getLogger(__name__)

# A literal split at special-token boundaries: an ``int`` is a special-token id,
# a ``str`` is ordinary text.
_Piece = Union[int, str]
_Pieces = Tuple[_Piece, ...]

# Any character but NUL; a byte regex never matches a special token.
_TEXT_CHAR = "[^\\x00]"
_TEXT_RUN = f"/{_TEXT_CHAR}+/"
_ANY_TEXT = f"/{_TEXT_CHAR}*/"


def structural_tag_to_lark(
    tag: Dict[str, Any],
    *,
    special_token_ids: Mapping[str, int],
    eos_token_ids: FrozenSet[int],
    max_enforced_string_length: Optional[int] = None,
) -> str:
    """Translate a serialized ``{"type": "structural_tag", "format": ...}``.

    ``special_token_ids`` maps each special token's text to its id.
    ``eos_token_ids`` are never admitted inside the grammar; llguidance allows
    them once the grammar is complete. String ``minLength``/``maxLength`` bounds
    above ``max_enforced_string_length`` are not enforced (``None`` enforces
    all of them).
    """
    builder = _LarkBuilder(
        special_token_ids=special_token_ids,
        eos_token_ids=eos_token_ids,
        max_enforced_string_length=max_enforced_string_length,
    )
    start = builder.emit(tag["format"], _Context())
    builder.log_relaxations()
    return "\n".join([f"start: {start}", *builder.rules])


class _Context(msgspec.Struct, frozen=True):
    """What surrounds the node being emitted."""

    # Strings that end the enclosing tag; trailing free text stops before them.
    stops: Tuple[_Pieces, ...] = ()
    # Whether the node is always followed by a special token or the grammar end.
    special_follow: bool = True
    # The enclosing tag's end when it is a single token.
    token_end: Optional[int] = None


class _FreeTextPlan(msgspec.Struct, frozen=True):
    """Free text with excluded strings, broken down by their leading token."""

    # Text no byte run may contain.
    banned_text: FrozenSet[str]
    # Special tokens that may not appear at all.
    banned_tokens: FrozenSet[int]
    # token -> texts the run right after it may not start with.
    prefixes: Mapping[int, FrozenSet[str]]
    # token -> run text -> tokens that may not follow ``token`` + exactly that run.
    pairs: Mapping[int, Mapping[str, FrozenSet[int]]]


def _plan_free_text(forbidden: Tuple[_Pieces, ...]) -> _FreeTextPlan:
    banned_text, banned_tokens = set(), set()
    prefixes: Dict[int, set] = {}
    pairs: Dict[int, Dict[str, set]] = {}
    for pieces in forbidden:
        head, rest = pieces[0], pieces[1:]
        if isinstance(head, str):
            if rest:
                raise ValueError(
                    "Excluded string must start with a special token when it "
                    f"contains one: {pieces!r}"
                )
            banned_text.add(head)
        elif not rest:
            banned_tokens.add(head)
        elif len(rest) == 1 and isinstance(rest[0], str):
            prefixes.setdefault(head, set()).add(rest[0])
        elif len(rest) == 1:
            pairs.setdefault(head, {}).setdefault("", set()).add(rest[0])
        elif len(rest) == 2 and isinstance(rest[0], str) and isinstance(rest[1], int):
            pairs.setdefault(head, {}).setdefault(rest[0], set()).add(rest[1])
        else:
            raise ValueError(f"Unsupported excluded string shape: {pieces!r}")
    return _FreeTextPlan(
        banned_text=frozenset(banned_text),
        banned_tokens=frozenset(banned_tokens),
        prefixes={k: frozenset(v) for k, v in prefixes.items()},
        pairs={k: {t: frozenset(s) for t, s in v.items()} for k, v in pairs.items()},
    )


# llguidance's ``%json`` compiler rejects these keywords ("Unimplemented keys"),
# failing the whole grammar. They are dropped so the remaining constraints still
# compile; ``oneOf`` becomes ``anyOf`` (the alternatives stay, exclusivity goes).
_DROP_KEYS = ("uniqueItems", "not", "propertyNames")
_DROP_GROUPS = (("if", "then", "else"), ("contains", "minContains", "maxContains"))
_SUBSCHEMA_MAP_KEYS = (
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
)
_SUBSCHEMA_LIST_KEYS = ("anyOf", "allOf", "prefixItems")
_SUBSCHEMA_SINGLE_KEYS = (
    "items",
    "additionalProperties",
    "unevaluatedItems",
    "unevaluatedProperties",
)
_STRING_LENGTH_KEYS = ("minLength", "maxLength")
# Keys kept beside ``const``/``enum``: annotations and definitions, no assertions.
_LITERAL_NODE_KEEP = frozenset(
    ("$defs", "definitions", "title", "description", "default", "examples", "$comment")
)


class _RelaxCounts(msgspec.Struct):
    unsupported_keywords: int = 0
    oversized_lengths: int = 0
    lengths_beside_pattern: int = 0
    literal_nodes: int = 0


def _child_subschemas(schema: Dict[str, Any]):
    for key in _SUBSCHEMA_MAP_KEYS:
        nested = schema.get(key)
        if isinstance(nested, dict):
            yield from nested.values()
    for key in _SUBSCHEMA_LIST_KEYS:
        nested = schema.get(key)
        if isinstance(nested, list):
            yield from nested
    for key in _SUBSCHEMA_SINGLE_KEYS:
        if key in schema:
            yield schema[key]


def _is_oversized_length(value: Any, max_enforced: Optional[int]) -> bool:
    return (
        max_enforced is not None
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value > max_enforced
    )


def _relax_json_schema(
    schema: Any, *, max_enforced_string_length: Optional[int], counts: _RelaxCounts
) -> None:
    """Rewrite ``schema`` in place into a form llguidance compiles.

    * ``const``/``enum`` win over their sibling assertions, as in xgrammar;
      llguidance intersects them and reports e.g. ``{"type": "boolean",
      "enum": ["true"]}`` as unsatisfiable.
    * Keywords in ``_DROP_KEYS``/``_DROP_GROUPS`` are dropped.
    * ``minLength``/``maxLength`` above ``max_enforced_string_length`` are
      dropped: llguidance lexes ``{0,N}`` with one lexer state per byte, so a
      large bound slows every mask fill and can exhaust the lexer state limit.
    * ``minLength``/``maxLength`` beside a ``pattern`` are dropped: llguidance
      compiles the intersection under a fixed DFA budget that ordinary
      hostname-like patterns exceed. xgrammar also enforces only the pattern.

    Only subschema positions are visited, so data such as ``default`` stays
    intact.
    """
    if isinstance(schema, list):
        for item in schema:
            _relax_json_schema(
                item,
                max_enforced_string_length=max_enforced_string_length,
                counts=counts,
            )
        return
    if not isinstance(schema, dict):
        return
    literal_key = next((k for k in ("const", "enum") if k in schema), None)
    if literal_key is not None and "$ref" not in schema:
        dropped = [
            k for k in schema if k != literal_key and k not in _LITERAL_NODE_KEEP
        ]
        for key in dropped:
            del schema[key]
        counts.literal_nodes += bool(dropped)
    if "oneOf" in schema:
        schema.setdefault("anyOf", schema.pop("oneOf"))
        counts.unsupported_keywords += 1
    for key in _DROP_KEYS:
        if schema.pop(key, None) is not None:
            counts.unsupported_keywords += 1
    for group in _DROP_GROUPS:
        if group[0] in schema:
            for key in group:
                if schema.pop(key, None) is not None:
                    counts.unsupported_keywords += 1
    has_pattern = isinstance(schema.get("pattern"), str)
    for key in _STRING_LENGTH_KEYS:
        if key not in schema:
            continue
        if has_pattern:
            del schema[key]
            counts.lengths_beside_pattern += 1
        elif _is_oversized_length(schema[key], max_enforced_string_length):
            del schema[key]
            counts.oversized_lengths += 1
    for child in _child_subschemas(schema):
        _relax_json_schema(
            child,
            max_enforced_string_length=max_enforced_string_length,
            counts=counts,
        )


_LARK_STRING_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r"}


def _lark_string_literal(value: str) -> str:
    """Render ``value`` as a double-quoted Lark literal.

    Control characters are escaped: a raw newline inside ``"..."`` is a Lark
    syntax error.
    """
    out: List[str] = []
    for ch in value:
        escape = _LARK_STRING_ESCAPES.get(ch)
        if escape is not None:
            out.append(escape)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


# A backslash escape pair, or a bare ``/``. Matching the pair first keeps an
# already-escaped ``\/`` (common in JSON Schema patterns) intact.
_REGEX_ESCAPE_OR_SLASH = re.compile(r"(\\.)|/", re.DOTALL)


def _lark_regex(pattern: str) -> str:
    """Render ``pattern`` as an inline ``/.../`` Lark regex."""
    escaped = _REGEX_ESCAPE_OR_SLASH.sub(lambda m: m.group(1) or r"\/", pattern)
    return "/" + escaped + "/"


def _token_set(token_ids, *, negate: bool = False) -> str:
    """Render token ids as a Lark token set such as ``<[1-3,7]>``."""
    ranges: List[str] = []
    ids = sorted(set(token_ids))
    i = 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        ranges.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return "<[" + ("^" if negate else "") + ",".join(ranges) + "]>"


def _raw_scalar(value: Any) -> str:
    """The unquoted wire form of a qwen_xml scalar."""
    return value if isinstance(value, str) else json.dumps(value)


# Characters with a meaning in a Rust regex, plus ``/``, which ends an inline
# Lark regex.
_REGEX_META = frozenset("\\.+*?()|[]{}^$#&-~/")


def _regex_escape(text: str) -> str:
    out: List[str] = []
    for ch in text:
        if ch in _REGEX_META:
            out.append("\\" + ch)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    return "".join(out)


def _regex_alternation(texts) -> str:
    return "(?:" + "|".join(_regex_escape(t) for t in sorted(texts)) + ")"


class _LarkBuilder:
    """Accumulate Lark rules while walking a serialized ``Format`` tree.

    ``emit`` returns a Lark atom (a rule or terminal name, a literal, a token
    reference or an inline regex) usable inside a rule body.
    """

    def __init__(
        self,
        *,
        special_token_ids: Mapping[str, int],
        eos_token_ids: FrozenSet[int],
        max_enforced_string_length: Optional[int],
    ) -> None:
        self.rules: List[str] = []
        self._counter = 0
        self._special_token_ids = {t: i for t, i in special_token_ids.items() if t}
        self._special_ids = frozenset(self._special_token_ids.values())
        self._eos_token_ids = eos_token_ids
        self._special_pattern = (
            re.compile(
                "("
                + "|".join(
                    re.escape(t)
                    for t in sorted(self._special_token_ids, key=len, reverse=True)
                )
                + ")"
            )
            if self._special_token_ids
            else None
        )
        self._max_enforced_string_length = max_enforced_string_length
        self._relax_counts = _RelaxCounts()
        self._free_text_rules: Dict[FrozenSet[_Pieces], str] = {}
        self._run_terminals: Dict[Tuple[FrozenSet[str], ...], str] = {}
        self._handlers: Dict[str, Callable[[Dict[str, Any], _Context], str]] = {
            "const_string": self._emit_const_string,
            "regex": self._emit_regex,
            "json_schema": self._emit_json_schema,
            "any_text": self._emit_any_text,
            "tag": self._emit_tag,
            "sequence": self._emit_sequence,
            "or": self._emit_or,
            "optional": self._emit_optional,
            "star": self._emit_star,
            "triggered_tags": self._emit_triggered_tags,
            "tags_with_separator": self._emit_tags_with_separator,
            "token": self._emit_token,
            "exclude_token": self._emit_exclude_token,
            "any_tokens": self._emit_any_tokens,
        }

    def emit(self, node: Dict[str, Any], ctx: _Context) -> str:
        node_type = node["type"]
        handler = self._handlers.get(node_type)
        if handler is None:
            raise ValueError(f"Unsupported structural tag node: {node_type!r}")
        return handler(node, ctx)

    def log_relaxations(self) -> None:
        counts = self._relax_counts
        if counts.unsupported_keywords:
            logger.warning(
                "Relaxed %d JSON-schema keyword(s) that llguidance cannot compile",
                counts.unsupported_keywords,
            )
        if counts.oversized_lengths:
            logger.warning(
                "Dropped %d string length bound(s) above %s",
                counts.oversized_lengths,
                self._max_enforced_string_length,
            )
        if counts.lengths_beside_pattern:
            logger.warning(
                "Dropped %d string length bound(s) beside a pattern",
                counts.lengths_beside_pattern,
            )
        if counts.literal_nodes:
            logger.debug(
                "Reduced %d const/enum schema node(s) to their literal values",
                counts.literal_nodes,
            )

    def _new_name(self, hint: str) -> str:
        self._counter += 1
        return f"r{self._counter}_{hint}"

    def _add_rule(self, body: str, hint: str) -> str:
        name = self._new_name(hint)
        self.rules.append(f"{name}: {body}")
        return name

    # ----- literals -----

    def _split(self, value: str) -> _Pieces:
        """Split ``value`` into special-token ids and the text between them."""
        if not value:
            return ()
        if self._special_pattern is None:
            return (value,)
        pieces: List[_Piece] = []
        for i, part in enumerate(self._special_pattern.split(value)):
            if i % 2:
                pieces.append(self._special_token_ids[part])
            elif part:
                pieces.append(part)
        return tuple(pieces)

    @staticmethod
    def _render(pieces: _Pieces) -> str:
        if not pieces:
            return '""'
        return " ".join(
            _token_set([p]) if isinstance(p, int) else _lark_string_literal(p)
            for p in pieces
        )

    def _token_id(self, token: Union[int, str]) -> int:
        if isinstance(token, int):
            return token
        token_id = self._special_token_ids.get(token)
        if token_id is None:
            raise ValueError(f"Token {token!r} is not a special token")
        return token_id

    def _boundary(self, value: Any) -> Tuple[str, Tuple[_Pieces, ...]]:
        """Render a tag ``begin``/``end``; also return the strings it can spell."""
        if isinstance(value, dict):
            if value.get("type") != "token":
                raise ValueError(f"Unsupported tag boundary: {value!r}")
            pieces: _Pieces = (self._token_id(value["token"]),)
            return self._render(pieces), (pieces,)
        if isinstance(value, list):
            options = tuple(self._split(v) for v in value)
            if not options or not all(options):
                raise ValueError(f"Unsupported tag end list: {value!r}")
            body = " | ".join(self._render(p) for p in options)
            return f"({body})", options
        pieces = self._split(value)
        return self._render(pieces), ((pieces,) if pieces else ())

    @staticmethod
    def _starts_special(options: Tuple[_Pieces, ...]) -> bool:
        return all(isinstance(pieces[0], int) for pieces in options)

    # ----- what may follow a node -----

    def _lead(self, node: Dict[str, Any]) -> Tuple[bool, bool]:
        """Return (non-empty matches start with a special token, may be empty)."""
        node_type = node["type"]
        if node_type == "const_string":
            pieces = self._split(node["value"])
            return not pieces or isinstance(pieces[0], int), not pieces
        if node_type == "token":
            return True, False
        if node_type == "tag":
            return self._lead_tag(node)
        if node_type == "sequence":
            elements = node["elements"]
            nullable = all(self._lead(e)[1] for e in elements)
            return self._follows_special(elements, special_follow=True), nullable
        if node_type == "or":
            leads = [self._lead(e) for e in node["elements"]]
            return all(s for s, _ in leads), any(n for _, n in leads)
        if node_type in ("optional", "star"):
            return self._lead(node["content"])[0], True
        if node_type == "tags_with_separator":
            special = all(self._lead(tag)[0] for tag in node["tags"])
            return special, not node.get("at_least_one", False)
        if node_type == "triggered_tags" and node.get("at_least_one", False):
            return all(self._lead(tag)[0] for tag in node["tags"]), False
        if node_type in ("json_schema", "exclude_token"):
            return False, False
        return False, True

    def _lead_tag(self, node: Dict[str, Any]) -> Tuple[bool, bool]:
        _, begins = self._boundary(node["begin"])
        if begins:
            return self._starts_special(begins), False
        special, nullable = self._lead(node["content"])
        if not nullable:
            return special, False
        _, ends = self._boundary(node["end"])
        return special and self._starts_special(ends), not ends

    def _follows_special(
        self, nodes: List[Dict[str, Any]], special_follow: bool
    ) -> bool:
        """Whether ``nodes``, then the context, always start specially."""
        for node in nodes:
            special, nullable = self._lead(node)
            if not special:
                return False
            if not nullable:
                return True
        return special_follow

    # ----- free text -----

    def _free_text(
        self, forbidden: Tuple[_Pieces, ...], *, special_follow: bool
    ) -> str:
        """Text spelling none of ``forbidden``, as a rule name.

        Runs of ordinary bytes alternate with special tokens; each forbidden
        string is kept out at the special token it starts with.
        """
        if not special_follow:
            raise ValueError(
                "Free text must be followed by a special token or the end of "
                "the grammar"
            )
        key = frozenset(forbidden)
        cached = self._free_text_rules.get(key)
        if cached is not None:
            return cached
        plan = _plan_free_text(tuple(sorted(key, key=repr)))
        allowed = self._special_ids - self._eos_token_ids - plan.banned_tokens
        heads = sorted((set(plan.prefixes) | set(plan.pairs)) & allowed)
        others = allowed.difference(heads)
        run = self._run_terminal(banned_text=plan.banned_text)
        head_rules = {head: self._new_name("text_head") for head in heads}

        def segments(excluded: FrozenSet[int]) -> List[str]:
            alts = []
            if others - excluded:
                alts.append(f"{_token_set(others - excluded)} {run}?")
            alts.extend(head_rules[h] for h in heads if h not in excluded)
            return alts

        endings: List[str] = []
        for head in heads:
            body, head_endings = self._head_segment(head, plan=plan, segments=segments)
            self.rules.append(f"{head_rules[head]}: {body}")
            endings.extend(head_endings)
        parts = [f"{run}?"]
        if segments(frozenset()):
            seg = self._add_rule(" | ".join(segments(frozenset())), "text_seg")
            parts.append(f"{seg}*")
        if endings:
            parts.append(f"({' | '.join(endings)})?")
        name = self._add_rule(" ".join(parts), "text")
        self._free_text_rules[key] = name
        return name

    def _head_segment(
        self,
        head: int,
        *,
        plan: _FreeTextPlan,
        segments: Callable[[FrozenSet[int]], List[str]],
    ) -> Tuple[str, List[str]]:
        """Alternatives starting at ``head``, and those that may end the text.

        A run equal to a ``pairs`` key is matched as a literal, so the segment
        after it can be restricted; the text may also end right after it.
        """
        token = _token_set([head])
        starts = plan.prefixes.get(head, frozenset())
        pairs = plan.pairs.get(head, {})
        run = self._run_terminal(
            banned_text=plan.banned_text,
            starts=starts,
            exact=frozenset(text for text in pairs if text),
        )
        alts = [f"{token} {run}" + ("" if "" in pairs else "?")]
        endings: List[str] = []
        for text, excluded in sorted(pairs.items()):
            if text and (
                any(banned in text for banned in plan.banned_text)
                or any(text.startswith(prefix) for prefix in starts)
            ):
                continue
            atom = f"{token} {_lark_string_literal(text)}" if text else token
            follow = segments(excluded)
            if follow:
                alts.append(f"{atom} ({' | '.join(follow)})")
            endings.append(atom)
        return " | ".join(alts), endings

    def _run_terminal(
        self,
        *,
        banned_text: FrozenSet[str],
        starts: FrozenSet[str] = frozenset(),
        exact: FrozenSet[str] = frozenset(),
    ) -> str:
        """A non-empty byte run that contains none of ``banned_text``, starts
        with none of ``starts`` and equals none of ``exact``."""
        key = (banned_text, starts, exact)
        cached = self._run_terminals.get(key)
        if cached is not None:
            return cached
        parts = [_TEXT_RUN]
        if banned_text:
            alts = _regex_alternation(banned_text)
            parts.append(f"~/{_TEXT_CHAR}*{alts}{_TEXT_CHAR}*/")
        if starts:
            parts.append(f"~/{_regex_alternation(starts)}{_TEXT_CHAR}*/")
        if exact:
            parts.append(f"~/{_regex_alternation(exact)}/")
        if len(parts) == 1:
            name = _TEXT_RUN
        else:
            self._counter += 1
            name = f"TEXT{self._counter}"
            self.rules.append(f"{name}: {' & '.join(parts)}")
        self._run_terminals[key] = name
        return name

    # ----- nodes -----

    def _excludes(self, node: Dict[str, Any]) -> Tuple[_Pieces, ...]:
        return tuple(p for p in map(self._split, node.get("excludes", [])) if p)

    def _emit_const_string(self, node: Dict[str, Any], ctx: _Context) -> str:
        return self._render(self._split(node["value"]))

    def _emit_regex(self, node: Dict[str, Any], ctx: _Context) -> str:
        return _lark_regex(node["pattern"])

    def _emit_any_text(self, node: Dict[str, Any], ctx: _Context) -> str:
        return self._free_text(
            self._excludes(node) + ctx.stops, special_follow=ctx.special_follow
        )

    def _emit_tag(self, node: Dict[str, Any], ctx: _Context) -> str:
        begin, _ = self._boundary(node["begin"])
        end, stops = self._boundary(node["end"])
        end_node = node["end"]
        content_ctx = _Context(
            stops=stops,
            special_follow=(
                self._starts_special(stops) if stops else ctx.special_follow
            ),
            token_end=(
                self._token_id(end_node["token"])
                if isinstance(end_node, dict)
                else None
            ),
        )
        content = self.emit(node["content"], content_ctx)
        parts = [part for part in (begin, content, end) if part != '""']
        return self._add_rule(" ".join(parts) or '""', "tag")

    def _emit_sequence(self, node: Dict[str, Any], ctx: _Context) -> str:
        elements = node["elements"]
        parts = [
            self.emit(
                element,
                msgspec.structs.replace(
                    ctx,
                    special_follow=self._follows_special(
                        elements[i + 1 :], ctx.special_follow
                    ),
                ),
            )
            for i, element in enumerate(elements)
        ]
        return self._add_rule(" ".join(parts) or '""', "seq")

    def _emit_or(self, node: Dict[str, Any], ctx: _Context) -> str:
        parts = [self.emit(element, ctx) for element in node["elements"]]
        return self._add_rule(" | ".join(parts), "or")

    def _emit_optional(self, node: Dict[str, Any], ctx: _Context) -> str:
        return self._add_rule(f"({self.emit(node['content'], ctx)})?", "opt")

    def _emit_star(self, node: Dict[str, Any], ctx: _Context) -> str:
        content = node["content"]
        # Each repetition is followed by the next one or by the context.
        content_ctx = msgspec.structs.replace(
            ctx, special_follow=ctx.special_follow and self._lead(content)[0]
        )
        return self._add_rule(f"({self.emit(content, content_ctx)})*", "star")

    def _emit_tag_alternatives(self, tags: List[Dict[str, Any]], ctx: _Context) -> str:
        return self._add_rule(" | ".join(self.emit(tag, ctx) for tag in tags), "tags")

    def _emit_triggered_tags(self, node: Dict[str, Any], ctx: _Context) -> str:
        trigger_texts = node["triggers"]
        triggers = tuple(map(self._split, trigger_texts))
        if not triggers or not all(triggers) or not self._starts_special(triggers):
            raise ValueError(
                f"Every trigger must start with a special token: {trigger_texts!r}"
            )
        for tag in node["tags"]:
            begin = tag["begin"]
            if not isinstance(begin, str) or not begin.startswith(tuple(trigger_texts)):
                raise ValueError(f"Tag begin {begin!r} does not start with a trigger")
        tags = self._emit_tag_alternatives(node["tags"], ctx)
        at_least_one = node.get("at_least_one", False)
        if node.get("stop_after_first", False) and at_least_one:
            return tags
        # Text between tags may not spell a trigger: a trigger starts a tag.
        text = self._free_text(
            self._excludes(node) + triggers + ctx.stops,
            special_follow=ctx.special_follow,
        )
        if node.get("stop_after_first", False):
            body = f"{text} {tags}?"
        elif at_least_one:
            body = f"{tags} {text} ({tags} {text})*"
        else:
            body = f"{text} ({tags} {text})*"
        return self._add_rule(body, "triggered")

    def _emit_tags_with_separator(self, node: Dict[str, Any], ctx: _Context) -> str:
        tags = self._emit_tag_alternatives(node["tags"], ctx)
        separator = self._split(node["separator"])
        at_least_one = node.get("at_least_one", False)
        if node.get("stop_after_first", False):
            body = tags if at_least_one else f"{tags}?"
        elif not separator:
            body = f"{tags}+" if at_least_one else f"{tags}*"
        else:
            repeated = f"{tags} ({self._render(separator)} {tags})*"
            body = repeated if at_least_one else f"({repeated})?"
        return self._add_rule(body, "tags_sep")

    def _emit_token(self, node: Dict[str, Any], ctx: _Context) -> str:
        return _token_set([self._token_id(node["token"])])

    def _token_excluding(self, node: Dict[str, Any], ctx: _Context) -> str:
        """Any single token except the listed ones, EOS and the tag's end."""
        excluded = {self._token_id(t) for t in node.get("exclude_tokens", [])}
        excluded |= self._eos_token_ids
        if ctx.token_end is not None:
            excluded.add(ctx.token_end)
        if not excluded:
            raise ValueError("A token-level node needs an EOS or end token")
        return _token_set(excluded, negate=True)

    def _emit_exclude_token(self, node: Dict[str, Any], ctx: _Context) -> str:
        return self._token_excluding(node, ctx)

    def _emit_any_tokens(self, node: Dict[str, Any], ctx: _Context) -> str:
        return f"{self._token_excluding(node, ctx)}*"

    def _emit_json_schema(self, node: Dict[str, Any], ctx: _Context) -> str:
        schema = node.get("json_schema")
        if schema is None:
            schema = {}
        style = node.get("style", "json")
        if style == "qwen_xml":
            return self._qwen_xml_string(schema, ctx)
        if style != "json":
            raise ValueError(f"Unsupported json_schema style: {style!r}")
        schema = copy.deepcopy(schema)
        _relax_json_schema(
            schema,
            max_enforced_string_length=self._max_enforced_string_length,
            counts=self._relax_counts,
        )
        return self._add_rule("%json" + json.dumps(schema), "json")

    def _qwen_xml_string(self, schema: Any, ctx: _Context) -> str:
        """A raw (unquoted) qwen_xml string value."""
        if schema is True:
            return self._free_text(ctx.stops, special_follow=ctx.special_follow)
        if not isinstance(schema, dict) or schema.get("type", "string") != "string":
            raise ValueError(f"Unsupported qwen_xml value schema: {schema!r}")
        options = schema.get("anyOf", schema.get("oneOf"))
        if isinstance(options, list):
            rest = {k: v for k, v in schema.items() if k not in ("anyOf", "oneOf")}
            alts = [
                self._qwen_xml_string({**rest, **o} if isinstance(o, dict) else o, ctx)
                for o in options
            ]
            return self._add_rule(" | ".join(alts), "string_any")
        if "const" in schema:
            return self._render(self._split(_raw_scalar(schema["const"])))
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            alts = [self._render(self._split(_raw_scalar(v))) for v in enum]
            return self._add_rule(" | ".join(alts), "enum")
        if isinstance(schema.get("pattern"), str):
            return _lark_regex(schema["pattern"])
        low, high = (
            self._enforced_length(schema.get(key)) for key in _STRING_LENGTH_KEYS
        )
        if low or high is not None:
            bound = "" if high is None else str(high)
            return _lark_regex(f"{_TEXT_CHAR}{{{low or 0},{bound}}}")
        return self._free_text(ctx.stops, special_follow=ctx.special_follow)

    def _enforced_length(self, value: Any) -> Optional[int]:
        """``value`` as a length bound, or ``None`` when it is not enforced."""
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        if _is_oversized_length(value, self._max_enforced_string_length):
            self._relax_counts.oversized_lengths += 1
            return None
        return value
