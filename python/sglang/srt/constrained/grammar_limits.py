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
"""Length and complexity validation for structured output grammar parameters."""

import json
import re

from sglang.srt.environ import envs

try:
    import re._parser as sre_parse
except ImportError:
    import sre_parse  # Python < 3.11

MAX_REGEX_AST_NODES = 4096
MAX_REGEX_NESTING = 64
MAX_REGEX_QUANTIFIER_DEPTH = 5
MAX_REGEX_REPEAT_PRODUCT = 10000

MAX_EBNF_RULES = 512
MAX_EBNF_ALTERNATIONS = 4096
MAX_EBNF_NESTING = 64

MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 10000
MAX_STRUCTURAL_TAG_NODES = 40000

_REPEAT_OPS = (
    sre_parse.MAX_REPEAT,
    sre_parse.MIN_REPEAT,
    getattr(sre_parse, "POSSESSIVE_REPEAT", None),
)
_ATOMIC_GROUP = getattr(sre_parse, "ATOMIC_GROUP", None)


def validate_regex_complexity(
    pattern: str,
    *,
    max_ast_nodes: int = MAX_REGEX_AST_NODES,
    max_nesting: int = MAX_REGEX_NESTING,
    max_quantifier_depth: int = MAX_REGEX_QUANTIFIER_DEPTH,
    max_repeat_product: int = MAX_REGEX_REPEAT_PRODUCT,
) -> None:
    """Reject regex patterns whose AST size, nesting, or repetitions can cause DoS."""
    try:
        parsed = sre_parse.parse(pattern)
    except (OverflowError, RecursionError) as err:
        raise ValueError(f"regex is too complex to compile: {err}") from err
    except (re.error, TypeError, ValueError):
        # Malformed regex syntax is left to the backend compiler to report.
        return

    nodes = 0
    stack: list[tuple[sre_parse.SubPattern, int, int, int]] = [(parsed, 0, 0, 1)]
    while stack:
        subpattern, depth, q_depth, repeat_product = stack.pop()
        if depth > max_nesting:
            raise ValueError(f"regex nesting depth exceeds limit ({max_nesting})")

        for op, arg in subpattern:
            nodes += 1
            if nodes > max_ast_nodes:
                raise ValueError(
                    f"regex AST node count exceeds limit ({max_ast_nodes})"
                )

            if op == sre_parse.SUBPATTERN:
                stack.append((arg[-1], depth + 1, q_depth, repeat_product))
            elif _ATOMIC_GROUP is not None and op == _ATOMIC_GROUP:
                stack.append((arg, depth + 1, q_depth, repeat_product))
            elif op == sre_parse.BRANCH:
                _, branches = arg
                for branch in branches:
                    nodes += 1
                    if nodes > max_ast_nodes:
                        raise ValueError(
                            f"regex AST node count exceeds limit ({max_ast_nodes})"
                        )
                    stack.append((branch, depth + 1, q_depth, repeat_product))
            elif op in _REPEAT_OPS:
                minimum, maximum, body = arg
                next_q_depth = q_depth + 1
                if next_q_depth > max_quantifier_depth:
                    raise ValueError(
                        f"regex quantifier nesting depth exceeds limit ({max_quantifier_depth})"
                    )
                factor = minimum if maximum == sre_parse.MAXREPEAT else maximum
                next_product = repeat_product * max(1, factor)
                if next_product > max_repeat_product:
                    raise ValueError(
                        f"regex repetition bound product ({next_product}) exceeds "
                        f"limit ({max_repeat_product})"
                    )
                stack.append((body, depth + 1, next_q_depth, next_product))
            elif op in (sre_parse.ASSERT, sre_parse.ASSERT_NOT):
                stack.append((arg[1], depth + 1, q_depth, repeat_product))
            elif op == sre_parse.GROUPREF_EXISTS:
                stack.append((arg[1], depth + 1, q_depth, repeat_product))
                if arg[2] is not None:
                    stack.append((arg[2], depth + 1, q_depth, repeat_product))


def validate_ebnf_complexity(
    grammar: str,
    *,
    max_rules: int = MAX_EBNF_RULES,
    max_alternations: int = MAX_EBNF_ALTERNATIONS,
    max_nesting: int = MAX_EBNF_NESTING,
) -> None:
    """Reject EBNF grammars with excessive rules, alternations, or grouping depth."""
    rules = 0
    alternations = 0
    depth = 0

    n = len(grammar)
    i = 0
    in_quote: str | None = None
    in_char_class = False
    in_comment = False

    while i < n:
        ch = grammar[i]

        if in_comment:
            if ch == "\n":
                in_comment = False
            i += 1
            continue

        if in_quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
            i += 1
            continue

        if in_char_class:
            if ch == "\\":
                i += 2
                continue
            if ch == "]":
                in_char_class = False
            i += 1
            continue

        if ch == "#":
            in_comment = True
            i += 1
            continue
        if ch in ('"', "'"):
            in_quote = ch
            i += 1
            continue
        if ch == "[":
            in_char_class = True
            i += 1
            continue

        if ch == ":" and grammar.startswith("::=", i):
            rules += 1
            if rules > max_rules:
                raise ValueError(f"ebnf rule count exceeds limit ({max_rules})")
            i += 3
            continue

        if ch == "|":
            alternations += 1
            if alternations > max_alternations:
                raise ValueError(
                    f"ebnf alternation count exceeds limit ({max_alternations})"
                )
        elif ch == "(":
            depth += 1
            if depth > max_nesting:
                raise ValueError(
                    f"ebnf grouping nesting depth exceeds limit ({max_nesting})"
                )
        elif ch == ")" and depth > 0:
            depth -= 1

        i += 1


def validate_json_grammar_complexity(
    payload: str,
    *,
    kind: str,
    max_depth: int = MAX_JSON_DEPTH,
    max_nodes: int = MAX_JSON_NODES,
) -> None:
    """Reject JSON-encoded grammars whose depth, node count, or embedded patterns exceed bounds."""
    if kind == "json_schema" and payload == "$$ANY$$":
        return

    try:
        root = json.loads(payload)
    except RecursionError as err:
        raise ValueError(f"{kind} nesting depth exceeds limit ({max_depth})") from err
    except ValueError:
        # Malformed JSON is left to the backend compiler to report.
        return

    nodes = 0
    stack: list[tuple[object, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            raise ValueError(f"{kind} nesting depth exceeds limit ({max_depth})")
        nodes += 1
        if nodes > max_nodes:
            raise ValueError(f"{kind} node count exceeds limit ({max_nodes})")

        if isinstance(node, dict):
            pattern = node.get("pattern")
            if isinstance(pattern, str):
                validate_regex_complexity(pattern)

            if kind == "structural_tag":
                if node.get("type") == "grammar":
                    grammar_text = node.get("grammar")
                    if isinstance(grammar_text, str):
                        validate_ebnf_complexity(grammar_text)

                raw_schema = node.get("schema")
                if isinstance(raw_schema, str):
                    try:
                        parsed_schema = json.loads(raw_schema)
                    except RecursionError as err:
                        raise ValueError(
                            f"{kind} nesting depth exceeds limit ({max_depth})"
                        ) from err
                    except ValueError:
                        parsed_schema = None
                    if parsed_schema is not None:
                        stack.append((parsed_schema, depth + 1))

            for val in node.values():
                stack.append((val, depth + 1))
        elif isinstance(node, list):
            for item in node:
                stack.append((item, depth + 1))


def validate_grammar_limits(key_type: str, value: str) -> None:
    """Validate byte size and structural complexity for a grammar constraint."""
    if key_type in ("json", "json_schema"):
        label = "json_schema"
        max_bytes = envs.SGLANG_GRAMMAR_MAX_BYTES.get()
    elif key_type == "regex":
        label = "regex"
        max_bytes = envs.SGLANG_GRAMMAR_MAX_BYTES.get()
    elif key_type in ("ebnf", "full_assistant_ebnf"):
        label = key_type
        max_bytes = envs.SGLANG_GRAMMAR_MAX_BYTES.get()
    elif key_type == "structural_tag":
        label = "structural_tag"
        max_bytes = envs.SGLANG_GRAMMAR_STRUCTURAL_TAG_MAX_BYTES.get()
    else:
        return

    if not isinstance(value, str):
        return

    if max_bytes is not None and max_bytes > 0:
        size = len(value.encode("utf-8", "replace"))
        if size > max_bytes:
            raise ValueError(
                f"{label} is {size} bytes, exceeding the {max_bytes}-byte limit"
            )

    if label == "json_schema":
        validate_json_grammar_complexity(
            value,
            kind="json_schema",
            max_depth=MAX_JSON_DEPTH,
            max_nodes=MAX_JSON_NODES,
        )
    elif label == "regex":
        validate_regex_complexity(value)
    elif label in ("ebnf", "full_assistant_ebnf"):
        validate_ebnf_complexity(value)
    elif label == "structural_tag":
        validate_json_grammar_complexity(
            value,
            kind="structural_tag",
            max_depth=MAX_JSON_DEPTH,
            max_nodes=MAX_STRUCTURAL_TAG_NODES,
        )
