"""Unit tests for length and complexity bounds on structured output grammars."""

import json
import time
import unittest

from sglang.srt.constrained.grammar_limits import (
    MAX_EBNF_ALTERNATIONS,
    MAX_EBNF_NESTING,
    MAX_EBNF_RULES,
    MAX_JSON_DEPTH,
    MAX_JSON_NODES,
    MAX_REGEX_AST_NODES,
    MAX_REGEX_NESTING,
    MAX_REGEX_QUANTIFIER_DEPTH,
    MAX_STRUCTURAL_TAG_NODES,
    validate_ebnf_complexity,
    validate_grammar_limits,
    validate_json_grammar_complexity,
    validate_regex_complexity,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


def _nested_json_schema(depth: int) -> str:
    node: dict = {"type": "string"}
    for _ in range(depth):
        node = {"type": "object", "properties": {"nested": node}}
    return json.dumps(node)


class TestGrammarLimitsValidBaselines(unittest.TestCase):
    def test_valid_regex_baselines(self):
        for pattern in (
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}",
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            r"\d{4}-\d{2}-\d{2}",
            r"^[0-9a-f]{128}$",
            r"^([a-z]+){5}$",
        ):
            with self.subTest(pattern=pattern):
                validate_grammar_limits("regex", pattern)

    def test_valid_ebnf_baselines(self):
        # String literals and comments containing '|', '::=', '(' must not trip counters.
        quoted_pipes = "|".join(f'"v{i}"' for i in range(100))
        grammar = (
            "# comment with (((( and ::= and " + "|" * 5000 + "\n"
            'root ::= expr "\\"::=||||(((("\n'
            f'expr ::= [|::=\\]] | {quoted_pipes} | "a" | "b"\n'
        )
        validate_grammar_limits("ebnf", grammar)
        validate_grammar_limits("full_assistant_ebnf", grammar)

    def test_valid_json_schema_and_any_sentinel(self):
        validate_grammar_limits("json_schema", "$$ANY$$")
        validate_grammar_limits("json", "$$ANY$$")
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
                    "items": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["id"],
            }
        )
        validate_grammar_limits("json_schema", schema)

    def test_valid_structural_tags(self):
        legacy_tag = json.dumps(
            {
                "type": "structural_tag",
                "structures": [
                    {
                        "begin": f"<tool_{i}>",
                        "schema": json.dumps(
                            {
                                "type": "object",
                                "properties": {
                                    f"arg_{j}": {"type": "string"} for j in range(5)
                                },
                            }
                        ),
                        "end": f"</tool_{i}>",
                    }
                    for i in range(25)
                ],
                "triggers": [f"<tool_{i}>" for i in range(25)],
            }
        )
        native_tag = json.dumps(
            {
                "type": "structural_tag",
                "format": {
                    "type": "or",
                    "elements": [
                        {"type": "regex", "pattern": r"^[a-z]{1,16}$"},
                        {"type": "grammar", "grammar": 'root ::= "ok" | "done"'},
                        {
                            "type": "json_schema",
                            "json_schema": {
                                "type": "object",
                                "properties": {"a": {"type": "integer"}},
                            },
                        },
                    ],
                },
            }
        )
        validate_grammar_limits("structural_tag", legacy_tag)
        validate_grammar_limits("structural_tag", native_tag)


class TestGrammarLimitsAttackVectors(unittest.TestCase):
    def _assert_rejected_fast(self, key_type: str, payload: str, expected_msg: str):
        t0 = time.perf_counter()
        with self.assertRaisesRegex(ValueError, expected_msg):
            validate_grammar_limits(key_type, payload)
        self.assertLess(time.perf_counter() - t0, 0.5)

    def test_v1_regex_byte_and_alternation_bombs(self):
        # 380 KB, ~20k alternation branches from #51833.
        bomb_380kb = "|".join(f"token_{i:06d}" for i in range(28000))
        self.assertGreater(len(bomb_380kb), 350_000)
        self._assert_rejected_fast("regex", bomb_380kb, "byte limit")

        # Compact regex under 64 KiB byte cap but exceeding AST node cap via branches.
        compact_branches = "|".join(f"x{i}" for i in range(MAX_REGEX_AST_NODES + 50))
        self.assertLess(len(compact_branches.encode("utf-8")), 65536)
        self._assert_rejected_fast("regex", compact_branches, "AST node count")

    def test_v1_regex_nesting_and_quantifier_bombs(self):
        deep_groups = (
            "(" * (MAX_REGEX_NESTING + 2) + "a" + ")" * (MAX_REGEX_NESTING + 2)
        )
        self._assert_rejected_fast("regex", deep_groups, "nesting depth")

        nested_q = (
            "(" * (MAX_REGEX_QUANTIFIER_DEPTH + 1)
            + "a"
            + ")+" * (MAX_REGEX_QUANTIFIER_DEPTH + 1)
        )
        self._assert_rejected_fast("regex", nested_q, "quantifier nesting depth")

        product_bomb = "(a{100}){200}"
        self._assert_rejected_fast("regex", product_bomb, "repetition bound product")

    def test_v1_ebnf_byte_and_complexity_bombs(self):
        oversized_ebnf = 'root ::= "a"\n' + ("r ::= 'x' | 'y'\n" * 5000)
        self._assert_rejected_fast("ebnf", oversized_ebnf, "byte limit")

        too_many_rules = "\n".join(f'r{i} ::= "a"' for i in range(MAX_EBNF_RULES + 1))
        self.assertLess(len(too_many_rules.encode("utf-8")), 65536)
        self._assert_rejected_fast("ebnf", too_many_rules, "rule count")

        too_many_alts = "root ::= " + " | ".join(['"a"'] * (MAX_EBNF_ALTERNATIONS + 2))
        self._assert_rejected_fast("ebnf", too_many_alts, "alternation count")

        deep_parens = (
            "root ::= "
            + "(" * (MAX_EBNF_NESTING + 1)
            + '"a"'
            + ")" * (MAX_EBNF_NESTING + 1)
        )
        self._assert_rejected_fast("ebnf", deep_parens, "nesting depth")

    def test_v2_json_schema_bombs(self):
        # > 361 KB schema from #51833.
        oversized_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    f"field_{i}": {"type": "string", "description": "x" * 64}
                    for i in range(4200)
                },
            }
        )
        self.assertGreater(len(oversized_schema), 360_000)
        self._assert_rejected_fast("json_schema", oversized_schema, "byte limit")

        # Compact schema triggering depth limit.
        with envs.SGLANG_GRAMMAR_MAX_BYTES.override(0):
            self._assert_rejected_fast(
                "json", _nested_json_schema(MAX_JSON_DEPTH // 2 + 5), "nesting depth"
            )
            many_nodes = json.dumps({f"k{i}": i for i in range(MAX_JSON_NODES + 1)})
            self._assert_rejected_fast("json_schema", many_nodes, "node count")

        # Deeply nested JSON triggering json.loads RecursionError under byte cap.
        deep_brackets = "[" * 2000 + "]" * 2000
        self._assert_rejected_fast("json_schema", deep_brackets, "nesting depth")

        # Explosive embedded pattern inside valid-size JSON schema.
        explosive_pattern = json.dumps({"type": "string", "pattern": "(a{100}){200}"})
        self._assert_rejected_fast(
            "json_schema", explosive_pattern, "repetition bound product"
        )

    def test_v3_structural_tag_bombs(self):
        # > 509 KB tag from #51833.
        oversized_tag = json.dumps(
            {
                "type": "structural_tag",
                "structures": [
                    {
                        "begin": "<t>",
                        "schema": {"description": "x" * 512000},
                        "end": "</t>",
                    }
                ],
                "triggers": ["<t>"],
            }
        )
        self.assertGreater(len(oversized_tag), 509_800)
        self._assert_rejected_fast("structural_tag", oversized_tag, "byte limit")

        # Compact structural_tag with explosive embedded regex, grammar, and double-encoded schema.
        bad_regex_tag = json.dumps(
            {
                "type": "structural_tag",
                "format": {"type": "regex", "pattern": "(a{100}){200}"},
            }
        )
        self._assert_rejected_fast(
            "structural_tag", bad_regex_tag, "repetition bound product"
        )

        bad_grammar_tag = json.dumps(
            {
                "type": "structural_tag",
                "format": {
                    "type": "grammar",
                    "grammar": "\n".join(
                        f'r{i} ::= "a"' for i in range(MAX_EBNF_RULES + 1)
                    ),
                },
            }
        )
        self._assert_rejected_fast("structural_tag", bad_grammar_tag, "rule count")

        bad_double_encoded_tag = json.dumps(
            {
                "type": "structural_tag",
                "structures": [
                    {
                        "begin": "<t>",
                        "schema": json.dumps(
                            {"type": "string", "pattern": "(a{100}){200}"}
                        ),
                        "end": "</t>",
                    }
                ],
                "triggers": ["<t>"],
            }
        )
        self._assert_rejected_fast(
            "structural_tag", bad_double_encoded_tag, "repetition bound product"
        )

        with envs.SGLANG_GRAMMAR_STRUCTURAL_TAG_MAX_BYTES.override(0):
            too_many_tag_nodes = json.dumps(
                {"items": list(range(MAX_STRUCTURAL_TAG_NODES + 5))}
            )
            self._assert_rejected_fast(
                "structural_tag", too_many_tag_nodes, "node count"
            )


class TestGrammarLimitsBoundariesAndOverrides(unittest.TestCase):
    def test_utf8_byte_boundary_and_env_override(self):
        with envs.SGLANG_GRAMMAR_MAX_BYTES.override(10):
            # 10 ASCII bytes pass; 11 fail.
            validate_grammar_limits("regex", "a" * 10)
            with self.assertRaisesRegex(
                ValueError, "11 bytes, exceeding the 10-byte limit"
            ):
                validate_grammar_limits("regex", "a" * 11)

            # Multi-byte UTF-8 is measured in bytes, not characters: 4 * 3 = 12 bytes > 10.
            with self.assertRaisesRegex(
                ValueError, "12 bytes, exceeding the 10-byte limit"
            ):
                validate_grammar_limits("regex", "\u4e2d" * 4)

        # Non-positive values disable the byte limit.
        large_json_schema = json.dumps({"type": "string", "description": "a" * 80_000})
        large_ebnf = 'root ::= "' + "a" * 80_000 + '"'
        for disabled in (0, -1):
            with (
                self.subTest(disabled=disabled),
                envs.SGLANG_GRAMMAR_MAX_BYTES.override(disabled),
            ):
                validate_grammar_limits("json_schema", large_json_schema)
                validate_grammar_limits("ebnf", large_ebnf)

    def test_structural_tag_limit_is_independent(self):
        with envs.SGLANG_GRAMMAR_MAX_BYTES.override(32):
            tag = json.dumps({"triggers": ["x" * 64]})
            self.assertGreater(len(tag), 32)
            validate_grammar_limits("structural_tag", tag)

        with (
            envs.SGLANG_GRAMMAR_STRUCTURAL_TAG_MAX_BYTES.override(32),
            self.assertRaisesRegex(ValueError, "byte limit"),
        ):
            validate_grammar_limits(
                "structural_tag", json.dumps({"triggers": ["x" * 40]})
            )

        large_tag = json.dumps({"triggers": ["x" * 300_000]})
        for disabled in (0, -1):
            with (
                self.subTest(disabled=disabled),
                envs.SGLANG_GRAMMAR_STRUCTURAL_TAG_MAX_BYTES.override(disabled),
            ):
                validate_grammar_limits("structural_tag", large_tag)

    def test_malformed_syntax_and_unknown_type_pass_through(self):
        # Malformed payloads under size limits must reach backend error handlers.
        validate_grammar_limits("json", "{not valid json")
        validate_grammar_limits("json", "schema")
        validate_grammar_limits("structural_tag", "{not valid json")
        validate_grammar_limits("regex", "[unclosed")

        # Unknown key types are left to BaseGrammarBackend.dispatch_fallback.
        validate_grammar_limits("unknown_type", "x" * 100_000)

    def test_direct_helpers_enforce_bounds(self):
        with self.assertRaises(ValueError):
            validate_regex_complexity("aa|bb|cc|dd", max_ast_nodes=2)
        with self.assertRaises(ValueError):
            validate_ebnf_complexity('a ::= "1"\nb ::= "2"', max_rules=1)
        with self.assertRaises(ValueError):
            validate_json_grammar_complexity(
                '{"a": 1, "b": 2}', kind="json_schema", max_nodes=2
            )


if __name__ == "__main__":
    unittest.main()
