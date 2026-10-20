"""Unit tests for dropping structural tag patterns that XGrammar cannot compile."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import xgrammar as xgr

from sglang.srt.constrained import xgrammar_backend
from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import (
    XGrammarGrammar,
    XGrammarGrammarBackend,
    _drop_uncompilable_patterns,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

# Fails under qwen_xml in xgrammar 0.2.7; tests using it go red once
# mlc-ai/xgrammar#1001 lands, which is the signal to remove the relaxation.
LOOKAHEAD = r"^(?!\.)[a-z]+$"
BACKREFERENCE = r"(a)\1"


def _make_backend(*, relax: bool) -> XGrammarGrammarBackend:
    vocab = [chr(i) for i in range(32, 127)] + ["\n", "<eos>"]
    info = xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, stop_token_ids=[len(vocab) - 1])
    tokenizer = SimpleNamespace(init_xgrammar=lambda: (info, None))
    with envs.SGLANG_ENABLE_STRUCTURAL_TAG_PATTERN_RELAXATION.override(relax):
        return XGrammarGrammarBackend(tokenizer, len(vocab))


def _dispatch(key: str, *, relax: bool):
    return _make_backend(relax=relax).dispatch_structural_tag(key)


def _tool(name: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        },
    }


def _string(pattern: str) -> dict:
    return {"type": "string", "pattern": pattern}


def _model_tag(tools: list, model="qwen_3_coder", tool_choice="required") -> str:
    return xgr.get_model_structural_tag(
        model=model, tools=tools, tool_choice=tool_choice, reasoning=False
    ).model_dump_json()


def _legacy_tag(schema: dict) -> str:
    return json.dumps(
        {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": '<tool_call>{"name": "ls", "arguments": ',
                    "schema": schema,
                    "end": "}</tool_call>",
                }
            ],
            "triggers": ["<tool_call>"],
        }
    )


class TestStructuralTagPatternDropping(CustomTestCase):
    def test_flag_off_still_fails(self):
        key = _model_tag([_tool("ls", {"path": _string(LOOKAHEAD)})])

        result = _dispatch(key, relax=False)

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertIn("Lookahead", result.error_message)

    def test_native_tag_compiles_and_names_the_tool(self):
        """One tool with an uncompilable pattern must not fail the whole tool set (#45224)."""
        key = _model_tag(
            [
                _tool("ls", {"path": _string(LOOKAHEAD)}),
                _tool("cat", {"file": _string("^[a-z]+$")}),
            ]
        )

        with self.assertLogs(xgrammar_backend.logger, level="WARNING") as logs:
            result = _dispatch(key, relax=True)

        self.assertIsInstance(result, XGrammarGrammar)
        self.assertIn("<function=ls>", logs.output[0])
        self.assertIn("dropped pattern at #/properties/path", logs.output[0])
        self.assertNotIn("<function=cat>", logs.output[0])
        self.assertNotIn(LOOKAHEAD.replace("\\", "\\\\"), result.key_string)

    def test_only_failing_patterns_are_dropped(self):
        tag = json.loads(
            _model_tag(
                [
                    _tool(
                        "ls",
                        {"path": _string(LOOKAHEAD), "mode": _string("^[rw]$")},
                    ),
                    _tool("cat", {"file": _string(BACKREFERENCE)}),
                    _tool("rm", {"file": _string("^[a-z]+$")}),
                ]
            )
        )
        before = copy.deepcopy(tag)

        changes = _drop_uncompilable_patterns(tag)

        schemas = [t["content"]["json_schema"] for t in tag["format"]["tags"]]
        self.assertNotIn("pattern", schemas[0]["properties"]["path"])
        self.assertEqual(schemas[0]["properties"]["mode"]["pattern"], "^[rw]$")
        self.assertNotIn("pattern", schemas[1]["properties"]["file"])
        self.assertEqual(
            schemas[2], before["format"]["tags"][2]["content"]["json_schema"]
        )
        self.assertEqual(len(changes), 2)

    def test_nested_patterns_are_checked_in_place(self):
        """Patterns are dropped only where they fail: XML styles compile nested values
        as JSON, and XGrammar ignores `not`, so only `a` loses its pattern."""
        tool = _tool(
            "ls",
            {
                "a": _string(BACKREFERENCE),
                "b": {"type": "array", "items": _string(LOOKAHEAD)},
                "c": {"type": "string", "not": _string(BACKREFERENCE)},
            },
        )
        tag = json.loads(_model_tag([tool]))

        changes = _drop_uncompilable_patterns(tag)

        properties = tag["format"]["tags"][0]["content"]["json_schema"]["properties"]
        self.assertNotIn("pattern", properties["a"])
        self.assertEqual(properties["b"]["items"]["pattern"], LOOKAHEAD)
        self.assertEqual(properties["c"]["not"]["pattern"], BACKREFERENCE)
        self.assertEqual(len(changes), 1)

    def test_pattern_inside_defs_is_dropped(self):
        tool = _tool("ls", {"path": {"$ref": "#/$defs/Path"}})
        tool["function"]["parameters"]["$defs"] = {"Path": _string(LOOKAHEAD)}

        changes = _drop_uncompilable_patterns(json.loads(_model_tag([tool])))

        self.assertEqual(len(changes), 1)
        self.assertIn("#/$defs/Path", changes[0])

    def test_json_style_keeps_patterns_it_can_compile(self):
        """JSON style ignores only the lookahead, so that pattern is kept."""
        key = _legacy_tag(
            {
                "type": "object",
                "properties": {
                    "name": _string(LOOKAHEAD),
                    "id": _string(BACKREFERENCE),
                },
            }
        )
        self.assertIsInstance(_dispatch(key, relax=False), InvalidGrammarObject)

        tag = json.loads(key)
        changes = _drop_uncompilable_patterns(tag)

        properties = tag["structures"][0]["schema"]["properties"]
        self.assertEqual(properties["name"]["pattern"], LOOKAHEAD)
        self.assertNotIn("pattern", properties["id"])
        self.assertEqual(len(changes), 1)
        with self.assertLogs(xgrammar_backend.logger, level="WARNING"):
            self.assertIsInstance(_dispatch(key, relax=True), XGrammarGrammar)

    def test_patterns_are_checked_in_the_tag_style(self):
        """Covers a style stricter than plain regex and a schema nested in `optional`."""
        cases = [
            # minimax_m3_xml rejects every pattern, even one plain regex accepts.
            ("minimax_m3", "required", _string("^[a-z]+$")),
            # kimi_k3 puts tool schemas under an `optional` format with tool_choice=auto.
            ("kimi_k3", "auto", _string(BACKREFERENCE)),
        ]
        for model, tool_choice, prop in cases:
            with self.subTest(model=model):
                key = _model_tag([_tool("ls", {"x": prop})], model, tool_choice)
                self.assertIsInstance(_dispatch(key, relax=False), InvalidGrammarObject)
                with self.assertLogs(xgrammar_backend.logger, level="WARNING"):
                    result = _dispatch(key, relax=True)
                self.assertIsInstance(result, XGrammarGrammar)

    def test_other_schema_errors_still_fail(self):
        """Only patterns are relaxed; other schema errors fail without a recompile."""
        backend = _make_backend(relax=True)
        backend.grammar_compiler = MagicMock(wraps=backend.grammar_compiler)
        key = _model_tag(
            [
                _tool(
                    "ls", {"path": {"type": "string", "minLength": 5, "maxLength": 1}}
                ),
                _tool("cat", {"file": _string(LOOKAHEAD)}),
            ]
        )

        result = backend.dispatch_structural_tag(key)

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertEqual(backend.grammar_compiler.compile_structural_tag.call_count, 1)

    def test_schema_with_too_many_patterns_keeps_its_error(self):
        """Checking is quadratic, so a schema with too many patterns keeps its 400."""
        backend = _make_backend(relax=True)
        backend.grammar_compiler = MagicMock(wraps=backend.grammar_compiler)
        cap = xgrammar_backend._MAX_PATTERNS_PER_SCHEMA
        properties = {f"p{i}": _string("^[a-z]+$") for i in range(cap)}
        properties["bad"] = _string(BACKREFERENCE)

        result = backend.dispatch_structural_tag(_model_tag([_tool("ls", properties)]))

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertEqual(backend.grammar_compiler.compile_structural_tag.call_count, 1)

    def test_nothing_to_drop_returns_original_error_without_recompiling(self):
        backend = _make_backend(relax=True)
        backend.grammar_compiler = MagicMock(wraps=backend.grammar_compiler)
        key = _legacy_tag(
            {"type": "object", "patternProperties": {"(?<=a)b": {"type": "string"}}}
        )

        result = backend.dispatch_structural_tag(key)

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertEqual(backend.grammar_compiler.compile_structural_tag.call_count, 1)


if __name__ == "__main__":
    unittest.main()
