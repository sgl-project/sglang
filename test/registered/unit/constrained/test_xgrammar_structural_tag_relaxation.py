"""Unit tests for relaxing structural tag schemas that XGrammar cannot compile."""

import copy
import json
import unittest
from unittest.mock import MagicMock, patch

import xgrammar as xgr

from sglang.srt.constrained import xgrammar_backend
from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import (
    XGrammarGrammar,
    XGrammarGrammarBackend,
    _relax_schema_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")

LOOKAHEAD = r"^(?!\.)[a-z]+$"
BACKREFERENCE = r"(a)\1"


def _make_backend(*, relax: bool) -> XGrammarGrammarBackend:
    vocab = [chr(i) for i in range(32, 127)] + ["\n", "<eos>"]
    tokenizer_info = xgr.TokenizerInfo(
        vocab, xgr.VocabType.RAW, stop_token_ids=[len(vocab) - 1]
    )
    backend = object.__new__(XGrammarGrammarBackend)
    backend.grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    backend.vocab_size = len(vocab)
    backend.override_stop_tokens = None
    backend.relax_structural_tag_schemas = relax
    return backend


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


def _qwen3_coder_tag(tools: list) -> str:
    return xgr.get_model_structural_tag(
        model="qwen_3_coder", tools=tools, tool_choice="required", reasoning=False
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


class TestStructuralTagSchemaRelaxation(unittest.TestCase):
    def test_flag_off_still_fails(self):
        key = _qwen3_coder_tag(
            [_tool("ls", {"path": {"type": "string", "pattern": LOOKAHEAD}})]
        )

        result = _make_backend(relax=False).dispatch_structural_tag(key)

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertIn("Lookahead", result.error_message)

    def test_native_tag_compiles_and_names_the_tool(self):
        """One tool with an uncompilable pattern must not fail the whole tool set (#45224)."""
        key = _qwen3_coder_tag(
            [
                _tool("ls", {"path": {"type": "string", "pattern": LOOKAHEAD}}),
                _tool("cat", {"file": {"type": "string", "pattern": "^[a-z]+$"}}),
            ]
        )

        with self.assertLogs(xgrammar_backend.logger, level="WARNING") as logs:
            result = _make_backend(relax=True).dispatch_structural_tag(key)

        self.assertIsInstance(result, XGrammarGrammar)
        self.assertIn("<function=ls>", logs.output[0])
        self.assertIn("dropped pattern at #/properties/path", logs.output[0])
        self.assertNotIn("<function=cat>", logs.output[0])

    def test_only_failing_patterns_are_dropped(self):
        tag = json.loads(
            _qwen3_coder_tag(
                [
                    _tool(
                        "ls",
                        {
                            "path": {"type": "string", "pattern": LOOKAHEAD},
                            "mode": {"type": "string", "pattern": "^[rw]$"},
                        },
                    ),
                    _tool(
                        "cat", {"file": {"type": "string", "pattern": BACKREFERENCE}}
                    ),
                    _tool("rm", {"file": {"type": "string", "pattern": "^[a-z]+$"}}),
                ]
            )
        )
        before = copy.deepcopy(tag)

        changes = _relax_schema_slots(tag)

        schemas = [t["content"]["json_schema"] for t in tag["format"]["tags"]]
        self.assertNotIn("pattern", schemas[0]["properties"]["path"])
        self.assertEqual(schemas[0]["properties"]["mode"]["pattern"], "^[rw]$")
        self.assertNotIn("pattern", schemas[1]["properties"]["file"])
        self.assertEqual(
            schemas[2], before["format"]["tags"][2]["content"]["json_schema"]
        )
        self.assertEqual(len(changes), 2)

    def test_legacy_tag_is_relaxed(self):
        key = _legacy_tag(
            {
                "type": "object",
                "properties": {"name": {"type": "string", "pattern": BACKREFERENCE}},
            }
        )

        self.assertIsInstance(
            _make_backend(relax=False).dispatch_structural_tag(key),
            InvalidGrammarObject,
        )
        with self.assertLogs(xgrammar_backend.logger, level="WARNING"):
            result = _make_backend(relax=True).dispatch_structural_tag(key)
        self.assertIsInstance(result, XGrammarGrammar)

    def test_schema_left_unconstrained_when_dropping_patterns_is_not_enough(self):
        tag = json.loads(
            _legacy_tag(
                {"type": "object", "patternProperties": {"(?<=a)b": {"type": "string"}}}
            )
        )

        changes = _relax_schema_slots(tag)

        self.assertEqual(tag["structures"][0]["schema"], {})
        self.assertEqual(len(changes), 1)
        self.assertIn("left unconstrained", changes[0])

    def test_no_relaxable_schema_returns_original_error_without_recompiling(self):
        backend = _make_backend(relax=True)
        backend.grammar_compiler = MagicMock(wraps=backend.grammar_compiler)
        key = _qwen3_coder_tag(
            [_tool("ls", {"path": {"type": "string", "pattern": LOOKAHEAD}})]
        )

        with patch.object(xgrammar_backend, "_schema_slot_compiles", return_value=True):
            result = backend.dispatch_structural_tag(key)

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertIn("Lookahead", result.error_message)
        self.assertEqual(backend.grammar_compiler.compile_structural_tag.call_count, 1)

    def test_relaxation_compiles_with_the_tokenizer_at_most_twice(self):
        backend = _make_backend(relax=True)
        backend.grammar_compiler = MagicMock(wraps=backend.grammar_compiler)
        tools = [
            _tool(f"t{i}", {"x": {"type": "string", "pattern": "^[a-z]+$"}})
            for i in range(50)
        ]
        tools.append(_tool("bad", {"x": {"type": "string", "pattern": LOOKAHEAD}}))

        with self.assertLogs(xgrammar_backend.logger, level="WARNING"):
            result = backend.dispatch_structural_tag(_qwen3_coder_tag(tools))

        self.assertIsInstance(result, XGrammarGrammar)
        self.assertEqual(backend.grammar_compiler.compile_structural_tag.call_count, 2)


if __name__ == "__main__":
    unittest.main()
