"""Unit tests for sglang.srt.constrained.xgrammar_backend (no server, no model files)."""

import json
import unittest
from types import SimpleNamespace

import torch
from xgrammar import TokenizerInfo

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import (
    XGrammarGrammar,
    XGrammarGrammarBackend,
    _unmask_rows_that_allow_no_token,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(2.0, "base-a-test-cpu")

# Printable ASCII plus one stop token keeps the grammars independent of model files.
_VOCAB = [chr(i) for i in range(128)] + ["<eos>"]
_EOS_ID = len(_VOCAB) - 1


def _make_backend() -> XGrammarGrammarBackend:
    info = TokenizerInfo(_VOCAB, stop_token_ids=[_EOS_ID])
    tokenizer = SimpleNamespace(init_xgrammar=lambda: (info, None))
    return XGrammarGrammarBackend(tokenizer, len(_VOCAB))


class TestGrammarThatAllowsNoToken(CustomTestCase):
    """A grammar whose start state allows no token must be rejected at compile
    time. Applying its mask sets every logit to -inf, and the sampler then hits
    NaN probabilities (a device-side assert in torch.multinomial on CUDA)."""

    def setUp(self):
        self.backend = _make_backend()

    def tearDown(self):
        self.backend.executor.shutdown(wait=True)

    def assert_rejected(self, grammar):
        self.assertIsInstance(grammar, InvalidGrammarObject)
        self.assertIn("allows no token", grammar.error_message)

    def test_json_schema_ref_to_itself_is_rejected(self):
        schema = {"$defs": {"a": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"}
        self.assert_rejected(self.backend.dispatch_json(json.dumps(schema)))

    def test_ebnf_root_refers_only_to_itself_is_rejected(self):
        self.assert_rejected(self.backend.dispatch_ebnf("root ::= root"))

    def test_valid_grammars_still_compile(self):
        grammars = [
            self.backend.dispatch_json(json.dumps({"type": "integer"})),
            self.backend.dispatch_ebnf('root ::= "a" root | "b"'),
            self.backend.dispatch_regex("[0-9]+"),
            # Matches only the empty output: the stop token is the one allowed token.
            self.backend.dispatch_ebnf('root ::= ""'),
        ]
        for grammar in grammars:
            self.assertIsInstance(grammar, XGrammarGrammar)
            self.assertFalse(grammar.allows_no_token())

    def test_terminated_grammar_is_not_reported(self):
        grammar = self.backend.dispatch_ebnf('root ::= "a"')
        grammar.accept_token(ord("a"))
        grammar.accept_token(_EOS_ID)
        self.assertTrue(grammar.is_terminated())
        self.assertFalse(grammar.allows_no_token())


class TestMaskRowThatAllowsNoToken(CustomTestCase):
    """A grammar can also reach a state that allows no token after some output.
    Its mask row must not turn every logit into -inf."""

    def setUp(self):
        self.backend = _make_backend()
        self.vocab_size = len(_VOCAB)

    def tearDown(self):
        self.backend.executor.shutdown(wait=True)

    def test_all_zero_rows_become_unconstrained(self):
        mask = torch.tensor([[0, 0, 0], [5, 0, 0], [-1, -1, -1]], dtype=torch.int32)
        out = _unmask_rows_that_allow_no_token(mask)
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.tolist(), [[-1, -1, -1], [5, 0, 0], [-1, -1, -1]])
        # The caller's mask is left as it was.
        self.assertEqual(mask[0].tolist(), [0, 0, 0])

    def test_state_without_allowed_token_keeps_logits_finite(self):
        # After "a", rule b refers only to itself, so no token is allowed.
        grammar = self.backend.dispatch_ebnf('root ::= "a" b\nb ::= b')
        self.assertIsInstance(grammar, XGrammarGrammar)
        grammar.accept_token(ord("a"))
        self.assertTrue(grammar.allows_no_token())

        mask = grammar.allocate_vocab_mask(self.vocab_size, 1, "cpu")
        grammar.fill_vocab_mask(mask, 0)
        self.assertFalse(bool(mask.any()))

        logits = torch.zeros(1, self.vocab_size)
        grammar.apply_vocab_mask(logits, mask)
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertFalse(bool(torch.isnan(torch.softmax(logits, dim=-1)).any()))

        # The sampled token is still rejected, which aborts only this request.
        with self.assertRaises(ValueError):
            grammar.accept_token(ord("b"))

    def test_constrained_rows_are_still_masked(self):
        grammar = self.backend.dispatch_ebnf('root ::= "a"')
        mask = grammar.allocate_vocab_mask(self.vocab_size, 1, "cpu")
        grammar.fill_vocab_mask(mask, 0)
        logits = torch.zeros(1, self.vocab_size)
        grammar.apply_vocab_mask(logits, mask)
        finite = torch.isfinite(logits[0]).nonzero().flatten().tolist()
        self.assertEqual(finite, [ord("a")])


if __name__ == "__main__":
    unittest.main()
