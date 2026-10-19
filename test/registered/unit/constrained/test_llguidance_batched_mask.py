"""Bitwise parity tests for llguidance's regular-decode batched mask fill."""

import unittest
from unittest.mock import MagicMock

import torch
from llguidance import LLTokenizer, grammar_from

from sglang.srt.constrained.base_grammar_backend import GrammarRow
from sglang.srt.constrained.llguidance_backend import GuidanceBackend, GuidanceGrammar
from sglang.srt.runtime_context import get_resources
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_REGEX = r"[0-9]{1,8}"


class TestLLGuidanceBatchedMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = GuidanceGrammar(
            llguidance_tokenizer=LLTokenizer("byte"),
            serialized_grammar=grammar_from("regex", _REGEX),
        )

    def _fresh(self, n):
        return [self.template.copy() for _ in range(n)]

    def _allocate(self, grammars):
        return grammars[0].allocate_vocab_mask(
            self.template.llguidance_tokenizer.vocab_size, len(grammars), "cpu"
        )

    def _serial(self, grammars):
        mask = self._allocate(grammars)
        for row, grammar in enumerate(grammars):
            if not grammar.finished and not grammar.is_terminated():
                grammar.fill_vocab_mask(mask, row)
        return mask

    def _batched(self, grammars):
        mask = self._allocate(grammars)
        entries = [
            GrammarRow(row=row, grammar=grammar)
            for row, grammar in enumerate(grammars)
            if not grammar.finished and not grammar.is_terminated()
        ]
        grammars[0].fill_vocab_mask_batched(entries, mask)
        return mask

    def test_batched_matches_serial(self):
        for batch_size in (1, 4, 10):
            with self.subTest(batch_size=batch_size):
                serial = self._serial(self._fresh(batch_size))
                batched = self._batched(self._fresh(batch_size))
                self.assertTrue(torch.equal(serial, batched))
                self.assertTrue((batched[0] != -1).any())

    def test_finished_row_stays_all_allow(self):
        serial_grammars = self._fresh(3)
        batched_grammars = self._fresh(3)
        serial_grammars[1].finished = True
        batched_grammars[1].finished = True

        serial = self._serial(serial_grammars)
        batched = self._batched(batched_grammars)

        self.assertTrue(torch.equal(serial, batched))
        self.assertTrue((batched[1] == -1).all())

    def test_termination_survives_scheduler_finished_sync(self):
        tokenizer = self.template.llguidance_tokenizer
        grammar = GuidanceGrammar(
            llguidance_tokenizer=tokenizer,
            serialized_grammar=grammar_from("regex", "ab"),
        )
        for token in tokenizer.tokenize_str("ab"):
            grammar.accept_token(token)
        before_eos = self._serial([grammar])
        grammar.accept_token(tokenizer.eos_token)
        self.assertTrue(grammar.is_terminated())

        # The scheduler writes the request's finish state into grammar.finished;
        # a request that keeps decoding (ignore_eos) must stay unconstrained.
        grammar.finished = False
        self.assertTrue(grammar.is_terminated())
        self.assertTrue((self._batched([grammar]) == -1).all())

        grammar.rollback(1)
        self.assertFalse(grammar.is_terminated())
        self.assertTrue(torch.equal(self._serial([grammar]), before_eos))

    def test_rollback_across_eos_handled_outside_matcher(self):
        # ``ab`` stops ll_matcher after "ab", so the EOS never reaches it; a
        # multi-token rollback must count that EOS once, outside the matcher.
        tokenizer = self.template.llguidance_tokenizer
        a, b = tokenizer.tokenize_str("ab")
        references = []
        for prefix in ([], [a], [a, b]):
            reference = GuidanceGrammar(
                llguidance_tokenizer=tokenizer,
                serialized_grammar=grammar_from("regex", "ab"),
            )
            for token in prefix:
                reference.accept_token(token)
            references.append(self._serial([reference]))

        for steps, expected in ((2, references[1]), (3, references[0])):
            with self.subTest(steps=steps):
                grammar = GuidanceGrammar(
                    llguidance_tokenizer=tokenizer,
                    serialized_grammar=grammar_from("regex", "ab"),
                )
                for token in (a, b, tokenizer.eos_token):
                    grammar.accept_token(token)
                self.assertTrue(grammar.is_terminated())

                grammar.rollback(steps)
                self.assertFalse(grammar.is_terminated())
                self.assertTrue(torch.equal(self._serial([grammar]), expected))

                # The restored grammar accepts the rest of the string again.
                for token in (a, b)[3 - steps :]:
                    grammar.accept_token(token)
                self.assertTrue(torch.equal(self._serial([grammar]), references[2]))

    def test_extensible_grammar_terminates_on_first_eos(self):
        # ``ab+`` can still extend after "ab", so ll_matcher consumes the EOS
        # itself and stops; that first EOS terminates the grammar.
        tokenizer = self.template.llguidance_tokenizer
        grammar = GuidanceGrammar(
            llguidance_tokenizer=tokenizer,
            serialized_grammar=grammar_from("regex", "ab+"),
        )
        a, b = tokenizer.tokenize_str("ab")
        grammar.accept_token(a)
        after_a = self._serial([grammar])
        grammar.accept_token(b)
        after_ab = self._serial([grammar])

        grammar.accept_token(tokenizer.eos_token)
        self.assertTrue(grammar.is_terminated())
        grammar.finished = False
        self.assertTrue((self._batched([grammar]) == -1).all())

        # ll_matcher tracked the EOS, so rollback must not skip a token.
        grammar.rollback(1)
        self.assertFalse(grammar.is_terminated())
        self.assertTrue(torch.equal(self._serial([grammar]), after_ab))

        grammar.accept_token(tokenizer.eos_token)
        grammar.rollback(2)
        self.assertFalse(grammar.is_terminated())
        self.assertTrue(torch.equal(self._serial([grammar]), after_a))

    def test_unsupported_entry_uses_serial_fill(self):
        mask = self._allocate(self._fresh(1))
        fallback = MagicMock()
        fallback.fill_vocab_mask.side_effect = lambda vocab_mask, row: vocab_mask[
            row
        ].zero_()
        entries = [GrammarRow(row=0, grammar=fallback)]

        self.template.fill_vocab_mask_batched(entries, mask)

        fallback.fill_vocab_mask.assert_called_once_with(mask, 0)
        self.assertTrue((mask == 0).all())

    def test_backend_initializes_fixed_mask_buffer(self):
        name = "test_llguidance_vocab_mask"
        get_resources().buffers.pop(name, None)
        backend = object.__new__(GuidanceBackend)
        backend.llguidance_tokenizer = self.template.llguidance_tokenizer

        try:
            mask = backend.initialize_vocab_mask_buffer(
                name=name,
                vocab_size=self.template.llguidance_tokenizer.vocab_size,
                max_rows=4,
                device="cpu",
            )
            same_mask = backend.initialize_vocab_mask_buffer(
                name=name,
                vocab_size=self.template.llguidance_tokenizer.vocab_size,
                max_rows=4,
                device="cpu",
            )

            self.assertEqual(mask.shape[0], 4)
            self.assertEqual(mask.data_ptr(), same_mask.data_ptr())
        finally:
            get_resources().buffers.pop(name, None)


if __name__ == "__main__":
    unittest.main()
