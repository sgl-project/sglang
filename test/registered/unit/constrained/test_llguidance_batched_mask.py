"""Bitwise parity tests for llguidance's regular-decode batched mask fill."""

import unittest
from unittest.mock import MagicMock

import torch
from llguidance import LLTokenizer, grammar_from

from sglang.srt.constrained.base_grammar_backend import GrammarRow
from sglang.srt.constrained.llguidance_backend import GuidanceBackend, GuidanceGrammar
from sglang.srt.runtime_context import get_resources
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

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
