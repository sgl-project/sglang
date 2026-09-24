"""Bitwise parity tests for xgrammar's batched mask fill."""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from xgrammar import TokenizerInfo

from sglang.srt.constrained import xgrammar_backend
from sglang.srt.constrained.base_grammar_backend import GrammarRow
from sglang.srt.constrained.xgrammar_backend import (
    XGrammarGrammar,
    XGrammarGrammarBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_VOCAB_SIZE = 130
_DOC = '{"name": "Alice", "age": 31}'
_SCHEMA = (
    '{"type": "object", "properties": {"name": {"type": "string"}, '
    '"age": {"type": "integer"}}, "required": ["name", "age"]}'
)


class TestXGrammarBatchedMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ASCII vocab plus EOS keeps the check independent of model files.
        vocab = [chr(i) for i in range(128)] + ["", "</s>"]
        info = TokenizerInfo(vocab, stop_token_ids=[129])
        tokenizer = SimpleNamespace(init_xgrammar=lambda: (info, None))
        cls.backend = XGrammarGrammarBackend(tokenizer, _VOCAB_SIZE)

    def _grammars(self, n, seed=0):
        rng = random.Random(seed)
        grammars = []
        for i in range(n):
            if i % 2:
                grammar = self.backend.dispatch_regex(r"[0-9]{1,6}-[a-z]{2,4}")
                prefix = "12345-abcd"
            else:
                grammar = self.backend.dispatch_json(_SCHEMA)
                prefix = _DOC
            for ch in prefix[: rng.randrange(len(prefix))]:
                grammar.accept_token(ord(ch))
            grammars.append(grammar)
        return grammars

    def _entries(self, grammars, skip=()):
        return [
            GrammarRow(row=row, grammar=grammar)
            for row, grammar in enumerate(grammars)
            if row not in skip
        ]

    def _serial(self, entries, rows):
        mask = XGrammarGrammar.allocate_vocab_mask(None, _VOCAB_SIZE, rows, "cpu")
        for entry in entries:
            entry.grammar.fill_vocab_mask(mask, entry.row)
        return mask

    def _batched(self, entries, rows):
        mask = XGrammarGrammar.allocate_vocab_mask(None, _VOCAB_SIZE, rows, "cpu")
        XGrammarGrammar.fill_vocab_mask_batched(entries, mask)
        return mask

    def test_batched_matches_serial(self):
        gate = xgrammar_backend._BATCH_FILL_MIN_ROWS
        for n in (1, gate - 1, gate, 3 * gate + 5):
            with self.subTest(n=n):
                grammars = self._grammars(n, seed=n)
                skip = set(range(0, n, 7)) if n >= gate + 7 else set()
                entries = self._entries(grammars, skip)
                serial = self._serial(entries, n)
                batched = self._batched(entries, n)
                self.assertTrue(torch.equal(serial, batched))
                for row in skip:
                    self.assertTrue((batched[row] == -1).all())

    def test_unlisted_padding_rows_stay_all_allow(self):
        n = xgrammar_backend._BATCH_FILL_MIN_ROWS
        entries = self._entries(self._grammars(n))
        batched = self._batched(entries, 2 * n)
        self.assertTrue(torch.equal(batched, self._serial(entries, 2 * n)))
        self.assertTrue((batched[n:] == -1).all())

    def test_mixed_batch_uses_serial_fill(self):
        grammars = self._grammars(xgrammar_backend._BATCH_FILL_MIN_ROWS)
        fallback = MagicMock()
        fallback.fill_vocab_mask.side_effect = lambda vocab_mask, row: vocab_mask[
            row
        ].zero_()
        grammars.append(fallback)
        entries = self._entries(grammars)
        mask = self._batched(entries, len(grammars))

        fallback.fill_vocab_mask.assert_called_once_with(mask, len(grammars) - 1)
        self.assertTrue(torch.equal(mask, self._serial(entries, len(grammars))))


if __name__ == "__main__":
    unittest.main()
