"""
Unit tests for the device-agnostic fallback in
sglang.srt.constrained.xgrammar_backend.apply_vocab_mask.

Devices with no fused token-bitmask kernel (cpu, tpu, ...) used to raise
`RuntimeError: Unsupported device: <type>`, which killed the scheduler on the
first structured-output request. They now go through xgrammar's device-agnostic
kernel. These tests run on CPU, which takes that same fallback branch.

Usage:
    python -m pytest test_xgrammar_vocab_mask_fallback.py -v
"""

import math
import unittest

import torch

from sglang.srt.constrained.xgrammar_backend import (
    XGrammarGrammar,
    XGrammarGrammarBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "base-a-test-cpu")


def _pack_allowed(allowed_per_row, vocab_size):
    """Pack per-row allowed token ids into xgrammar's int32 bitmask (set bit = keep)."""
    num_words = math.ceil(vocab_size / 32)
    mask = torch.zeros((len(allowed_per_row), num_words), dtype=torch.int32)
    for row, allowed in enumerate(allowed_per_row):
        for token_id in allowed:
            mask[row, token_id // 32] |= 1 << (token_id % 32)
    return mask


def _expected(logits, allowed_per_row):
    out = logits.clone()
    for row, allowed in enumerate(allowed_per_row):
        for token_id in range(logits.shape[-1]):
            if token_id not in allowed:
                out[row, token_id] = float("-inf")
    return out


class TestXGrammarVocabMaskFallback(unittest.TestCase):
    def test_grammar_object_masks_disallowed_tokens(self):
        vocab_size = 40
        allowed_per_row = [[0, 5, 33], [1, 39]]
        vocab_mask = _pack_allowed(allowed_per_row, vocab_size)

        torch.manual_seed(0)
        logits = torch.randn(len(allowed_per_row), vocab_size)
        expected = _expected(logits, allowed_per_row)

        grammar = XGrammarGrammar.__new__(XGrammarGrammar)
        grammar.apply_vocab_mask(logits, vocab_mask)

        torch.testing.assert_close(logits, expected)

    def test_backend_static_method_masks_disallowed_tokens(self):
        vocab_size = 64
        allowed_per_row = [[5, 6, 7, 8]]
        vocab_mask = _pack_allowed(allowed_per_row, vocab_size)

        logits = torch.zeros(1, vocab_size)
        logits[0, 16] = 22.125
        logits[0, 5] = 10.0

        XGrammarGrammarBackend.apply_vocab_mask(logits, vocab_mask)

        self.assertFalse(torch.isfinite(logits[0, 16]))
        self.assertEqual(int(torch.argmax(logits[0]).item()), 5)


if __name__ == "__main__":
    unittest.main()
