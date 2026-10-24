"""
Unit tests for token filter operations (Triton and Torch paths).

Verifies that both implementations produce identical bitmask output
for the same inputs, ensuring parity across GPU and CPU paths.
"""

import unittest

import torch

from sglang.srt.constrained.torch_ops.token_filter_torch_ops import (
    set_token_filter_torch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "base-a-test-cpu")


def _get_allowed_tokens(vocab_mask, batch_idx, max_token_id):
    """Extract allowed token IDs from a bitmask row."""
    allowed = []
    for token_id in range(max_token_id):
        elem = token_id // 32
        bit = token_id % 32
        val = int(vocab_mask[batch_idx, elem].item())
        if val & (1 << bit):
            allowed.append(token_id)
    return allowed


class TestSetTokenFilterTorch(unittest.TestCase):
    """Tests for the Torch token filter implementation."""

    @staticmethod
    def _reference(vocab_mask, token_ids, batch_idx, is_allowed, reset_vocab_mask):
        """Bit-by-bit reference for the cached implementation."""
        row = vocab_mask[batch_idx]
        if reset_vocab_mask:
            row.fill_(-1 if not is_allowed else 0)
        for token_id in token_ids:
            elem, bit = token_id // 32, token_id % 32
            val = int(row[elem].item()) & 0xFFFFFFFF
            val = (val | (1 << bit)) if is_allowed else (val & ~(1 << bit))
            if val >= 2**31:
                val -= 2**32
            row[elem] = val

    def test_repeated_calls_match_reference(self):
        # The cached template and index/bitmask arrays must give the same rows as
        # a bit-by-bit update, on first use and on cache hits, with and without
        # a reset, including tokens in the int32 sign bit and duplicates.
        torch.manual_seed(0)
        lists = [[1, 128815], [31, 63, 31], [0], [5, 7, 9, 200, 201], [4095]]
        for token_ids in lists:
            for is_allowed in (True, False):
                for reset in (True, False):
                    for _ in range(2):  # second pass is a cache hit
                        # 4096 words cover a 131k vocabulary, so 128815 fits.
                        base = torch.randint(
                            -(2**31), 2**31 - 1, (3, 4096), dtype=torch.int32
                        )
                        got, want = base.clone(), base.clone()
                        set_token_filter_torch(
                            got,
                            token_ids,
                            1,
                            is_allowed=is_allowed,
                            reset_vocab_mask=reset,
                        )
                        self._reference(want, token_ids, 1, is_allowed, reset)
                        self.assertTrue(
                            torch.equal(got, want),
                            f"{token_ids=} {is_allowed=} {reset=}",
                        )
                        # Other rows are untouched.
                        self.assertTrue(torch.equal(got[0], base[0]))
                        self.assertTrue(torch.equal(got[2], base[2]))

    def test_cached_template_is_not_aliased_by_the_row(self):
        vocab_mask = torch.zeros((2, 4), dtype=torch.int32)
        set_token_filter_torch(vocab_mask, [3, 5], 0, is_allowed=False)
        vocab_mask[0].fill_(0)  # caller scribbles over its row
        set_token_filter_torch(vocab_mask, [3, 5], 1, is_allowed=False)
        allowed = _get_allowed_tokens(vocab_mask, 1, 128)
        self.assertEqual(allowed, [t for t in range(128) if t not in (3, 5)])

    def test_pinned_host_row(self):
        if not torch.cuda.is_available():
            self.skipTest("pinned memory needs CUDA")
        vocab_mask = torch.zeros((1, 4), dtype=torch.int32, pin_memory=True)
        set_token_filter_torch(vocab_mask, [0, 5, 31, 32, 63], 0, is_allowed=True)
        self.assertEqual(_get_allowed_tokens(vocab_mask, 0, 64), [0, 5, 31, 32, 63])

    def test_allow_tokens_from_blank_mask(self):
        vocab_mask = torch.zeros((1, 4), dtype=torch.int32)  # 128 tokens
        set_token_filter_torch(vocab_mask, [0, 5, 31, 32, 63], 0, is_allowed=True)

        allowed = _get_allowed_tokens(vocab_mask, 0, 64)
        self.assertEqual(allowed, [0, 5, 31, 32, 63])

    def test_block_tokens_from_full_mask(self):
        vocab_mask = torch.full((1, 4), -1, dtype=torch.int32)  # all bits set
        set_token_filter_torch(
            vocab_mask, [3, 5], 0, is_allowed=False, reset_vocab_mask=False
        )

        allowed = _get_allowed_tokens(vocab_mask, 0, 64)
        self.assertNotIn(3, allowed)
        self.assertNotIn(5, allowed)
        self.assertIn(0, allowed)
        self.assertIn(1, allowed)

    def test_reset_then_allow(self):
        vocab_mask = torch.full((1, 2), -1, dtype=torch.int32)
        set_token_filter_torch(
            vocab_mask, [7], 0, is_allowed=True, reset_vocab_mask=True
        )

        allowed = _get_allowed_tokens(vocab_mask, 0, 64)
        self.assertEqual(allowed, [7])

    def test_reset_then_block(self):
        vocab_mask = torch.zeros((1, 2), dtype=torch.int32)
        set_token_filter_torch(
            vocab_mask, [3, 5], 0, is_allowed=False, reset_vocab_mask=True
        )

        allowed = _get_allowed_tokens(vocab_mask, 0, 64)
        self.assertNotIn(3, allowed)
        self.assertNotIn(5, allowed)
        # All other tokens should be allowed (reset to -1 for block mode)
        self.assertIn(0, allowed)
        self.assertIn(7, allowed)

    def test_empty_token_list(self):
        vocab_mask = torch.zeros((1, 2), dtype=torch.int32)
        set_token_filter_torch(
            vocab_mask, [], 0, is_allowed=True, reset_vocab_mask=True
        )

        allowed = _get_allowed_tokens(vocab_mask, 0, 64)
        self.assertEqual(allowed, [])

    def test_batch_indexing(self):
        vocab_mask = torch.zeros((3, 2), dtype=torch.int32)
        set_token_filter_torch(vocab_mask, [1], 0, is_allowed=True)
        set_token_filter_torch(vocab_mask, [2], 1, is_allowed=True)
        set_token_filter_torch(vocab_mask, [3], 2, is_allowed=True)

        self.assertEqual(_get_allowed_tokens(vocab_mask, 0, 64), [1])
        self.assertEqual(_get_allowed_tokens(vocab_mask, 1, 64), [2])
        self.assertEqual(_get_allowed_tokens(vocab_mask, 2, 64), [3])


if __name__ == "__main__":
    unittest.main()
