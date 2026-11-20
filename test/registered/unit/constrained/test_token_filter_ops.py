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
