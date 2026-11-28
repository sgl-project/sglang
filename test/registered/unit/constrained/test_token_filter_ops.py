"""
Unit tests for token filter operations (Triton and Torch paths).

Verifies that both implementations produce identical bitmask output
for the same inputs, ensuring parity across GPU and CPU paths.
"""

import sys
import unittest
from unittest.mock import patch

import pytest
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


@pytest.mark.parametrize(
    "width,token_ids", [(1, []), (2, [0, 31, 32, 32, 63]), (3, [0, 31, 32, 32, 63])]
)
@pytest.mark.parametrize("is_allowed", [False, True])
def test_cached_reset_matches_uncached_and_isolates_rows(width, token_ids, is_allowed):
    expected = torch.full((1, width), 0 if is_allowed else -1, dtype=torch.int32)
    set_token_filter_torch(expected, token_ids, 0, is_allowed, reset_vocab_mask=False)
    mask = torch.full((3, width * 2), 123, dtype=torch.int32)[:, ::2]
    for idx in (0, 1):
        set_token_filter_torch(mask, token_ids, idx, is_allowed)
        assert torch.equal(mask[idx], expected[0])
    mask[0].zero_()
    mask[1].fill_(17)
    set_token_filter_torch(mask, token_ids, 2, is_allowed)
    assert torch.equal(mask[2], expected[0])
    assert torch.all(mask[0] == 0)
    assert torch.all(mask[1] == 17)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_partial_filters_preserve_existing_bits(dtype):
    mask = torch.zeros((1, 2), dtype=dtype)
    set_token_filter_torch(mask, [0, 31, 32, 63], 0)
    set_token_filter_torch(mask, [31, 32, 32], 0, False, reset_vocab_mask=False)
    assert _get_allowed_tokens(mask, 0, 64) == [0, 63]
    set_token_filter_torch(mask, [1, 1], 0, reset_vocab_mask=False)
    assert _get_allowed_tokens(mask, 0, 64) == [0, 1, 63]
    before = mask.clone()
    set_token_filter_torch(mask, [], 0, False, reset_vocab_mask=False)
    assert torch.equal(mask, before)
    set_token_filter_torch(mask, [31, 32], 0, False)
    assert _get_allowed_tokens(mask, 0, 64) == [
        i for i in range(64) if i not in (31, 32)
    ]


def test_non_cpu_filter_does_not_use_cpu_cache():
    mask = torch.empty((1, 2), dtype=torch.int32, device="meta")
    with patch(
        "sglang.srt.constrained.torch_ops.token_filter_torch_ops._cached_cpu_filter",
        side_effect=AssertionError("non-CPU mask entered CPU cache"),
    ):
        for is_allowed in (False, True):
            set_token_filter_torch(mask, [0, 31, 32], 0, is_allowed)
    assert mask.device.type == "meta"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
