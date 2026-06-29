"""Unit tests for bugfix #26286: top_logprobs tensor crash in PD mode.

detokenize_top_logprobs_tokens crashes with RuntimeError when
token_logprobs_val contains multi-element GPU tensors (no_copy_to_cpu=True).
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    TokenizerManager,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDetokenizeTopLogprobsTensorCrash(CustomTestCase):
    """Test for bug #26286: detokenize_top_logprobs_tokens crashes on multi-element tensor."""

    def setUp(self):
        self.tm = TokenizerManager.__new__(TokenizerManager)
        self.tm.tokenizer = MagicMock()
        self.tm.tokenizer.batch_decode = MagicMock(
            side_effect=lambda x: ["token"] * len(x)
        )

    def test_multi_element_tensor_no_crash(self):
        """Multi-element GPU tensor should not crash with 'Boolean value of Tensor is ambiguous'."""
        token_logprobs_val = [
            torch.tensor([-0.1, -0.2, -0.3]),
            torch.tensor([-0.4]),
        ]
        token_logprobs_idx = [
            [1, 2, 3],
            [4],
        ]
        result = self.tm.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=True
        )
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])

    def test_empty_list_returns_none(self):
        """Empty list entry should return None."""
        token_logprobs_val = [
            [],
            [0.1],
        ]
        token_logprobs_idx = [
            [],
            [5],
        ]
        result = self.tm.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=True
        )
        self.assertIsNone(result[0])
        self.assertIsNotNone(result[1])

    def test_cpu_tensor_list_works(self):
        """Regular CPU float lists should still work as before."""
        token_logprobs_val = [
            [0.1, 0.2],
            [0.3],
        ]
        token_logprobs_idx = [
            [1, 2],
            [3],
        ]
        result = self.tm.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=True
        )
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])

    def test_all_empty_returns_all_none(self):
        """All empty entries should return all None."""
        result = self.tm.detokenize_top_logprobs_tokens(
            [[], []], [[], []], decode_to_text=True
        )
        self.assertEqual(result, [None, None])

    def test_mixed_tensor_and_empty(self):
        """Mix of tensor entries and empty entries."""
        token_logprobs_val = [
            torch.tensor([-0.5, -0.6]),
            [],
            torch.tensor([-0.7]),
        ]
        token_logprobs_idx = [
            [10, 20],
            [],
            [30],
        ]
        result = self.tm.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=True
        )
        self.assertEqual(len(result), 3)
        self.assertIsNotNone(result[0])
        self.assertIsNone(result[1])
        self.assertIsNotNone(result[2])

    def test_decode_to_text_false_skips_tokenizer(self):
        """With decode_to_text=False, tokenizer should not be called."""
        token_logprobs_val = [
            torch.tensor([-0.1, -0.2]),
        ]
        token_logprobs_idx = [
            [1, 2],
        ]
        result = self.tm.detokenize_top_logprobs_tokens(
            token_logprobs_val, token_logprobs_idx, decode_to_text=False
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [(-0.1, 1, None), (-0.2, 2, None)])
        self.tm.tokenizer.batch_decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
