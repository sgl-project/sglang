"""CPU tests for linear-attention prefill graph input trimming."""

import unittest

import torch

from sglang.srt.layers.radix_linear_attention import (
    _trim_linear_attention_gate_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestUnifiedLinearAttentionInputTrimming(unittest.TestCase):
    def test_kda_trims_singleton_batch_gate_token_dimension(self):
        a, b = _trim_linear_attention_gate_tokens(
            torch.empty(1, 4, 2, 3), torch.empty(1, 4, 2), 3
        )

        self.assertEqual(a.shape, (1, 3, 2, 3))
        self.assertEqual(b.shape, (1, 3, 2))

    def test_gdn_keeps_token_leading_trimming(self):
        a, b = _trim_linear_attention_gate_tokens(
            torch.empty(4, 2), torch.empty(4, 2), 3
        )

        self.assertEqual(a.shape, (3, 2))
        self.assertEqual(b.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
