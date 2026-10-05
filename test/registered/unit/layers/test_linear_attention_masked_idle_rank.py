import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import radix_linear_attention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _EmptyBatchRejectingBackend:
    """Stand-in with the Blackwell GDN prefill contract: an empty batch raises."""

    def __init__(self):
        self.calls = 0

    def forward(self, *, layer, forward_batch, mixed_qkv, a, b, linear_attn_output):
        self.calls += 1
        if mixed_qkv.shape[0] == 0:
            raise RuntimeError("max_seqlen must be positive, got 0")
        linear_attn_output.fill_(1.0)
        return linear_attn_output


class TestLinearAttentionMaskedIdleRank(CustomTestCase):
    def test_fully_masked_extend_skips_the_kernel(self):
        """An idle DP rank's fully masked extend must not launch linear attention
        on an empty batch, and its output rows stay finite."""
        backend = _EmptyBatchRejectingBackend()
        rows = 4
        output = torch.full((1, rows, 2, 3), float("nan"))
        forward_batch = SimpleNamespace(
            global_num_token_non_padded_cpu=0, out_cache_loc=torch.arange(rows)
        )
        with patch.object(
            radix_linear_attention, "get_attn_backend", return_value=backend
        ):
            radix_linear_attention._linear_attention_with_output_impl(
                mixed_qkv=torch.randn(rows, 8),
                a=torch.randn(rows, 2),
                b=torch.randn(rows, 2),
                output=output,
                attention_layer=None,
                forward_batch=forward_batch,
            )
        self.assertEqual(backend.calls, 0)
        torch.testing.assert_close(output, torch.zeros_like(output))


if __name__ == "__main__":
    unittest.main()
