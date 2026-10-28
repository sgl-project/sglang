"""Tests for the DeepSeek-V4 fused indexer Q RoPE + Hadamard + FP8 quant kernel."""

import pytest
import sgl_kernel
import torch


@pytest.mark.parametrize("batch_size", [1, 8])
def test_fused_q_indexer_rope_hadamard_quant_runs(batch_size):
    """Basic launch coverage with finite output checks."""
    torch.manual_seed(42)
    num_heads = 4
    head_dim = 128
    rope_dim = 64
    max_pos = 256

    q_input = torch.randn(
        batch_size, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    q_fp8 = torch.empty(
        batch_size, num_heads, head_dim, dtype=torch.uint8, device="cuda"
    )
    weight = torch.randn(batch_size, num_heads, dtype=torch.bfloat16, device="cuda")
    weights_out = torch.empty(
        batch_size, num_heads, 1, dtype=torch.float32, device="cuda"
    )
    freqs_cis = torch.randn(max_pos, rope_dim, dtype=torch.float32, device="cuda")
    positions = torch.randint(
        0, max_pos, (batch_size,), dtype=torch.int32, device="cuda"
    )
    weight_scale = 0.5

    sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant(
        q_input, q_fp8, weight, weights_out, weight_scale, freqs_cis, positions
    )

    assert torch.isfinite(weights_out).all(), "weights_out contains non-finite values"
    assert q_fp8.any(), "q_fp8 should not be all zeros"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
