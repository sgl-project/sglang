"""Tests for DeepSeek-V4 fused q_indexer_rope_hadamard_quant kernels."""

import pytest
import torch

from sglang.jit_kernel.dsv4.elementwise import fused_q_indexer_rope_hadamard_quant
from sglang.jit_kernel.dsv4.fused_q_indexer_rope_hadamard_quant_torch import (
    fused_q_indexer_rope_hadamard_quant_torch,
)
from sglang.srt.utils import get_device


@pytest.mark.parametrize("batch_size", [1, 8])
def test_fused_q_indexer_rope_hadamard_quant_runs(batch_size):
    """Smoke test: kernel runs without errors and produces finite results."""
    torch.manual_seed(42)
    num_heads = 4
    head_dim = 128
    rope_dim = 64
    max_pos = 256

    q_input = torch.randn(
        batch_size, num_heads, head_dim, dtype=torch.bfloat16, device=get_device()
    )
    weight = torch.randn(
        batch_size, num_heads, dtype=torch.bfloat16, device=get_device()
    )
    freqs_cis = torch.randn(
        max_pos, rope_dim // 2, dtype=torch.complex64, device=get_device()
    )
    positions = torch.randint(
        0, max_pos, (batch_size,), dtype=torch.int32, device=get_device()
    )
    weight_scale = 0.5

    q_fp8, weights_out = fused_q_indexer_rope_hadamard_quant(
        q_input, weight, weight_scale, freqs_cis, positions
    )

    # Reference
    ref_q_fp8 = torch.empty(
        q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device
    )
    ref_weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    fused_q_indexer_rope_hadamard_quant_torch(
        q_input,
        ref_q_fp8,
        weight,
        ref_weights_out,
        float(weight_scale),
        freqs_real,
        positions,
    )

    assert torch.isfinite(weights_out).all(), "weights_out contains non-finite values"
    torch.testing.assert_close(q_fp8, ref_q_fp8)
    torch.testing.assert_close(weights_out, ref_weights_out)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
