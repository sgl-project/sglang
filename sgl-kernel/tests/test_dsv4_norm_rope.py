"""Tests for DeepSeek-V4 fused norm + RoPE kernels."""

import math

import pytest
import sgl_kernel
import torch


def _ref_rmsnorm_self(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference: RMSNorm without weight (identity weight)."""
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms).to(x.dtype)


def _ref_rope_interleaved(
    x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    """Reference: apply RoPE to the last `rope_dim` elements (interleaved re/im)."""
    out = x.clone()
    B = x.size(0)
    head_dim = x.size(-1)
    nope_dim = head_dim - rope_dim

    for b in range(B):
        pos = positions[b].item()
        freq = freqs_cis[pos]  # (rope_dim,) interleaved [re0, im0, re1, im1, ...]
        rope_part = out[b, ..., nope_dim:].float()
        # Reshape to pairs
        pairs = rope_part.reshape(*rope_part.shape[:-1], rope_dim // 2, 2)
        x_real = pairs[..., 0]
        x_imag = pairs[..., 1]
        freq_pairs = freq.reshape(rope_dim // 2, 2)
        f_real = freq_pairs[:, 0]
        f_imag = freq_pairs[:, 1]
        rot_real = x_real * f_real - x_imag * f_imag
        rot_imag = x_real * f_imag + x_imag * f_real
        result = torch.stack([rot_real, rot_imag], dim=-1).reshape(rope_part.shape)
        out[b, ..., nope_dim:] = result.to(x.dtype)
    return out


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("head_dim", [128, 192])
def test_fused_q_norm_rope_correctness(batch_size, num_heads, head_dim):
    """Test Q norm + rope against reference."""
    torch.manual_seed(42)
    rope_dim = 64
    max_pos = 512
    eps = 1e-6

    q_input = torch.randn(
        batch_size, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    freqs_cis = torch.randn(max_pos, rope_dim, dtype=torch.float32, device="cuda")
    positions = torch.randint(
        0, max_pos, (batch_size,), dtype=torch.int32, device="cuda"
    )

    q_output = sgl_kernel.dsv4_fused_q_norm_rope(q_input, freqs_cis, positions, eps)

    # Reference
    normed = _ref_rmsnorm_self(q_input, eps)
    expected = _ref_rope_interleaved(normed, freqs_cis, positions, rope_dim)

    torch.testing.assert_close(q_output.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_fused_q_norm_rope_zero_batch():
    """Empty batch should not crash."""
    q_input = torch.empty(0, 8, 192, dtype=torch.bfloat16, device="cuda")
    freqs_cis = torch.randn(512, 64, dtype=torch.float32, device="cuda")
    positions = torch.empty(0, dtype=torch.int32, device="cuda")
    q_output = sgl_kernel.dsv4_fused_q_norm_rope(q_input, freqs_cis, positions)
    assert q_output.shape == q_input.shape


def test_fused_q_norm_rope_preallocated_output():
    """Test with pre-allocated output tensor."""
    torch.manual_seed(42)
    B, H, D = 4, 8, 192
    q_input = torch.randn(B, H, D, dtype=torch.bfloat16, device="cuda")
    freqs_cis = torch.randn(512, 64, dtype=torch.float32, device="cuda")
    positions = torch.randint(0, 512, (B,), dtype=torch.int32, device="cuda")
    q_output = torch.empty_like(q_input)

    result = sgl_kernel.dsv4_fused_q_norm_rope(
        q_input, freqs_cis, positions, q_output=q_output
    )
    assert result is q_output


@pytest.mark.parametrize("batch_size", [1, 8])
def test_fused_q_indexer_rope_hadamard_quant_runs(batch_size):
    """Smoke test: kernel runs without errors and produces finite results."""
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
