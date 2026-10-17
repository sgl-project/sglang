"""Tests for DeepSeek-V4 fused norm + RoPE kernels."""

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


@pytest.mark.parametrize("page_size", [1, 16])
def test_fused_k_norm_rope_flashmla_fp8_bytes(page_size):
    """Check the INSTALLED AOT binary, not just an independently JITed helper.

    Unit KV and eps=0 make RMSNorm exactly one. Identity RoPE plus BF16-exact
    weights expose literal FN encoding/rounding cases; each 64-value quant
    group has absmax=448, hence UE8M0 scale=1. This catches stale .hip/.so builds.
    """
    cases = [
        (256.0, 0x78),
        (288.0, 0x79),
        (320.0, 0x7A),
        (352.0, 0x7B),
        (384.0, 0x7C),
        (416.0, 0x7D),
        (448.0, 0x7E),
        (1.0625, 0x38),  # even halfway rounds down
        (1.1875, 0x3A),  # odd halfway rounds up
        (0.0146484375, 0x08),  # largest subnormal carries to normal
        (0.0009765625, 0x00),  # half min-subnormal rounds to even zero
        (0.00146484375, 0x01),  # below min-subnormal but rounds up
        (0.001953125, 0x01),
        (0.0, 0x00),
    ]
    cases += [(-value, code | 0x80) for value, code in cases]
    group = (cases * 3)[:64]
    weights = torch.tensor(
        [value for value, _ in group] * 8, dtype=torch.bfloat16, device="cuda"
    )
    batch = 2
    kv = torch.ones((batch, 512), dtype=torch.bfloat16, device="cuda")
    freqs = torch.tensor([1.0, 0.0] * 32, device="cuda").unsqueeze(0)
    positions = torch.zeros(batch, dtype=torch.int32, device="cuda")
    locations = [0, page_size + page_size - 1]
    out_loc = torch.tensor(locations, dtype=torch.int32, device="cuda")
    page_bytes = ((584 * page_size + 575) // 576) * 576
    sentinel = 0xA5
    cache = torch.full((2 * page_bytes,), sentinel, dtype=torch.uint8, device="cuda")
    sgl_kernel.dsv4_fused_k_norm_rope_flashmla(
        kv, weights, freqs, positions, out_loc, cache, eps=0.0, page_size=page_size
    )
    expected = torch.full((2 * page_bytes,), sentinel, dtype=torch.uint8)
    encoded = torch.tensor([code for _, code in group] * 7, dtype=torch.uint8)
    rope = weights[448:].cpu().view(torch.uint8)
    for location in locations:
        page, offset = divmod(location, page_size)
        start = page * page_bytes + offset * 576
        expected[start : start + 448] = encoded
        expected[start + 448 : start + 576] = rope
        scale = page * page_bytes + 576 * page_size + offset * 8
        expected[scale : scale + 7] = 127
    torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
