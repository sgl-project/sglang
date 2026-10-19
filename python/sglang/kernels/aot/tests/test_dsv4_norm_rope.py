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
