"""Tests for the SM90 fused mega-MoE pre-dispatch kernel.

The kernel is the SM90 (Hopper) variant of `mega_moe_pre_dispatch` and is
expected to match the legacy fallback's topk writes exactly and its FP8
quantization numerically within FP8 rounding tolerance:

    1) sglang_per_token_group_quant_fp8(x, group_size=128, scale_ue8m0=False)
       -> writes buf.x[:M], buf.x_sf[:M]
    2) buf.topk_idx[:M].copy_(topk_idx); buf.topk_weights[:M].copy_(weights * alpha)
    3) buf.topk_idx[M:].fill_(-1); buf.topk_weights[M:].zero_()

with the additional capability of folding a `routed_scaling_factor` (= alpha)
into the `buf.topk_weights` write so the caller can drop the post-mega
`y.mul_(alpha)` kernel.
"""

from __future__ import annotations

import pytest
import torch


def _has_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


pytestmark = pytest.mark.skipif(
    not _has_hopper(),
    reason="SM90 mega-moe pre-dispatch requires a Hopper (sm_90) GPU",
)


def _reference_pre_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    padded_max: int,
    routed_scaling_factor: float,
):
    """Mirror `_run_mega_routed`'s non-fused SM90 path."""
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    num_tokens, hidden = x.shape
    top_k = topk_idx.shape[1]
    num_groups = hidden // 128

    buf_x = torch.empty(
        (padded_max, hidden), dtype=torch.float8_e4m3fn, device=x.device
    )
    buf_x_sf = torch.empty(
        (padded_max, num_groups), dtype=torch.float32, device=x.device
    )
    buf_topk_idx = torch.empty(
        (padded_max, top_k), dtype=torch.int64, device=x.device
    )
    buf_topk_weights = torch.empty(
        (padded_max, top_k), dtype=torch.float32, device=x.device
    )

    if num_tokens > 0:
        x_fp8, x_sf = sglang_per_token_group_quant_fp8(
            x, group_size=128, scale_ue8m0=False
        )
        buf_x[:num_tokens].copy_(x_fp8)
        buf_x_sf[:num_tokens].copy_(x_sf)
        buf_topk_idx[:num_tokens].copy_(topk_idx)
        buf_topk_weights[:num_tokens].copy_(topk_weights * routed_scaling_factor)

    if num_tokens < padded_max:
        buf_topk_idx[num_tokens:].fill_(-1)
        buf_topk_weights[num_tokens:].zero_()

    return buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights


def _run_kernel(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    padded_max: int,
    routed_scaling_factor: float,
):
    from sglang.jit_kernel.deepseek_v4 import mega_moe_pre_dispatch_sm90

    num_tokens, hidden = x.shape
    top_k = topk_idx.shape[1]
    num_groups = hidden // 128

    buf_x = torch.empty(
        (padded_max, hidden), dtype=torch.float8_e4m3fn, device=x.device
    )
    buf_x_sf = torch.empty(
        (padded_max, num_groups), dtype=torch.float32, device=x.device
    )
    buf_topk_idx = torch.empty(
        (padded_max, top_k), dtype=torch.int64, device=x.device
    )
    buf_topk_weights = torch.empty(
        (padded_max, top_k), dtype=torch.float32, device=x.device
    )

    # Poison the padded region so we can confirm the kernel actually wrote it.
    if num_tokens < padded_max:
        buf_topk_idx[num_tokens:].fill_(0x4242)
        buf_topk_weights[num_tokens:].fill_(float("nan"))

    if num_tokens > 0:
        topk_idx_in = topk_idx
        topk_weights_in = topk_weights
    else:
        topk_idx_in = x.new_empty((0, top_k), dtype=torch.int32)
        topk_weights_in = x.new_empty((0, top_k), dtype=torch.float32)

    mega_moe_pre_dispatch_sm90(
        x if num_tokens > 0 else x.new_empty((0, hidden), dtype=x.dtype),
        topk_idx_in,
        topk_weights_in,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        quant_group_size=128,
    )
    return buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights


@pytest.mark.parametrize(
    "num_tokens,hidden,top_k,padded_max,scale",
    [
        (0, 2048, 8, 32, 1.0),
        (1, 1024, 4, 8, 2.5),
        (7, 2048, 8, 16, 1.0),
        (7, 2048, 8, 16, 2.5),
        (32, 4096, 8, 32, 2.5),
        (128, 7168, 8, 256, 2.5),
        (128, 7168, 8, 128, 1.0),  # exact-fill (no padding rows)
    ],
)
def test_sm90_pre_dispatch_matches_reference(
    num_tokens: int,
    hidden: int,
    top_k: int,
    padded_max: int,
    scale: float,
):
    assert hidden % 128 == 0 and hidden % 8 == 0
    assert num_tokens <= padded_max
    device = torch.device("cuda")
    torch.manual_seed(0xC0FFEE ^ num_tokens ^ hidden ^ top_k)

    # Mix of small and large magnitudes so absmax / clamp logic is exercised.
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device) * 4.0
    topk_idx = torch.randint(
        0, 256, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    topk_weights = torch.randn(
        num_tokens, top_k, dtype=torch.float32, device=device
    )

    ref = _reference_pre_dispatch(x, topk_idx, topk_weights, padded_max, scale)
    out = _run_kernel(x, topk_idx, topk_weights, padded_max, scale)

    ref_x, ref_sf, ref_idx, ref_w = ref
    out_x, out_sf, out_idx, out_w = out

    # Note: neither the kernel nor the reference initializes buf_x / buf_x_sf
    # past `num_tokens`, so we only compare the valid prefix for those.
    if num_tokens > 0:
        # Scales use the same formula as the reference, but the fused kernel and
        # reference quantizer can differ by a few fp32 ULPs.
        torch.testing.assert_close(
            out_sf[:num_tokens], ref_sf[:num_tokens], rtol=1e-6, atol=0
        )
        # FP8 codes: the reference uses `y / y_s` while the CUDA kernel uses
        # `y * (1 / y_s)`, which can differ by 1 ULP at quantization boundaries.
        # Compare the dequantized values instead.
        num_groups = x.shape[1] // 128

        def _dequant(buf_x, buf_sf):
            return (
                buf_x[:num_tokens]
                .float()
                .reshape(num_tokens, num_groups, 128)
                * buf_sf[:num_tokens].unsqueeze(-1)
            )

        out_deq = _dequant(out_x, out_sf)
        ref_deq = _dequant(ref_x, ref_sf)
        # Two outputs differ by at most one fp8-e4m3 mantissa step (1/8 of the
        # local magnitude).
        torch.testing.assert_close(out_deq, ref_deq, rtol=1.0 / 8, atol=0)
        # Sanity: dequantized values must approximate the original input.
        torch.testing.assert_close(
            out_deq,
            x.float().reshape(num_tokens, num_groups, 128),
            rtol=0.25,
            atol=1e-4,
        )

    # topk_idx / topk_weights are written for the entire padded buffer.
    torch.testing.assert_close(out_idx, ref_idx, rtol=0, atol=0)
    torch.testing.assert_close(out_w, ref_w, rtol=1e-6, atol=0)


def test_sm90_pre_dispatch_alpha_one_equals_unscaled():
    """`routed_scaling_factor=1.0` must reproduce the original (unfused) output."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    M, P, H, K = 5, 16, 2048, 8
    x = torch.randn(M, H, dtype=torch.bfloat16, device=device)
    topk_idx = torch.randint(0, 256, (M, K), dtype=torch.int32, device=device)
    topk_weights = torch.randn(M, K, dtype=torch.float32, device=device)

    out = _run_kernel(x, topk_idx, topk_weights, P, routed_scaling_factor=1.0)
    _, _, _, out_w = out

    # Valid rows: bit-equal to the input weights (single fp32 store).
    torch.testing.assert_close(out_w[:M], topk_weights, rtol=0, atol=0)
    # Padded rows: zeros (kernel-emitted, overwriting the NaN poison).
    assert torch.all(out_w[M:] == 0.0)


def test_sm90_pre_dispatch_zero_tokens_only_pads():
    """num_tokens=0 must still produce a fully padded buffer."""
    device = torch.device("cuda")
    P, H, K = 8, 1024, 4
    x = torch.empty(0, H, dtype=torch.bfloat16, device=device)
    topk_idx = torch.empty(0, K, dtype=torch.int32, device=device)
    topk_weights = torch.empty(0, K, dtype=torch.float32, device=device)

    _, _, idx, w = _run_kernel(x, topk_idx, topk_weights, P, routed_scaling_factor=2.5)
    assert torch.all(idx == -1)
    assert torch.all(w == 0.0)


def test_sm90_pre_dispatch_no_padding_branch():
    """num_tokens == padded_max: kernel must not launch any pad blocks
    yet still produce a valid buffer for the active rows."""
    device = torch.device("cuda")
    M = P = 16
    H, K = 2048, 8
    torch.manual_seed(11)
    x = torch.randn(M, H, dtype=torch.bfloat16, device=device)
    topk_idx = torch.randint(0, 256, (M, K), dtype=torch.int32, device=device)
    topk_weights = torch.randn(M, K, dtype=torch.float32, device=device)

    ref = _reference_pre_dispatch(x, topk_idx, topk_weights, P, 1.0)
    out = _run_kernel(x, topk_idx, topk_weights, P, 1.0)
    ref_x, ref_sf, ref_idx, ref_w = ref
    out_x, out_sf, out_idx, out_w = out
    # Scales use the same formula as the reference, with fp32 ULP-level drift.
    torch.testing.assert_close(out_sf, ref_sf, rtol=1e-6, atol=0)
    # topk fully written.
    torch.testing.assert_close(out_idx, ref_idx, rtol=0, atol=0)
    torch.testing.assert_close(out_w, ref_w, rtol=0, atol=0)
    # FP8 quant matches within one e4m3 mantissa step via dequant.
    num_groups = H // 128
    out_deq = out_x.float().reshape(M, num_groups, 128) * out_sf.unsqueeze(-1)
    ref_deq = ref_x.float().reshape(M, num_groups, 128) * ref_sf.unsqueeze(-1)
    torch.testing.assert_close(out_deq, ref_deq, rtol=1.0 / 8, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
