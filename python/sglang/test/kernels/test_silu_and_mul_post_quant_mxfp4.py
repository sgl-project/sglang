"""Tests for ``silu_and_mul_post_quant_mxfp4`` (TP, 2D MXFP4)."""

import sys

import pytest
import torch

from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="PPU-only kernel")


_TP_SHAPES = [
    # --- (N, H)  N=tokens, H=moe_intermediate_size/tp_size ---
    # N=4096: large prefill
    (4096, 512),  # DS-Flash TP=4
    (4096, 256),  # Qwen TP=4
    # N=2048: medium prefill
    (2048, 1024),  # DS-Flash TP=2
    (2048, 512),  # DS-Flash TP=4
    # N=1024: small prefill
    (1024, 512),  # DS-Flash TP=4
    (1024, 256),  # Qwen TP=4
    # N=512: decode batch
    (512, 512),  # DS-Flash TP=4
    (512, 384),  # DS-Pro TP=8 (non-aligned)
    # N=256: small decode
    (256, 256),  # Qwen TP=4
    (256, 96),  # Minimax TP=16 (non-aligned)
    # N=128: tiny decode
    (128, 128),  # Qwen TP=8
    (128, 48),  # Minimax TP=64 (non-aligned)
    # N=64: minimal
    (64, 64),  # Qwen TP=16
    # N=32: edge case
    (32, 128),  # Qwen TP=8
]


@pytest.mark.parametrize("N,H", _TP_SHAPES)
@pytest.mark.parametrize("swiglu_limit", [None, 4.0, 8.0, 12.0])
def test_shapes(N: int, H: int, swiglu_limit: float | None) -> None:
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_quant_mxfp4,
    )

    torch.manual_seed(N * 17 + H)
    gateup = torch.randn((N, 2 * H), dtype=torch.bfloat16, device="cuda")
    out_quant, out_scale_view = silu_and_mul_post_quant_mxfp4(
        gateup, swiglu_limit=swiglu_limit
    )

    assert out_quant.shape == (N, H // 2)
    assert out_quant.dtype == torch.uint8

    S_valid = (H + 63) // 64
    assert out_scale_view.shape == (N, S_valid)
    assert out_scale_view.dtype == torch.uint16


def test_empty() -> None:
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_quant_mxfp4,
    )

    gateup = torch.empty((0, 2048), dtype=torch.bfloat16, device="cuda")
    out_quant, out_scale_view = silu_and_mul_post_quant_mxfp4(gateup)
    assert out_quant.shape == (0, 512)
    S_valid = (1024 + 63) // 64
    assert out_scale_view.shape == (0, S_valid)


def test_determinism() -> None:
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_quant_mxfp4,
    )

    torch.manual_seed(7)
    gateup = torch.randn((2048, 4096), dtype=torch.bfloat16, device="cuda")
    for swiglu in [4.0, 8.0, 10.0, 12.0]:
        a_q, a_s = silu_and_mul_post_quant_mxfp4(gateup, swiglu_limit=swiglu)
        b_q, b_s = silu_and_mul_post_quant_mxfp4(gateup, swiglu_limit=swiglu)
        assert torch.equal(a_q, b_q), f"quant determinism fail swiglu={swiglu}"
        assert torch.equal(a_s.contiguous(), b_s.contiguous()), (
            f"scale determinism fail swiglu={swiglu}"
        )


def test_clamp_no_op() -> None:
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_quant_mxfp4,
    )

    torch.manual_seed(123)
    gateup = torch.randn((512, 2048), dtype=torch.bfloat16, device="cuda") * 0.1
    a_q, a_s = silu_and_mul_post_quant_mxfp4(gateup, swiglu_limit=None)
    b_q, b_s = silu_and_mul_post_quant_mxfp4(gateup, swiglu_limit=1000.0)
    # With fp32 clamp path, large limit is a no-op for clamp but uses fp32
    # product/absmax/quant, so allow minor e2m1 rounding differences.
    match_q = (a_q == b_q).float().mean().item()
    assert match_q > 0.98, f"quant match {match_q:.4f} < 0.98"
    match_s = (a_s.contiguous() == b_s.contiguous()).float().mean().item()
    assert match_s > 0.98, f"scale match {match_s:.4f} < 0.98"


def test_scale_layout_for_deepgemm() -> None:
    """Verify scale output layout matches DeepGemm grouped GEMM expectations.

    TP path: DeepGemm expects scale shape (N, S) where S = ceil(H/64).
    The returned scale must be viewable as (N, S_valid) after .t().
    """
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_quant_mxfp4,
    )

    # Use DeepSeek V3 shape: N=1024, H=2048
    N, H = 1024, 2048
    torch.manual_seed(42)
    gateup = torch.randn((N, 2 * H), dtype=torch.bfloat16, device="cuda")
    out_quant, out_scale = silu_and_mul_post_quant_mxfp4(gateup)

    S_valid = (H + 63) // 64  # = 32
    assert out_scale.shape == (
        N,
        S_valid,
    ), f"Expected ({N}, {S_valid}), got {out_scale.shape}"
    # Scale should be a transposed view (non-contiguous) for zero-copy
    assert out_scale.stride() == (
        1,
        N,
    ), f"Expected stride (1, {N}), got {out_scale.stride()}"


def test_scale_layout_nonaligned() -> None:
    """Verify scale layout for non-aligned H (Minimax TP=16: H=96)."""
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_quant_mxfp4,
    )

    N, H = 256, 96  # 96/64=1.5, non-aligned
    torch.manual_seed(99)
    gateup = torch.randn((N, 2 * H), dtype=torch.bfloat16, device="cuda")
    out_quant, out_scale = silu_and_mul_post_quant_mxfp4(gateup)

    S_valid = (H + 63) // 64  # = 2
    assert out_scale.shape == (
        N,
        S_valid,
    ), f"Expected ({N}, {S_valid}), got {out_scale.shape}"
    assert out_quant.shape == (N, H // 2)  # = (32, 48)


def test_large_scale() -> None:
    """Large-scale stress test with many tokens."""
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_quant_mxfp4,
    )

    N, H = 4096, 2048
    torch.manual_seed(2024)
    gateup = torch.randn((N, 2 * H), dtype=torch.bfloat16, device="cuda")
    for swiglu in [None, 8.0]:
        out_quant, out_scale = silu_and_mul_post_quant_mxfp4(
            gateup, swiglu_limit=swiglu
        )
        assert out_quant.shape == (N, H // 2)
        S_valid = (H + 63) // 64
        assert out_scale.shape == (N, S_valid)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
