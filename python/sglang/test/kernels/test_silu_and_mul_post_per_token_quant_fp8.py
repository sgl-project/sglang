"""Tests for ``silu_and_mul_post_per_token_quant_fp8`` (TP, 2D per-token FP8 E4M3)."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="PPU-only kernel")


def _swiglu_ref(gateup: torch.Tensor, swiglu_limit: float | None) -> torch.Tensor:
    last = gateup.shape[-1]
    assert last % 2 == 0
    half = last // 2
    gate, up = gateup[..., :half], gateup[..., half:]
    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate.float()) * up.float()


def _oai_swiglu_ref(
    gateup: torch.Tensor, alpha: float, clamp_limit: float
) -> torch.Tensor:
    last = gateup.shape[-1]
    assert last % 2 == 0
    half = last // 2
    gate, up = gateup[..., :half].float(), gateup[..., half:].float()
    gate = gate.clamp(max=clamp_limit)
    up = up.clamp(min=-clamp_limit, max=clamp_limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1)


@pytest.mark.parametrize(
    "M,N",
    [(1, 8), (8, 64), (33, 512), (128, 1024), (256, 4096)],
)
@pytest.mark.parametrize("swiglu_limit", [None, 7.0])
def test_correctness(M: int, N: int, swiglu_limit: float | None) -> None:
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_per_token_quant_fp8,
    )

    torch.manual_seed(M * 100 + N)
    gateup = torch.randn((M, 2 * N), dtype=torch.bfloat16, device="cuda")

    out_fp8, out_scale = silu_and_mul_post_per_token_quant_fp8(
        gateup, swiglu_limit=swiglu_limit
    )

    assert out_fp8.shape == (M, N)
    assert out_fp8.dtype == torch.float8_e4m3fn
    assert out_scale.shape == (M, 1)
    assert out_scale.dtype == torch.float32

    ref = _swiglu_ref(gateup, swiglu_limit)
    deq = out_fp8.float() * out_scale.float()

    row_absmax = ref.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    err = (deq - ref).abs() / row_absmax
    assert err.max().item() < 8e-2, f"max relerr = {err.max().item():.4f} exceeds 8%"


def test_empty() -> None:
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_per_token_quant_fp8,
    )

    gateup = torch.empty((0, 64), dtype=torch.bfloat16, device="cuda")
    out_fp8, out_scale = silu_and_mul_post_per_token_quant_fp8(gateup)
    assert out_fp8.shape == (0, 32)
    assert out_scale.shape == (0, 1)


def test_determinism() -> None:
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_per_token_quant_fp8,
    )

    torch.manual_seed(42)
    gateup = torch.randn((64, 1024), dtype=torch.bfloat16, device="cuda")
    a_q, a_s = silu_and_mul_post_per_token_quant_fp8(gateup, swiglu_limit=8.0)
    b_q, b_s = silu_and_mul_post_per_token_quant_fp8(gateup, swiglu_limit=8.0)
    assert torch.equal(a_q.view(torch.uint8), b_q.view(torch.uint8))
    torch.testing.assert_close(a_s, b_s, atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,N",
    [(1, 8), (8, 64), (33, 512), (128, 1024), (256, 4096)],
)
@pytest.mark.parametrize("alpha,clamp", [(1.702, 7.0), (1.0, 8.0), (0.5, 15.0)])
def test_oai_swiglu_correctness(M: int, N: int, alpha: float, clamp: float) -> None:
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_per_token_quant_fp8,
    )

    torch.manual_seed(M * 1000 + N + int(alpha * 10))
    # Scale so a fraction of the values actually hits the clamp limits.
    gateup = torch.randn((M, 2 * N), dtype=torch.bfloat16, device="cuda") * 4.0

    out_fp8, out_scale = silu_and_mul_post_per_token_quant_fp8(
        gateup, gemm1_alpha=alpha, gemm1_clamp_limit=clamp
    )

    assert out_fp8.shape == (M, N)
    assert out_fp8.dtype == torch.float8_e4m3fn
    assert out_scale.shape == (M, 1)
    assert out_scale.dtype == torch.float32

    ref = _oai_swiglu_ref(gateup, alpha, clamp)
    deq = out_fp8.float() * out_scale.float()

    row_absmax = ref.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    err = (deq - ref).abs() / row_absmax
    assert err.max().item() < 8e-2, f"max relerr = {err.max().item():.4f} exceeds 8%"


def test_invalid_args() -> None:
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_post_per_token_quant_fp8,
    )

    gateup = torch.randn((8, 64), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="mutually exclusive"):
        silu_and_mul_post_per_token_quant_fp8(gateup, swiglu_limit=7.0, gemm1_alpha=1.0)
    with pytest.raises(AssertionError, match="requires gemm1_clamp_limit"):
        silu_and_mul_post_per_token_quant_fp8(gateup, gemm1_alpha=1.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
