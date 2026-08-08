import pytest
import torch

from sglang.kernels.ops.diffusion.triton.native_bf16_rmsnorm import (
    rmsnorm_scale,
    rmsnorm_tanh_residual,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")

EPS = 1e-5


def _native_bf16_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    square = (x * x).to(torch.bfloat16)
    mean_square = square.mean(dim=-1, keepdim=True).to(torch.bfloat16)
    rstd = torch.rsqrt((mean_square + EPS).to(torch.bfloat16).float()).to(
        torch.bfloat16
    )
    return ((x * rstd).to(torch.bfloat16) * weight).to(torch.bfloat16)


def test_native_bf16_rmsnorm_rejects_unsupported_inputs():
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    weight = torch.randn(16, dtype=torch.bfloat16)
    modulation = torch.randn(2, 1, 16, dtype=torch.bfloat16)
    residual = torch.randn_like(x)

    assert rmsnorm_scale(x, weight, modulation, EPS) is None
    assert rmsnorm_tanh_residual(x, modulation, residual, weight, EPS) is None
    assert rmsnorm_scale(x, weight[:-1], modulation, EPS) is None
    assert rmsnorm_tanh_residual(x, modulation, residual[..., :-1], weight, EPS) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(1, 32, 2560), (2, 17, 256)])
def test_rmsnorm_scale_matches_native_bf16(shape):
    torch.manual_seed(0)
    batch, _, dim = shape
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(batch, 1, dim, device="cuda", dtype=torch.bfloat16)

    actual = rmsnorm_scale(x, weight, scale, EPS)
    expected = (_native_bf16_rmsnorm(x, weight) * scale).to(torch.bfloat16)

    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(1, 32, 2560), (2, 17, 256)])
def test_rmsnorm_tanh_residual_matches_native_bf16(shape):
    torch.manual_seed(0)
    batch, _, dim = shape
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(batch, 1, dim, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)

    actual = rmsnorm_tanh_residual(x, gate, residual, weight, EPS)
    norm = _native_bf16_rmsnorm(x, weight)
    gated = (torch.tanh(gate.float()).to(torch.bfloat16) * norm).to(torch.bfloat16)
    expected = (residual + gated).to(torch.bfloat16)

    assert actual is not None
    # Triton's exp-based tanh can differ slightly from torch.tanh in BF16.
    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_bf16_rmsnorm_rejects_hidden_size_above_limit():
    dim = 8448
    x = torch.empty(1, 1, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(dim, device="cuda", dtype=torch.bfloat16)
    modulation = torch.empty(1, 1, dim, device="cuda", dtype=torch.bfloat16)
    residual = torch.empty_like(x)

    assert rmsnorm_scale(x, weight, modulation, EPS) is None
    assert rmsnorm_tanh_residual(x, modulation, residual, weight, EPS) is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
