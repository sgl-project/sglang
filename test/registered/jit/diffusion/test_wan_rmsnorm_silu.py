import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.jit_kernel.diffusion.triton.wan_rmsnorm_silu import (
    triton_wan_rmsnorm_silu,
)
from sglang.jit_kernel.diffusion.wan_rmsnorm_silu import apply_wan_rmsnorm_silu
from sglang.multimodal_gen.runtime.models.vaes.wanvae import WanRMS_norm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)


DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = [
    pytest.param((1, 96, 3, 8, 8), id="c96"),
    pytest.param((1, 192, 2, 6, 10), id="c192"),
]


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not hasattr(torch, "channels_last_3d"):
        pytest.skip("channels_last_3d required")
    torch.cuda.manual_seed(0)


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 5e-3, 5e-3
    return 1.5e-1, 3e-2


def _make_input(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=dtype).contiguous(
        memory_format=torch.channels_last_3d
    )


def _reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    y = F.normalize(x, dim=1) * (x.shape[1] ** 0.5) * gamma
    if bias is not None:
        y = y + bias
    return F.silu(y)


@torch.no_grad()
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_bias", [False, True])
def test_triton_wan_rmsnorm_silu(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    has_bias: bool,
) -> None:
    x = _make_input(shape, dtype)
    gamma = torch.randn((shape[1], 1, 1, 1), device="cuda", dtype=dtype).contiguous()
    bias = (
        torch.randn((shape[1], 1, 1, 1), device="cuda", dtype=dtype).contiguous()
        if has_bias
        else None
    )

    actual = triton_wan_rmsnorm_silu(x, gamma, bias)
    expected = _reference(x, gamma, bias)

    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    assert actual.stride() == x.stride()
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
def test_apply_wan_rmsnorm_silu_matches_module() -> None:
    dtype = torch.bfloat16
    x = _make_input((1, 96, 3, 8, 8), dtype)
    norm = WanRMS_norm(96, images=False).to(device="cuda", dtype=dtype)
    activation = nn.SiLU()

    actual = apply_wan_rmsnorm_silu(x, norm, activation)
    expected = activation(norm(x))

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
