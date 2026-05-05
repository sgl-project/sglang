import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.jit_kernel.diffusion.group_norm_silu import apply_group_norm_silu
from sglang.jit_kernel.diffusion.triton.group_norm_silu import triton_group_norm_silu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
TEST_CASES = [
    pytest.param((2, 64, 32, 32), 32, id="image_2d"),
    pytest.param((1, 64, 4, 16, 16), 32, id="video_3d"),
    pytest.param((4, 128), 32, id="token_2d"),
]
LARGE_TILE_CASE = ((1, 128, 20, 256, 256), 32)


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.bfloat16:
        return 7e-2, 2e-2
    return 3e-3, 3e-3


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    return F.silu(F.group_norm(x, num_groups, weight=weight, bias=bias, eps=eps))


@torch.no_grad()
@pytest.mark.parametrize("shape,num_groups", TEST_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_group_norm_silu(
    shape: tuple[int, ...], num_groups: int, dtype: torch.dtype
) -> None:
    channels = shape[1]
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    weight = torch.randn(channels, device=DEVICE, dtype=dtype)
    bias = torch.randn(channels, device=DEVICE, dtype=dtype)

    actual = triton_group_norm_silu(x, weight, bias, num_groups=num_groups)
    expected = _reference(x, weight, bias, num_groups)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
@pytest.mark.parametrize("shape,num_groups", TEST_CASES[:2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_apply_group_norm_silu(
    shape: tuple[int, ...],
    num_groups: int,
    dtype: torch.dtype,
) -> None:
    norm = nn.GroupNorm(num_groups, shape[1], eps=1e-5, affine=True).to(
        device=DEVICE, dtype=dtype
    )
    activation = nn.SiLU()
    hidden_states = torch.randn(shape, device=DEVICE, dtype=dtype)

    actual = apply_group_norm_silu(hidden_states, norm, activation)
    expected = activation(norm(hidden_states))

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
def test_triton_group_norm_silu_large_tile_bf16() -> None:
    shape, num_groups = LARGE_TILE_CASE
    x = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(shape[1], device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(shape[1], device=DEVICE, dtype=torch.bfloat16)

    actual = triton_group_norm_silu(x, weight, bias, num_groups=num_groups)
    expected = _reference(x, weight, bias, num_groups)

    atol, rtol = _tol(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
