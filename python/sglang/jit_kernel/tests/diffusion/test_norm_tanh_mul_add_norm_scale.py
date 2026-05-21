import sys

import pytest
import torch

from sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale import (
    fused_norm_tanh_mul_add,
    fused_norm_tanh_mul_add_norm_scale,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=180, suite="nightly-kernel-1-gpu", nightly=True)

BSD_CONFIG = [
    (1, 3648, 3840),  # Z-image
    (1, 4128, 3840),  # Z-image
    (3, 7, 256),  # bound
    (7, 1, 8192),  # bound
]


def _randn(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    elif actual.dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=8e-2)
    else:
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("B,S,D", BSD_CONFIG)
@pytest.mark.parametrize("norm_type", ["rms", "layer"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_norm_tanh_mul_add(
    B: int, S: int, D: int, norm_type: str, dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    eps = 1e-5
    x = _randn(B, S, D, dtype=dtype)
    weight = _randn(D, dtype=dtype)
    bias = _randn(D, dtype=dtype) if norm_type == "layer" else None
    scale = _randn(B, 1, D, dtype=dtype)
    shift = _randn(B, 1, D, dtype=dtype)

    y = fused_norm_tanh_mul_add(x, weight, bias, scale, shift, norm_type, eps)
    if norm_type == "rms":
        normed = torch.rms_norm(x, x.shape[-1:], weight=weight, eps=eps)
    else:
        normed = torch.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=eps)
    ref_y = normed * torch.tanh(scale) + shift
    _assert_close(y, ref_y)


@pytest.mark.parametrize("B,S,D", BSD_CONFIG)
@pytest.mark.parametrize("norm_type", ["rms", "layer"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_norm_tanh_mul_add_norm_scale(
    B: int, S: int, D: int, norm_type: str, dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    eps = 1e-5
    x = _randn(B, S, D, dtype=dtype)
    weight = _randn(D, dtype=dtype)
    bias = _randn(D, dtype=dtype) if norm_type == "layer" else None
    scale = _randn(B, 1, D, dtype=dtype)
    shift = _randn(B, 1, D, dtype=dtype)
    weight2 = _randn(D, dtype=dtype)
    bias2 = _randn(D, dtype=dtype) if norm_type == "layer" else None
    scale2 = _randn(B, 1, D, dtype=dtype)

    y, y2 = fused_norm_tanh_mul_add_norm_scale(
        x, weight, bias, scale, shift, weight2, bias2, scale2, norm_type, eps
    )
    if norm_type == "rms":
        normed = torch.rms_norm(x, x.shape[-1:], weight=weight, eps=eps)
    else:
        normed = torch.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=eps)
    ref_y = normed * torch.tanh(scale) + shift
    if norm_type == "rms":
        normed2 = torch.rms_norm(ref_y, ref_y.shape[-1:], weight=weight2, eps=eps)
    else:
        normed2 = torch.layer_norm(
            ref_y, ref_y.shape[-1:], weight=weight2, bias=bias2, eps=eps
        )
    ref_y2 = normed2 * (1 + scale2)
    _assert_close(y, ref_y)
    _assert_close(y2, ref_y2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
