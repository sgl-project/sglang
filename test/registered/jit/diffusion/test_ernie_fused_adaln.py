import sys

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.models.dits.ernie_image import (
    _ernie_norm_scale_shift,
    _ernie_residual_gate_add,
    _ernie_scale_residual_norm_scale_shift,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=18, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    return (1e-5, 1e-5) if dtype == torch.float32 else (5e-2, 5e-2)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ernie_norm_scale_shift_helper(dtype: torch.dtype) -> None:
    x = torch.randn((1, 17, 256), device="cuda", dtype=dtype)
    norm = RMSNorm(256, eps=1e-6).to(device="cuda", dtype=dtype)
    shift = torch.randn((1, 1, 256), device="cuda", dtype=dtype)
    scale = torch.randn((1, 1, 256), device="cuda", dtype=dtype)

    actual = _ernie_norm_scale_shift(x, norm, shift, scale)
    expected = norm(x) * (1 + scale) + shift

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ernie_residual_adaln_helpers(dtype: torch.dtype) -> None:
    residual = torch.randn((1, 17, 256), device="cuda", dtype=dtype)
    update = torch.randn_like(residual)
    norm = RMSNorm(256, eps=1e-6).to(device="cuda", dtype=dtype)
    gate = torch.randn((1, 1, 256), device="cuda", dtype=dtype)
    shift = torch.randn((1, 1, 256), device="cuda", dtype=dtype)
    scale = torch.randn((1, 1, 256), device="cuda", dtype=dtype)

    actual, residual_out = _ernie_scale_residual_norm_scale_shift(
        residual, update, gate, norm, shift, scale
    )
    expected_residual = residual + gate * update
    expected = norm(expected_residual) * (1 + scale) + shift

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(residual_out, expected_residual, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    final = _ernie_residual_gate_add(residual_out, update, gate)
    torch.testing.assert_close(
        final,
        residual_out + update * gate,
        atol=0 if dtype != torch.float32 else atol,
        rtol=0 if dtype != torch.float32 else rtol,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
