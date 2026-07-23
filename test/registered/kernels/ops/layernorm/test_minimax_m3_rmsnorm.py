# SPDX-License-Identifier: Apache-2.0
"""Reference tests for MiniMax-M3 ROCm Gemma RMSNorm Triton kernels."""

import pytest
import torch

from sglang.srt.utils import is_hip

if not is_hip():
    pytest.skip(
        "MiniMax-M3 Gemma RMSNorm Triton kernels are ROCm-only.",
        allow_module_level=True,
    )
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (  # noqa: E402
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"
EPS = 1e-6


def _gemma_rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f = x.float()
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    out = x_f * torch.rsqrt(variance + EPS)
    out = out * (1.0 + weight.float())
    return out.to(orig_dtype)


@pytest.mark.parametrize("shape", [(1, 512), (64, 6144), (257, 6144)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_gemma_rmsnorm_matches_reference(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    weight = torch.randn(shape[-1], device=DEVICE, dtype=torch.float32)

    got = gemma_rmsnorm(x, weight, EPS)
    ref = _gemma_rmsnorm_ref(x, weight)

    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_gemma_rmsnorm_accepts_strided_2d_input(dtype):
    torch.manual_seed(0)
    base = torch.randn(128, 1024, device=DEVICE, dtype=dtype)
    x = base[:, ::2]
    weight = torch.randn(x.shape[-1], device=DEVICE, dtype=torch.float32)

    assert not x.is_contiguous()
    got = gemma_rmsnorm(x, weight, EPS)
    ref = _gemma_rmsnorm_ref(x, weight)

    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("shape", [(1, 512), (64, 6144), (257, 6144)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_gemma_fused_add_rmsnorm_matches_reference(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    residual = torch.randn(*shape, device=DEVICE, dtype=dtype)
    weight = torch.randn(shape[-1], device=DEVICE, dtype=torch.float32)

    got, residual_out = gemma_fused_add_rmsnorm(x, residual, weight, EPS)
    ref_residual = x + residual
    ref = _gemma_rmsnorm_ref(ref_residual, weight)

    torch.testing.assert_close(residual_out, ref_residual, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
