# SPDX-License-Identifier: Apache-2.0
"""Reference tests for MiniMax-M3 ROCm Gemma RMSNorm Triton kernels."""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip

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


@pytest.mark.skipif(not is_gfx95_supported(), reason="fp8 emission is gfx950-only")
@pytest.mark.parametrize("m", [1, 128])
@torch.inference_mode()
def test_gemma_fused_add_rmsnorm_fp8_matches_per_token_quant_of_output(m):
    """The fp8 pair must quantize the bf16-rounded output, as the unfused per_token_quant does."""
    torch.manual_seed(0)
    x = torch.randn(m, 6144, device=DEVICE, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(6144, device=DEVICE, dtype=torch.bfloat16) * 0.1

    got, _ = gemma_fused_add_rmsnorm(x, residual, weight, EPS, emit_fp8=True)
    q8, scale = got._fp8_qinput
    ref_scale = got.float().abs().amax(dim=-1, keepdim=True) / 448.0
    ref_q8 = (got.float() / ref_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    torch.testing.assert_close(scale, ref_scale, atol=0, rtol=1e-6)
    torch.testing.assert_close(q8.float(), ref_q8.float(), atol=0, rtol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
