"""``diffusion.norm``: the FlyDSL fused norm + scale/shift kernels (ROCm).

Split out of ``test_norm.py`` rather than merged with the other norm backends:
FlyDSL is an AMD gfx950-only compiler, so these run on the AMD CI lane and
nothing else in that file does.  Keeping them together forced the CUDA-only
CuTe-DSL cases onto the ROCm runner, where cuda-python does not exist.

Oracle: an fp32 reference chain, with a tolerance -- the kernel keeps fp32
statistics but reorders the reduction.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

DEVICE = "cuda"

FLYDSL_D = 5120
FLYDSL_EPS = 1e-6


def _require_rocm():
    if not torch.version.hip:
        pytest.skip("ROCm/HIP required for FlyDSL kernels")


def _flydsl_reference(residual, x, gate, weight, bias, scale, shift, norm_type, eps):
    if residual is not None:
        x = (residual.float() + x.float() * gate.float()).to(torch.bfloat16)
        residual_out = x
    else:
        residual_out = None
    if norm_type == "layer":
        normed = F.layer_norm(x.float(), (FLYDSL_D,), weight, bias, eps)
    else:
        var = x.float().pow(2).mean(-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + eps) * weight.float()
    y = (normed * (1.0 + scale.float()) + shift.float()).to(torch.bfloat16)
    return y, residual_out


@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize(
    "norm_type,B,L",
    [("rms", 1, 16), ("rms", 2, 16), ("layer", 2, 16), ("rms", 1, 90000)],
)
def test_flydsl_norm_scale_shift(with_residual, norm_type, B, L):
    _require_rocm()
    # Imported inside the test: the FlyDSL compiler only exists on ROCm, and
    # the facade resolves an export the moment it is named -- a module-level
    # import here would fail collection of this whole file on CUDA.
    from sglang.kernels.ops.diffusion import (
        flydsl_fused_residual_norm_scale_shift,
        flydsl_norm_scale_shift,
    )

    torch.manual_seed(42)
    shape = (B, L, FLYDSL_D)
    x = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(FLYDSL_D, device=DEVICE, dtype=torch.float32)
    bias = (
        torch.randn(FLYDSL_D, device=DEVICE, dtype=torch.float32)
        if norm_type == "layer"
        else None
    )
    scale = torch.randn(B, 1, FLYDSL_D, device=DEVICE, dtype=torch.bfloat16)
    shift = torch.randn(B, 1, FLYDSL_D, device=DEVICE, dtype=torch.bfloat16)

    if with_residual:
        residual = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
        gate = torch.randn(B, 1, FLYDSL_D, device=DEVICE, dtype=torch.bfloat16)
        y, res = flydsl_fused_residual_norm_scale_shift(
            residual, x, gate, weight, bias, scale, shift, norm_type, FLYDSL_EPS
        )
        y_ref, res_ref = _flydsl_reference(
            residual, x, gate, weight, bias, scale, shift, norm_type, FLYDSL_EPS
        )
        torch.testing.assert_close(res, res_ref, atol=5e-2, rtol=5e-2)
    else:
        y = flydsl_norm_scale_shift(
            x, weight, bias, scale, shift, norm_type, FLYDSL_EPS
        )
        y_ref, _ = _flydsl_reference(
            None, x, None, weight, bias, scale, shift, norm_type, FLYDSL_EPS
        )
    torch.testing.assert_close(y, y_ref, atol=1.0, rtol=5e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
