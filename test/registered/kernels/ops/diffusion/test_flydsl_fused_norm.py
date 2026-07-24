import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"
D = 5120
EPS = 1e-6


def _ref_rms_norm(x_f32, weight, eps):
    var = x_f32.pow(2).mean(-1, keepdim=True)
    return x_f32 * torch.rsqrt(var + eps)


def _ref_fused_residual_norm_ss(
    residual, x, gate, weight, bias, scale, shift, norm_type, eps
):
    ref_res = residual.float() + x.float() * (gate.float() if gate is not None else 1)
    ref_res_bf16 = ref_res.to(torch.bfloat16)
    if norm_type == "layer":
        normed = F.layer_norm(ref_res_bf16.float(), (D,), weight, bias, eps)
    else:
        normed = _ref_rms_norm(ref_res_bf16.float(), weight, eps) * weight.float()
    y = (normed * (1.0 + scale.float()) + shift.float()).to(torch.bfloat16)
    return y, ref_res_bf16


def _ref_norm_ss(x, weight, bias, scale, shift, norm_type, eps):
    if norm_type == "layer":
        normed = F.layer_norm(x.float(), (D,), weight, bias, eps)
    else:
        normed = _ref_rms_norm(x.float(), weight, eps) * weight.float()
    return (normed * (1.0 + scale.float()) + shift.float()).to(torch.bfloat16)


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not hasattr(torch.version, "hip") or not torch.version.hip:
        pytest.skip("ROCm/HIP required for FlyDSL kernels")
    torch.manual_seed(42)


FUSED_CASES = [
    ("rms", 1, 16),
    ("rms", 2, 16),
    ("layer", 2, 16),
    ("rms", 1, 90000),
]


@pytest.mark.parametrize("norm_type,B,L", FUSED_CASES)
def test_fused_residual_norm_scale_shift(norm_type, B, L):
    from sglang.kernels.ops.diffusion.flydsl.fused_residual_norm import (
        flydsl_fused_residual_norm_scale_shift,
    )

    residual = torch.randn(B, L, D, device=DEVICE, dtype=torch.bfloat16)
    x = torch.randn(B, L, D, device=DEVICE, dtype=torch.bfloat16)
    gate = torch.randn(B, 1, D, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(D, device=DEVICE, dtype=torch.float32)
    bias = (
        torch.randn(D, device=DEVICE, dtype=torch.float32)
        if norm_type == "layer"
        else None
    )
    scale = torch.randn(B, 1, D, device=DEVICE, dtype=torch.bfloat16)
    shift = torch.randn(B, 1, D, device=DEVICE, dtype=torch.bfloat16)

    y, res_out = flydsl_fused_residual_norm_scale_shift(
        residual,
        x,
        gate,
        weight,
        bias,
        scale,
        shift,
        norm_type,
        EPS,
    )
    y_ref, res_ref = _ref_fused_residual_norm_ss(
        residual,
        x,
        gate,
        weight,
        bias,
        scale,
        shift,
        norm_type,
        EPS,
    )
    torch.testing.assert_close(res_out, res_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(y, y_ref, atol=1.0, rtol=5e-2)


NSS_CASES = [
    ("rms", 2, 16),
    ("layer", 2, 16),
    ("rms", 1, 90000),
    ("layer", 1, 90000),
]


@pytest.mark.parametrize("norm_type,B,L", NSS_CASES)
def test_norm_scale_shift(norm_type, B, L):
    from sglang.kernels.ops.diffusion.flydsl.fused_residual_norm import (
        flydsl_norm_scale_shift,
    )

    x = torch.randn(B, L, D, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(D, device=DEVICE, dtype=torch.float32)
    bias = (
        torch.randn(D, device=DEVICE, dtype=torch.float32)
        if norm_type == "layer"
        else None
    )
    scale = torch.randn(B, 1, D, device=DEVICE, dtype=torch.bfloat16)
    shift = torch.randn(B, 1, D, device=DEVICE, dtype=torch.bfloat16)

    y = flydsl_norm_scale_shift(x, weight, bias, scale, shift, norm_type, EPS)
    y_ref = _ref_norm_ss(x, weight, bias, scale, shift, norm_type, EPS)
    torch.testing.assert_close(y, y_ref, atol=1.0, rtol=5e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
