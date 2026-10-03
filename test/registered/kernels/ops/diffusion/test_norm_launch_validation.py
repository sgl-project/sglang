"""Reject malformed fused-normalization inputs before any GPU kernel runs."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    fused_qk_head_layernorm,
    fused_rmsnorm_scale_shift_bitexact,
    fused_scale_residual_rmsnorm_scale_shift_bitexact,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("bad_input", ["shape", "dtype", "device", "layout"])
def test_qk_layernorm_rejects_invalid_key(bad_input):
    q = torch.randn(2, 17, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    if bad_input == "shape":
        k = k.view(1, 34, 4, 128)
    elif bad_input == "dtype":
        k = k.float()
    elif bad_input == "device":
        k = k.cpu()
    else:
        k = k.transpose(1, 2)
    with pytest.raises(RuntimeError):
        fused_qk_head_layernorm(q, k, 1e-6)


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("bad_input", ["weight", "scale", "shift"])
def test_rmsnorm_rejects_invalid_affine_inputs(residual, bad_input):
    x = torch.randn(2, 17, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(2048, device=x.device, dtype=x.dtype)
    scale = torch.zeros(2, 1, 2048, device=x.device, dtype=x.dtype)
    shift = torch.zeros_like(scale)
    if bad_input == "weight":
        weight = weight.float()
    elif bad_input == "scale":
        scale = scale.view(1, 2, 2048)
    else:
        shift = shift.cpu()
    with pytest.raises(RuntimeError):
        if residual:
            gate = torch.ones(2, 1, 2048, device=x.device, dtype=x.dtype)
            fused_scale_residual_rmsnorm_scale_shift_bitexact(
                x, x, gate, weight, scale, shift, 1e-6
            )
        else:
            fused_rmsnorm_scale_shift_bitexact(x, weight, scale, shift, 1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
