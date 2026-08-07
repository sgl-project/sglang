"""ERNIE fused norm/scale/shift fast paths must stay bit-exact vs eager."""

import sys

import pytest
import torch

import sglang.multimodal_gen.runtime.models.dits.ernie_image as ernie_image
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.models.dits.ernie_image import (
    _ernie_gated_norm_scale_shift,
    _ernie_norm_scale_shift,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shape", [(1, 4216, 4096), (2, 1140, 4096), (1, 128, 2048)])
def test_fused_norm_scale_shift_is_bit_exact(shape):
    # (1, 4216, 4096) is the real ERNIE-Image shape (1024^2 image + text
    # tokens, hidden 4096); 2048 covers the threads_per_row=32 regime.
    torch.manual_seed(0)
    batch, seq, hidden = shape
    norm = RMSNorm(hidden, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden))
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    update = torch.randn_like(x)
    scale = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    shift = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out = _ernie_norm_scale_shift(norm, x, scale, shift)
        ref = norm(x) * (1 + scale) + shift
        assert torch.equal(out, ref)

        out2, res = _ernie_gated_norm_scale_shift(
            norm, residual, update, gate, scale, shift
        )
        res_ref = residual + gate * update
        ref2 = norm(res_ref) * (1 + scale) + shift
        assert torch.equal(res, res_ref)
        assert torch.equal(out2, ref2)

    # the fast paths must actually be in use (not silently disabled)
    assert ernie_image._ERNIE_FUSED_NORM_VERIFIED
    assert ernie_image._ERNIE_FUSED_GATED_NORM_VERIFIED
    assert not ernie_image._ERNIE_FUSED_NORM_DISABLED
    assert not ernie_image._ERNIE_FUSED_GATED_NORM_DISABLED


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
