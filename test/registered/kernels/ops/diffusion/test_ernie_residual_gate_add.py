"""ERNIE residual-gate fast path must stay bit-exact vs the eager pair."""

import sys

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.ernie_image import (
    _ernie_residual_gate_add,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_residual_gate_add_is_bit_exact(dtype):
    # Real ERNIE-Image shapes: hidden 4096, 1024^2 image tokens + text tokens.
    # fp32 exercises the eager fallback (fast path is half-dtype only).
    torch.manual_seed(0)
    residual = torch.randn(1, 4216, 4096, device="cuda", dtype=dtype)
    update = torch.randn_like(residual)
    gate = torch.randn(1, 1, 4096, device="cuda", dtype=dtype)
    out = _ernie_residual_gate_add(residual, update, gate)
    assert torch.equal(out, residual + gate * update)

    # Full-shape gate takes the same kernel path and must stay exact too.
    gate_full = gate.expand_as(residual).contiguous()
    out_full = _ernie_residual_gate_add(residual, update, gate_full)
    assert torch.equal(out_full, residual + gate_full * update)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
