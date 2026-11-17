"""Triton Gemma RMSNorm with ``round_sum`` matches the eager ``forward_native``
rounding (the HIP path without aiter or vllm)."""

import sys

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")

rocm = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hip()), reason="needs a ROCm GPU"
)


@rocm
@pytest.mark.parametrize("m", [1, 8, 300])
def test_gemma_fused_add_rmsnorm_matches_eager(m):
    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
    )

    torch.manual_seed(0)
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    r = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(5120, device="cuda", dtype=torch.bfloat16) * 0.1

    def eager(v):
        y = v.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (y * (1.0 + w.float())).to(torch.bfloat16)

    out, res = gemma_fused_add_rmsnorm(x, r, w, 1e-6, round_sum=True)
    assert torch.equal(res, x + r)
    assert (out.float() - eager(x + r).float()).abs().max().item() <= 2**-6
    assert (
        gemma_rmsnorm(x, w, 1e-6).float() - eager(x).float()
    ).abs().max().item() <= 2**-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
