"""Compensated DeepGEMM mHC prenorm + sinkhorn against an fp64 reference."""

import sys

import pytest
import torch

import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.mhc import hc_split_sinkhorn_torch

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not get_platform().is_sm100,
    reason="Blackwell DeepGEMM prenorm",
)


@pytest.mark.parametrize("num_tokens", [32, 576, 1024, 4095])
def test_deepgemm_compensated_mhc_slices(num_tokens):
    torch.manual_seed(7 + num_tokens)
    x = torch.randn(num_tokens, 4 * 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(24, 4 * 5120, device="cuda") * 0.01
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(24, device="cuda")
    parts = mhc.split_tf32_hc_weight(w)
    actual = mhc.hc_mix_stats_sinkhorn_deepgemm(x, parts, scale, base, 20, 1e-6, 1e-6)
    xd = x.double()
    z = (xd @ w.double().T) * torch.rsqrt(xd.square().mean(-1, keepdim=True) + 1e-6)
    expected = hc_split_sinkhorn_torch(z[None], scale, base, 4, 20, 1e-6)
    for result, ref in zip(actual, expected):
        torch.testing.assert_close(result, ref[0].float(), rtol=1e-5, atol=2e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
