"""The M=1 skinny fp8 PTPC GEMV must read the aiter (16,16)-preshuffled weight layout exactly."""

import sys

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not is_gfx95_supported(), reason="skinny_ptpc_gemv targets gfx950 MFMA fp8"
)


# the dispatch gate's N bounds: M3 qkv at TP8 and TP4
@pytest.mark.parametrize("n,k", [(1280, 6144), (2560, 6144)])
@pytest.mark.parametrize("scale", [1e-2, 1.0, 1e2])
def test_matches_reference(n, k, scale):
    from aiter.ops.shuffle import shuffle_weight

    from sglang.kernels.ops.gemm.skinny_ptpc_gemv import skinny_ptpc_gemv

    torch.manual_seed(0)
    w32 = torch.randn(n, k, device="cuda") * 0.02 * scale
    w_scale = w32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 448.0
    wq = (w32 / w_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    aq = (torch.randn(1, k, device="cuda") * scale).clamp(-448, 448)
    aq = aq.to(torch.float8_e4m3fn)
    x_scale = torch.rand(1, 1, device="cuda").clamp(min=1e-6)

    ref = (aq.float() @ wq.float().T) * x_scale * w_scale.T
    got = skinny_ptpc_gemv(aq, shuffle_weight(wq, (16, 16)), x_scale, w_scale)
    assert got.shape == (1, n) and got.dtype == torch.bfloat16
    rel = (got.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 5e-3, f"n={n} scale={scale}: rel_err={rel:.2e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
