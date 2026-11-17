"""The fused GDN qkvz/ba split kernel handles head ratio 3 (Qwen3.5/3.8 dense) bitwise,
as enabled on HIP."""

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
@pytest.mark.parametrize("m", [1, 8, 37])
def test_gdn_fused_split_ratio_three(m):
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )

    nk, nv, dk, dv = 16, 48, 128, 128
    qkvz = torch.randn(
        m, 2 * nk * dk + 2 * nv * dv, device="cuda", dtype=torch.bfloat16
    )
    ba = torch.randn(m, 2 * nv, device="cuda", dtype=torch.bfloat16)
    q, k, v, z = qkvz.split([nk * dk, nk * dk, nv * dv, nv * dv], dim=-1)
    b, a = ba.split([nv, nv], dim=-1)
    mixed, zz, bb, aa = fused_qkvzba_split_reshape_cat_contiguous(
        qkvz, ba, nk, nv, dk, dv
    )
    assert torch.equal(mixed.reshape(m, -1), torch.cat((q, k, v), dim=-1))
    assert torch.equal(zz.reshape(m, nv, dv), z.reshape(m, nv, dv))
    assert torch.equal(bb.reshape(m, nv), b) and torch.equal(aa.reshape(m, nv), a)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
