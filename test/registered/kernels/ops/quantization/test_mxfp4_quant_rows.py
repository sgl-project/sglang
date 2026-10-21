"""mxfp4_quant_rows (runtime-M MXFP4 activation quant) against aiter's dynamic_mxfp4_quant."""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="MXFP4 kernels target gfx95x."
)


@pytest.mark.parametrize("m", [1, 7, 300, 4096])
@pytest.mark.parametrize("n", [4096, 8192])
def test_matches_aiter_dynamic_mxfp4_quant(m, n):
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant

    from sglang.kernels.ops.quantization.mxfp4_quant_rows import mxfp4_quant_rows

    torch.manual_seed(m * 31 + n)
    x = (torch.randn(m, n, device="cuda") * 3).to(torch.bfloat16)
    q, s = mxfp4_quant_rows(x)
    q_ref, s_ref = dynamic_mxfp4_quant(x, shuffle=False)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)[:m])
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)[:m, : n // 32])
