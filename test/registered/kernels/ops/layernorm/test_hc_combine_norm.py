"""The fused hc_combine_norm reproduces the <=8-row kernel across row bands."""

import sys

import pytest
import torch

from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="hc_combine_norm requires CUDA",
)

HIDDEN = 5120
STREAMS = 4
EPS = 1e-6


def _inputs(m: int):
    torch.manual_seed(0)
    x = torch.randn(m, STREAMS * HIDDEN, device="cuda", dtype=torch.bfloat16)
    pre = torch.randn(m, STREAMS, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)
    return x, pre, weight


@pytest.mark.parametrize("m", [8, 9, 48, 49, 96])
def test_row_bands_match_small_band(m: int):
    # Row statistics are per row and the column partitions are disjoint, so the
    # mid-band dispatch (2 parts up to 48 rows, 1 part above) has to reproduce
    # the <=8-row kernel (4 parts) for the same rows.
    x, pre, weight = _inputs(m)
    got = hc_combine_norm(x, pre, weight, EPS)
    ref = torch.cat(
        [
            hc_combine_norm(x[i : i + 8], pre[i : i + 8], weight, EPS)
            for i in range(0, m, 8)
        ]
    )
    assert got.shape == (m, HIDDEN)
    assert torch.allclose(got.float(), ref.float(), rtol=2**-7, atol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
