"""row_sum must stay bit-identical to the tl.sum reference it replaces.

Column counts that are multiples of 16 are excluded: Triton specializes such a
K and splits each row across lanes, which reorders its sum. The router top-k
that row_sum serves is 8.
"""

import pytest
import torch

from sglang.kernels.ops.elementwise.row_sum import row_sum
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.triton_sum import triton_sum

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("num_rows", [1, 7, 129, 4096])
@pytest.mark.parametrize("num_cols", [1, 6, 8, 13, 31])
def test_row_sum_matches_triton_sum_bitwise(num_rows, num_cols):
    torch.manual_seed(num_rows * 64 + num_cols)
    x = torch.rand(num_rows, num_cols, device="cuda", dtype=torch.float32)
    ref = triton_sum(x, dim=-1)
    assert torch.equal(row_sum(x, dim=-1), ref)
    assert torch.equal(row_sum(x, dim=1, keepdim=True), ref.unsqueeze(-1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
