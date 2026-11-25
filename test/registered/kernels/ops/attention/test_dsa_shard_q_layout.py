import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.shard_q_layout import swap_leading_dims
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


# Shard-Q q (E4M3, 576-wide) and output (bf16, 512-wide) layouts; 12 heads is
# Kimi-K3.1 at DCP8. The 4x3x5 fp32 case leaves a row shorter than one block.
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((8, 37, 12, 576), torch.float8_e4m3fn),
        ((37, 8, 12, 512), torch.bfloat16),
        ((4, 3, 5), torch.float32),
    ],
)
def test_swap_leading_dims_matches_transpose(shape, dtype):
    src = torch.randn(shape, device="cuda").to(dtype)
    out = torch.empty((shape[1], shape[0], *shape[2:]), device="cuda", dtype=dtype)
    swap_leading_dims(src, out)
    ref = src.transpose(0, 1).contiguous()
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
