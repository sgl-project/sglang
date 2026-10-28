import sys

import pytest
import torch

from sglang.kernels.ops.embeddings.engram_gather import engram_rows_to_token_major
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("n", [0, 1, 37, 4096])
def test_matches_transpose(world, n):
    chunk = 24 // world * 256
    recv = torch.randn(world * n, chunk, device="cuda", dtype=torch.bfloat16)
    expected = recv.view(world, n, chunk).transpose(0, 1).reshape(n, world * chunk)
    assert torch.equal(engram_rows_to_token_major(recv, world), expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
