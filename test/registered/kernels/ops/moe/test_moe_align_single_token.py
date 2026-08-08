import sys

import pytest
import torch

from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("topk,block_size", [(1, 16), (8, 32), (32, 64)])
def test_moe_align_single_token(topk: int, block_size: int) -> None:
    torch.manual_seed(topk)
    topk_ids = torch.randperm(256, device="cuda", dtype=torch.int32)[:topk][None]

    sorted_ids, expert_ids, num_post = moe_align_single_token(topk_ids, block_size)

    expected_expert_ids, source_indices = torch.sort(topk_ids.flatten())
    expected_sorted_ids = torch.full_like(sorted_ids, topk)
    expected_sorted_ids[::block_size] = source_indices.to(torch.int32)

    assert torch.equal(sorted_ids, expected_sorted_ids)
    assert torch.equal(expert_ids, expected_expert_ids)
    assert num_post.item() == topk * block_size


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
