from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import CompressorPrefillPlan
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("active_batch_size", [0, 3])
def test_online_c128_mtp_plan(active_batch_size: int) -> None:
    prefix_lens = torch.tensor([112, 120, 124, 128], dtype=torch.int64, device="cuda")
    req_pool_indices = torch.tensor([3, 5, 7, 9], dtype=torch.int64, device="cuda")

    def generate_plan():
        return CompressorPrefillPlan.generate_online_mtp(
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
            num_draft_tokens=8,
            state_slot_offset=128,
            active_batch_size=active_batch_size,
        )

    # Warm up JIT compilation before capture, then verify replay overwrites stale plans.
    generate_plan()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan = generate_plan()
    plan.plan_c.zero_()
    plan.plan_w.zero_()
    graph.replay()

    invalid = [-1, 0, -1, -1]
    expected_c = torch.tensor(
        [
            invalid,
            [128, (8 << 16) | 15, 133, 5],
            [128, (4 << 16) | 19, 135, 7],
            invalid,
        ],
        dtype=torch.int32,
    )
    expected_w = torch.tensor(
        [
            [120, (8 << 16) | 7, 131, 3],
            invalid,
            [132, (4 << 16) | 23, 135, 7],
            invalid,
        ],
        dtype=torch.int32,
    )

    if active_batch_size == 0:
        expected_c[:] = torch.tensor(invalid)
        expected_w[:] = torch.tensor(invalid)
    torch.testing.assert_close(plan.plan_c.view(torch.int32).cpu(), expected_c)
    torch.testing.assert_close(plan.plan_w.view(torch.int32).cpu(), expected_w)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
