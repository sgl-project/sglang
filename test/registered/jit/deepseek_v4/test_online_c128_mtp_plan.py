from __future__ import annotations

import torch

from sglang.jit_kernel.dsv4 import CompressorPrefillPlan
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=25, suite="nightly-kernel-1-gpu", nightly=True)


def test_online_c128_mtp_plan() -> None:
    prefix_lens = torch.tensor([112, 120, 124, 128], dtype=torch.int32, device="cuda")
    req_pool_indices = torch.tensor([3, 5, 7, 9], dtype=torch.int64, device="cuda")

    plan = CompressorPrefillPlan.generate_online_mtp(
        prefix_lens=prefix_lens,
        req_pool_indices=req_pool_indices,
        num_draft_tokens=8,
        state_slot_offset=128,
        active_batch_size=3,
    )

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

    for output in (plan.plan_c, plan.plan_w):
        assert output.shape == (4, 16)
        assert output.dtype == torch.uint8 and output.is_cuda
        assert output.is_contiguous()
    torch.testing.assert_close(plan.plan_c.view(torch.int32).cpu(), expected_c)
    torch.testing.assert_close(plan.plan_w.view(torch.int32).cpu(), expected_w)
    assert plan.pin_buffer is None
