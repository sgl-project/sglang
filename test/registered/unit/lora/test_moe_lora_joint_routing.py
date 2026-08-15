"""Focused contracts for the shared-factor joint route builder."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from sglang.srt.lora.moe.joint_routing import (  # noqa: E402
    _plan_scratch,
    build_joint_shared_routes,
)
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def test_joint_route_counts_are_initialized_only_on_first_allocation() -> None:
    """Reusing count storage must not hide a missing scan-side reset."""
    workspace = MoeLoraWorkspace()
    first = _plan_scratch(
        workspace,
        prefix="joint_route:test",
        num_buckets=7,
        capacity=32,
        block_size=8,
        device=torch.device("cpu"),
    )
    assert first["counts"].tolist() == [0] * 7

    first["counts"].fill_(9)
    second = _plan_scratch(
        workspace,
        prefix="joint_route:test",
        num_buckets=7,
        capacity=32,
        block_size=8,
        device=torch.device("cpu"),
    )

    assert second["counts"].data_ptr() == first["counts"].data_ptr()
    assert second["counts"].tolist() == [9] * 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA route kernel required")
def test_joint_route_scan_restores_both_count_buffers_between_calls() -> None:
    """Each scan must leave both histograms ready for the next invocation."""
    device = torch.device("cuda")
    workspace = MoeLoraWorkspace()
    num_local_experts = 3
    max_loras = 2
    block_size = 8
    topk_ids = torch.tensor(
        [[0, 1], [2, 0], [1, 2], [0, 2], [2, 1], [1, 0]],
        dtype=torch.int32,
        device=device,
    )

    traffic = (
        torch.tensor([0, 1, -1, 0, 1, -1], dtype=torch.int32, device=device),
        torch.tensor([1, -1, 0, 1, -1, 0], dtype=torch.int32, device=device),
    )
    for token_slots in traffic:
        build_joint_shared_routes(
            topk_ids,
            token_slots,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            workspace=workspace,
            use_pdl=False,
        )

        per_expert_counts = workspace.tensor(
            "joint_route:per_expert:counts",
            (num_local_experts * max_loras + 1,),
            dtype=torch.int32,
            device=device,
            zero_on_first_allocation=True,
        )
        shared_counts = workspace.tensor(
            "joint_route:shared:counts",
            (max_loras + 1,),
            dtype=torch.int32,
            device=device,
            zero_on_first_allocation=True,
        )

        assert torch.count_nonzero(per_expert_counts).item() == 0
        assert torch.count_nonzero(shared_counts).item() == 0
