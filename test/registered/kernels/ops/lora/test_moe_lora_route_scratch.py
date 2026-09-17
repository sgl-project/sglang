from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from sglang.kernels.ops.lora.common.routing import (  # noqa: E402
    _aligned_route_scratch,
    _build_large_route,
)
from sglang.srt.lora.workspace import LoraWorkspace  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_route_counts_are_initialized_only_on_first_allocation() -> None:
    """Reusing count storage must not hide a missing scan-side reset."""
    workspace = LoraWorkspace()
    first = _aligned_route_scratch(
        workspace,
        prefix="route:test",
        num_buckets=7,
        capacity=32,
        block_size=8,
        device=torch.device("cpu"),
    )
    assert first["counts"].tolist() == [0] * 7

    first["counts"].fill_(9)
    second = _aligned_route_scratch(
        workspace,
        prefix="route:test",
        num_buckets=7,
        capacity=32,
        block_size=8,
        device=torch.device("cpu"),
    )

    assert second["counts"].data_ptr() == first["counts"].data_ptr()
    assert second["counts"].tolist() == [9] * 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA route kernel required")
def test_route_scan_restores_both_count_buffers_between_calls() -> None:
    device = torch.device("cuda")
    workspace = LoraWorkspace()
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
    for token_lora_mapping in traffic:
        for is_shared_outer in (False, True):
            _build_large_route(
                token_lora_mapping,
                topk_ids,
                groups_per_slot=1 if is_shared_outer else num_local_experts,
                max_loras=max_loras,
                block_size=block_size,
                workspace=workspace,
                tensor_prefix="route:aligned",
            )

        per_expert_counts = workspace.tensor(
            f"route:aligned:groups{num_local_experts}:counts",
            (num_local_experts * max_loras + 1,),
            dtype=torch.int32,
            device=device,
            zero_on_first_allocation=True,
        )
        shared_counts = workspace.tensor(
            "route:aligned:groups1:counts",
            (max_loras + 1,),
            dtype=torch.int32,
            device=device,
            zero_on_first_allocation=True,
        )

        assert torch.count_nonzero(per_expert_counts).item() == 0
        assert torch.count_nonzero(shared_counts).item() == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
