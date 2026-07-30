"""Distributed correctness and CUDA-graph replay for Shared-DCP Output/LSE."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=90,
    stage="base-b",
    runner_config="4-gpu-b200",
)

WORLD_SIZE = int(os.environ.get("DCP_TEST_WORLD_SIZE", "4"))
ROWS = (1, 8, 32, 64)
LOCAL_HEADS = 4
HEAD_DIM = 64


def _launch_worker() -> None:
    result = subprocess.run(
        ["torchrun", f"--nproc_per_node={WORLD_SIZE}", __file__],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_shared_output_vmm() -> None:
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"Requires {WORLD_SIZE} GPUs")
    _launch_worker()


def _reference(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    rank: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    gathered_outputs = [torch.empty_like(partial_output) for _ in range(WORLD_SIZE)]
    gathered_lses = [torch.empty_like(partial_lse) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_outputs, partial_output, group=group)
    dist.all_gather(gathered_lses, partial_lse, group=group)

    head_start = rank * LOCAL_HEADS
    head_end = head_start + LOCAL_HEADS
    outputs = torch.stack(gathered_outputs)[:, :, head_start:head_end]
    lses = torch.stack(gathered_lses)[:, :, head_start:head_end]
    lses = torch.where(
        torch.isnan(lses) | (lses == float("inf")),
        torch.full_like(lses, -float("inf")),
        lses,
    )
    lse_max = lses.max(dim=0).values
    lse_max = torch.where(lse_max == -float("inf"), torch.zeros_like(lse_max), lse_max)
    weights = torch.exp2(lses - lse_max)
    denominator = weights.sum(dim=0)
    weights = torch.where(
        denominator == 0,
        torch.zeros_like(weights),
        weights / denominator,
    )
    weighted = torch.where(
        weights[..., None] == 0,
        torch.zeros_like(outputs, dtype=torch.float32),
        outputs.float() * weights[..., None],
    )
    return weighted.sum(dim=0).to(torch.bfloat16)


def _make_inputs(rows: int, rank: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1000 + rank * 31 + step)
    output = torch.randn(
        rows,
        WORLD_SIZE * LOCAL_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    lse = torch.randn(
        rows,
        WORLD_SIZE * LOCAL_HEADS,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    if rows > 1:
        lse[0, rank * LOCAL_HEADS] = -float("inf")
        output[0, rank * LOCAL_HEADS] = float("nan")
    return output, lse


def _worker() -> None:
    from sglang.srt.layers.dcp.shared_output import (
        create_dcp_output_vmm_workspace,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coordinator = ps.init_world_group(
        ranks=list(range(WORLD_SIZE)),
        local_rank=rank,
        backend="nccl",
    )
    workspace = create_dcp_output_vmm_workspace(
        max_rows=max(ROWS),
        total_heads=WORLD_SIZE * LOCAL_HEADS,
        head_dim=HEAD_DIM,
        group=coordinator,
    )

    for step, rows in enumerate(ROWS):
        partial_output, partial_lse = _make_inputs(rows, rank, step)
        expected = _reference(
            partial_output, partial_lse, rank, coordinator.device_group
        )
        actual = workspace.merge(partial_output, partial_lse, is_lse_base_on_e=False)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    rows = 32
    static_output, static_lse = _make_inputs(rows, rank, 100)
    for _ in range(2):
        workspace.merge(static_output, static_lse, is_lse_base_on_e=False)
    torch.cuda.synchronize()
    dist.barrier(group=coordinator.cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = workspace.merge(
            static_output, static_lse, is_lse_base_on_e=False
        )
    torch.cuda.synchronize()
    dist.barrier(group=coordinator.cpu_group)

    for step in range(3):
        new_output, new_lse = _make_inputs(rows, rank, 200 + step)
        static_output.copy_(new_output)
        static_lse.copy_(new_lse)
        expected = _reference(static_output, static_lse, rank, coordinator.device_group)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, expected, atol=2e-2, rtol=2e-2)

    workspace.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _worker()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
