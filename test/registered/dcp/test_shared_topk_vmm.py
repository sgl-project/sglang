"""Distributed correctness and CUDA-graph replay for Shared-DCP Top-K VMM."""

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
    est_time=120,
    stage="base-b",
    runner_config="4-gpu-b200",
)

WORLD_SIZE = int(os.environ.get("DCP_TEST_WORLD_SIZE", "4"))
ROWS = (1, 8, 32, 64)
LOCAL_LENGTH = 4096
TOPK = 512
TIED_PER_OWNER = TOPK // WORLD_SIZE + 64


def _launch_worker() -> None:
    result = subprocess.run(
        ["torchrun", f"--nproc_per_node={WORLD_SIZE}", __file__],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_shared_topk_vmm() -> None:
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"Requires {WORLD_SIZE} GPUs")
    _launch_worker()


def _make_logits(rows: int, rank: int, step: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(30_000 + rank * 101 + step)
    logits = torch.randn(
        rows,
        LOCAL_LENGTH,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    # More than TOPK candidates share the maximum score globally. The exact
    # result must therefore exercise the lower-global-ID cutoff tie-break, not
    # merely include every tied candidate.
    logits.clamp_max_(9.0)
    logits[:, :TIED_PER_OWNER] = 10.0
    return logits


def _reference(
    logits: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    gathered = [torch.empty_like(logits) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, logits, group=group)
    global_logits = (
        torch.stack(gathered, dim=0).permute(1, 2, 0).reshape(logits.shape[0], -1)
    )
    return torch.argsort(
        global_logits,
        dim=1,
        descending=True,
        stable=True,
    )[
        :, :TOPK
    ].to(torch.int32)


def _assert_exact_selected_set(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    # The CUDA selector atomically emits the exact stable-key Top-K set. Sparse
    # attention does not require score order, so compare canonicalized IDs.
    torch.testing.assert_close(
        actual.sort(dim=1).values,
        expected.sort(dim=1).values,
        rtol=0,
        atol=0,
    )
    # The constructed tie spans the cutoff, so this also checks that lower
    # global IDs win rather than an arbitrary subset of equal-score entries.
    expected_tied_ids = torch.arange(
        TOPK,
        dtype=torch.int32,
        device=expected.device,
    ).expand(expected.shape[0], -1)
    torch.testing.assert_close(expected, expected_tied_ids, rtol=0, atol=0)


def _worker() -> None:
    from sglang.srt.layers.dcp.shared_topk_vmm import (
        create_dcp_topk_vmm_workspace,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coordinator = ps.init_world_group(
        ranks=list(range(WORLD_SIZE)),
        local_rank=rank,
        backend="nccl",
    )
    workspace = create_dcp_topk_vmm_workspace(
        max_rows=max(ROWS),
        local_candidates=TOPK,
        group=coordinator,
    )

    try:
        for step, rows in enumerate(ROWS):
            logits = _make_logits(rows, rank, step)
            expected = _reference(logits, coordinator.device_group)
            local_indices = torch.topk(logits, TOPK, dim=1, sorted=False).indices.to(
                torch.int32
            )
            actual = workspace.merge(
                logits,
                local_indices,
                TOPK,
                dcp_rank=rank,
                dcp_size=WORLD_SIZE,
            )
            torch.cuda.synchronize()
            _assert_exact_selected_set(actual, expected)

        rows = 32
        static_logits = _make_logits(rows, rank, 100)
        static_indices = torch.topk(
            static_logits, TOPK, dim=1, sorted=False
        ).indices.to(torch.int32)
        workspace.merge(
            static_logits,
            static_indices,
            TOPK,
            dcp_rank=rank,
            dcp_size=WORLD_SIZE,
        )
        torch.cuda.synchronize()
        dist.barrier(group=coordinator.cpu_group)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = workspace.merge(
                static_logits,
                static_indices,
                TOPK,
                dcp_rank=rank,
                dcp_size=WORLD_SIZE,
            )
        torch.cuda.synchronize()
        dist.barrier(group=coordinator.cpu_group)

        for step in range(3):
            new_logits = _make_logits(rows, rank, 200 + step)
            static_logits.copy_(new_logits)
            static_indices.copy_(
                torch.topk(static_logits, TOPK, dim=1, sorted=False).indices
            )
            expected = _reference(static_logits, coordinator.device_group)
            graph.replay()
            torch.cuda.synchronize()
            _assert_exact_selected_set(graph_output, expected)

        pipelined_workspaces = [
            create_dcp_topk_vmm_workspace(
                max_rows=max(ROWS),
                local_candidates=TOPK,
                group=coordinator,
            )
            for _ in range(2)
        ]
        pipelined_inputs = []
        for step in range(6):
            logits = _make_logits(rows, rank, 300 + step)
            pipelined_inputs.append(
                (
                    logits,
                    torch.topk(logits, TOPK, dim=1, sorted=False).indices.to(
                        torch.int32
                    ),
                    _reference(logits, coordinator.device_group),
                )
            )
        pipelined_outputs = [
            pipelined_workspaces[step % 2].merge(
                logits,
                local_indices,
                TOPK,
                dcp_rank=rank,
                dcp_size=WORLD_SIZE,
                pipelined=True,
            )
            for step, (logits, local_indices, _) in enumerate(pipelined_inputs)
        ]
        torch.cuda.synchronize()
        for actual, (_, _, expected) in zip(
            pipelined_outputs, pipelined_inputs, strict=True
        ):
            _assert_exact_selected_set(actual, expected)
        for pipelined_workspace in pipelined_workspaces:
            pipelined_workspace.close()
    finally:
        workspace.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _worker()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
