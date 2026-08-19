"""Multi-GPU parity for DeepSeek V4 DCP top-k and attention collectives.

This file doubles as its torchrun worker. The pytest process launches two- and
four-rank workers; each worker uses the production GroupCoordinator, C4
candidate merge, paged decode kernel, and both DCP reduction backends.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)
from sglang.srt.layers.attention.dsv4.dcp import (
    local_c4_topk_candidates,
    local_compressed_lens,
    merge_c4_topk_candidates,
)
from sglang.srt.layers.dcp import cp_lse_ag_out_rs_mha, dcp_a2a_lse_reduce
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="sgl-kernel-unit", runner_config="2-gpu-amd")


def _launch_workers(nproc: int) -> None:
    command = ["torchrun", f"--nproc_per_node={nproc}", __file__]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"torchrun (nproc={nproc}) failed with rc={result.returncode}\n"
        f"{result.stdout}"
    )


@pytest.mark.parametrize("nproc", [2, 4])
def test_dsv4_dcp_collectives(nproc: int) -> None:
    if torch.cuda.device_count() < nproc:
        pytest.skip(f"Requires {nproc} GPUs")
    _launch_workers(nproc)


def _init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend="gloo")
    coordinator = parallel_state.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    return rank, world_size, device, coordinator


def _flat_indices(
    padded_indices: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pieces = []
    indptr = [0]
    for row, length in zip(padded_indices, lengths.tolist()):
        pieces.append(row[:length])
        indptr.append(indptr[-1] + length)
    return (
        torch.cat(pieces).to(torch.int32),
        torch.tensor(indptr, dtype=torch.int32, device=padded_indices.device),
    )


@torch.inference_mode()
def _worker_test(rank: int, world_size: int, device, coordinator) -> None:
    batch = 3
    local_heads = 2
    global_heads = local_heads * world_size
    head_dim = 512
    global_width = 67
    topk = 24
    c4_page_size = 8

    generator = torch.Generator(device=device).manual_seed(20260819)
    q_global = torch.randn(
        batch,
        global_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    kv_global = torch.randn(
        global_width,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    global_scores = torch.randn(
        batch,
        global_width,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    global_scores[1, 40:] += 8.0
    global_scores[2, 4:12] = 6.0
    global_lens = torch.tensor([17, 53, global_width], device=device)
    sink = torch.randn(
        global_heads, dtype=torch.float32, device=device, generator=generator
    )

    local_q = q_global[:, rank * local_heads : (rank + 1) * local_heads]
    gathered_q = coordinator.all_gather(local_q.contiguous(), dim=1).contiguous()
    torch.testing.assert_close(gathered_q, q_global)

    local_kv = kv_global[rank::world_size].contiguous()
    local_scores = global_scores[:, rank::world_size].contiguous()
    local_lens = local_compressed_lens(
        global_lens * 4, 4, world_size, rank
    )
    candidate_scores, candidate_ids = local_c4_topk_candidates(
        local_scores, local_lens, topk, world_size, rank
    )
    gathered_scores = coordinator.all_gather(
        candidate_scores.contiguous(), dim=1
    )
    gathered_ids = coordinator.all_gather(candidate_ids.contiguous(), dim=1)

    local_pages = (local_kv.shape[0] + c4_page_size - 1) // c4_page_size
    page_table = torch.arange(
        local_pages, dtype=torch.int32, device=device
    ).expand(batch, -1)
    topk_result = merge_c4_topk_candidates(
        gathered_scores,
        gathered_ids,
        topk,
        world_size,
        rank,
        page_table,
        c4_page_size,
    )
    local_indices, local_indptr = _flat_indices(
        topk_result.page_indices, topk_result.local_lens
    )

    reference_indices, reference_indptr = _flat_indices(
        topk_result.global_indices,
        (topk_result.global_indices >= 0).sum(dim=1),
    )
    scale = head_dim**-0.5
    reference_out = _sparse_attn_v4_paged_decode_triton(
        q_global,
        kv_global,
        reference_indices,
        reference_indptr,
        sink,
        scale,
    )
    reference_local = reference_out[
        :, rank * local_heads : (rank + 1) * local_heads
    ]

    shifted_sink = sink - torch.log(
        torch.tensor(float(world_size), dtype=torch.float32, device=device)
    )
    partial_out, partial_lse = _sparse_attn_v4_paged_decode_triton(
        gathered_q,
        local_kv,
        local_indices,
        local_indptr,
        shifted_sink,
        scale,
        return_lse=True,
    )

    ag_rs = cp_lse_ag_out_rs_mha(
        partial_out.clone(), partial_lse.clone(), coordinator
    )
    a2a = dcp_a2a_lse_reduce(
        partial_out.clone(),
        partial_lse.clone(),
        coordinator,
        is_lse_base_on_e=True,
        comm_backend="a2a",
    )
    torch.testing.assert_close(
        ag_rs.float(), reference_local.float(), atol=4e-2, rtol=4e-2
    )
    torch.testing.assert_close(
        a2a.float(), reference_local.float(), atol=4e-2, rtol=4e-2
    )
    torch.testing.assert_close(a2a.float(), ag_rs.float(), atol=3e-2, rtol=3e-2)


def _worker_main() -> None:
    rank, world_size, device, coordinator = _init_distributed()
    try:
        _worker_test(rank, world_size, device, coordinator)
        dist.barrier(group=coordinator.cpu_group)
    finally:
        coordinator.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _worker_main()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))