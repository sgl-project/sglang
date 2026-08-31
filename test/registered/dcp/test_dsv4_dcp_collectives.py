"""Multi-GPU parity for DeepSeek V4 DCP top-k and attention collectives.

This file doubles as its torchrun worker. Registered CI launches two ranks;
larger MI355X validation can invoke the worker directly with torchrun. Each
worker uses the production GroupCoordinator, C4 candidate merge, paged decode
kernel, and both DCP reduction backends.
"""

from __future__ import annotations

import math
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
    DSV4DCPCombinedQTopKWorkspace,
    combined_q_topk_candidate_view,
    local_c4_topk_candidates,
    local_compressed_lens,
    merge_c4_topk_candidates,
    run_combined_q_c4_topk,
    run_packed_c4_topk,
)
from sglang.srt.layers.dcp import cp_lse_ag_out_rs_mla, dcp_a2a_lse_reduce
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


def test_dsv4_dcp_collectives() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires 2 GPUs")
    _launch_workers(2)


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
    local_heads = 16
    global_heads = local_heads * world_size
    head_dim = 512
    global_width = 1153
    topk = 1024
    c4_page_size = 64

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
    global_lens = torch.tensor([257, 1025, global_width], device=device)
    sink = torch.randn(
        global_heads, dtype=torch.float32, device=device, generator=generator
    )

    local_q = q_global[:, rank * local_heads : (rank + 1) * local_heads]
    gathered_q = coordinator.all_gather(local_q.contiguous(), dim=1).contiguous()
    torch.testing.assert_close(gathered_q, q_global)

    local_kv = kv_global[rank::world_size].contiguous()
    local_scores = global_scores[:, rank::world_size].contiguous()
    local_lens = local_compressed_lens(global_lens * 4, 4, world_size, rank)
    candidate_scores, candidate_ids = local_c4_topk_candidates(
        local_scores, local_lens, topk, world_size, rank
    )
    gathered_scores = coordinator.all_gather(candidate_scores.contiguous(), dim=1)
    gathered_ids = coordinator.all_gather(candidate_ids.contiguous(), dim=1)

    local_pages = (local_kv.shape[0] + c4_page_size - 1) // c4_page_size
    page_table = torch.arange(local_pages, dtype=torch.int32, device=device).expand(
        batch, -1
    )
    topk_result = merge_c4_topk_candidates(
        gathered_scores,
        gathered_ids,
        topk,
        world_size,
        rank,
        page_table,
        c4_page_size,
    )

    packed_page_indices = torch.empty_like(topk_result.page_indices)
    packed_local_raw = torch.empty_like(topk_result.local_raw_indices)
    packed_local_lens = torch.empty_like(topk_result.local_lens)
    local_candidates = torch.empty((batch, topk), dtype=torch.int64, device=device)
    gathered_candidates = torch.empty(
        (world_size * batch, topk), dtype=torch.int64, device=device
    )
    run_packed_c4_topk(
        logits=local_scores,
        local_lens=local_lens.to(torch.int32),
        local_page_table=page_table,
        local_candidates=local_candidates,
        gathered_candidates=gathered_candidates,
        out_page_indices=packed_page_indices,
        out_local_lens=packed_local_lens,
        c4_page_size=c4_page_size,
        dcp_size=world_size,
        dcp_rank=rank,
        dcp_group=coordinator,
        out_local_raw_indices=packed_local_raw,
    )
    for row in range(batch):
        expected_count = int(topk_result.local_lens[row])
        actual_count = int(packed_local_lens[row])
        assert actual_count == expected_count
        torch.testing.assert_close(
            torch.sort(packed_local_raw[row, :actual_count]).values,
            torch.sort(topk_result.local_raw_indices[row, :expected_count]).values,
        )
        torch.testing.assert_close(
            torch.sort(packed_page_indices[row, :actual_count]).values,
            torch.sort(topk_result.page_indices[row, :expected_count]).values,
        )

    combined_heads = local_heads + topk * 4 // head_dim
    local_combined = torch.empty(
        (batch, combined_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    local_combined[:, :local_heads, :].copy_(local_q)
    gathered_combined = torch.empty(
        (world_size * batch, combined_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    combined_workspace = DSV4DCPCombinedQTopKWorkspace(
        local_combined=local_combined,
        gathered_combined=gathered_combined,
    )
    combined_page_indices = torch.empty_like(packed_page_indices)
    combined_local_raw = torch.empty_like(packed_local_raw)
    combined_local_lens = torch.empty_like(packed_local_lens)
    rank_major_q = run_combined_q_c4_topk(
        logits=local_scores,
        local_lens=local_lens.to(torch.int32),
        local_page_table=page_table,
        workspace=combined_workspace,
        local_heads=local_heads,
        out_page_indices=combined_page_indices,
        out_local_lens=combined_local_lens,
        c4_page_size=c4_page_size,
        dcp_size=world_size,
        dcp_rank=rank,
        dcp_group=coordinator,
        out_local_raw_indices=combined_local_raw,
    )
    torch.testing.assert_close(
        rank_major_q.reshape(batch, global_heads, head_dim), q_global, rtol=0, atol=0
    )
    combined_candidate_rows = combined_q_topk_candidate_view(
        gathered_combined,
        local_heads=local_heads,
        topk=topk,
    )
    torch.testing.assert_close(
        torch.sort(combined_candidate_rows, dim=1).values,
        torch.sort(gathered_candidates, dim=1).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(combined_local_lens, packed_local_lens)
    for row in range(batch):
        count = int(combined_local_lens[row])
        torch.testing.assert_close(
            torch.sort(combined_local_raw[row, :count]).values,
            torch.sort(packed_local_raw[row, :count]).values,
        )
        torch.testing.assert_close(
            torch.sort(combined_page_indices[row, :count]).values,
            torch.sort(packed_page_indices[row, :count]).values,
        )

    local_indices, local_indptr = _flat_indices(packed_page_indices, packed_local_lens)

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
    reference_local = reference_out[:, rank * local_heads : (rank + 1) * local_heads]

    sink_logit_shift = -math.log(float(world_size))
    shifted_sink = sink + sink_logit_shift
    partial_out, partial_lse = _sparse_attn_v4_paged_decode_triton(
        rank_major_q,
        local_kv,
        local_indices,
        local_indptr,
        shifted_sink,
        scale,
        return_lse=True,
    )
    inline_shift_out, inline_shift_lse = _sparse_attn_v4_paged_decode_triton(
        rank_major_q,
        local_kv,
        local_indices,
        local_indptr,
        sink,
        scale,
        return_lse=True,
        attn_sink_logit_shift=sink_logit_shift,
    )
    torch.testing.assert_close(
        inline_shift_out.float(), partial_out.float(), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(inline_shift_lse, partial_lse, atol=1e-6, rtol=0)
    fused_shifted_out, fused_shifted_lse = _sparse_attn_v4_paged_decode_triton(
        rank_major_q,
        local_kv,
        local_indices,
        local_indptr,
        shifted_sink,
        scale,
        kv_splits=1,
        return_lse=True,
    )
    fused_inline_out, fused_inline_lse = _sparse_attn_v4_paged_decode_triton(
        rank_major_q,
        local_kv,
        local_indices,
        local_indptr,
        sink,
        scale,
        kv_splits=1,
        return_lse=True,
        attn_sink_logit_shift=sink_logit_shift,
    )
    torch.testing.assert_close(
        fused_inline_out.float(), fused_shifted_out.float(), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(fused_inline_lse, fused_shifted_lse, atol=1e-6, rtol=0)

    ag_rs = (
        cp_lse_ag_out_rs_mla(
            partial_out,
            partial_lse.clone(),
            coordinator,
            is_lse_base_on_e=True,
        )
        .transpose(0, 1)
        .contiguous()
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
    assert ag_rs.is_contiguous()
    ag_rs.view(batch, -1)

    graph_scores = local_scores.clone()
    graph_local_combined = torch.empty_like(local_combined)
    graph_local_combined[:, :local_heads, :].copy_(local_q)
    graph_gathered_combined = torch.empty_like(gathered_combined)
    graph_combined_workspace = DSV4DCPCombinedQTopKWorkspace(
        local_combined=graph_local_combined,
        gathered_combined=graph_gathered_combined,
    )
    graph_page_indices = torch.empty_like(packed_page_indices)
    graph_local_raw = torch.empty_like(packed_local_raw)
    graph_local_lens = torch.empty_like(packed_local_lens)
    topk_graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with coordinator.graph_capture() as graph_context:
        with torch.cuda.graph(topk_graph, stream=graph_context.stream):
            graph_rank_major_q = run_combined_q_c4_topk(
                logits=graph_scores,
                local_lens=local_lens.to(torch.int32),
                local_page_table=page_table,
                workspace=graph_combined_workspace,
                local_heads=local_heads,
                out_page_indices=graph_page_indices,
                out_local_lens=graph_local_lens,
                c4_page_size=c4_page_size,
                dcp_size=world_size,
                dcp_rank=rank,
                dcp_group=coordinator,
                out_local_raw_indices=graph_local_raw,
            )
    torch.cuda.synchronize()
    for _ in range(3):
        topk_graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_local_lens, packed_local_lens)
        torch.testing.assert_close(
            graph_rank_major_q.reshape(batch, global_heads, head_dim),
            q_global,
            rtol=0,
            atol=0,
        )
        for row in range(batch):
            count = int(graph_local_lens[row])
            torch.testing.assert_close(
                torch.sort(graph_local_raw[row, :count]).values,
                torch.sort(packed_local_raw[row, :count]).values,
            )

    graph_partial_out = torch.empty_like(partial_out)
    graph_partial_lse = torch.empty_like(partial_lse)
    graph_partial_out.copy_(partial_out)
    graph_partial_lse.copy_(partial_lse)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with coordinator.graph_capture() as graph_context:
        with torch.cuda.graph(graph, stream=graph_context.stream):
            graph_ag_rs = (
                cp_lse_ag_out_rs_mla(
                    graph_partial_out,
                    graph_partial_lse,
                    coordinator,
                    is_lse_base_on_e=True,
                )
                .transpose(0, 1)
                .contiguous()
            )
    torch.cuda.synchronize()

    replay_results = []
    for _ in range(3):
        graph_partial_out.copy_(partial_out)
        graph_partial_lse.copy_(partial_lse)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replay_results.append(graph_ag_rs.clone())

    for replay_result in replay_results:
        torch.testing.assert_close(
            replay_result.float(), ag_rs.float(), atol=3e-2, rtol=3e-2
        )
        assert replay_result.is_contiguous()
        replay_result.view(batch, -1)
    for replay_result in replay_results[1:]:
        torch.testing.assert_close(replay_result, replay_results[0], atol=0, rtol=0)


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
