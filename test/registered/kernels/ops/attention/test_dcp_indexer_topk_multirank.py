"""Multi-rank correctness test for the DCP-sharded DSA indexer top-k pipeline:
per-rank local top-k -> pack -> a REAL torch.distributed all_gather (NCCL,
not a single-process simulation) -> CuteDSL merge.

Validates against a brute-force reference that computes the true global
top-k directly, and checks that every rank's merged result agrees.

Usage::

    python test/registered/kernels/ops/attention/test_dcp_indexer_topk_multirank.py
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.attention.dsa.dcp_localize_index_kv import (
    dcp_local_capacity,
    dcp_localize_page_table,
)
from sglang.kernels.ops.attention.dsa.dcp_topk_merge_cutedsl import (
    pack_dcp_topk_candidates_cutedsl,
    stable_topk_from_gathered_candidates_cutedsl,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="base-c", runner_config="4-gpu-b200")

TOPK = 512


@cache_once
def _init_group_once() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    return dist.group.WORLD


def _rank_local_topk(
    scores: torch.Tensor,
    page_table_1: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    seq_lens: torch.Tensor,
    topk: int,
    page_size: int,
):
    """Mirrors what dsa_indexer.py's sharded _get_topk_paged computes: score
    only this rank's local (owned) KV shard, then top-k locally."""
    capacity = dcp_local_capacity(page_table_1.shape[1], dcp_size, page_size)
    local_page_table, local_to_global, local_causal_count = dcp_localize_page_table(
        page_table_1, dcp_size, dcp_rank, capacity, page_size
    )
    local_len = torch.gather(
        local_causal_count, 1, (seq_lens.long() - 1).clamp(min=0).unsqueeze(1)
    ).squeeze(1)
    local_len = torch.where(seq_lens > 0, local_len, torch.zeros_like(local_len))

    global_idx = torch.where(
        local_to_global >= 0,
        local_to_global.long(),
        torch.zeros_like(local_to_global.long()),
    )
    local_logits = torch.gather(scores, 1, global_idx)
    col_ids = torch.arange(local_logits.shape[1], device=scores.device).unsqueeze(0)
    valid_col = col_ids < local_len.unsqueeze(1)
    local_logits = torch.where(
        valid_col, local_logits, torch.full_like(local_logits, float("-inf"))
    )

    k_eff = min(topk, local_logits.shape[1])
    local_topk_idx = local_logits.topk(k_eff, dim=-1).indices.to(torch.int32)
    keep = torch.gather(valid_col, 1, local_topk_idx.long())
    local_topk_idx = torch.where(
        keep, local_topk_idx, torch.full_like(local_topk_idx, -1)
    )
    if k_eff < topk:
        pad = torch.full(
            (page_table_1.shape[0], topk - k_eff),
            -1,
            dtype=torch.int32,
            device=scores.device,
        )
        local_topk_idx = torch.cat([local_topk_idx, pad], dim=1)
    return (
        local_logits.contiguous(),
        local_topk_idx.contiguous(),
        local_to_global.contiguous(),
    )


def _reference_topk(
    scores: torch.Tensor, page_table_1: torch.Tensor, seq_lens: torch.Tensor, topk: int
) -> list[set[int]]:
    # page_table_1[row, i] is the physical slot for logical position i;
    # seq_len bounds the logical position, not the physical slot's numeric
    # value, so the causal window must go through page_table_1.
    ref_sets = []
    for row in range(page_table_1.shape[0]):
        sl = int(seq_lens[row].item())
        k_row = min(topk, sl)
        valid_slots = page_table_1[row, :sl].long()
        valid_scores = scores[row, valid_slots]
        top_local = valid_scores.topk(k_row).indices
        ref_sets.append(set(valid_slots[top_local].tolist()))
    return ref_sets


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sharded_indexer_topk_matches_reference_and_ranks_agree(seed: int) -> None:
    group = _init_group_once()
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    dcp_size = world_size

    num_rows = 6
    page_size = 64
    num_pages = 63  # deliberately not a multiple of world_size (uneven ownership)
    max_seq_len = num_pages * page_size
    # Physical address space pages are scattered within (must cover whatever
    # random page ids get picked below, since those become physical slot
    # values indexing into `scores`).
    physical_num_pages = 500
    physical_size = physical_num_pages * page_size

    torch.manual_seed(1000 + seed)  # identical across ranks: same replicated inputs
    scores = torch.randn(num_rows, physical_size, device=device)
    # Realistic paged layout: each page's page_size rows are internally
    # contiguous, but pages are scattered (non-adjacent) physical addresses --
    # exactly what a paged allocator guarantees (blocks can be anywhere;
    # that's the whole point of block_tables indirection). A flat
    # torch.randperm over individual slots (no page structure) would not
    # exercise the page-contiguity requirement at all.
    page_table_1 = torch.stack(
        [
            torch.cat(
                [
                    torch.arange(
                        p.item() * page_size,
                        p.item() * page_size + page_size,
                        dtype=torch.int32,
                        device=device,
                    )
                    for p in torch.randperm(physical_num_pages, device=device)[
                        :num_pages
                    ]
                ]
            )
            for _ in range(num_rows)
        ]
    )
    seq_lens = torch.randint(200, max_seq_len + 1, (num_rows,), device=device)
    seq_lens[0] = 300  # force a short-context row (< TOPK)

    local_logits, local_topk_idx, local_to_global = _rank_local_topk(
        scores, page_table_1, dcp_size, rank, seq_lens, TOPK, page_size
    )

    # Contiguity check on real hardware: every page_size-row window of this
    # rank's compacted local page table must be one physically contiguous
    # page (the property that broke in production under per-token ownership).
    capacity = dcp_local_capacity(max_seq_len, dcp_size, page_size)
    local_page_table, _, _ = dcp_localize_page_table(
        page_table_1, dcp_size, rank, capacity, page_size
    )
    num_windows = capacity // page_size
    for row in range(num_rows):
        for k in range(num_windows):
            window = local_page_table[row, k * page_size : (k + 1) * page_size]
            if bool((window < 0).all()):
                continue
            expected = torch.arange(
                int(window[0]),
                int(window[0]) + page_size,
                device=device,
                dtype=window.dtype,
            )
            assert torch.equal(window, expected), (
                f"rank {rank} row {row} window {k} is not one contiguous "
                f"physical page: {window.tolist()}"
            )
    packed = torch.empty((num_rows, TOPK, 2), dtype=torch.float32, device=device)
    pack_dcp_topk_candidates_cutedsl(
        local_logits, local_topk_idx, local_to_global, packed, None
    )

    gathered = torch.empty(
        (num_rows, TOPK * world_size, 2), dtype=torch.float32, device=device
    )
    gathered_list = list(gathered.view(world_size, num_rows, TOPK, 2).unbind(0))
    dist.all_gather(gathered_list, packed.contiguous(), group=group)
    gathered = torch.cat(gathered_list, dim=1).contiguous()

    merged = stable_topk_from_gathered_candidates_cutedsl(gathered, TOPK)
    torch.cuda.synchronize()

    ref_sets = _reference_topk(scores, page_table_1, seq_lens, TOPK)
    for row in range(num_rows):
        merged_set = set(merged[row].tolist()) - {-1}
        assert merged_set == ref_sets[row], (
            f"rank {rank} row {row}: sharded pack+merge diverged from the "
            "true global top-k"
        )

    merged_cpu = merged.cpu()
    gathered_check = [torch.empty_like(merged_cpu) for _ in range(world_size)]
    dist.all_gather_object(gathered_check, merged_cpu, group=group)
    for other_rank, other_merged in enumerate(gathered_check):
        for row in range(num_rows):
            a = set(merged_cpu[row].tolist()) - {-1}
            b = set(other_merged[row].tolist()) - {-1}
            assert a == b, f"rank {rank} and rank {other_rank} disagree on row {row}"


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
