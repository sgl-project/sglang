"""DCP sparse-attention top-2048 on AMD (csrc/dsa/dcp_topk_hip.cuh).

One launch picks a rank's local candidates; after the scores alone are all-gathered, one picks
the global top-2048 and keeps the rank's own winners as KV slots. Positions never leave the
rank. Row lengths are read on the device, so the logits buffer may be wider than any row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

TOPK = 2048
# Scores per thread held in registers (1024 threads); longer rows re-read the excess
# per pass. 16 covers a DCP8 rank's share of a 131K context; 32 spills on gfx950.
_LOCAL_PER_THREAD = 16


@cache_once
def _local_module(per: int) -> Module:
    args = make_cpp_args(per)
    return load_jit(
        "dcp_local_candidates_hip",
        *args,
        cuda_files=["dsa/dcp_topk_hip.cuh"],
        cuda_wrappers=[("run", f"DcpLocalCandidatesHip<{args}>::run")],
    )


@cache_once
def _global_module(world: int) -> Module:
    args = make_cpp_args(world)
    return load_jit(
        "dcp_global_slots_hip",
        *args,
        cuda_files=["dsa/dcp_topk_hip.cuh"],
        cuda_wrappers=[("run", f"DcpGlobalSlotsHip<{args}>::run")],
    )


def supports_dcp_topk_hip(topk: int, world: int) -> bool:
    return topk == TOPK and world in (2, 4, 8)


def dcp_local_candidates_hip(
    logits: torch.Tensor, local_lens: torch.Tensor, world: int, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The top-2048 of logits[i, :local_lens[i]]: [rows, 2048] fp32 scores (-inf padded), to
    all-gather, and their [rows, 2048] int32 global positions (-1 padded), kept on the rank.
    Local position j is global position j * world + rank."""
    rows = local_lens.shape[0]
    scores = torch.empty((rows, TOPK), dtype=torch.float32, device=logits.device)
    positions = torch.empty((rows, TOPK), dtype=torch.int32, device=logits.device)
    _local_module(_LOCAL_PER_THREAD).run(
        logits, local_lens.to(torch.int32), scores, positions, world, rank
    )
    return scores, positions


def dcp_global_slots_hip(
    gathered: torch.Tensor,
    positions: torch.Tensor,
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    span: int,
    world: int,
    rank: int,
) -> torch.Tensor:
    """This rank's share of the top-2048 per row of the rank-major [world * rows, 2048]
    gathered scores, with ``positions`` this rank's from dcp_local_candidates_hip.

    Written to out [rows, 2048] as slots page_table[p // span] * span + p % span where the
    winner is this rank's, -1 elsewhere: every rank computes the same winner set, so the
    ranks' outputs are disjoint and their union is the global top-k."""
    _global_module(world).run(
        gathered, positions, global_lens, page_table, out, span, rank
    )
    return out
