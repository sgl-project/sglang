"""DSA indexer top-k under DCP on ROCm: the exact merge of ``dsa_dcp_sharded``
over the group all-gather, without multicast or the v2 plan."""

from functools import partial
from typing import Callable, Optional

import torch
import triton

from sglang.kernels.ops.attention.dsa.dcp_topk_hip import (
    _all_visible_slots_kernel,
    _candidates_to_slots_kernel,
    _gather_candidates_kernel,
    _pack_candidates_kernel,
    dcp_global_slots_hip,
    dcp_local_candidates_hip,
    supports_dcp_topk_hip,
)
from sglang.srt.runtime_context import get_exec, get_parallel


def _local_candidates(
    logits: torch.Tensor, local_lens: torch.Tensor, topk: int
) -> torch.Tensor:
    # sgl-kernel's HIP top-k emits local positions (-1 padded); ties resolve in
    # its own order rather than by position as the CUDA v2 kernel does.
    from sgl_kernel import fast_topk_v2

    parallel = get_parallel()
    rows = local_lens.shape[0]
    out = torch.empty((rows, topk, 2), dtype=torch.int32, device=local_lens.device)
    if rows == 0:
        return out
    if logits.shape[1] == 0:
        out[..., 1] = -1
        return out
    local_idx = fast_topk_v2(logits, local_lens.to(torch.int32), topk)
    _pack_candidates_kernel[(rows,)](
        logits,
        logits.stride(0),
        local_idx,
        out,
        WORLD=parallel.attn_dcp_size,
        RANK=parallel.attn_dcp_rank,
        TOPK=topk,
    )
    return out


def dcp_global_topk(
    logits: torch.Tensor,
    local_lens: torch.Tensor,
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    span: int,
    all_gather: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """``dsa_dcp_sharded.dcp_global_topk`` into ``out`` for its non-empty rows.

    The global top-k is the top-k of the ranks' local top-k candidates, so it selects
    over those (WORLD * topk per row) instead of a context-wide dense buffer."""
    from sgl_kernel import fast_topk_v2

    parallel = get_parallel()
    world = parallel.attn_dcp_size
    rows, topk = out.shape
    if all_gather is None:
        all_gather = partial(parallel.dcp_group.all_gather, dim=0)
    # indexer top-k capture needs every rank's slots
    if supports_dcp_topk_hip(topk, world) and not (
        get_exec().features.enable_return_indexer_topk
    ):
        # only the scores travel: every rank selects the same winners from them and
        # emits its own, which is all dcp_localize_compact keeps
        scores, positions = dcp_local_candidates_hip(
            logits, local_lens, world=world, rank=parallel.attn_dcp_rank
        )
        return dcp_global_slots_hip(
            all_gather(scores),
            positions,
            global_lens,
            page_table,
            out,
            span=span,
            world=world,
            rank=parallel.attn_dcp_rank,
        )
    candidates = _local_candidates(logits, local_lens, topk)
    # AITER's custom all-gather takes float dtypes only; the int32 (score bits,
    # position) pairs travel bit-exact as float32, else they would go to RCCL.
    gathered = all_gather(candidates.view(torch.float32)).view(torch.int32)
    num_candidates = world * topk
    scores = torch.empty(
        (rows, num_candidates), dtype=torch.float32, device=logits.device
    )
    positions = torch.empty_like(scores, dtype=torch.int32)
    lengths = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    block = 1024
    _gather_candidates_kernel[(gathered.shape[0], triton.cdiv(topk, block))](
        gathered,
        rows,
        global_lens,
        scores,
        positions,
        lengths,
        WORLD=world,
        TOPK=topk,
        BLOCK=block,
    )
    selected = fast_topk_v2(scores, lengths, topk)
    _candidates_to_slots_kernel[(rows,)](
        selected,
        positions,
        page_table,
        page_table.stride(0),
        out,
        span,
        NUM_CANDIDATES=num_candidates,
        TOPK=topk,
    )
    return out


def dcp_all_visible(
    global_lens: torch.Tensor, page_table: torch.Tensor, topk: int, span: int
) -> torch.Tensor:
    """``dsa_dcp_sharded.dcp_all_visible`` without the v2 top-k pass."""
    rows = global_lens.shape[0]
    out = torch.empty((rows, topk), dtype=torch.int32, device=global_lens.device)
    if rows == 0:
        return out
    _all_visible_slots_kernel[(rows,)](
        global_lens.to(torch.int32),
        page_table,
        page_table.stride(0),
        out,
        span,
        TOPK=topk,
    )
    return out
