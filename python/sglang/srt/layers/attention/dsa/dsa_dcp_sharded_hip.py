"""DSA under DCP on ROCm: the indexer top-k as the exact merge of
``dsa_dcp_sharded`` over the group all-gather, without multicast or the v2 plan,
and the prefill gather of every request's cache rows."""

from typing import Optional

import torch
import triton

from sglang.kernels.ops.attention.dsa.dcp_topk_hip import (
    _candidates_to_slots_kernel,
    _gather_candidates_kernel,
    _pack_candidates_kernel,
    dcp_global_slots_hip,
    dcp_local_candidates_hip,
    supports_dcp_topk_hip,
)
from sglang.srt.runtime_context import get_parallel


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


class DcpTopKHip:
    """ROCm's DCP top-k in ``dsa_dcp_sharded``'s fused top-k slot: local select,
    exchange and exact merge, then this rank's slots compacted with their counts,
    as ``DcpFusedTopK`` returns them. Any row count and context length."""

    def __init__(self, page_size: int):
        self.page_size = page_size

    def usable(self, rows: int) -> bool:
        return True

    def __call__(
        self,
        logits: torch.Tensor,
        local_lens: torch.Tensor,
        global_lens: torch.Tensor,
        page_table: torch.Tensor,
        slots: torch.Tensor,
        out_len: torch.Tensor,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import (
            dcp_localize_compact,
        )

        span = self.page_size * get_parallel().attn_dcp_size
        global_lens = global_lens.to(torch.int32)
        dcp_global_topk(logits, local_lens, global_lens, page_table, slots, span)
        return dcp_localize_compact(slots, out_len)[0]


def dcp_global_topk(
    logits: torch.Tensor,
    local_lens: torch.Tensor,
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    span: int,
) -> torch.Tensor:
    """``dsa_dcp_sharded.dcp_global_topk`` into ``out`` for non-empty rows; only this
    rank's slots are set, which is all ``dcp_localize_compact`` keeps.

    The global top-k is the top-k of the ranks' local top-k candidates, so it selects
    over those (WORLD * topk per row) instead of a context-wide dense buffer."""
    from sgl_kernel import fast_topk_v2

    parallel = get_parallel()
    world = parallel.attn_dcp_size
    rows, topk = out.shape
    if supports_dcp_topk_hip(topk, world):
        # only the scores travel: every rank selects the same winners from them and
        # emits its own, which is all dcp_localize_compact keeps
        scores, positions = dcp_local_candidates_hip(
            logits, local_lens, world=world, rank=parallel.attn_dcp_rank
        )
        return dcp_global_slots_hip(
            parallel.dcp_group.all_gather(scores, dim=0),
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
    gathered = parallel.dcp_group.all_gather(candidates.view(torch.float32), dim=0)
    gathered = gathered.view(torch.int32)
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


# (seq_lens, dcp_size, dcp_rank, device) -> plan; every layer of a batch reuses its batch's plan
_REQUEST_ROWS_PLAN: dict = {}


def request_rows_plan(
    seq_lens_cpu: torch.Tensor, dcp_size: int, dcp_rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Index maps of the DCP prefill gather (positions from 0, one-token interleave).

    Every rank sends ceil(seq_len / dcp_size) slots per request. Returns
    ``send``: this rank's local row for each slot (pad slots read row 0),
    ``recv``: the all-gathered row of each request position, request after
    request, and the slot count per rank.
    """
    key = (tuple(seq_lens_cpu.tolist()), dcp_size, dcp_rank, device)
    plan = _REQUEST_ROWS_PLAN.get(key)
    if plan is not None:
        return plan
    lens = seq_lens_cpu.to(torch.int64)
    slots = (lens + dcp_size - 1) // dcp_size
    owned = (lens - dcp_rank + dcp_size - 1) // dcp_size
    num_slots, num_rows = int(slots.sum()), int(lens.sum())

    def starts(x: torch.Tensor) -> torch.Tensor:
        return (torch.cumsum(x, 0) - x).to(device)

    reqs = torch.arange(len(lens), device=device)
    lens_d, slots_d, owned_d = lens.to(device), slots.to(device), owned.to(device)
    req = torch.repeat_interleave(reqs, slots_d, output_size=num_slots)
    j = torch.arange(num_slots, device=device) - starts(slots)[req]
    send = torch.where(j < owned_d[req], starts(owned)[req] + j, 0)
    req = torch.repeat_interleave(reqs, lens_d, output_size=num_rows)
    pos = torch.arange(num_rows, device=device) - starts(lens)[req]
    recv = (pos % dcp_size) * num_slots + starts(slots)[req] + pos // dcp_size
    _REQUEST_ROWS_PLAN.clear()
    _REQUEST_ROWS_PLAN[key] = plan = (send, recv, num_slots)
    return plan


def _as_words(rows: torch.Tensor) -> torch.Tensor:
    # index_select moves whole rows faster as 8- or 4-byte words than as bytes
    width = rows.shape[-1]
    for dtype, size in ((torch.int64, 8), (torch.int32, 4)):
        if width % size == 0 and rows.is_contiguous():
            return rows.view(dtype)
    return rows


def dcp_gather_request_rows(
    local_rows: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    *,
    row_ids: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    """Prefill: every request's cache rows from all ranks, in position order.

    ``local_rows``: [sum of this rank's shares, W] uint8, request by request
    in local position order; with ``row_ids``, the whole cache, of which this
    rank's rows are ``local_rows[row_ids]``. Returns [sum(seq_lens), W]:
    request after request, the rows of positions 0..seq_len-1 (the layout of
    the ragged top-k indices), zero-padded to whole ``page_size`` pages.
    """
    parallel = get_parallel()
    send_idx, recv_idx, num_slots = request_rows_plan(
        seq_lens_cpu, parallel.dcp_size, parallel.dcp_rank, local_rows.device
    )
    width = local_rows.shape[1]
    owned = local_rows.shape[0] if row_ids is None else row_ids.shape[0]
    if owned == 0:  # every slot of this rank is padding
        send = local_rows.new_empty((num_slots, width))
    else:
        src = send_idx if row_ids is None else row_ids[send_idx]
        send = _as_words(local_rows).index_select(0, src).view(torch.uint8)
    gathered = local_rows.new_empty((parallel.dcp_size * num_slots, width))
    parallel.dcp_group.all_gather_into_tensor(gathered, send.view(torch.uint8))
    num_rows = recv_idx.shape[0]
    out = local_rows.new_empty((num_rows + (-num_rows % page_size), width))
    torch.index_select(_as_words(gathered), 0, recv_idx, out=_as_words(out)[:num_rows])
    out[num_rows:].zero_()
    return out
