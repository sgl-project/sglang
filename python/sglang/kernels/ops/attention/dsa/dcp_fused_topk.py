"""Fused global top-k of the DSA indexer for decode context parallelism over 8
ranks; the algorithm is described in ``jit/csrc/dsa/dcp_fused_topk.cuh``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

DCP_FUSED_TOPK_RANKS = 8
DCP_FUSED_TOPK_K = 2048
DCP_FUSED_TOPK_PAGE_SIZE = 64
# the kernel keeps a row's page ids (131072 local keys) in shared memory
DCP_FUSED_TOPK_MAX_CONTEXT = 1 << 20
# Local ranks [lo, hi] each rank sends around its expected share k / 8 = 256;
# measured on real rows: no certificate failure in 1.09M shard-calls.
DCP_FUSED_TOPK_WINDOW = (176, 336)


@cache_once
def _jit_dcp_fused_topk_module() -> Module:
    return load_jit(
        "dcp_fused_topk",
        cuda_files=["dsa/dcp_fused_topk.cuh"],
        cuda_wrappers=[
            ("dcp_fused_topk", "DcpFusedTopKKernel::run"),
            ("buffer_words", "DcpFusedTopKKernel::buffer_words"),
        ],
    )


def dcp_fused_topk_buffer_numel(max_rows: int) -> int:
    """int32 entries of the symmetric buffer."""
    return _jit_dcp_fused_topk_module().buffer_words(max_rows)


def dcp_fused_topk(
    logits: torch.Tensor,
    local_lens: torch.Tensor,
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    out_local: torch.Tensor,
    out_slots: torch.Tensor,
    out_len: torch.Tensor,
    buffer: torch.Tensor,
    multicast_ptr: int,
    counters: torch.Tensor,
    fallbacks: torch.Tensor,
    dcp_rank: int,
    window: tuple[int, int] = DCP_FUSED_TOPK_WINDOW,
) -> None:
    """Global top-k of one batch of decode rows; every rank of the group must
    call it with the same rows, in the same order, and at most one rank's SM
    count of rows (all CTAs of all ranks are resident at once).

    ``logits``: [rows, >= max local_lens] fp32, row-strided, this rank's scores
    of local positions 0 .. local_lens[i]-1 (global position p = 8 * local +
    dcp_rank); NaN ranks as -inf. ``local_lens`` / ``global_lens``: [rows] int32,
    read before the kernel's PDL wait, so no kernel still running ahead of it in
    the stream may write them. Contract: context length <= 1M (local_lens <=
    131072); the kernel reads only the first 131072 keys of a longer row.
    ``page_table``: [>= rows, pages] int32 this rank's page ids of 64 local
    positions. ``out_local``: [rows, 2048] int32 this rank's keys of the global
    top-k as local KV slots ``page_table[i // 64] * 64 + i % 64``, -1 after
    them; ``out_slots`` the same keys as widened slots ``8 * local + dcp_rank``;
    ``out_len``: [rows] int32 their count. ``buffer``: the int32 symmetric
    buffer (``dcp_fused_topk_buffer_numel``), zero-initialized on every rank
    before the rendezvous, ``multicast_ptr`` its multicast address.
    ``counters`` / ``fallbacks``: [max_rows] int32 exchanges / exact fallbacks
    per row slot, zero-initialized together with the buffer; always the same
    full tensors, since their length sets the buffer layout. ``window``: local
    ranks each rank sends; a narrower one only makes the fallback more likely.
    """
    _jit_dcp_fused_topk_module().dcp_fused_topk(
        logits,
        local_lens,
        global_lens,
        page_table,
        out_local,
        out_slots,
        out_len,
        buffer,
        multicast_ptr,
        counters,
        fallbacks,
        dcp_rank,
        window[0],
        window[1],
    )
