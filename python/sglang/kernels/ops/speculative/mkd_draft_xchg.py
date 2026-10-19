from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# CTAs per exchange; each owns hidden / CHUNKS columns of the candidate and partial rows.
CHUNKS = 16


@cache_once
def _module(hidden: int, world: int, memo: bool, chunks: int) -> Module:
    args = make_cpp_args(hidden, world, memo, chunks, is_arch_support_pdl())
    return load_jit(
        "mkd_draft_xchg",
        *args,
        cuda_files=["speculative/mkd_draft_xchg.cuh"],
        cuda_wrappers=[
            ("run", f"MkdXchg<{args}>::run"),
            ("zero", f"MkdXchg<{args}>::zero"),
        ],
    )


# Rotating epoch slots of the exchange buffer (must match mkd_xchg::kSlots).
SLOTS = 3


def mkd_xchg_words(hidden: int, world: int, memo: bool) -> int:
    """64-bit words of the symmetric exchange buffer (mirrors MkdXchg::kWords)."""
    cols = hidden // CHUNKS
    rec = 2 + (cols if memo else cols // 2) + cols
    return SLOTS * world * CHUNKS * rec


def mkd_xchg_zero(hidden: int, world: int, memo: bool, uc_ptr: int) -> None:
    """Zero this rank's copy of the exchange buffer (tags 0 = nothing arrived)."""
    _module(hidden, world, memo, CHUNKS).zero(uc_ptr)


def mkd_xchg(
    logits: torch.Tensor,
    cand: torch.Tensor,
    hot_ids: torch.Tensor,
    own_partial: torch.Tensor,
    epoch_done: torch.Tensor,
    out_cand: torch.Tensor,
    out_partials: torch.Tensor,
    topk_index: torch.Tensor,
    topk_p: torch.Tensor,
    draft_tokens: torch.Tensor | None,
    positions: torch.Tensor | None,
    mrope_positions: torch.Tensor | None,
    mc_ptr: int,
    uc_ptr: int,
    rank: int,
    world: int,
    draft_column: int,
    nan_key: int,
    trace: torch.Tensor | None = None,
    prefetch: torch.Tensor | None = None,
    prefetch_bytes: int = 0,
    fc_out: torch.Tensor | None = None,
    token_only: bool = False,
    partial_flag: torch.Tensor | None = None,
    scan_part: torch.Tensor | None = None,
    head_part: torch.Tensor | None = None,
) -> None:
    """Shard argmax + multicast exchange + selection for one draft token (see the .cuh).
    ``hot_ids`` is the full sorted hot-vocabulary id list; this rank's rows are its
    ``rank``-th contiguous slice.
    ``cand`` holds bf16 embedding rows or fp32 memo rows of the local hot shard; the winner's
    row lands in ``out_cand`` and the hidden-slice fc partials of every hidden rank in
    ``out_partials`` [world / 2, H], or with memo rows and ``fc_out`` [1, H] bf16 the next
    step's fc output directly. ``token_only`` exchanges the keys alone (last draft step,
    draft extend). ``partial_flag`` (int32, 1 element) makes a hidden rank wait until it equals
    the call's epoch before reading ``own_partial`` (written by a kernel on another stream).
    ``scan_part`` (int64 [SLOTS * CHUNKS], zeroed once) lets large shards be scanned in parts.
    ``head_part`` (int32, the per-CTA maxima of mkd_draft_head on ``logits``) replaces the scan.
    ``trace`` is an int64 ring [slots, CHUNKS, 9] of
    %globaltimer stamps (device or mapped pinned host memory), slot = call epoch % slots."""
    memo = cand.dtype == torch.float32
    empty = topk_index.new_empty(0)
    _module(cand.shape[-1], world, memo, CHUNKS).run(
        logits,
        cand,
        hot_ids,
        own_partial,
        epoch_done,
        out_cand,
        out_partials,
        fc_out
        if fc_out is not None
        else out_partials.new_empty(0, dtype=torch.bfloat16),
        topk_index,
        topk_p,
        draft_tokens if draft_tokens is not None else empty,
        positions if positions is not None else empty,
        mrope_positions if mrope_positions is not None else empty,
        mc_ptr,
        uc_ptr,
        rank,
        draft_column,
        nan_key,
        int(token_only),
        partial_flag.data_ptr() if partial_flag is not None else 0,
        scan_part.data_ptr() if scan_part is not None else 0,
        head_part.data_ptr() if head_part is not None else 0,
        head_part.numel() if head_part is not None else 0,
        trace.data_ptr() if trace is not None else 0,
        trace.numel() // (CHUNKS * 9) if trace is not None else 1,
        prefetch.data_ptr() if prefetch is not None else 0,
        min(prefetch_bytes, prefetch.numel() * prefetch.element_size())
        if prefetch is not None
        else 0,
    )
