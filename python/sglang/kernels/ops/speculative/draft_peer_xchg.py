from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.speculative.draft_vocab_argmax import META_ELEMS, PACK_BLOCKS

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Arrival markers the slots hold until the peer's store lands (see draft_peer_xchg.cuh).
ROW_MARKER = 0x80008000
FP32_MARKER = 0xFFFFFFFF


def as_int32_word(word: int) -> int:
    return word - (1 << 32) if word >= 1 << 31 else word


@cache_once
def _argmax_module(hidden: int, blocks: int) -> Module:
    args = make_cpp_args(hidden, blocks, is_arch_support_pdl())
    return load_jit(
        "draft_peer_xchg_argmax",
        *args,
        cuda_files=["speculative/draft_peer_xchg.cuh"],
        cuda_wrappers=[("run", f"DraftArgmaxXchg<{args}>::run")],
    )


@cache_once
def _fc_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "draft_peer_xchg_fc",
        *args,
        cuda_files=["speculative/draft_peer_xchg.cuh"],
        cuda_wrappers=[("run", f"FcXchgSum<{args}>::run")],
    )


def argmax_slots_shape(max_rows: int, hidden: int) -> tuple:
    """[2 halves, rows, 2 ranks, hidden + META_ELEMS] bf16."""
    return (2, max_rows, 2, hidden + META_ELEMS)


def fc_slots_shape(max_rows: int, cols: int) -> tuple:
    """[2 halves, rows, cols] fp32."""
    return (2, max_rows, cols)


def fc_epoch_shape(max_rows: int, cols: int) -> tuple:
    return (max_rows, cols // 1024)


def draft_argmax_xchg(
    logits: torch.Tensor,
    embed: torch.Tensor,
    part_key: torch.Tensor,
    part_idx: torch.Tensor,
    counter: torch.Tensor,
    token_map: torch.Tensor,
    slots: torch.Tensor,
    peer_slots: int,
    epoch: torch.Tensor,
    topk_index: torch.Tensor,
    topk_p: torch.Tensor,
    draft_tokens: torch.Tensor,
    positions: torch.Tensor,
    mrope_positions: torch.Tensor,
    next_embeds: torch.Tensor,
    vocab_start: int,
    nan_key: int,
    rank: int,
    draft_column: int,
) -> None:
    """``vocab_argmax_pack`` + packed-row all-gather + ``vocab_argmax_select`` of a two-rank
    group in one launch; ``slots`` is this rank's symmetric buffer and ``peer_slots`` the
    peer's (data pointer), both filled with ROW_MARKER before first use. Empty tensors skip
    ``token_map`` / ``draft_tokens`` / ``positions`` / ``mrope_positions``."""
    _argmax_module(embed.shape[-1], part_key.shape[-1]).run(
        logits,
        embed,
        part_key,
        part_idx,
        counter,
        token_map,
        slots,
        peer_slots,
        epoch,
        topk_index,
        topk_p,
        draft_tokens,
        positions,
        mrope_positions,
        next_embeds,
        vocab_start,
        nan_key,
        rank,
        draft_column,
    )


def fc_xchg_sum(
    partial: torch.Tensor,
    out: torch.Tensor,
    slots: torch.Tensor,
    peer_slots: int,
    epoch: torch.Tensor,
) -> torch.Tensor:
    """``out`` = bf16(rank-0 partial + rank-1 partial) of a two-rank group, the peer's fp32
    partial pushed into this rank's ``slots`` (filled with FP32_MARKER before first use)."""
    _fc_module().run(partial, out, slots, peer_slots, epoch)
    return out


__all__ = [
    "FP32_MARKER",
    "PACK_BLOCKS",
    "ROW_MARKER",
    "argmax_slots_shape",
    "as_int32_word",
    "draft_argmax_xchg",
    "fc_epoch_shape",
    "fc_slots_shape",
    "fc_xchg_sum",
]
