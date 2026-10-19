from __future__ import annotations

import struct
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

META_ELEMS = (
    8  # packed row tail (bf16 slots): max as an order key, global index, padding
)
# Tunable: CTAs scanning one vocab shard.
PACK_BLOCKS = 32


def _order_key(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else (bits | 0x80000000)


# NaN handling of the reference argmax: the split argmax of draft steps reads NaN
# as -1e30; torch.argmax ranks NaN above every value.
NAN_KEY_SPLIT_ARGMAX = _order_key(-1e30)
NAN_KEY_TORCH_ARGMAX = 0xFFFFFFFF


@cache_once
def _module(hidden: int, blocks: int, world: int) -> Module:
    args = make_cpp_args(hidden, blocks, world, is_arch_support_pdl())
    return load_jit(
        "draft_vocab_argmax",
        *args,
        cuda_files=["speculative/draft_vocab_argmax.cuh"],
        cuda_wrappers=[
            ("pack", f"DraftVocabArgmax<{args}>::pack"),
            ("select", f"DraftVocabArgmax<{args}>::select"),
        ],
    )


def vocab_argmax_pack(
    logits: torch.Tensor,
    embed: torch.Tensor,
    payload: torch.Tensor,
    part_key: torch.Tensor,
    part_idx: torch.Tensor,
    counter: torch.Tensor,
    vocab_start: int,
    world: int,
    nan_key: int = NAN_KEY_SPLIT_ARGMAX,
    token_map: torch.Tensor | None = None,
) -> None:
    """Per-row shard argmax of bf16 ``logits`` [M, V_local] plus the candidate's
    ``embed`` row, packed into ``payload`` [M, H + META_ELEMS]; int32 scratch
    part_key / part_idx [M, blocks], counter [M] (zero between calls).
    ``token_map`` (int64 [V_local], increasing) gives each local row's vocabulary id
    for a vocabulary-subset shard; otherwise the id is ``vocab_start`` + row."""
    if token_map is None:
        token_map = logits.new_empty(0, dtype=torch.int64)
    _module(embed.shape[-1], part_key.shape[-1], world).pack(
        logits,
        embed,
        payload,
        part_key,
        part_idx,
        counter,
        token_map,
        vocab_start,
        nan_key,
    )


def vocab_argmax_select(
    gathered: torch.Tensor,
    topk_index: torch.Tensor,
    topk_p: torch.Tensor,
    draft_tokens: torch.Tensor,
    positions: torch.Tensor,
    next_embeds: torch.Tensor,
    draft_column: int,
    blocks: int,
    world: int,
    mrope_positions: torch.Tensor | None = None,
) -> None:
    """Per row, pick the token over gathered packed rows [M, world * (H + META_ELEMS)];
    write it to ``topk_index`` / ``draft_tokens[:, draft_column]``, advance
    ``positions`` and the [3, M] ``mrope_positions`` (empty tensors / None skip them)
    and copy its embedding row to ``next_embeds``."""
    if mrope_positions is None:
        mrope_positions = positions.new_empty(0)
    _module(next_embeds.shape[-1], blocks, world).select(
        gathered,
        topk_index,
        topk_p,
        draft_tokens,
        positions,
        mrope_positions,
        next_embeds,
        draft_column,
    )


@cache_once
def _lm_head_module(hidden: int, rows: int, stages: int, ctas_per_sm: int) -> Module:
    args = make_cpp_args(hidden, rows, stages, ctas_per_sm, is_arch_support_pdl())
    return load_jit(
        "lm_head_argmax",
        *args,
        cuda_files=["speculative/draft_vocab_argmax.cuh"],
        cuda_wrappers=[("run", f"LmHeadArgmax<{args}>::run")],
    )


def lm_head_argmax_pack(
    x: torch.Tensor,
    lm_head_w: torch.Tensor,
    embed: torch.Tensor,
    payload: torch.Tensor,
    part_key: torch.Tensor,
    part_idx: torch.Tensor,
    counter: torch.Tensor,
    vocab_start: int,
    stages: int = 3,
    rows: int = 8,
    ctas_per_sm: int = 1,
    nan_key: int = NAN_KEY_SPLIT_ARGMAX,
) -> None:
    """Shard lm_head GEMV of x [1, H] fused with the argmax and candidate pack
    (same payload as ``vocab_argmax_pack``); int32 part buffers sized >= SM count."""
    _lm_head_module(x.shape[-1], rows, stages, ctas_per_sm).run(
        x, lm_head_w, embed, payload, part_key, part_idx, counter, vocab_start, nan_key
    )
