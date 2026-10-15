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

META_ELEMS = 8  # packed row tail (bf16 slots): max as fp32 bits, global index, padding
# Tunable: CTAs scanning one vocab shard.
PACK_BLOCKS = 32


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
    part_val: torch.Tensor,
    part_idx: torch.Tensor,
    counter: torch.Tensor,
    vocab_start: int,
    world: int,
) -> None:
    """Shard argmax of bf16 ``logits`` [1, V_local] plus the candidate's ``embed``
    row, packed into ``payload`` [1, H + META_ELEMS]."""
    _module(embed.shape[-1], part_val.shape[0], world).pack(
        logits, embed, payload, part_val, part_idx, counter, vocab_start
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
) -> None:
    """Pick the token over gathered packed rows [1, world * (H + META_ELEMS)];
    write it to ``topk_index`` / ``draft_tokens[:, draft_column]`` (empty tensor
    to skip), advance ``positions`` and copy its embedding row to ``next_embeds``."""
    _module(next_embeds.shape[-1], blocks, world).select(
        gathered, topk_index, topk_p, draft_tokens, positions, next_embeds, draft_column
    )
