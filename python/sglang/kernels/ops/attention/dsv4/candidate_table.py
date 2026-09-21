"""Candidate-table kernels of the two-level indexer: level-one block keys and
the sorted, page-transformed block table."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from .utils import make_name

if TYPE_CHECKING:
    pass


@cache_once
def _jit_block_amax_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("block_amax"),
        *args,
        cuda_files=["deepseek_v4/block_amax.cuh"],
        cuda_wrappers=[("amax8_varlen", f"BlockAmaxKernel<{args}>::amax8_varlen")],
    )


def amax8_varlen(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int = 0,
    *,
    max_seqlen: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Level-one keys of the two-level indexer: ``out[b, i]`` is the max of
    ``scores[b, 8 i : 8 i + 8]`` for ``i < ceil(seq_lens[b] / 8)``, the last of
    them ``+inf`` (the newest block is always selected), nothing written past
    that count. Rows with at most ``topk`` blocks are skipped (every block is
    selected anyway); ``topk=0`` never skips. ``out`` is allocated as
    ``[rows, ceil(max_seqlen / 8)]`` when not given, ``max_seqlen`` defaulting to
    the width of ``scores``; every ``seq_lens[b]`` must fit in ``8 * out.shape[1]``.
    fp32 only for now; ``scores`` rows must be 32-byte aligned (stride a multiple
    of 8). Returns ``out``.
    """
    if out is None:
        num_tokens, max_len = scores.shape
        if max_seqlen == 0:
            max_seqlen = max_len
        out = scores.new_empty(num_tokens, (max_seqlen + 7) // 8)
    _jit_block_amax_module().amax8_varlen(scores, seq_lens, out, topk)
    return out


@cache_once
def _jit_candidate_block_table_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("candidate_block_table"),
        *args,
        cuda_files=["deepseek_v4/candidate_block_table.cuh"],
        cuda_wrappers=[("transform", f"CandidateBlockTableKernel<{args}>::transform")],
    )


def sort_candidate_blocks(
    blocks: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    *,
    out_pages: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The block table of the two-level indexer from a row's selected blocks,
    in place: ``blocks`` ``[rows, k]`` int32 block ids in any order, ``-1``
    padded, become the same ids ascending with ``INT32_MAX`` past ``min(k,
    ceil(seq_lens[b] / 8))``; the matching pool slots / 8 (``page_table[b, id //
    bpp] * bpp + id % bpp``, ``bpp = page_size // 8``, same padding) go to
    ``out_pages``. A row with at most ``k`` blocks gets the identity table
    regardless of its input. Returns ``out_pages``.
    """
    if out_pages is None:
        out_pages = torch.empty_like(blocks)
    _jit_candidate_block_table_module().transform(
        blocks, seq_lens, page_table, out_pages, page_size
    )
    return out_pages
