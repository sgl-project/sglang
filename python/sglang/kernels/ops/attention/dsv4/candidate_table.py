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

from .topk import topk_transform_bf16_small
from .utils import make_name

if TYPE_CHECKING:
    pass

CANDIDATE_BLOCK_SIZE = 8  # positions per block; DeepGEMM accepts 8 or 16


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


def build_sparse_indexer_schedule(
    blocks: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    q_dtype: torch.dtype,
    request_ids: torch.Tensor,
) -> torch.Tensor:
    """DeepGEMM's schedule for the published blocks: ``seq_lens`` ``[rows]``
    int32, ``page_table`` ``[rows, pages]`` int32 at the index pool's page size.
    ``request_ids`` ``[rows]`` int32 lets DeepGEMM pair two rows of a request on
    one KV pass; each row keeps its own block list and output layout, and paired
    rows must share their page-table row."""
    import deep_gemm

    return deep_gemm.get_paged_sparse_mqa_logits_metadata(
        seq_lens.contiguous(),
        page_table,
        request_ids,
        page_size,
        blocks,
        q_dtype,
        CANDIDATE_BLOCK_SIZE,
    )


def sparse_logits(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    schedule: torch.Tensor,
    topk_blocks: int,
) -> torch.Tensor:
    """bf16 logits ``[rows, topk_blocks * 8]`` of the published blocks: ``q_fp4``
    ``[rows, 1, heads, 64]`` int8 with ``q_sf`` ``[rows, 1, heads]`` int32 (packed
    ue8m0), ``k_cache`` ``[pages, page_size, 1, 68]`` uint8 whose page stride is
    a multiple of 512 bytes, ``weights`` ``[rows, heads]`` bf16, ``schedule`` the
    ``build_sparse_indexer_schedule`` of the ``topk_blocks`` published blocks."""
    import deep_gemm

    return deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
        (q_fp4, q_sf),
        k_cache,
        weights,
        schedule,
        topk_blocks,
        CANDIDATE_BLOCK_SIZE,
    )


def topk_transform_sparse(
    logits: torch.Tensor,
    valid_lens: torch.Tensor,
    phys_blocks: torch.Tensor,
    page_indices: torch.Tensor,
) -> None:
    """Top-``k`` (``k = page_indices.shape[1]``) of every row of the sparse
    ``logits`` (bf16 ``[rows, topk_blocks * 8]``) within its first ``valid_lens[b]``
    columns, written as pool slots, ``-1`` where a row has fewer than ``k`` valid
    columns, in no particular order; ``phys_blocks`` ``[rows, topk_blocks]`` int32
    holds the published blocks as pool slots / 8."""
    topk_transform_bf16_small(
        logits, valid_lens, phys_blocks, page_indices, CANDIDATE_BLOCK_SIZE
    )
