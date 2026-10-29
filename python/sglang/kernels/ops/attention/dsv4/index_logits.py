"""The index logits of the DeepSeek V4.1 ratio-1/2 index layers, by the layout of
the index K they read: the flattened K of a prefill chunk (tile by tile under a
memory budget), the paged pool, or the published blocks of a sparse table."""

from __future__ import annotations

import functools
from typing import Iterator, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.mqa_logits_utils import (
    mqa_logits_row_bytes,
    mqa_logits_rows_per_chunk,
)
from sglang.srt.utils.common import ceil_align

from .candidate_table import CANDIDATE_BLOCK_SIZE


@functools.cache
def _is_sm100() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability()[0] == 10
    )


@functools.cache
def index_logits_dtype() -> torch.dtype:
    """bf16 on SM100, where the reference kernels score in bf16; fp32 elsewhere.
    SGLANG_OPT_DSV41_BF16_INDEX_LOGITS, when set, overrides the choice."""
    forced = envs.SGLANG_OPT_DSV41_BF16_INDEX_LOGITS.get()
    bf16 = _is_sm100() if forced is None else forced
    return torch.bfloat16 if bf16 else torch.float32


@functools.cache
def index_weights_dtype() -> torch.dtype:
    """DeepGEMM reduces in the head weights' dtype and takes bf16 weights only on
    SM100 with bf16 logits."""
    if index_logits_dtype() == torch.bfloat16 and _is_sm100():
        return torch.bfloat16
    return torch.float32


# Width alignment of the flat logits: covers block-wise reads of up to 64 columns, and 128 B of bf16, which
# DeepSelect reads past a row's end (DeepGEMM pads the stride but its tensor's storage ends at the last row)
FLAT_LOGITS_WIDTH_ALIGN = 64


def flat_index_logits_rows_per_tile(
    rows: int,
    width: int,
    *,
    heads: int,
    budget_bytes: int,
) -> int:
    """Rows per logits tile within ``budget_bytes``, at the kernel's row alignment.
    Tiles are sized as fp32 logits whatever the logits dtype: bf16 tiles then fill
    half the budget, which leaves room for the per-row selection scratch that
    grows with the tile."""
    row_alignment = 128 // heads
    rows_per_chunk = mqa_logits_rows_per_chunk(
        num_rows=ceil_align(rows, row_alignment),
        row_bytes=mqa_logits_row_bytes(width),
        budget_bytes=budget_bytes,
    )
    if rows_per_chunk is None:
        return rows
    return max(row_alignment, rows_per_chunk // row_alignment * row_alignment)


def flat_index_logits_tiles(
    *,
    q: tuple[torch.Tensor, torch.Tensor],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    context_lengths: list[int],
    budget_bytes: int,
) -> Iterator[tuple[slice, torch.Tensor]]:
    """``(rows, logits)`` per row tile, each tile within ``budget_bytes``:
    ``logits[i, j]`` scores query row ``rows.start + i`` against ``kv[starts + j]``,
    garbage past the row's ``lengths``; the width is ``max(context_lengths)``
    aligned to ``FLAT_LOGITS_WIDTH_ALIGN``."""
    from deep_gemm import fp8_fp4_mqa_logits

    rows = q[0].shape[0]
    width = ceil_align(max(context_lengths, default=0), FLAT_LOGITS_WIDTH_ALIGN)
    if rows == 0 or width == 0:
        return
    rows_per_chunk = flat_index_logits_rows_per_tile(
        rows, width, heads=q[0].shape[1], budget_bytes=budget_bytes
    )
    weights = weights.to(index_weights_dtype())
    logits_dtype = index_logits_dtype()
    for offset in range(0, rows, rows_per_chunk):
        tile = slice(offset, min(offset + rows_per_chunk, rows))
        tile_starts = starts[tile]
        yield (
            tile,
            fp8_fp4_mqa_logits(
                (q[0][tile], q[1][tile]),
                kv,
                weights[tile],
                tile_starts,
                tile_starts + lengths[tile],
                False,
                width,
                logits_dtype=logits_dtype,
            ),
        )


def deep_gemm_fp4_paged_mqa_logits(
    q_fp4: Tuple[torch.Tensor, torch.Tensor],
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata,
    max_seq_len: int,
) -> torch.Tensor:
    """DeepGEMM paged fp4 logits; no hadamard, the reference does not apply one."""
    from deep_gemm import fp8_fp4_paged_mqa_logits

    sl = seq_lens.to(torch.int32)
    if sl.dim() == 1:
        sl = sl.unsqueeze(-1)
    return fp8_fp4_paged_mqa_logits(
        q_fp4,
        k_cache,
        weights.to(index_weights_dtype()),
        sl,
        page_table,
        deep_gemm_metadata,
        max_seq_len,
        False,
        logits_dtype=index_logits_dtype(),
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
