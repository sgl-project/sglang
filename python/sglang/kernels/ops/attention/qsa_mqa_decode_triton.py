"""Triton paged decode scoring for the QSA indexer, used on ROCm.

``qsa_mqa_decode_triton`` returns what ``tilelang_qsa_mqa_decode`` returns for
bf16 operands: fp32 logits ``[rows, max_model_len]`` where each scored position
holds ``relu(q . k)`` summed over the first ``num_heads`` query heads and divided
by ``score_scale`` (``sqrt(head_dim)`` by default).

A program scores ``rows_per_group`` consecutive rows that share one page-table
row, such as one request's target-verify draft rows, against a slice of their
keys. It reads each key tile once for the whole group, and the group's rows x
heads form the N dimension of one dot, padded to the 16 columns ROCm MFMA needs.
``rows_per_group=1`` scores row by row.
"""

from __future__ import annotations

import functools
import math
from typing import Optional

import torch
import triton
import triton.language as tl

# Tuned on MI355X at the Qwen3.8 target-verify shape (44 requests x 4 draft rows,
# 4 index heads, head_dim 128): 128-key tiles, 8 warps, no software pipelining,
# and about 11 programs per compute unit.
_BLOCK_N = 128
_NUM_WARPS = 8
_NUM_STAGES = 1
_PROGRAMS_PER_CU = 11
_MAX_SPLITS = 128
_MAX_COLS = 128


@triton.jit
def _qsa_mqa_decode_kernel(
    q_ptr,
    stride_q_row,
    stride_q_head,
    k_ptr,
    table_ptr,
    stride_table,
    lens_ptr,
    out_ptr,
    stride_out,
    max_len,
    inv_scale,
    min_scored_len,
    ROWS: tl.constexpr,
    ROWS_POW2: tl.constexpr,
    HEADS: tl.constexpr,
    COLS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid (groups, splits). The group's rows read the page-table row of its first
    # row and differ only in length: the program scores its split of keys
    # [0, longest) against every row and stores each row's scores below its length.
    group = tl.program_id(0)
    split = tl.program_id(1)
    num_splits = tl.num_programs(1)
    row0 = group * ROWS
    slots = tl.arange(0, ROWS_POW2)
    lens = tl.load(lens_ptr + row0 + slots, mask=slots < ROWS, other=0)
    lens = tl.where((slots < ROWS) & (lens > min_scored_len), lens, 0)
    lens = tl.minimum(lens, max_len)
    longest = tl.max(lens, axis=0)
    tiles = tl.cdiv(longest, BLOCK_N)
    tiles_per_split = tl.cdiv(tiles, num_splits)
    begin = split * tiles_per_split
    end = tl.minimum(begin + tiles_per_split, tiles)

    # Column c of the dot is head c % HEADS of row c // HEADS; padding columns
    # load zeros.
    cols = tl.arange(0, COLS)
    dims = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr
        + (row0 + cols // HEADS)[:, None] * stride_q_row
        + (cols % HEADS)[:, None] * stride_q_head
        + dims[None, :],
        mask=(cols // HEADS < ROWS)[:, None],
        other=0.0,
    )
    q_t = tl.trans(q)
    table = table_ptr + row0 * stride_table
    offs = tl.arange(0, BLOCK_N)
    for tile in range(begin, end):
        pos = tile * BLOCK_N + offs
        valid = pos < longest
        page = tl.load(table + pos // PAGE_SIZE, mask=valid, other=0)
        k = tl.load(
            k_ptr
            + (page * PAGE_SIZE + pos % PAGE_SIZE)[:, None] * HEAD_DIM
            + dims[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.maximum(tl.dot(k, q_t), 0.0) * inv_scale
        for r in tl.static_range(ROWS):
            length = tl.sum(tl.where(slots == r, lens, 0), axis=0)
            row_scores = tl.sum(
                tl.where((cols // HEADS == r)[None, :], scores, 0.0), axis=1
            )
            tl.store(
                out_ptr + (row0 + r) * stride_out + pos,
                row_scores,
                mask=pos < length,
            )


@functools.lru_cache(maxsize=None)
def _num_compute_units(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def can_use_qsa_mqa_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    rows_per_group: int = 1,
    num_heads: Optional[int] = None,
) -> bool:
    """Whether ``qsa_mqa_decode_triton`` serves these inputs: bf16 operands, a
    ``[pages, page_size, 1, head_dim]`` cache with a power-of-two head_dim, whole
    groups of at most 128 row-heads, and offsets that fit in 32 bits."""
    if q.ndim != 3 or k_cache.ndim != 4:
        return False
    heads = num_heads or q.shape[1]
    return (
        q.dtype == torch.bfloat16
        and k_cache.dtype == torch.bfloat16
        and q.stride(2) == 1
        and k_cache.shape[2] == 1
        and k_cache.is_contiguous()
        and q.shape[2] == k_cache.shape[3] == triton.next_power_of_2(q.shape[2])
        and 0 < heads <= q.shape[1]
        and rows_per_group >= 1
        and rows_per_group * heads <= _MAX_COLS
        and q.shape[0] % rows_per_group == 0
        and page_table.ndim == 2
        and page_table.shape[0] == q.shape[0] == context_lens.numel()
        and k_cache.numel() < 2**31
        and q.shape[0] * max(max_model_len, 1) < 2**31
    )


def qsa_mqa_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
    *,
    min_scored_len: int = 0,
    rows_per_group: int = 1,
    num_heads: Optional[int] = None,
) -> torch.Tensor:
    """Paged decode MQA logits, as ``tilelang_qsa_mqa_decode``.

    Each run of ``rows_per_group`` consecutive rows must share its first row's
    page-table row; only the first ``num_heads`` heads of ``q`` are read. With
    ``min_scored_len`` > 0, rows whose context length is at most that are not
    scored and logits past each row's context length are left unset; otherwise
    they are -inf. Check the inputs with ``can_use_qsa_mqa_decode_triton``."""
    rows, head_dim = q.shape[0], q.shape[2]
    heads = num_heads or q.shape[1]
    if min_scored_len > 0:
        logits = torch.empty(
            (rows, max_model_len), dtype=torch.float32, device=q.device
        )
    else:
        logits = torch.full(
            (rows, max_model_len), -float("inf"), dtype=torch.float32, device=q.device
        )
    if not rows or not max_model_len:
        return logits
    groups = rows // rows_per_group
    programs = _PROGRAMS_PER_CU * _num_compute_units(q.device.index or 0)
    splits = min(_MAX_SPLITS, triton.next_power_of_2(triton.cdiv(programs, groups)))
    page_size = k_cache.shape[1]
    table = page_table.to(device=q.device, dtype=torch.int32).contiguous()
    _qsa_mqa_decode_kernel[(groups, splits)](
        q,
        q.stride(0),
        q.stride(1),
        k_cache.view(-1, head_dim),
        table,
        table.stride(0),
        context_lens.to(device=q.device, dtype=torch.int32).contiguous(),
        logits,
        logits.stride(0),
        min(max_model_len, table.shape[1] * page_size),
        1.0 / float(score_scale or math.sqrt(head_dim)),
        min_scored_len,
        ROWS=rows_per_group,
        ROWS_POW2=triton.next_power_of_2(rows_per_group),
        HEADS=heads,
        COLS=max(16, triton.next_power_of_2(rows_per_group * heads)),
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return logits


__all__ = ["can_use_qsa_mqa_decode_triton", "qsa_mqa_decode_triton"]
