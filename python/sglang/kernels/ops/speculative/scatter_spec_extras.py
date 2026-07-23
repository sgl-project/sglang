from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_rows_kernel(
    idx_ptr,
    idx_stride,
    s0,
    d0,
    n0: tl.constexpr,
    s1,
    d1,
    n1: tl.constexpr,
    s2,
    d2,
    n2: tl.constexpr,
    s3,
    d3,
    n3: tl.constexpr,
    s4,
    d4,
    n4: tl.constexpr,
    s5,
    d5,
    n5: tl.constexpr,
    HAS_TOPK: tl.constexpr,
    HAS_HIDDEN: tl.constexpr,
    HAS_DRAFT_PROBS: tl.constexpr,
    HAS_DSA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scatter one source row into several request-pool-indexed buffers."""
    row = tl.program_id(0)
    row_offset = row.to(tl.int64)
    column_block = tl.program_id(1)
    dst_row = tl.load(idx_ptr + row_offset * idx_stride).to(tl.int64)
    columns = column_block * BLOCK + tl.arange(0, BLOCK)

    mask0 = columns < n0
    values0 = tl.load(s0 + row_offset * n0 + columns, mask=mask0)
    tl.store(
        d0 + dst_row * n0 + columns,
        values0.to(d0.dtype.element_ty),
        mask=mask0,
    )

    if HAS_TOPK:
        mask1 = columns < n1
        values1 = tl.load(s1 + row_offset * n1 + columns, mask=mask1)
        tl.store(
            d1 + dst_row * n1 + columns,
            values1.to(d1.dtype.element_ty),
            mask=mask1,
        )

        mask2 = columns < n2
        values2 = tl.load(s2 + row_offset * n2 + columns, mask=mask2)
        tl.store(
            d2 + dst_row * n2 + columns,
            values2.to(d2.dtype.element_ty),
            mask=mask2,
        )

    if HAS_HIDDEN:
        mask3 = columns < n3
        values3 = tl.load(s3 + row_offset * n3 + columns, mask=mask3)
        tl.store(
            d3 + dst_row * n3 + columns,
            values3.to(d3.dtype.element_ty),
            mask=mask3,
        )

    if HAS_DRAFT_PROBS:
        mask4 = columns < n4
        values4 = tl.load(s4 + row_offset * n4 + columns, mask=mask4)
        tl.store(
            d4 + dst_row * n4 + columns,
            values4.to(d4.dtype.element_ty),
            mask=mask4,
        )

    if HAS_DSA:
        mask5 = columns < n5
        values5 = tl.load(s5 + row_offset * n5 + columns, mask=mask5)
        tl.store(
            d5 + dst_row * n5 + columns,
            values5.to(d5.dtype.element_ty),
            mask=mask5,
        )


def _row_width(tensor: torch.Tensor) -> int:
    """Flattened element count per leading-dimension row."""
    return math.prod(tensor.shape[1:])


def _pair_metadata(
    destination: torch.Tensor, source: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    row_width = _row_width(destination)
    return source, destination, row_width


def scatter_spec_extras(
    indices: torch.Tensor,
    *,
    output_tokens_buf: torch.Tensor,
    bonus_tokens: torch.Tensor,
    topk_p_buf: Optional[torch.Tensor] = None,
    topk_p: Optional[torch.Tensor] = None,
    topk_index_buf: Optional[torch.Tensor] = None,
    topk_index: Optional[torch.Tensor] = None,
    hidden_states_buf: Optional[torch.Tensor] = None,
    hidden_states: Optional[torch.Tensor] = None,
    draft_probs_buf: Optional[torch.Tensor] = None,
    draft_probs: Optional[torch.Tensor] = None,
    dsa_topk_indices_buf: Optional[torch.Tensor] = None,
    dsa_topk_indices: Optional[torch.Tensor] = None,
) -> None:
    """Scatter contiguous speculative-decoding relay fields by request-pool row.

    This CUDA-only launcher is the inverse of :func:`gather_spec_extras`. The
    caller owns shape, layout, device, aliasing, index-range, and dtype
    invariants. Destination indices must be unique.
    """
    if indices.shape[0] == 0:
        return

    output = _pair_metadata(output_tokens_buf, bonus_tokens)
    topk_p_pair = _pair_metadata(topk_p_buf, topk_p) if topk_p_buf is not None else None
    topk_index_pair = (
        _pair_metadata(topk_index_buf, topk_index)
        if topk_index_buf is not None
        else None
    )
    hidden = (
        _pair_metadata(hidden_states_buf, hidden_states)
        if hidden_states_buf is not None
        else None
    )
    draft = (
        _pair_metadata(draft_probs_buf, draft_probs)
        if draft_probs_buf is not None
        else None
    )
    dsa = (
        _pair_metadata(dsa_topk_indices_buf, dsa_topk_indices)
        if dsa_topk_indices_buf is not None
        else None
    )
    enabled_pairs = [
        pair
        for pair in (output, topk_p_pair, topk_index_pair, hidden, draft, dsa)
        if pair is not None
    ]
    max_row_width = max(pair[2] for pair in enabled_pairs)
    block = min(256, triton.next_power_of_2(max_row_width))
    num_warps = 1 if block <= 32 else 4 if block <= 128 else 8

    dummy = (indices, indices, 1)
    topk_p_pair = topk_p_pair or dummy
    topk_index_pair = topk_index_pair or dummy
    hidden = hidden or dummy
    draft = draft or dummy
    dsa = dsa or dummy

    grid = (indices.shape[0], triton.cdiv(max_row_width, block))
    # HAS_* values are tl.constexpr specialization keys. Triton removes every
    # disabled copy branch; the launched kernel does not test these flags at runtime.
    _scatter_rows_kernel[grid](
        indices,
        indices.stride(0),
        *output,
        *topk_p_pair,
        *topk_index_pair,
        *hidden,
        *draft,
        *dsa,
        HAS_TOPK=topk_p_buf is not None,
        HAS_HIDDEN=hidden_states_buf is not None,
        HAS_DRAFT_PROBS=draft_probs_buf is not None,
        HAS_DSA=dsa_topk_indices_buf is not None,
        BLOCK=block,
        num_warps=num_warps,
    )
