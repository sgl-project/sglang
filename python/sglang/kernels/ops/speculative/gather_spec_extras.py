from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_rows_kernel(
    idx_ptr,
    s0,
    d0,
    n0,
    s1,
    d1,
    n1,
    s2,
    d2,
    n2,
    s3,
    d3,
    n3,
    HAS3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program == one (output row, column block). All buffers share the
    # same gather index, so a single launch copies every buffer's row and
    # the per-kernel launch bubbles between the old separate gathers vanish.
    row = tl.program_id(0)
    cb = tl.program_id(1)
    src = tl.load(idx_ptr + row).to(tl.int64)
    cols = cb * BLOCK + tl.arange(0, BLOCK)

    m0 = cols < n0
    tl.store(d0 + row * n0 + cols, tl.load(s0 + src * n0 + cols, mask=m0), mask=m0)

    m1 = cols < n1
    tl.store(d1 + row * n1 + cols, tl.load(s1 + src * n1 + cols, mask=m1), mask=m1)

    m2 = cols < n2
    tl.store(d2 + row * n2 + cols, tl.load(s2 + src * n2 + cols, mask=m2), mask=m2)

    if HAS3:
        m3 = cols < n3
        tl.store(d3 + row * n3 + cols, tl.load(s3 + src * n3 + cols, mask=m3), mask=m3)


def _row_width(buf: torch.Tensor) -> int:
    """Flattened per-row element count (trailing dims), 1 for a 1-D buffer."""
    return buf[0].numel() if buf.dim() > 1 else 1


def _empty_like_rows(buf: torch.Tensor, m: int) -> torch.Tensor:
    """Output buffer for `m` gathered rows of `buf` (same trailing dims/dtype/device)."""
    return torch.empty((m, *buf.shape[1:]), dtype=buf.dtype, device=buf.device)


def gather_spec_extras(
    indices: torch.Tensor,
    topk_p_buf: torch.Tensor,
    topk_index_buf: torch.Tensor,
    output_tokens_buf: torch.Tensor,
    hidden_states_buf: Optional[torch.Tensor],
):
    """Gather spec extras (topk_p / topk_index / bonus_tokens / optional hidden
    states) by a shared row index in a single fused Triton launch (one kernel
    for all buffers) instead of one advanced-index gather per buffer.
    `hidden_states_buf` is None when the build does not capture hidden states."""
    # Source buffers are allocated once (torch.empty/full) and only ever mutated
    # in place, so they are guaranteed row-contiguous. `indices` flows from
    # several producers (req_pool_indices, filtered/merged future_indices); the
    # kernel addresses it linearly, so normalize layout here (no-op when already
    # contiguous) to avoid a silent wrong-result on a strided index tensor.
    indices = indices.contiguous()
    m = indices.shape[0]
    has_hidden = hidden_states_buf is not None

    topk_p = _empty_like_rows(topk_p_buf, m)
    topk_index = _empty_like_rows(topk_index_buf, m)
    bonus_tokens = _empty_like_rows(output_tokens_buf, m)
    hidden_states = _empty_like_rows(hidden_states_buf, m) if has_hidden else None
    if m == 0:
        return topk_p, topk_index, bonus_tokens, hidden_states

    n0 = _row_width(topk_p_buf)
    n1 = _row_width(topk_index_buf)
    n2 = _row_width(output_tokens_buf)
    n3 = _row_width(hidden_states_buf) if has_hidden else 1
    max_n = max(n0, n1, n2, n3)

    # Dummy operands for the disabled hidden-states slot: the pointers must be
    # valid even though the kernel never dereferences them (gated off by HAS3).
    s3 = hidden_states_buf if has_hidden else indices
    d3 = hidden_states if has_hidden else indices

    block = min(1024, triton.next_power_of_2(max_n))
    grid = (m, triton.cdiv(max_n, block))
    _gather_rows_kernel[grid](
        indices,
        topk_p_buf,
        topk_p,
        n0,
        topk_index_buf,
        topk_index,
        n1,
        output_tokens_buf,
        bonus_tokens,
        n2,
        s3,
        d3,
        n3,
        HAS3=has_hidden,
        BLOCK=block,
    )
    return topk_p, topk_index, bonus_tokens, hidden_states


@triton.jit
def _scatter_rows_kernel(
    idx_ptr,
    s0,
    d0,
    n0,
    s1,
    d1,
    n1,
    s2,
    d2,
    n2,
    s3,
    d3,
    n3,
    HAS12: tl.constexpr,
    HAS3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Inverse of _gather_rows_kernel: source row `row` of every buffer lands in
    # buffer row idx[row], cast to the buffer dtype as `.to(buf.dtype)` would.
    row = tl.program_id(0)
    cb = tl.program_id(1)
    dst = tl.load(idx_ptr + row).to(tl.int64)
    cols = cb * BLOCK + tl.arange(0, BLOCK)

    m0 = cols < n0
    v0 = tl.load(s0 + row * n0 + cols, mask=m0)
    tl.store(d0 + dst * n0 + cols, v0.to(d0.dtype.element_ty), mask=m0)

    if HAS12:
        m1 = cols < n1
        v1 = tl.load(s1 + row * n1 + cols, mask=m1)
        tl.store(d1 + dst * n1 + cols, v1.to(d1.dtype.element_ty), mask=m1)
        m2 = cols < n2
        v2 = tl.load(s2 + row * n2 + cols, mask=m2)
        tl.store(d2 + dst * n2 + cols, v2.to(d2.dtype.element_ty), mask=m2)

    if HAS3:
        m3 = cols < n3
        v3 = tl.load(s3 + row * n3 + cols, mask=m3)
        tl.store(d3 + dst * n3 + cols, v3.to(d3.dtype.element_ty), mask=m3)


_INT_DTYPES = (torch.int32, torch.int64)


def scatter_spec_extras(
    indices: torch.Tensor,
    bonus_tokens: torch.Tensor,
    output_tokens_buf: torch.Tensor,
    topk_p: Optional[torch.Tensor],
    topk_p_buf: Optional[torch.Tensor],
    topk_index: Optional[torch.Tensor],
    topk_index_buf: Optional[torch.Tensor],
    hidden_states: Optional[torch.Tensor],
    hidden_states_buf: Optional[torch.Tensor],
) -> bool:
    """`buf[indices] = src.to(buf.dtype)` for every given (src, buf) pair in one
    launch; top-k is taken only as the (topk_p, topk_index) pair. Returns False
    without writing when a pair needs more than a copy or an int32/int64 cast.
    Rows of `indices` must be distinct, as for the advanced-index assignment."""
    has12 = topk_p is not None
    has3 = hidden_states is not None
    pairs = [(bonus_tokens, output_tokens_buf)]
    if has12:
        if topk_index is None:
            return False
        pairs += [(topk_p, topk_p_buf), (topk_index, topk_index_buf)]
    if has3:
        pairs.append((hidden_states, hidden_states_buf))
    m = indices.shape[0]
    for src, buf in pairs:
        if (
            buf is None
            or src.shape[0] != m
            or src.shape[1:] != buf.shape[1:]
            or not src.is_contiguous()
            or not buf.is_contiguous()
            or src.device != buf.device
            or (
                src.dtype != buf.dtype
                and not (src.dtype in _INT_DTYPES and buf.dtype in _INT_DTYPES)
            )
        ):
            return False
    if m == 0:
        return True
    indices = indices.contiguous()

    # Disabled slots point at the bonus pair; the kernel never touches them.
    off = (bonus_tokens, output_tokens_buf)
    slots = [
        off,
        (topk_p, topk_p_buf) if has12 else off,
        (topk_index, topk_index_buf) if has12 else off,
        (hidden_states, hidden_states_buf) if has3 else off,
    ]
    args = []
    for src, buf in slots:
        args += (src, buf, _row_width(buf))
    max_n = max(_row_width(buf) for _, buf in pairs)
    block = min(1024, triton.next_power_of_2(max_n))
    grid = (m, triton.cdiv(max_n, block))
    _scatter_rows_kernel[grid](indices, *args, HAS12=has12, HAS3=has3, BLOCK=block)
    return True
