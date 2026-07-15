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
