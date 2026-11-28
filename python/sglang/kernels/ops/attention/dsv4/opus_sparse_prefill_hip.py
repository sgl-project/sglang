"""Sparse MLA prefill attention on gfx950 through aiter's OPUS kernel.

OPUS attends over two bf16 row sources, each addressed by a CSR index list. The
DeepSeek-V4 caller dequantizes rows into bf16 workspaces and builds padded
per-query lists (``SparsePrefillChunkCache``); the DSA backend passes its
page-size-1 top-k table. This module turns those lists into CSR and runs the
kernel.
"""

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# (int32 indices, int32 indptr [T + 1])
Csr = Tuple[torch.Tensor, torch.Tensor]


@triton.jit
def _combined_to_csr_kernel(
    indices_ptr,
    indices_stride,
    lens_ptr,
    indptr_ptr,
    out_ptr,
    row_start,
    row_end,
    BLOCK: tl.constexpr,
):
    # OPUS reads every CSR entry, so -1 holes inside the scanned prefix and
    # entries outside [row_start, row_end) are dropped here.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    length = tl.load(lens_ptr + row)
    idx = tl.load(
        indices_ptr + row * indices_stride + cols, mask=cols < length, other=-1
    )
    keep = (idx >= row_start) & (idx < row_end)
    dst = tl.cumsum(keep.to(tl.int32), axis=0) - 1
    start = tl.load(indptr_ptr + row)
    tl.store(out_ptr + start + dst, idx - row_start, mask=keep)


def combined_to_csr(
    indices: torch.Tensor, lens: torch.Tensor, row_start: int, row_end: int
) -> Csr:
    """int32 [T, W] padded index lists with per-row scanned lengths [T] -> the CSR
    (T * W capacity) of their entries in [row_start, row_end), rebased to row_start."""
    num_rows, width = indices.shape
    device = indices.device
    cols = torch.arange(width, device=device, dtype=torch.int32)
    keep = (
        (cols[None, :] < lens[:, None]) & (indices >= row_start) & (indices < row_end)
    )
    indptr = torch.zeros(num_rows + 1, dtype=torch.int32, device=device)
    torch.cumsum(keep.sum(dim=1, dtype=torch.int32), dim=0, out=indptr[1:])
    csr = torch.empty(max(num_rows * width, 1), dtype=torch.int32, device=device)
    if num_rows > 0:
        _combined_to_csr_kernel[(num_rows,)](
            indices,
            indices.stride(0),
            lens,
            indptr,
            csr,
            row_start,
            row_end,
            BLOCK=triton.next_power_of_2(width),
        )
    return csr, indptr


@functools.cache
def opus_sparse_prefill_available() -> bool:
    try:
        from aiter.ops.pa_sparse_prefill_opus import (  # noqa: F401
            pa_sparse_prefill_opus,
        )
    except ImportError:
        return False
    return True


@functools.cache
def _mla_sparse_prefill():
    """aiter's sink-optional entry (aiter >= 0.1.28), else None."""
    try:
        from aiter.ops.mla_sparse_prefill import mla_sparse_prefill
    except ImportError:
        return None
    return mla_sparse_prefill


@functools.cache
def _no_sink(num_heads: int, device: torch.device) -> torch.Tensor:
    # exp(-1e30 - max) is 0, so the sink adds nothing to the softmax denominator
    return torch.full((num_heads,), -1e30, dtype=torch.float32, device=device)


def opus_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    csr: Csr,
    attn_sink: Optional[torch.Tensor],
    softmax_scale: float,
    extend_kv: Optional[torch.Tensor] = None,
    extend_csr: Optional[Csr] = None,
) -> torch.Tensor:
    """Attention of q [T, H, 512] bf16 over the bf16 rows of kv [R, 512] named by
    csr, then those of extend_kv named by extend_csr, with a per-head sink, or
    none when attn_sink is None. A row with no keys gives zeros."""
    indices, indptr = csr
    splits = {}
    if attn_sink is None:
        # Rows with no keys (the DSA caller's padding rows) must give zeros;
        # aiter's split-key kernel can leave them non-finite.
        splits = {"num_kv_splits": 1}
        mla_sparse_prefill = _mla_sparse_prefill()
        if mla_sparse_prefill is not None:
            if extend_kv is None:
                # an empty extend list, so aiter may pick a single-source kernel
                extend_kv = kv[:1]
                extend_csr = (indices[:0], torch.zeros_like(indptr))
            return mla_sparse_prefill(
                q.contiguous(),
                kv,
                indices,
                indptr,
                extend_kv,
                *extend_csr,
                None,
                softmax_scale,
                **splits,
            )
        attn_sink = _no_sink(q.shape[1], q.device)

    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus

    if extend_kv is None:
        # every query reads an empty row of the unused extend source
        extend_kv = kv[:1]
        extend_csr = (indices[:1], torch.zeros_like(indptr))
    extend_indices, extend_indptr = extend_csr
    return pa_sparse_prefill_opus(
        q.contiguous(),
        kv,
        indices,
        indptr,
        extend_kv,
        extend_indices,
        extend_indptr,
        attn_sink,
        softmax_scale,
        **splits,
    )
