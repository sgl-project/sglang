"""Bounded FP8 -> query-dtype reads for the absorbed-MLA paged KV pool.

When a FlashInfer MLA kernel requires matching query and KV dtypes, the backend
must convert an FP8 KV cache. Previously, the attention backend used
to upcast the *whole* per-layer pool (``get_key_buffer(...).to(q.dtype)``) on
every layer of every forward, which costs O(pool size) memory and bandwidth and
can OOM when the pool is large. These helpers convert only the rows that the
current plan actually references.

Two layouts are provided:

* ``gather_mla_kv_rows`` (eager): copy the referenced rows into a compact
  buffer of exactly ``len(kv_indices)`` rows (byte gather + cast). The caller
  then plans the FlashInfer wrapper with ``kv_indices = arange(len(kv_indices))``.
* ``refresh_mla_kv_shadow`` (CUDA-graph safe): rewrite the referenced rows of
  a persistent, pool-indexed shadow buffer in place. Physical row ids are kept,
  so a captured plan is still valid on replay. The grid is fixed and the number
  of live entries is read on the device from ``kv_indptr[bs]``, so the same
  launch can be replayed with different batch contents.

Both convert the referenced FP8 values to the query dtype without changing
their scaling.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _refresh_mla_kv_shadow_kernel(
    src_ptr,
    dst_ptr,
    idx_ptr,
    n_live_ptr,
    max_rows,
    src_stride,
    dst_stride,
    row_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    n_live = tl.minimum(tl.load(n_live_ptr).to(tl.int64), max_rows)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < row_dim
    start = pid.to(tl.int64) * ROWS_PER_PROG
    step = nprog.to(tl.int64) * ROWS_PER_PROG
    while start < n_live:
        for r in tl.static_range(ROWS_PER_PROG):
            i = start + r
            if i < n_live:
                row = tl.load(idx_ptr + i).to(tl.int64)
                vals = tl.load(src_ptr + row * src_stride + offs, mask=mask)
                if (
                    vals.dtype == tl.float8e5
                    and dst_ptr.dtype.element_ty == tl.bfloat16
                ):
                    # E5M2 is the high byte of FP16. Decode that exact layout to
                    # preserve Inf/NaN; direct E5M2 -> BF16 lowering can rebias
                    # these encodings into finite values.
                    bits = vals.to(tl.uint8, bitcast=True).to(tl.uint16) << 8
                    vals = bits.to(tl.uint16).to(tl.float16, bitcast=True)
                tl.store(
                    dst_ptr + row * dst_stride + offs,
                    vals.to(dst_ptr.dtype.element_ty),
                    mask=mask,
                )
        start += step


def _row_view(buf: torch.Tensor) -> torch.Tensor:
    return buf.view(buf.shape[0], -1)


def gather_mla_kv_rows(
    kv_buffer: torch.Tensor,
    kv_indices: torch.Tensor,
    dst_dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``kv_buffer[kv_indices].to(dst_dtype)`` without materialising the pool.

    ``kv_buffer`` has shape ``[pool_rows, 1, dim]``; the result has shape
    ``[len(kv_indices), 1, dim]``. The indices are physical token rows. Rows are
    gathered as raw bytes and then converted with the same ``.to()`` the old
    full-pool path used, so the values are bit-identical; the only temporaries
    are ``len(kv_indices)`` rows, never the pool.
    """
    src_dtype = kv_buffer.dtype
    rows = torch.index_select(kv_buffer.view(torch.uint8), 0, kv_indices)
    return rows.view(src_dtype).to(dst_dtype)


def refresh_mla_kv_shadow(
    kv_buffer: torch.Tensor,
    shadow: torch.Tensor,
    kv_indices: torch.Tensor,
    n_live: torch.Tensor,
    num_programs: int = 1024,
    rows_per_program: int = 4,
) -> None:
    """Convert ``kv_buffer[kv_indices[:n_live]]`` into ``shadow`` at the same rows.

    ``n_live`` is a 1-element integer device tensor (normally
    ``kv_indptr[bs:bs+1]``). The fixed launch grid means the call can be
    captured in a CUDA graph and replayed with a different live count. Rows not
    referenced in this step keep stale contents and are never read by the plan.
    """
    max_rows = kv_indices.numel()
    if max_rows == 0:
        return
    dim = kv_buffer.shape[-1]
    src = _row_view(kv_buffer)
    dst = _row_view(shadow)
    _refresh_mla_kv_shadow_kernel[(num_programs,)](
        src,
        dst,
        kv_indices,
        n_live,
        max_rows,
        src.stride(0),
        dst.stride(0),
        row_dim=dim,
        BLOCK_D=triton.next_power_of_2(dim),
        ROWS_PER_PROG=rows_per_program,
    )
