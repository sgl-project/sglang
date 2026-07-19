from __future__ import annotations

import torch
import triton
import triton.language as tl

# Column tile width. Each program copies a contiguous COL_BLOCK-wide slice of one
# request's segment: a coalesced int64 load -> int32 store, guarded by a single
# ``col < write_len`` mask. 1024 keeps every program busy without over-tiling the
# common (short-context) case.
_SCATTER_COL_BLOCK: int = 1024


def launch_scatter_req_token_ids_kernel(
    *,
    flat_in: torch.Tensor,
    offsets: torch.Tensor,
    req_pool_indices: torch.Tensor,
    pool_out: torch.Tensor,
) -> None:
    """Scatter a flat per-req int64 object sequence into a 2-D int32 pool.

    For each request ``r`` in ``[0, bs)``:

    - ``start, end = offsets[r], offsets[r + 1]``
    - ``rp = req_pool_indices[r]``
    - ``write_len = min(end - start, pool_max_context_len)``
    - ``pool_out[rp, :write_len] = flat_in[start:start + write_len].to(int32)``

    Args:
        flat_in: ``[total_tokens]`` int64 device tensor of objects, flattened
            per-req in req order.
        offsets: ``[bs + 1]`` int64 device tensor (host-computed cumsum of per-req
            lengths). ``offsets[bs] == total_tokens``.
        req_pool_indices: ``[bs]`` int64 device tensor of pool row indices.
        pool_out: ``[max_reqs, max_context_len]`` int32 device tensor of objects.
            Mutated in-place; rows not addressed by ``req_pool_indices`` are untouched.

    Implementation notes:
        - The grid is ``(bs, num_col_blocks)``: one axis per request, one per
          column tile. Each program loads ``offsets[r]`` / ``offsets[r + 1]`` once
          (``O(1)`` per program) and copies its slice directly -- no per-token
          owner search, and no cap on ``bs``.
    """
    if flat_in.dim() != 1:
        raise ValueError(
            f"kv-canary: scatter_req_token_ids flat_in must be 1-D, got shape "
            f"{tuple(flat_in.shape)}"
        )
    if offsets.dim() != 1:
        raise ValueError(
            f"kv-canary: scatter_req_token_ids offsets must be 1-D, got shape "
            f"{tuple(offsets.shape)}"
        )
    if req_pool_indices.dim() != 1:
        raise ValueError(
            f"kv-canary: scatter_req_token_ids req_pool_indices must be 1-D, got shape "
            f"{tuple(req_pool_indices.shape)}"
        )
    if pool_out.dim() != 2:
        raise ValueError(
            f"kv-canary: scatter_req_token_ids pool_out must be 2-D, got shape "
            f"{tuple(pool_out.shape)}"
        )
    if flat_in.dtype != torch.int64:
        raise TypeError(
            f"kv-canary: scatter_req_token_ids flat_in must be int64, got "
            f"{flat_in.dtype}"
        )
    if offsets.dtype != torch.int64:
        raise TypeError(
            f"kv-canary: scatter_req_token_ids offsets must be int64, got "
            f"{offsets.dtype}"
        )
    if req_pool_indices.dtype != torch.int64:
        raise TypeError(
            f"kv-canary: scatter_req_token_ids req_pool_indices must be int64, got "
            f"{req_pool_indices.dtype}"
        )
    if pool_out.dtype != torch.int32:
        raise TypeError(
            f"kv-canary: scatter_req_token_ids pool_out must be int32, got "
            f"{pool_out.dtype}"
        )

    bs = int(req_pool_indices.shape[0])
    if int(offsets.shape[0]) != bs + 1:
        raise ValueError(
            f"kv-canary: scatter_req_token_ids offsets length {offsets.shape[0]} != "
            f"bs+1 ({bs + 1})"
        )

    num_tokens = int(flat_in.shape[0])
    if bs == 0 or num_tokens == 0:
        return

    pool_stride0 = int(pool_out.stride(0))
    pool_max_context_len = int(pool_out.shape[1])

    # A request can write at most ``min(seg_len, pool_max_context_len)`` columns, and
    # no segment is longer than ``num_tokens``; grid only as many column tiles as the
    # tighter of the two bounds needs (all host-known, so no device sync).
    effective_cols = min(pool_max_context_len, num_tokens)
    if effective_cols <= 0:
        return

    num_col_blocks = triton.cdiv(effective_cols, _SCATTER_COL_BLOCK)
    grid = (bs, num_col_blocks)
    _scatter_req_token_ids_kernel[grid](
        flat_in,
        offsets,
        req_pool_indices,
        pool_out,
        pool_stride0=pool_stride0,
        pool_max_context_len=pool_max_context_len,
        COL_BLOCK=_SCATTER_COL_BLOCK,
    )


@triton.jit
def _scatter_req_token_ids_kernel(
    flat_in_ptr,  # [num_tokens] int64
    offsets_ptr,  # [num_batch + 1] int64
    req_pool_indices_ptr,  # [num_batch] int64
    pool_out_ptr,  # [num_rows, pool_max_context_len] int32, row stride = pool_stride0
    pool_stride0,  # scalar int32 (row stride of pool_out in elements)
    pool_max_context_len,  # scalar int32 (dim-1 length of pool_out)
    COL_BLOCK: tl.constexpr,
):
    r = tl.program_id(0)  # request index
    cblk = tl.program_id(1)  # column-block index within the request

    start = tl.load(offsets_ptr + r)  # int64
    end = tl.load(offsets_ptr + r + 1)  # int64
    seg_len = end - start  # int64

    # Bound writes by the pool's max_context_len so a token sequence longer than the
    # ReqToTokenPool row never spills into an adjacent row (matches the reference,
    # which truncates each segment at max_context_len).
    write_len = tl.minimum(seg_len, pool_max_context_len)  # int64

    col = cblk * COL_BLOCK + tl.arange(0, COL_BLOCK)  # [COL_BLOCK] int32
    mask = col < write_len  # [COL_BLOCK] bool

    val = tl.load(flat_in_ptr + start + col, mask=mask, other=0).to(
        tl.int32
    )  # [COL_BLOCK] int32
    rp = tl.load(req_pool_indices_ptr + r)  # int64
    tl.store(pool_out_ptr + rp * pool_stride0 + col, val, mask=mask)


def scatter_req_token_ids_torch_reference(
    *,
    flat_in: torch.Tensor,
    offsets: torch.Tensor,
    req_pool_indices: torch.Tensor,
    pool_out: torch.Tensor,
) -> None:
    """Plain-PyTorch reference for :func:`launch_scatter_req_token_ids_kernel`."""
    bs = int(req_pool_indices.shape[0])
    offsets_host = offsets.detach().cpu().tolist()
    req_pool_indices_host = req_pool_indices.detach().cpu().tolist()
    flat_host = flat_in.detach().cpu()
    pool_max_context_len = int(pool_out.shape[1])

    for r in range(bs):
        start = int(offsets_host[r])
        end = int(offsets_host[r + 1])
        if end <= start:
            continue
        rp = int(req_pool_indices_host[r])
        seg = flat_host[start:end].to(torch.int32)
        write_len = min(int(seg.shape[0]), pool_max_context_len)
        if write_len <= 0:
            continue
        pool_out[rp, :write_len] = seg[:write_len].to(pool_out.device)
