from __future__ import annotations

import torch
import triton
import triton.language as tl

_SCATTER_TOKEN_BLOCK: int = 256
# Upper bound on bs+1 the kernel can scan per program. Owner-req lookup uses an
# outer-product tile of shape ``[TOKEN_BLOCK, BATCH_BLOCK]``; keep this small so
# the tile stays in registers (256 x 512 = 128 KiB i1, well below the SM trap-
# inducing budget that bites at the 1M cell mark).
_SCATTER_BATCH_BLOCK: int = 512


def launch_scatter_req_token_ids_kernel(
    *,
    flat_in: torch.Tensor,
    offsets: torch.Tensor,
    req_pool_indices: torch.Tensor,
    pool_out: torch.Tensor,
) -> None:
    """Scatter a flat per-req int64 object sequence into a 2-D int32 pool.

    For each global object index ``t`` in ``[0, total_tokens)``:

    - find ``r`` = largest req index s.t. ``offsets[r] <= t``
    - ``pos = t - offsets[r]``
    - ``rp = req_pool_indices[r]``
    - if ``pos < pool_max_context_len``:
      ``pool_out[rp, pos] = flat_in[t].to(int32)``

    Args:
        flat_in: ``[total_tokens]`` int64 device tensor of objects, flattened
            per-req in req order.
        offsets: ``[bs + 1]`` int64 device tensor (host-computed cumsum of per-req
            lengths). ``offsets[bs] == total_tokens``.
        req_pool_indices: ``[bs]`` int64 device tensor of pool row indices.
        pool_out: ``[max_reqs, max_context_len]`` int32 device tensor of objects.
            Mutated in-place; rows not addressed by ``req_pool_indices`` are untouched.

    Implementation notes:
        - Linear scan over ``offsets`` (``BATCH_BLOCK >= bs + 1``); fits easily in
          registers for the workloads kv-canary handles (``bs <= a few thousand``).
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
    if bs + 1 > _SCATTER_BATCH_BLOCK:
        raise ValueError(
            f"kv-canary: scatter_req_token_ids bs+1={bs + 1} exceeds BATCH_BLOCK="
            f"{_SCATTER_BATCH_BLOCK}; bump _SCATTER_BATCH_BLOCK if real workloads need this"
        )

    num_tokens = int(flat_in.shape[0])
    if num_tokens == 0:
        return

    pool_stride0 = int(pool_out.stride(0))
    pool_max_context_len = int(pool_out.shape[1])

    grid = (triton.cdiv(num_tokens, _SCATTER_TOKEN_BLOCK),)
    _scatter_req_token_ids_kernel[grid](
        flat_in,
        offsets,
        req_pool_indices,
        pool_out,
        num_tokens=num_tokens,
        num_batch=bs,
        pool_stride0=pool_stride0,
        pool_max_context_len=pool_max_context_len,
        TOKEN_BLOCK=_SCATTER_TOKEN_BLOCK,
        BATCH_BLOCK=_SCATTER_BATCH_BLOCK,
    )


@triton.jit
def _scatter_req_token_ids_kernel(
    flat_in_ptr,  # [num_tokens] int64
    offsets_ptr,  # [num_batch + 1] int64
    req_pool_indices_ptr,  # [num_batch] int64
    pool_out_ptr,  # [num_rows, pool_max_context_len] int32, row stride = pool_stride0
    num_tokens,  # scalar int32
    num_batch,  # scalar int32
    pool_stride0,  # scalar int32 (row stride of pool_out in elements)
    pool_max_context_len,  # scalar int32 (dim-1 length of pool_out)
    TOKEN_BLOCK: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tids = pid * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)  # [TOKEN_BLOCK] int32
    tid_mask = tids < num_tokens  # [TOKEN_BLOCK] bool

    bs_offs = tl.arange(0, BATCH_BLOCK)  # [BATCH_BLOCK] int32
    bs_mask = bs_offs < (num_batch + 1)  # [BATCH_BLOCK] bool
    offs_vals = tl.load(  # [BATCH_BLOCK] int64
        offsets_ptr + bs_offs,
        mask=bs_mask,
        other=(1 << 62),
    )

    # find owning req for each tid via reduce-sum: req_idx = (count of offsets <= tid) - 1
    le = offs_vals[None, :] <= tids[:, None]  # [TOKEN_BLOCK, BATCH_BLOCK] bool
    req_idx = tl.sum(le.to(tl.int32), axis=1) - 1  # [TOKEN_BLOCK] int32

    safe_req_idx = tl.where(tid_mask, req_idx, 0)  # [TOKEN_BLOCK] int32
    starts = tl.load(
        offsets_ptr + safe_req_idx, mask=tid_mask, other=0
    )  # [TOKEN_BLOCK] int64
    pos = tids - starts  # [TOKEN_BLOCK] int64
    rp = tl.load(
        req_pool_indices_ptr + safe_req_idx, mask=tid_mask, other=0
    )  # [TOKEN_BLOCK] int64

    # Bound writes by the pool's max_context_len so a token sequence longer than the
    # ReqToTokenPool row never spills into an adjacent row.
    in_row = pos < pool_max_context_len  # [TOKEN_BLOCK] bool
    write_mask = tid_mask & in_row  # [TOKEN_BLOCK] bool

    val = tl.load(flat_in_ptr + tids, mask=tid_mask, other=0).to(
        tl.int32
    )  # [TOKEN_BLOCK] int32
    tl.store(pool_out_ptr + rp * pool_stride0 + pos, val, mask=write_mask)


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
