"""A DFLASH draft block's KV ring locs in one launch, bitwise equal to
``DFlashDraftKVRing.locs(req_pool_indices.repeat_interleave(block_size), positions)``."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dflash_ring_block_locs_kernel(
    req_pool_indices_ptr,
    positions_ptr,
    out_ptr,
    num_tokens,
    req_pool_indices_stride,
    positions_stride,
    block_size,
    rows_per_req,
    dcp_size,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_tokens
    req = tl.load(
        req_pool_indices_ptr + (offs // block_size) * req_pool_indices_stride,
        mask=mask,
        other=0,
    ).to(tl.int64)
    pos = tl.load(positions_ptr + offs * positions_stride, mask=mask, other=0).to(
        tl.int64
    )
    # Floor division and remainder as torch computes them (the divisors are positive).
    q = pos // dcp_size
    r = pos - q * dcp_size
    q = tl.where(r < 0, q - 1, q)
    r = tl.where(r < 0, r + dcp_size, r)
    row = q % rows_per_req
    row = tl.where(row < 0, row + rows_per_req, row)
    local = 1 + req * rows_per_req + row
    tl.store(out_ptr + offs, local * dcp_size + r, mask=mask)


def dflash_ring_block_locs(
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    *,
    block_size: int,
    rows_per_req: int,
    dcp_size: int,
) -> torch.Tensor:
    """Ring locs (int64 [num_tokens]) of ``positions``, ``block_size`` per request of
    ``req_pool_indices`` in order."""
    if req_pool_indices.dim() != 1 or positions.dim() != 1:
        raise ValueError(
            "DFLASH ring block locs take 1-D req_pool_indices and positions."
        )
    num_tokens = int(positions.numel())
    if int(req_pool_indices.numel()) * int(block_size) != num_tokens:
        raise ValueError(
            f"DFLASH ring block locs: {req_pool_indices.numel()} requests x block "
            f"{block_size} != {num_tokens} positions."
        )
    out = torch.empty(num_tokens, dtype=torch.int64, device=positions.device)
    if num_tokens == 0:
        return out
    block = min(1024, triton.next_power_of_2(num_tokens))
    _dflash_ring_block_locs_kernel[(triton.cdiv(num_tokens, block),)](
        req_pool_indices,
        positions,
        out,
        num_tokens,
        req_pool_indices.stride(0),
        positions.stride(0),
        int(block_size),
        int(rows_per_req),
        int(dcp_size),
        BLOCK=block,
        num_warps=1 if block <= 128 else 4,
    )
    return out
