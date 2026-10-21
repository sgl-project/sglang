"""Read FP8 Engram rows directly from immutable, NVLink-mapped TP shards."""

import torch
import triton
import triton.language as tl


@triton.jit
def _engram_peer_gather_kernel(
    weight_ptrs,
    scale_ptrs,
    row_bounds,
    ids,
    output,
    WORLD: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    index = tl.load(ids + row).to(tl.int64)
    weight_address = tl.full((), 0, tl.int64)
    scale_address = tl.full((), 0, tl.int64)
    local = tl.full((), 0, tl.int64)
    valid = tl.full((), False, tl.int1)
    for rank in tl.static_range(WORLD):
        lower = tl.load(row_bounds + rank)
        upper = tl.load(row_bounds + rank + 1)
        owned = (index >= lower) & (index < upper)
        weight_address = tl.where(owned, tl.load(weight_ptrs + rank), weight_address)
        scale_address = tl.where(owned, tl.load(scale_ptrs + rank), scale_address)
        local = tl.where(owned, index - lower, local)
        valid = valid | owned
    columns = tl.arange(0, DIM)
    weight = weight_address.to(tl.pointer_type(tl.float8e4nv))
    scale = scale_address.to(tl.pointer_type(tl.uint8))
    values = tl.load(weight + local * DIM + columns, mask=valid, other=0.0).to(
        tl.float32
    )
    exponents = tl.load(
        scale + local * (DIM // BLOCK_SIZE) + columns // BLOCK_SIZE,
        mask=valid,
        other=0,
    ).to(tl.int32)
    factors = (exponents << 23).to(tl.float32, bitcast=True)
    factors = tl.where(exponents == 0, 2.0**-127, factors)
    result = tl.where(valid, values * factors, 0.0)
    tl.store(output + row * DIM + columns, result.to(tl.bfloat16))


def engram_peer_gather(
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    row_bounds: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """Gather into [N, D] BF16; invalid IDs produce zero rows.

    The caller owns the mapped peer allocations and keeps them immutable until
    all readers finish. Bounds have WORLD + 1 entries and permit empty shards.
    Pointer arrays and bounds are initialized once, before graph capture.
    """
    assert output.ndim == 2 and block_size > 0
    assert all(
        value.device == ids.device
        and value.dtype == torch.int64
        and value.ndim == 1
        and value.is_contiguous()
        for value in (weight_ptrs, scale_ptrs, row_bounds)
    )
    dim = output.shape[1]
    assert dim > 0 and dim & (dim - 1) == 0 and dim % block_size == 0
    assert ids.is_cuda and ids.is_contiguous()
    assert ids.dtype in (torch.int32, torch.int64)
    assert output.device == ids.device and output.shape[0] == ids.numel()
    assert output.dtype == torch.bfloat16 and output.is_contiguous()
    world = weight_ptrs.numel()
    assert scale_ptrs.numel() == world and row_bounds.numel() == world + 1
    if ids.numel():
        _engram_peer_gather_kernel[(ids.numel(),)](
            weight_ptrs,
            scale_ptrs,
            row_bounds,
            ids,
            output,
            WORLD=world,
            DIM=dim,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
    return output
