"""DeepSeek byte packing; computational ownership and collectives stay in Engram."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant(
    ROWS,
    OUT,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, TILE)
    bits = tl.load(ROWS + row * WIDTH + col, col < DIM, 0)
    values = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    exponents = tl.load(ROWS + row * WIDTH + DIM + col // BLOCK, col < DIM, 0).to(
        tl.uint32
    )
    scales = (exponents << 23).to(tl.float32, bitcast=True)
    scales = tl.where(exponents == 0, 2.0**-127, scales)
    tl.store(OUT + row * DIM + col, values * scales, col < DIM)


def publish(module, block_size):
    packed = torch.cat(
        (
            module.weight.detach().view(torch.uint8),
            module.scale.detach().view(torch.uint8),
        ),
        dim=1,
    )
    module._hi_rows.initialize(packed)
    module.weight.data = module.weight.new_empty(0)
    module.scale.data = module.scale.new_empty(0)


def lookup(module, indices, block_size):
    owned = (indices >= module.row_start) & (indices < module.row_start + module.rows)
    ids = indices.masked_fill(~owned, -1)
    if module._hi_rows.remote is None:
        ids = torch.where(owned, indices - module.row_start, -1)
    packed = module._hi_rows.lookup(ids)
    output = torch.empty(
        (*indices.shape, module.dim), dtype=torch.bfloat16, device=indices.device
    )
    if indices.numel():
        _dequant[(indices.numel(),)](
            packed,
            output,
            module.dim,
            block_size,
            packed.shape[-1],
            triton.next_power_of_2(module.dim),
        )
    return output
