# Adapted from NVlabs/Sana sol-engine LTX2 Ada-value fusion.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from sglang.multimodal_gen.runtime.platforms import current_platform


@triton.jit
def _ltx2_ada_values9_kernel(
    temb_ptr,
    table_ptr,
    out_ptr,
    hidden: tl.constexpr,
    total_params: tl.constexpr,
    table_stride_p: tl.constexpr,
    table_stride_d: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    nblocks = tl.cdiv(hidden, BLOCK_N)
    row = pid // (total_params * nblocks)
    rem = pid % (total_params * nblocks)
    p = rem // nblocks
    block = rem % nblocks

    cols = block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < hidden
    rows = tl.num_programs(0) // (total_params * nblocks)

    table_vals = tl.load(
        table_ptr + p * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb_vals = tl.load(
        temb_ptr + row.to(tl.int64) * (total_params * hidden) + p * hidden + cols,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    tl.store(
        out_ptr + (p * rows + row).to(tl.int64) * hidden + cols,
        (table_vals + temb_vals).to(tl.bfloat16),
        mask=mask,
    )


def ltx2_ada_values9(
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if timestep.ndim != 3:
        raise ValueError("timestep must have shape [B, S, 9 * D]")
    if (
        not current_platform.tensor_on_device(timestep)
        or timestep.dtype != torch.bfloat16
    ):
        raise ValueError("timestep must be a CUDA bfloat16 tensor")
    if not timestep.is_contiguous():
        raise ValueError("timestep must be contiguous")
    if scale_shift_table.ndim != 2 or scale_shift_table.shape[0] != 9:
        raise ValueError("scale_shift_table must have shape [9, D]")
    if (
        not current_platform.tensor_on_device(scale_shift_table)
        or scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.stride(-1) != 1
    ):
        raise ValueError(
            "scale_shift_table must be CUDA, bf16/fp32, last-dim contiguous"
        )

    total_params = int(scale_shift_table.shape[0])
    hidden = int(scale_shift_table.shape[1])
    if hidden <= 0 or timestep.shape[-1] != total_params * hidden:
        raise ValueError("timestep last dim must equal 9 * hidden")
    if hidden % 256 != 0 or hidden > 8192:
        raise ValueError("hidden size is outside the supported LTX2 fast-path range")

    batch, seq, _ = timestep.shape
    rows = int(batch * seq)
    # Each returned output is a disjoint, contiguous view, so one allocation
    # avoids nine allocator round trips per transformer block.
    output_storage = torch.empty(
        (9, batch, seq, hidden), device=timestep.device, dtype=timestep.dtype
    )
    outs = tuple(output_storage.unbind(dim=0))

    # Flatten (row, slice, column-block) into one grid so tiny B*S still
    # launches enough CTAs and no CTA carries nine hidden-size tiles.
    block_n = 1024
    grid = (rows * total_params * triton.cdiv(hidden, block_n),)
    _ltx2_ada_values9_kernel[grid](
        timestep,
        scale_shift_table,
        output_storage,
        hidden,
        total_params,
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        BLOCK_N=block_n,
        num_warps=4,
    )
    return outs
