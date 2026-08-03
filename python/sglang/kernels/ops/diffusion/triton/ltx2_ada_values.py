# Adapted from NVlabs/Sana sol-engine LTX2 Ada-value fusion.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _ltx2_ada_values9_kernel(
    temb_ptr,
    table_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    out4_ptr,
    out5_ptr,
    out6_ptr,
    out7_ptr,
    out8_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    total_params: tl.constexpr,
    table_stride_p: tl.constexpr,
    table_stride_d: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < hidden
    temb_row = temb_ptr + row * total_params * hidden
    base = row * hidden + cols

    table0 = tl.load(
        table_ptr + 0 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb0 = tl.load(
        temb_row + (0 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table1 = tl.load(
        table_ptr + 1 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb1 = tl.load(
        temb_row + (1 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table2 = tl.load(
        table_ptr + 2 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb2 = tl.load(
        temb_row + (2 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table3 = tl.load(
        table_ptr + 3 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb3 = tl.load(
        temb_row + (3 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table4 = tl.load(
        table_ptr + 4 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb4 = tl.load(
        temb_row + (4 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table5 = tl.load(
        table_ptr + 5 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb5 = tl.load(
        temb_row + (5 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table6 = tl.load(
        table_ptr + 6 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb6 = tl.load(
        temb_row + (6 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table7 = tl.load(
        table_ptr + 7 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb7 = tl.load(
        temb_row + (7 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table8 = tl.load(
        table_ptr + 8 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb8 = tl.load(
        temb_row + (8 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)

    tl.store(out0_ptr + base, (table0 + temb0).to(tl.bfloat16), mask=mask)
    tl.store(out1_ptr + base, (table1 + temb1).to(tl.bfloat16), mask=mask)
    tl.store(out2_ptr + base, (table2 + temb2).to(tl.bfloat16), mask=mask)
    tl.store(out3_ptr + base, (table3 + temb3).to(tl.bfloat16), mask=mask)
    tl.store(out4_ptr + base, (table4 + temb4).to(tl.bfloat16), mask=mask)
    tl.store(out5_ptr + base, (table5 + temb5).to(tl.bfloat16), mask=mask)
    tl.store(out6_ptr + base, (table6 + temb6).to(tl.bfloat16), mask=mask)
    tl.store(out7_ptr + base, (table7 + temb7).to(tl.bfloat16), mask=mask)
    tl.store(out8_ptr + base, (table8 + temb8).to(tl.bfloat16), mask=mask)


def ltx2_ada_values9(
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if timestep.ndim != 3:
        raise ValueError("timestep must have shape [B, S, 9 * D]")
    if not timestep.is_cuda or timestep.dtype != torch.bfloat16:
        raise ValueError("timestep must be a CUDA bfloat16 tensor")
    if not timestep.is_contiguous():
        raise ValueError("timestep must be contiguous")
    if scale_shift_table.ndim != 2 or scale_shift_table.shape[0] != 9:
        raise ValueError("scale_shift_table must have shape [9, D]")
    if (
        not scale_shift_table.is_cuda
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
    outs = tuple(
        torch.empty((batch, seq, hidden), device=timestep.device, dtype=timestep.dtype)
        for _ in range(9)
    )
    _ltx2_ada_values9_kernel[(rows,)](
        timestep,
        scale_shift_table,
        *outs,
        rows,
        hidden,
        total_params,
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        BLOCK_N=triton.next_power_of_2(hidden),
        num_warps=4 if hidden >= 4096 else 8,
    )
    return outs
