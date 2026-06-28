# Adapted from NVlabs/Sana sol-engine LTX2 dual-modulation fusion.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _ltx2_rmsnorm_dual_modulate_kernel(
    x_ptr,
    y0_ptr,
    y1_ptr,
    scale0_ptr,
    shift0_ptr,
    scale1_ptr,
    shift1_ptr,
    rows: tl.constexpr,
    seq: tl.constexpr,
    hidden: tl.constexpr,
    scale0_stride_b: tl.constexpr,
    scale0_stride_s: tl.constexpr,
    scale0_stride_d: tl.constexpr,
    shift0_stride_b: tl.constexpr,
    shift0_stride_s: tl.constexpr,
    shift0_stride_d: tl.constexpr,
    scale1_stride_b: tl.constexpr,
    scale1_stride_s: tl.constexpr,
    scale1_stride_d: tl.constexpr,
    shift1_stride_b: tl.constexpr,
    shift1_stride_s: tl.constexpr,
    shift1_stride_d: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < hidden

    x = tl.load(x_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden
    normed = (x * tl.rsqrt(var + eps)).to(tl.bfloat16)

    batch = row // seq
    token = row - batch * seq

    scale0 = tl.load(
        scale0_ptr
        + batch * scale0_stride_b
        + token * scale0_stride_s
        + cols * scale0_stride_d,
        mask=mask,
        other=0.0,
    )
    shift0 = tl.load(
        shift0_ptr
        + batch * shift0_stride_b
        + token * shift0_stride_s
        + cols * shift0_stride_d,
        mask=mask,
        other=0.0,
    )
    scale1 = tl.load(
        scale1_ptr
        + batch * scale1_stride_b
        + token * scale1_stride_s
        + cols * scale1_stride_d,
        mask=mask,
        other=0.0,
    )
    shift1 = tl.load(
        shift1_ptr
        + batch * shift1_stride_b
        + token * shift1_stride_s
        + cols * shift1_stride_d,
        mask=mask,
        other=0.0,
    )

    one_plus_scale0 = (1.0 + scale0).to(tl.bfloat16)
    one_plus_scale1 = (1.0 + scale1).to(tl.bfloat16)
    y0 = ((normed * one_plus_scale0).to(tl.bfloat16) + shift0).to(tl.bfloat16)
    y1 = ((normed * one_plus_scale1).to(tl.bfloat16) + shift1).to(tl.bfloat16)
    tl.store(y0_ptr + row * hidden + cols, y0, mask=mask)
    tl.store(y1_ptr + row * hidden + cols, y1, mask=mask)


def _expand_param(
    param: torch.Tensor, batch: int, seq: int, hidden: int
) -> torch.Tensor:
    if param.ndim == 2:
        param = param[:, None, :]
    if param.ndim != 3:
        raise ValueError("scale/shift tensors must be [B,D], [B,1,D], or [B,S,D]")
    if param.shape[0] not in (1, batch) or param.shape[1] not in (1, seq):
        raise ValueError(
            f"scale/shift tensor shape {tuple(param.shape)} is not broadcastable "
            f"to {(batch, seq, hidden)}"
        )
    if param.shape[2] != hidden:
        raise ValueError(
            f"scale/shift hidden dim must be {hidden}, got {param.shape[2]}"
        )
    return param.expand(batch, seq, hidden)


def ltx2_rmsnorm_dual_modulate(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3:
        raise ValueError("x must have shape [B,S,D]")
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError("x must be bfloat16")

    batch, seq, hidden = x.shape
    params = tuple(
        _expand_param(t, batch, seq, hidden) for t in (scale0, shift0, scale1, shift1)
    )
    for param in params:
        if not param.is_cuda or param.dtype != x.dtype or param.stride(-1) != 1:
            raise ValueError(
                "scale/shift tensors must be CUDA, same dtype, last-dim contiguous"
            )

    y0 = torch.empty_like(x)
    y1 = torch.empty_like(x)
    rows = int(batch * seq)
    _ltx2_rmsnorm_dual_modulate_kernel[(rows,)](
        x,
        y0,
        y1,
        params[0],
        params[1],
        params[2],
        params[3],
        rows,
        seq,
        hidden,
        params[0].stride(0),
        params[0].stride(1),
        params[0].stride(2),
        params[1].stride(0),
        params[1].stride(1),
        params[1].stride(2),
        params[2].stride(0),
        params[2].stride(1),
        params[2].stride(2),
        params[3].stride(0),
        params[3].stride(1),
        params[3].stride(2),
        eps,
        BLOCK_N=triton.next_power_of_2(hidden),
        num_warps=4 if hidden >= 4096 else 8,
    )
    return y0, y1


@triton.jit
def _ltx2_rmsnorm_ca_dual_modulate_from_temb_kernel(
    x_ptr,
    y0_ptr,
    y1_ptr,
    temb_ptr,
    table_ptr,
    rows: tl.constexpr,
    seq: tl.constexpr,
    hidden: tl.constexpr,
    temb_stride_b: tl.constexpr,
    temb_stride_s: tl.constexpr,
    temb_stride_d: tl.constexpr,
    table_stride_p: tl.constexpr,
    table_stride_d: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < hidden

    x = tl.load(x_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden
    normed = (x * tl.rsqrt(var + eps)).to(tl.bfloat16)

    batch = row // seq
    token = row - batch * seq
    temb_base = temb_ptr + batch * temb_stride_b + token * temb_stride_s

    table0 = tl.load(
        table_ptr + 0 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table1 = tl.load(
        table_ptr + 1 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table2 = tl.load(
        table_ptr + 2 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table3 = tl.load(
        table_ptr + 3 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)

    scale0 = (
        table0
        + tl.load(
            temb_base + (0 * hidden + cols) * temb_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.bfloat16)
    ).to(tl.bfloat16)
    shift0 = (
        table1
        + tl.load(
            temb_base + (1 * hidden + cols) * temb_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.bfloat16)
    ).to(tl.bfloat16)
    scale1 = (
        table2
        + tl.load(
            temb_base + (2 * hidden + cols) * temb_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.bfloat16)
    ).to(tl.bfloat16)
    shift1 = (
        table3
        + tl.load(
            temb_base + (3 * hidden + cols) * temb_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.bfloat16)
    ).to(tl.bfloat16)

    one_plus_scale0 = (1.0 + scale0).to(tl.bfloat16)
    one_plus_scale1 = (1.0 + scale1).to(tl.bfloat16)
    y0 = ((normed * one_plus_scale0).to(tl.bfloat16) + shift0).to(tl.bfloat16)
    y1 = ((normed * one_plus_scale1).to(tl.bfloat16) + shift1).to(tl.bfloat16)
    tl.store(y0_ptr + row * hidden + cols, y0, mask=mask)
    tl.store(y1_ptr + row * hidden + cols, y1, mask=mask)


def ltx2_rmsnorm_ca_dual_modulate_from_temb(
    x: torch.Tensor,
    temb_scale_shift: torch.Tensor,
    scale_shift_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3:
        raise ValueError("x must have shape [B,S,D]")
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError("x must be bfloat16")

    batch, seq, hidden = x.shape
    if temb_scale_shift.ndim != 3:
        raise ValueError("temb_scale_shift must have shape [B,S,4*D]")
    if temb_scale_shift.shape[0] != batch or temb_scale_shift.shape[1] != seq:
        raise ValueError("temb_scale_shift batch/seq must match x")
    if temb_scale_shift.shape[2] != 4 * hidden:
        raise ValueError("temb_scale_shift last dim must be 4 * hidden")
    if (
        not temb_scale_shift.is_cuda
        or temb_scale_shift.dtype != x.dtype
        or temb_scale_shift.stride(-1) != 1
    ):
        raise ValueError("temb_scale_shift must be CUDA, bfloat16, last-dim contiguous")
    if scale_shift_table.ndim != 2 or scale_shift_table.shape[0] < 4:
        raise ValueError("scale_shift_table must have at least shape [4,D]")
    if scale_shift_table.shape[1] != hidden:
        raise ValueError("scale_shift_table hidden dim must match x")
    if (
        not scale_shift_table.is_cuda
        or scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.stride(-1) != 1
    ):
        raise ValueError(
            "scale_shift_table must be CUDA, bf16/fp32, last-dim contiguous"
        )

    y0 = torch.empty_like(x)
    y1 = torch.empty_like(x)
    rows = int(batch * seq)
    _ltx2_rmsnorm_ca_dual_modulate_from_temb_kernel[(rows,)](
        x,
        y0,
        y1,
        temb_scale_shift,
        scale_shift_table,
        rows,
        seq,
        hidden,
        temb_scale_shift.stride(0),
        temb_scale_shift.stride(1),
        temb_scale_shift.stride(2),
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        eps,
        BLOCK_N=triton.next_power_of_2(hidden),
        num_warps=4 if hidden >= 4096 else 8,
    )
    return y0, y1
