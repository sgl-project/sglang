# SPDX-License-Identifier: Apache-2.0
"""Ascend-shaped ``residual + update * gate`` for Qwen-Image 2.1 blocks.

Eager on Ascend is ``aclnnMul`` + ``aclnnAdd``: two full passes over a
``[1, S, C]`` bf16 tensor (32 MB at 1024^2) with a per-timestep ``[1, 1, C]``
gate, twice per block, 32 blocks, 40 steps. The CUDA fast path
(``residual_gate_add``) is CUDA-JIT only, so NPU always pays both passes.

This kernel fuses the pair into one pass while reproducing both eager bf16
boundaries: the product is rounded to bf16 first, then the sum is rounded once
by the store. It is pure elementwise, so it is bit-exact by construction -- no
reduction order is involved.

Launch shape follows the Ascend rules: a persistent grid of at most
``num_progs`` programs, ``[rows, cols]`` tiles that are contiguous for every
row (``cols`` must equal the last dimension and be 32-byte aligned), no masks.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.numerics import cast_rne_bf16_to_fp32

DEFAULT_ROWS = 16  # rows per tile, clamped so one tile stays inside Ascend UB
DEFAULT_PROGS = 40  # Ascend910 exposes 40 vector cores; more only queues
ELEMENTS_BUDGET = 8192  # fp32 elements per tile (4 live buffers ~ 128 KB of UB)

_MAX_I32_INDEX = 2**31


@triton.jit
def _residual_gate_add_kernel(
    out_ptr,
    residual_ptr,
    update_ptr,
    gate_ptr,
    num_tiles,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    NUM_PROGS: tl.constexpr,
    GATE_PER_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.arange(0, ROWS)[:, None]
    col = tl.arange(0, COLS)[None, :]
    for tile in range(pid, num_tiles, NUM_PROGS):
        offs = (tile * ROWS + row) * COLS + col
        residual = tl.load(residual_ptr + offs).to(tl.float32)
        update = tl.load(update_ptr + offs).to(tl.float32)
        if GATE_PER_ROW:
            gate = tl.load(gate_ptr + offs).to(tl.float32)
        else:
            gate = tl.load(gate_ptr + col).to(tl.float32)
        # eager rounds the product to bf16, then the add rounds once more
        product = cast_rne_bf16_to_fp32(update * gate)
        tl.store(out_ptr + offs, residual + product)


def residual_gate_add_npu(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    *,
    rows_per_tile: int = DEFAULT_ROWS,
    num_progs: int = DEFAULT_PROGS,
) -> torch.Tensor:
    """``residual + update * gate`` for ``[1, S, C]`` bf16 operands.

    ``gate`` is either the same shape as ``residual`` or a single broadcast row
    (``[1, 1, C]`` / ``[C]``), which is what the per-timestep modulation is.
    """
    if not (
        (residual.is_npu or residual.is_cuda)
        and residual.dtype is torch.bfloat16
        and update.dtype is residual.dtype
        and update.device == residual.device
        and update.shape == residual.shape
        and residual.ndim >= 2
        and residual.is_contiguous()
        and update.is_contiguous()
        and 0 < residual.numel() < _MAX_I32_INDEX
    ):
        raise RuntimeError("unsupported input for residual_gate_add_npu")
    cols = residual.shape[-1]
    if cols & (cols - 1):
        raise RuntimeError("residual_gate_add_npu needs a power-of-two last dim")
    rows = residual.numel() // cols
    if rows_per_tile < 1:
        raise RuntimeError("rows_per_tile must be positive")
    # a 16 x 4096 fp32 tile overflows UB (~5x the 192 KB budget); clamp to the budget
    limit = min(rows_per_tile, max(1, ELEMENTS_BUDGET // cols))
    while limit > 1 and rows % limit:
        limit //= 2
    if rows % limit:
        raise RuntimeError("rows_per_tile must divide the row count")
    rows_per_tile = limit
    gate_per_row = gate.numel() == residual.numel()
    if not gate_per_row and not (
        gate.numel() == cols and gate.stride(-1) == 1 and gate.dtype == residual.dtype
    ):
        raise RuntimeError("gate must be [1, S, C] or a single [1, 1, C] row")

    out = torch.empty_like(residual)
    num_tiles = rows // rows_per_tile
    progs = max(1, min(num_tiles, num_progs))
    _residual_gate_add_kernel[(progs,)](
        out,
        residual,
        update,
        gate,
        num_tiles,
        ROWS=rows_per_tile,
        COLS=cols,
        NUM_PROGS=progs,
        GATE_PER_ROW=gate_per_row,
    )
    return out


__all__ = ["DEFAULT_PROGS", "DEFAULT_ROWS", "residual_gate_add_npu"]
