# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Fused prologue of the sparse TARGET_VERIFY graph (SGLANG_OPT_DFLASH_VERIFY_PROLOGUE_FUSED, default off).

``K31SparseVerifyHostBinding.forward`` (kernel_plugins/nvidia/kimi_k31_verify_dsa_binding.py) opens the verify graph with
four torch kernels before the embedding gather: ``(slots >= 0).to(torch.int32)`` (two kernels), the de-expand of the
per-row page table into the per-request table (``table.copy_(real_page_table[: rows : q])``) and the expanded score
contexts (``cache_seqlens_int32[:b].view(b, 1).expand(b, q).contiguous()``). A fifth, ``slots >= 0`` passed to
``select_verify_intermediate_state_indices``, is discarded unless SGLANG_ENABLE_PP_SPEC stable rows are on; with the
knob the binding passes it as a callable, which that function evaluates only when it uses it.

``fused_verify_prologue`` writes the same three outputs with one Triton kernel. Every output is a plain copy or the same
integer comparison, so the values are bitwise those of the torch ops.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Page-table columns per program. A [b, ~16K] int32 table at 1M context / page 64 takes b x 17 programs.
_BLOCK = 1024


@triton.jit
def _verify_prologue_kernel(
    slots_ptr,
    valid_ptr,
    lens_ptr,
    contexts_ptr,
    src_ptr,
    table_ptr,
    pages,
    src_stride,
    q,
    QP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    cb = tl.program_id(1)
    if cb == 0:
        s = tl.load(slots_ptr + r)
        tl.store(valid_ptr + r, (s >= 0).to(tl.int32))
        length = tl.load(lens_ptr + r)
        offs = tl.arange(0, QP)
        tl.store(contexts_ptr + r * q + offs, tl.zeros([QP], dtype=tl.int32) + length, mask=offs < q)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    mask = cols < pages
    # Request r's rows are r * q .. r * q + q - 1 of the per-row table; all hold the same pages, row r * q is read.
    src_row = (r * q).to(tl.int64)
    x = tl.load(src_ptr + src_row * src_stride + cols, mask=mask)
    tl.store(table_ptr + r.to(tl.int64) * pages + cols, x, mask=mask)


def unsupported_reason(
    slots: torch.Tensor, page_table: torch.Tensor, lens: torch.Tensor, table_out: torch.Tensor, b: int, q: int
) -> Optional[str]:
    """None when ``fused_verify_prologue`` reproduces the binding's torch ops for these tensors, else why not."""
    if page_table.dtype != torch.int32 or lens.dtype != torch.int32 or table_out.dtype != torch.int32:
        return f"dtypes page_table {page_table.dtype} lens {lens.dtype} table {table_out.dtype} (int32 needed)"
    if slots.dtype not in (torch.int32, torch.int64):
        return f"slots dtype {slots.dtype}"
    if page_table.dim() != 2 or page_table.stride(1) != 1:
        return f"page_table shape {tuple(page_table.shape)} strides {page_table.stride()}"
    if slots.dim() != 1 or slots.stride(0) != 1 or lens.dim() != 1 or lens.stride(0) != 1:
        return "slots / lens not 1-D unit stride"
    if b < 1 or q < 1 or slots.shape[0] < b or lens.shape[0] < b or page_table.shape[0] < (b - 1) * q + 1:
        return f"too few rows for b {b} q {q}"
    if not table_out.is_contiguous() or table_out.numel() < b * page_table.shape[1]:
        return "table buffer too small or not contiguous"
    if len({t.device for t in (slots, page_table, lens, table_out)}) != 1:
        return "tensors on different devices"
    return None


def fused_verify_prologue(
    slots: torch.Tensor,
    page_table: torch.Tensor,
    lens: torch.Tensor,
    table_out: torch.Tensor,
    b: int,
    q: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (valid [b] int32, table [b, P] int32 view of ``table_out``, score contexts [b, q] int32): bitwise
    ``(slots[:b] >= 0).to(int32)``, ``page_table[: b * q : q]`` and ``lens[:b].view(b, 1).expand(b, q)``.
    Callers check ``unsupported_reason`` first."""
    pages = int(page_table.shape[1])
    valid = torch.empty((b,), dtype=torch.int32, device=slots.device)
    contexts = torch.empty((b, q), dtype=torch.int32, device=slots.device)
    table = table_out[: b * pages].view(b, pages)
    _verify_prologue_kernel[(b, triton.cdiv(pages, _BLOCK))](
        slots,
        valid,
        lens,
        contexts,
        page_table,
        table,
        pages,
        page_table.stride(0),
        q,
        QP=triton.next_power_of_2(q),
        BLOCK=_BLOCK,
        num_warps=4,
    )
    return valid, table, contexts
