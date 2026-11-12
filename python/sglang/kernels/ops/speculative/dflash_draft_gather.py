# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_DRAFT_HEAD_GATHER: the drafted rows of every draft block (rows 1..block-1 of each request; row 0 is the
anchor) copied into one compact [bs * slots, H] buffer for the draft head, as a PDL kernel.

Without it, FusedSelectorHead reshapes the non-contiguous [bs, slots, H] slice (bs >= 2), which runs a plain ATen copy that
waits for the whole last draft AR site and never lets the head launch early. (The selector's hidden projection keeps reading
the slice: F.linear on a compact view is not bitwise to F.linear on the slice, gate HG1.)
Here the copy is launched with PDL: it waits on its predecessor (griddepcontrol.wait) before its first read, then lets its
dependents launch; a dependent reads the buffer only after its own griddepcontrol.wait, which waits for this whole grid and
its memory. Pure copy: bitwise equal to the reshape.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _draft_rows_gather_kernel(src, dst, stride_b, stride_s, slots, H: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)  # compact row = b * slots + s
    c = tl.program_id(1)
    tl.extra.cuda.gdc_wait()  # launched with PDL: the predecessor's hidden rows are complete and visible past this point
    tl.extra.cuda.gdc_launch_dependents()  # dependents still wait for this whole grid before they read dst
    b = r // slots
    s = r % slots
    offs = c * BLOCK + tl.arange(0, BLOCK)
    m = offs < H
    v = tl.load(src + b.to(tl.int64) * stride_b + s.to(tl.int64) * stride_s + offs, mask=m)
    tl.store(dst + r.to(tl.int64) * H + offs, v, mask=m)


def draft_rows_gather_covers(hidden: torch.Tensor, out: torch.Tensor) -> str:
    """'' when draft_rows_gather takes (hidden, out), else the reason."""
    if hidden.dim() != 3 or hidden.stride(2) != 1:
        return "hidden is not a [bs, slots, H] view with unit inner stride"
    if hidden.is_contiguous():
        return "hidden already contiguous (bs 1): the reshape is a view"
    rows = hidden.shape[0] * hidden.shape[1]
    if out.dim() != 2 or not out.is_contiguous() or out.shape[0] < rows or out.shape[1] != hidden.shape[2]:
        return f"buffer {tuple(out.shape)} does not hold {rows} x {hidden.shape[2]} rows"
    if out.dtype != hidden.dtype or out.device != hidden.device:
        return "buffer dtype / device differs from hidden"
    return ""


def draft_rows_gather(hidden: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """hidden [bs, slots, H] (any row strides) -> out[: bs * slots] as [bs * slots, H], compact."""
    bs, slots, h = hidden.shape
    rows = bs * slots
    dst = out[:rows]
    _draft_rows_gather_kernel[(rows, triton.cdiv(h, _BLOCK))](
        hidden, dst, hidden.stride(0), hidden.stride(1), slots, H=h, BLOCK=_BLOCK, num_warps=4, launch_pdl=True
    )
    return dst
