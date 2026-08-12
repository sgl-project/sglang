"""Triton one-shot all-reduce over Intel XPU symmetric memory.

Every rank reads all peers' symmetric buffers (IPC-mapped by torch-xpu-ops'
``XPUSymmetricMemory``) and reduces locally. torch-xpu-ops registers no
``symm_mem::*`` reduce kernel to call instead, and Xe reports no multicast
support, so the ``multimem`` variant is not an option either.

Accumulation is fp32 in ascending rank order, so the result is bitwise identical
on every rank and stable across runs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 2048
_NUM_WARPS = 8

_TRITON_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}

SUPPORTED_DTYPES = tuple(_TRITON_DTYPE)


def pack_peer_ptrs(buffer_ptrs: Sequence[int], device: torch.device) -> torch.Tensor:
    """Pack peer buffer addresses into an int64 device tensor for Triton.

    Intel USM device pointers have the high bit set, so they are reinterpreted
    from uint64: ``torch.tensor(buffer_ptrs, dtype=torch.int64)`` overflows.
    """
    packed = np.array(list(buffer_ptrs), dtype=np.uint64).view(np.int64)
    return torch.from_numpy(packed).to(device)


@triton.jit
def _one_shot_all_reduce_kernel(
    peer_ptrs,
    out_ptr,
    numel,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for peer in tl.static_range(WORLD_SIZE):
        peer_base = tl.load(peer_ptrs + peer).to(tl.pointer_type(DTYPE))
        acc += tl.load(peer_base + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, acc.to(out_ptr.dtype.element_ty), mask=mask)


def one_shot_all_reduce(
    peer_ptrs: torch.Tensor, out: torch.Tensor, world_size: int
) -> None:
    """Sum ``world_size`` peer symmetric buffers into flat contiguous ``out``.

    The caller owns synchronization: every rank must have staged its input
    before this runs, and no rank may re-stage until all peers have read it.
    """
    numel = out.numel()
    _one_shot_all_reduce_kernel[(triton.cdiv(numel, _BLOCK_SIZE),)](
        peer_ptrs,
        out,
        numel,
        WORLD_SIZE=world_size,
        BLOCK_SIZE=_BLOCK_SIZE,
        DTYPE=_TRITON_DTYPE[out.dtype],
        num_warps=_NUM_WARPS,
    )
