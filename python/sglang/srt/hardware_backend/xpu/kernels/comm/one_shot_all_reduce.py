"""Triton one-shot all-reduce over Intel XPU symmetric memory.

Every rank stages its input in its own symmetric buffer, and after a barrier
reads all peers' buffers (IPC-mapped by torch-xpu-ops' ``XPUSymmetricMemory``)
and reduces locally. torch-xpu-ops registers no ``symm_mem::*`` reduce kernel to
call instead, and Xe reports no multicast support, so the ``multimem`` variant
is not an option either.

The buffer holds two staging halves, picked by a per-rank call counter that
every rank advances identically, so call ``k + 2`` reuses call ``k``'s half
only after the barrier of call ``k + 1``, which every peer reaches after it has
read call ``k``'s data. One barrier per call is then enough. The counter lives
on the device so a captured graph alternates halves on every replay.

Accumulation is fp32 in ascending rank order, so the result is bitwise identical
on every rank and stable across runs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Measured on Arc Pro B60: up to 4 elements per thread keeps the peer reads
# from stalling at 6 ranks, where 8 per thread was 2.5x slower at 64 KiB.
_BLOCK_SIZE = 256
_NUM_WARPS = 4

_TRITON_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}

SUPPORTED_DTYPES = tuple(_TRITON_DTYPE)


@triton.jit
def _stage_kernel(
    inp_ptr,
    staging_ptr,
    counter_ptr,
    snapshot_ptr,
    numel,
    half_numel,
    BLOCK_SIZE: tl.constexpr,
):
    call = tl.load(counter_ptr)
    # The reduce reads the snapshot and advances the counter, so neither
    # kernel reads a value the other is writing.
    if tl.program_id(0) == 0:
        tl.store(snapshot_ptr, call)
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    tl.store(
        staging_ptr + (call & 1) * half_numel + offsets,
        tl.load(inp_ptr + offsets, mask=mask),
        mask=mask,
    )


@triton.jit
def _one_shot_all_reduce_kernel(
    peer_ptrs,
    out_ptr,
    counter_ptr,
    snapshot_ptr,
    numel,
    half_numel,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    call = tl.load(snapshot_ptr)
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for peer in tl.static_range(WORLD_SIZE):
        peer_half = tl.load(peer_ptrs + peer).to(tl.pointer_type(DTYPE))
        peer_half += (call & 1) * half_numel
        acc += tl.load(peer_half + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, acc.to(out_ptr.dtype.element_ty), mask=mask)
    if tl.program_id(0) == 0:
        tl.store(counter_ptr, call + 1)


def stage(
    *,
    inp: torch.Tensor,
    staging: torch.Tensor,
    counter: torch.Tensor,
    snapshot: torch.Tensor,
) -> None:
    """Copy flat contiguous ``inp`` into this call's half of ``staging``.

    ``staging`` is this rank's symmetric buffer viewed as ``inp.dtype``; its
    first and second halves alternate between calls.
    """
    numel = inp.numel()
    _stage_kernel[(triton.cdiv(numel, _BLOCK_SIZE),)](
        inp,
        staging,
        counter,
        snapshot,
        numel,
        staging.numel() // 2,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
    )


def one_shot_all_reduce(
    *,
    peer_ptrs: torch.Tensor,
    out: torch.Tensor,
    world_size: int,
    half_numel: int,
    counter: torch.Tensor,
    snapshot: torch.Tensor,
) -> None:
    """Sum every rank's staged half into flat contiguous ``out``.

    The caller barriers between :func:`stage` and this call on every rank.
    """
    numel = out.numel()
    _one_shot_all_reduce_kernel[(triton.cdiv(numel, _BLOCK_SIZE),)](
        peer_ptrs,
        out,
        counter,
        snapshot,
        numel,
        half_numel,
        WORLD_SIZE=world_size,
        BLOCK_SIZE=_BLOCK_SIZE,
        DTYPE=_TRITON_DTYPE[out.dtype],
        num_warps=_NUM_WARPS,
    )
