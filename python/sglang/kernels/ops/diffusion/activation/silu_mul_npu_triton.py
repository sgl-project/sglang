# SPDX-License-Identifier: Apache-2.0
"""Ascend-shaped ``silu(a) * b`` for the Qwen-Image 2.1 FFN.

``silu_mul_bitexact`` launches one 1024-element program per block. On Ascend
that is thousands of programs queued onto 40-48 vector cores, and each one
moves only 2 KB -- measured ~60x slower than the two eager aclnn ops it
replaces. The rounding is not the problem; the launch shape is.

This variant keeps the identical arithmetic (``round_bf16(a * sigmoid(a))``
then one rounding on the multiply, exactly the two eager bf16 boundaries) and
reshapes the launch for Ascend:

* a persistent grid of at most ``num_progs`` programs (default 48), each
  walking several large contiguous tiles;
* no masks, no per-element index arithmetic -- the tail is one extra launch;
* a single 1D address space, so every load/store is one contiguous DMA.

``tl.sigmoid`` is still an approximation, so callers must verify the first call
(``BitExactFusionGate``) exactly as for the CUDA kernel.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.numerics import cast_rne_bf16_to_fp32

DEFAULT_TILE = 4096  # bf16 elements per program iteration (2 inputs + output in UB)
DEFAULT_PROGS = 40  # Ascend910 exposes 40 vector cores; more only queues

_MAX_I32_INDEX = 2**31


@triton.jit
def _silu_mul_kernel(
    out_ptr,
    a_ptr,
    b_ptr,
    num_tiles,
    NUM_PROGS: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, TILE)
    for tile in range(pid, num_tiles, NUM_PROGS):
        offs = tile * TILE + cols
        a = tl.load(a_ptr + offs).to(tl.float32)
        b = tl.load(b_ptr + offs).to(tl.float32)
        s = cast_rne_bf16_to_fp32(a * tl.sigmoid(a))
        tl.store(out_ptr + offs, s * b)  # store rounds the multiply


@triton.jit
def _silu_mul_tail_kernel(
    out_ptr,
    a_ptr,
    b_ptr,
    start,
    numel,
    TILE: tl.constexpr,
):
    offs = start + tl.arange(0, TILE)
    mask = offs < numel
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = cast_rne_bf16_to_fp32(a * tl.sigmoid(a))
    tl.store(out_ptr + offs, s * b, mask=mask)


def silu_mul_npu(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    tile: int = DEFAULT_TILE,
    num_progs: int = DEFAULT_PROGS,
) -> torch.Tensor:
    """``silu(a) * b`` for contiguous same-shape bf16 tensors."""
    if not (
        (a.is_npu or a.is_cuda)
        and a.dtype is torch.bfloat16
        and b.dtype is torch.bfloat16
        and a.device == b.device
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
        and 0 < a.numel() < _MAX_I32_INDEX
    ):
        raise RuntimeError("unsupported input for silu_mul_npu")
    numel = a.numel()
    num_tiles, tail = divmod(numel, tile)
    out = torch.empty_like(a)
    if num_tiles:
        progs = max(1, min(num_tiles, num_progs))
        _silu_mul_kernel[(progs,)](
            out,
            a,
            b,
            num_tiles,
            NUM_PROGS=progs,
            TILE=tile,
        )
    if tail:
        _silu_mul_tail_kernel[(1,)](
            out,
            a,
            b,
            num_tiles * tile,
            numel,
            TILE=triton.next_power_of_2(tail),
        )
    return out


__all__ = ["DEFAULT_PROGS", "DEFAULT_TILE", "silu_mul_npu"]
