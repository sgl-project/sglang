"""Gemma-4 fused TP all-reduce + layer epilogue + per-128 FP8 quant (bf16 in).

Two entry points over ``csrc/gemma4/comm/ar_fusion.cuh``, both NVLS pull
kernels on the CustomAllReduceV2 barrier plane (same semaphore protocol as the
Kimi-K3 pull kernels in :mod:`all_reduce_residual`)::

    site:  x = allreduce(partial); a = rmsnorm(x, w_a); r = (a + r) * s
           y = rmsnorm(r, w_b); out, scales = per_token_group_quant_fp8(y, 128)
    final: x = allreduce(partial); out = (rmsnorm(x, w_a) + r) * s          (bf16)

Token rows are sharded across the ranks (``row0`` / ``rows`` per rank): an owner
reads its rows summed over the team with ``multimem.ld_reduce`` and multicasts
its FP8 rows and scales (or bf16 rows for ``final``) to every rank. The
residual is this rank's ``[rows, H]`` slice of the stream and is updated in
place; it is never exchanged.

``partial``, ``out`` and ``scales`` are this rank's views of symmetric,
multicast-bound buffers whose multicast VAs the caller passes alongside (a
:func:`~sglang.srt.layers.communication.gemma4_ar_fusion.symm_buffer`). Every
rank must call with the same shapes and ``num_blocks``.

The hidden size and the block width are compiled in; :func:`threads_for` picks
the width. Needs :func:`register_comm` once per process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.communication.comm_registry import CommRegistry

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator

GROUP = 128  # quant group, the FP8 weights' block_k


def threads_for(hidden: int) -> int:
    """Block width for ``hidden``: the largest multiple of 32 up to 256 that
    divides ``hidden / 8`` (one 16-byte vector per thread per pass)."""
    if hidden % 256:
        raise ValueError(f"hidden must be a multiple of 256, got {hidden}")
    num_vecs = hidden // 8
    for threads in range(256, 0, -32):
        if num_vecs % threads == 0:
            return threads
    raise AssertionError("unreachable: hidden / 8 is a multiple of 32")


@cache_once
def _jit_module(hidden: int, threads: int) -> Module:
    args = make_cpp_args(hidden, threads, is_arch_support_pdl())
    cls = f"Gemma4ArFusionKernel<{args}>"
    return load_jit(
        "gemma4_ar_fusion",
        *args,
        cuda_files=["gemma4/comm/ar_fusion.cuh"],
        cuda_wrappers=[("site", f"{cls}::site"), ("final", f"{cls}::final")],
        extra_cuda_cflags=["-O3"],
    )


_COMMS = CommRegistry("gemma4_ar_fusion")


def register_comm(provider: Callable[[], Communicator], *, world_size: int) -> None:
    """Register the provider of the communicator these ops launch over; it runs on
    first launch (see ``CommRegistry``). The fused sites barrier on its pull semaphores.
    """
    _COMMS.register(provider, world_size=world_size)


def site(
    partial: torch.Tensor,
    partial_mc: int,
    out: torch.Tensor,
    out_mc: int,
    scales: torch.Tensor,
    scales_mc: int,
    residual: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    scalar: torch.Tensor,
    eps: float,
    row0: int,
    rows: int,
    round_a: bool,
    *,
    world_size: int,
    num_blocks: Optional[int] = None,
) -> None:
    """One fused site; see the module docstring. ``round_a`` selects the
    attention-site rounding (first norm rounded to bf16 before the add, second
    norm on the fp32 sum) over the MLP-site rounding (fp32 add, bf16 stream).
    ``num_blocks`` defaults to the barrier plane's capacity."""
    hidden = partial.shape[1]
    _jit_module(hidden, threads_for(hidden)).site(
        _COMMS.get(world_size),
        partial,
        partial_mc,
        out,
        out_mc,
        scales,
        scales_mc,
        residual,
        w_a,
        w_b,
        scalar,
        eps,
        row0,
        rows,
        round_a,
        num_blocks or 1 << 30,
    )


def final(
    partial: torch.Tensor,
    partial_mc: int,
    out: torch.Tensor,
    out_mc: int,
    residual: torch.Tensor,
    w_a: torch.Tensor,
    scalar: torch.Tensor,
    eps: float,
    row0: int,
    rows: int,
    *,
    world_size: int,
    num_blocks: Optional[int] = None,
) -> None:
    """Last layer: every row of ``(rmsnorm(allreduce(partial), w_a) + r) * s`` as
    bf16 in ``out``, which may be ``partial`` itself."""
    hidden = partial.shape[1]
    _jit_module(hidden, threads_for(hidden)).final(
        _COMMS.get(world_size),
        partial,
        partial_mc,
        out,
        out_mc,
        residual,
        w_a,
        scalar,
        eps,
        row0,
        rows,
        num_blocks or 1 << 30,
    )
