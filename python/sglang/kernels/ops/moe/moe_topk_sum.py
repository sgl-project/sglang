"""CUDA JIT top-k expert-output sum: out[M, K] = in[M, topk, K].sum(dim=1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_THREADS: int = 256


@cache_once
def _jit_topk_sum_module(threads: int) -> Module:
    args = make_cpp_args(threads, is_arch_support_pdl())
    return load_jit(
        "moe_topk_sum_" + str(threads),
        *args,
        cuda_files=["moe/topk_sum.cuh"],
        cuda_wrappers=[("run", f"TopkSumKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def moe_topk_sum(
    x: torch.Tensor, out: torch.Tensor, scale: float = 1.0, threads: int = _THREADS
) -> torch.Tensor:
    """out[M, K] = x[M, topk, K].sum(dim=1) * scale for contiguous bf16 tensors.

    ``threads`` sets the block width; the grid is ``(ceil(K / 8 / threads), M)``, so a narrow block
    buys CTAs at decode (K=2560, M=8: 32 threads turns 16 CTAs into 80).
    """
    _jit_topk_sum_module(threads).run(x, out, scale)
    return out
