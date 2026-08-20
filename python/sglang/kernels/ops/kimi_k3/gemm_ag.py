"""K3 column-parallel up_proj + multicast all-gather + add3 (bf16, TP8).

One entry point over ``csrc/kimi_k3/comm/gemm_ag.cuh``: for the latent MoE
up_proj ([M, 3584] x [3584, 7168]) at small decode M, every rank computes
only its 896-column slice of the replicated GEMM (the C++ side slices the
full weight itself), multicast-stores it into the CustomAllReduceV2 push
workspace (one more user of its double-buffer phase protocol), and a
Lamport-spin consumer assembles ``out = up_proj(x) + b (+ c)`` — reading
1/8 of the weight bytes per rank instead of all of them. Needs
:func:`sglang.kernels.ops.kimi_k3.all_reduce.register_comm` once beforehand
(the same registration the push all-reduce uses).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.kimi_k3.all_reduce import _COMM_MAP
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Kimi-K3 up_proj dims (the kernel template takes any K/N passing its
# static_asserts; this module instantiates the K3 shape).
K = 3584
N = 7168

# Largest decode batch the kernel wins at (crossover vs the replicated
# cublas GEMM + add3 tail is ~13-14 tokens on B200x8); also the GEMV
# function-table size.
MAX_TOKENS = 12


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(K, N, MAX_TOKENS, is_arch_support_pdl())
    cls = f"GEMMAGKernel<{args}>"
    return load_jit(
        "kimi_k3_gemm_ag",
        *args,
        cuda_files=["kimi_k3/comm/gemm_ag.cuh"],
        cuda_wrappers=[("run", f"{cls}::run")],
        extra_cuda_cflags=["-O3"],
    )


@register_custom_op(mutates_args=["out"])
def _gemm_ag_op(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
    out: torch.Tensor,
) -> None:
    _jit_module().run(_COMM_MAP[world_size], x, weight, b, c, out)


def gemm_ag_up_proj(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
    out: torch.Tensor,
) -> torch.Tensor:
    """``out = x @ weight.T (allgathered) + b (+ c)``, all bf16.

    ``x`` is [M, 3584] with M in [1, MAX_TOKENS]; ``weight`` is the FULL
    replicated [7168, 3584] up_proj weight (each rank reads only its own
    row block); ``b`` / ``c`` / ``out`` are [M, 7168] (``out`` is
    output-only)."""
    _gemm_ag_op(world_size, x, weight, b, c, out)
    return out
