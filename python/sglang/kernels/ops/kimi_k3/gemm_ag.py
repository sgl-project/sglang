"""K3 TP8 projections with multicast gather through the shared push workspace.

Call ``kimi_k3.all_reduce.register_comm`` first;
serialize these operations with other users of the communicator's push workspace.
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
def _front_gather_module() -> Module:
    return load_jit(
        "k3_front_gather",
        cuda_files=["kimi_k3/comm/front_gather.cuh"],
        cuda_wrappers=[("run", "front_gather::Gather::run")],
        extra_cuda_cflags=["-O3"],
    )


@register_custom_op(mutates_args=["out"])
def _front_gather_op(world_size: int, front: torch.Tensor, out: torch.Tensor) -> None:
    _front_gather_module().run(_COMM_MAP[world_size], front, out)


def gather_front_latent(*, world_size: int, front: torch.Tensor) -> torch.Tensor:
    """Gather FP32 latent slices from [M, 2880] fronts into [M, 3584]."""
    out = front.new_empty((front.shape[0], K))
    _front_gather_op(world_size=world_size, front=front, out=out)
    return out


def gemm_ag_front(
    *, world_size: int, x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm_out

    front = x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)
    cutedsl_bf16_gemm_out(x=x, weight=weight, out=front)
    return (
        front[:, :1536],
        front[:, 1536:2432],
        gather_front_latent(world_size=world_size, front=front),
    )


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
