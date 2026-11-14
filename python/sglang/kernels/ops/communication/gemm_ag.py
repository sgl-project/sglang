"""K3 column-parallel up_proj + multicast all-gather + add3 (bf16, TP2/4/8).

Each rank produces its ``7168 / world``-column slice of the latent MoE up_proj
([M, 3584] x [3584, 7168]) and multicast-stores it into the CustomAllReduceV2
push workspace, where a Lamport-spin consumer assembles
``out = up_proj(x) + b (+ c)``. :func:`gemm_ag_up_proj` fuses a GEMV producer;
:func:`gather_add3` takes a slice the caller computed. Needs
:func:`sglang.kernels.ops.communication.all_reduce_residual.register_comm` once
beforehand (the same registration the push all-reduce uses).
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
from sglang.kernels.ops.communication.all_reduce_residual import get_comm
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Kimi-K3 up_proj dims (the kernel template takes any K/N passing its
# static_asserts; this module instantiates the K3 shape).
K = 3584
N = 7168

WORLD_SIZES = (2, 4, 8)

# Largest decode batch the fused GEMV producer serves (crossover vs the
# replicated cublas GEMM + add3 tail is ~13-14 tokens on B200x8); also the
# GEMV function-table size.
GEMV_MAX_TOKENS = 12


@cache_once
def _jit_module(world_size: int) -> Module:
    args = make_cpp_args(K, N, world_size, GEMV_MAX_TOKENS, is_arch_support_pdl())
    cls = f"GEMMAGKernel<{args}>"
    return load_jit(
        "kimi_k3_gemm_ag",
        *args,
        cuda_files=["kimi_k3/comm/gemm_ag.cuh"],
        cuda_wrappers=[("run", f"{cls}::run"), ("gather", f"{cls}::gather")],
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
    _jit_module(world_size).run(get_comm(world_size), x, weight, b, c, out)


@register_custom_op(mutates_args=["out"])
def _gather_add3_op(
    world_size: int,
    local: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
    out: torch.Tensor,
) -> None:
    _jit_module(world_size).gather(get_comm(world_size), local, b, c, out)


def gemm_ag_up_proj(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
    out: torch.Tensor,
) -> torch.Tensor:
    """``out = x @ weight.T (allgathered) + b (+ c)``, all bf16.

    ``x`` is [M, 3584] with M in [1, GEMV_MAX_TOKENS]; ``weight`` is the FULL
    replicated [7168, 3584] up_proj weight (each rank reads only its own
    row block); ``b`` / ``c`` / ``out`` are [M, 7168] (``out`` is
    output-only)."""
    _gemm_ag_op(world_size, x, weight, b, c, out)
    return out


def gather_add3(
    world_size: int,
    local: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
    out: torch.Tensor,
) -> torch.Tensor:
    """``out = allgather(local) + b (+ c)``, all bf16.

    ``local`` is this rank's contiguous [M, 7168 / world] column slice of
    ``x @ weight.T`` (rank r holds columns ``[r * 7168 / world, (r + 1) *
    7168 / world)``); ``b`` / ``c`` / ``out`` are [M, 7168] (``out`` is
    output-only). The add order matches :func:`gemm_ag_up_proj` and the
    replicated ``bf16(up_proj + b) + c`` tail."""
    _gather_add3_op(world_size, local, b, c, out)
    return out
