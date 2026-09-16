from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator


_PRIMITIVES: Final = ["all_reduce", "all_gather", "reduce_scatter"]


def get_multicast_ptr(tensor: torch.Tensor) -> int:
    """Multicast alias of a symmetric-memory tensor. Collective on first call;
    torch caches the handle per allocation, so repeats stay cheap.
    """
    from torch._C._distributed_c10d import _SymmetricMemory

    ptr = _SymmetricMemory.rendezvous(tensor).multicast_ptr
    assert ptr != 0, "tensor has no multicast alias; was it allocated p2p?"
    return ptr


@cache_once
def _jit_pull_module(dtype: torch.dtype, num_unroll: int) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "nvlink_comm_pull",
        *args,
        f"unroll{num_unroll}",
        cuda_files=["distributed/nvlink_comm.cuh"],
        cuda_wrappers=[
            (n, f"NVLinkComm<{args}>::{n}_pull<{num_unroll}>") for n in _PRIMITIVES
        ],
    )


@cache_once
def _jit_push_module(dtype: torch.dtype, world_size: int) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "nvlink_comm_push",
        *args,
        f"world{world_size}",
        cuda_files=["distributed/nvlink_comm.cuh"],
        cuda_wrappers=[
            (n, f"NVLinkComm<{args}>::{n}_push<{world_size}>") for n in _PRIMITIVES
        ],
    )


# `residual` on any of these is folded into the reduction; it may be shaped like
# this rank's shard or like the whole tensor, of which this rank's slice is taken.
def all_reduce_push(
    comm: Communicator,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> None:
    _jit_push_module(input.dtype, comm.world_size).all_reduce(
        comm, input, output, residual
    )


def all_gather_push(
    comm: Communicator,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> None:
    _jit_push_module(input.dtype, comm.world_size).all_gather(
        comm, input, output, residual
    )


def reduce_scatter_push(
    comm: Communicator,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> None:
    _jit_push_module(input.dtype, comm.world_size).reduce_scatter(
        comm, input, output, residual
    )


def all_reduce_pull(
    comm: Communicator,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: torch.Tensor | None = None,
    *,
    in_mc_ptr: int = 0,
    out_mc_ptr: int = 0,
    num_unroll=4,
    num_blocks_hint: int = 0,
) -> None:
    _jit_pull_module(input.dtype, num_unroll).all_reduce(
        comm,
        input,
        output,
        residual,
        in_mc_ptr or get_multicast_ptr(input),
        out_mc_ptr or get_multicast_ptr(output),
        num_blocks_hint,
    )


def all_gather_pull(
    comm: Communicator,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: torch.Tensor | None = None,
    *,
    out_mc_ptr: int = 0,
    num_unroll=4,
    num_blocks_hint: int = 0,
) -> None:
    _jit_pull_module(input.dtype, num_unroll).all_gather(
        comm,
        input,
        output,
        residual,
        out_mc_ptr or get_multicast_ptr(output),
        num_blocks_hint,
    )


def reduce_scatter_pull(
    comm: Communicator,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: torch.Tensor | None = None,
    *,
    in_mc_ptr: int = 0,
    num_unroll=4,
    num_blocks_hint: int = 0,
) -> None:
    _jit_pull_module(input.dtype, num_unroll).reduce_scatter(
        comm,
        input,
        output,
        residual,
        in_mc_ptr or get_multicast_ptr(input),
        num_blocks_hint,
    )
