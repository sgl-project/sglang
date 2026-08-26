from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple

import torch
from tvm_ffi import Module

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.all_reduce import Communicator


class Partition(NamedTuple):
    num_prefix_tokens: int  # excluive prefix sum of tokens
    num_local_tokens: int  # number of tokens in this rank


def get_token_partion(num_tokens: int, comm: Communicator) -> Partition:
    rank = comm.rank
    world_size = comm.world_size
    avg_tokens = num_tokens // world_size
    rem_tokens = num_tokens % world_size
    return Partition(
        num_prefix_tokens=avg_tokens * rank + min(rank, rem_tokens),
        num_local_tokens=avg_tokens + (1 if rank < rem_tokens else 0),
    )


SUPPORTED_OPS: Final = ["all_reduce", "all_gather", "reduce_scatter"]


def get_multicast_ptr(tensor: torch.Tensor) -> int:
    """Multicast alias of a symmetric-memory tensor. Collective on first call.

    torch caches the handle per allocation, so this stays cheap on repeat; a
    cache of our own keyed by address would go stale the moment an allocation is
    freed and the address reused.
    """
    from torch._C._distributed_c10d import _SymmetricMemory

    ptr = _SymmetricMemory.rendezvous(tensor).multicast_ptr
    assert ptr != 0, "tensor has no multicast alias; was it allocated p2p?"
    return ptr


@cache_once
def _jit_misc_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "nvl_comm_misc",
        *args,
        cuda_files=["distributed/nvlink_comm.cuh"],
        cuda_wrappers=[
            ("barrier", f"nvlink_barrier<{args}>"),
            ("all_gather_copy_engine", f"all_gather_copy_engine<{args}>"),
        ],
    )


@cache_once
def _jit_pull_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "nvl_comm_pull",
        *args,
        cuda_files=["distributed/nvlink_comm.cuh"],
        cuda_wrappers=[(n, f"NVLinkComm<{args}>::{n}_pull") for n in SUPPORTED_OPS],
    )


@cache_once
def _jit_push_module(dtype: torch.dtype, world_size: int) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "nvl_comm_push",
        str(world_size),
        *args,
        cuda_files=["distributed/nvlink_comm.cuh"],
        cuda_wrappers=[
            (n, f"NVLinkComm<{args}>::{n}_push<{world_size}>") for n in SUPPORTED_OPS
        ],
    )


def all_reduce_push(comm: Communicator, input: torch.Tensor, output: torch.Tensor):
    _jit_push_module(input.dtype, comm.world_size).all_reduce(comm, input, output)


def all_gather_push(comm: Communicator, input: torch.Tensor, output: torch.Tensor):
    _jit_push_module(input.dtype, comm.world_size).all_gather(comm, input, output)


def reduce_scatter_push(comm: Communicator, input: torch.Tensor, output: torch.Tensor):
    _jit_push_module(input.dtype, comm.world_size).reduce_scatter(comm, input, output)


def all_reduce_pull(
    comm: Communicator,
    input: torch.Tensor,
    output: torch.Tensor,
    *,
    in_mc_ptr: int = 0,
    out_mc_ptr: int = 0,
    num_blocks_hint: int = 0,
) -> None:
    _jit_pull_module(input.dtype).all_reduce(
        comm,
        input,
        output,
        in_mc_ptr or get_multicast_ptr(input),
        out_mc_ptr or get_multicast_ptr(output),
        num_blocks_hint,
    )


def all_gather_pull(
    comm: Communicator,
    input: torch.Tensor,
    output: torch.Tensor,
    *,
    out_mc_ptr: int = 0,
    num_blocks_hint: int = 32,
) -> None:
    _jit_pull_module(input.dtype).all_gather(
        comm,
        input,
        output,
        out_mc_ptr or get_multicast_ptr(output),
        num_blocks_hint,
    )


def reduce_scatter_pull(
    comm: Communicator,
    input: torch.Tensor,
    output: torch.Tensor,
    *,
    in_mc_ptr: int = 0,
    num_blocks_hint: int = 0,
) -> None:
    _jit_pull_module(input.dtype).reduce_scatter(
        comm,
        input,
        output,
        in_mc_ptr or get_multicast_ptr(input),
        num_blocks_hint,
    )


def all_gather_copy_engine(
    comm: Communicator,
    input: torch.Tensor,
    output: torch.Tensor,
    *,
    out_mc_ptr: int = 0,
) -> None:
    _jit_misc_module().all_gather_copy_engine(
        comm,
        input,
        output,
        out_mc_ptr or get_multicast_ptr(output),
    )


def barrier(comm: Communicator, *, stream: int | None = None) -> None:
    """Multicast barrier across the plane's ranks.

    The stream is passed explicitly because this call has no tensor argument,
    and tvm-ffi only publishes the framework stream when one is present. Left
    to the environment, the launch lands on a stale stream and a graph capture
    silently drops it.
    """
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    _jit_misc_module().barrier(comm, stream)
