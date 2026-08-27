from __future__ import annotations

from typing import TYPE_CHECKING, Final, List, NamedTuple

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


SUPPORTED_OPS: Final = ["all_reduce", "all_gather", "reduce_scatter"]


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


class CopyEngineFlags(NamedTuple):
    flags: torch.Tensor  # [2 * world_size,] on symm-mem
    flags_ptr: int
    flags_mc_ptr: int


def make_ce_flags(
    group,
    world_size: int,
    *,
    flags: torch.Tensor | None = None,
) -> CopyEngineFlags:
    """Allocate the flag array the copy-engine barrier waits on."""
    from torch._C._distributed_c10d import _SymmetricMemory

    if flags is None:
        flags = _SymmetricMemory.empty_strided_p2p(
            (2 * world_size,),
            [1],
            torch.int32,
            torch.device("cuda", torch.cuda.current_device()),
            group.group_name,
        )
        flags.zero_()
    mc_ptr = get_multicast_ptr(flags)
    torch.cuda.synchronize()
    return CopyEngineFlags(flags, flags.data_ptr(), mc_ptr)


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
            ("all_gather_copy_engine_unicast", "all_gather_copy_engine_unicast"),
        ],
    )


@cache_once
def _jit_pull_module(dtype: torch.dtype, num_unroll: int) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "nvl_comm_pull",
        *args,
        f"unroll{num_unroll}",
        cuda_files=["distributed/nvlink_comm.cuh"],
        cuda_wrappers=[
            (n, f"NVLinkComm<{args}>::{n}_pull<{num_unroll}>") for n in SUPPORTED_OPS
        ],
    )


@cache_once
def _jit_push_module(dtype: torch.dtype, world_size: int) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "nvl_comm_push",
        *args,
        f"world{world_size}",
        cuda_files=["distributed/nvlink_comm.cuh"],
        cuda_wrappers=[
            (n, f"NVLinkComm<{args}>::{n}_push<{world_size}>") for n in SUPPORTED_OPS
        ],
    )


# `residual` on any of these is folded into the reduction rather than costing a
# separate pass. It may be shaped like this rank's shard or like the whole
# tensor; in the latter case this rank's slice is taken, so a ragged split needs
# no view on the caller's side.
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


def all_gather_copy_engine_multicast(
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


def all_gather_copy_engine_unicast(
    comm: Communicator,
    input: torch.Tensor,
    output: torch.Tensor,
    *,
    peer_out_ptrs: List[int] | None = None,
    stream: int | None = None,
    ce_flags: CopyEngineFlags,
) -> None:
    """All-gather that launches no kernel at all.

    The copy engine moves the payload -- one peer-to-peer copy per rank, started
    at this rank so the links are not all driven in the same order -- and the two
    barriers around it are stream memory ops. `output` must be symmetric memory,
    since this rank writes its shard straight into every peer's copy; `input` is
    read locally and can be an ordinary tensor. `group` is only needed on the
    first call for a given communicator, to allocate the barrier flags.
    """
    from torch._C._distributed_c10d import _SymmetricMemory

    if peer_out_ptrs is None:
        base = _SymmetricMemory.rendezvous(output).buffer_ptrs
        # `output` may be a view into the middle of the allocation, and the peer
        # pointers address its base; the same shift applies on every rank.
        shift = output.data_ptr() - int(base[comm.rank])
        peer_out_ptrs = [int(p) + shift for p in base]
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    _jit_misc_module().all_gather_copy_engine_unicast(
        comm, input, output, peer_out_ptrs, *ce_flags[1:], stream
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
