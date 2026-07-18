from __future__ import annotations

import enum
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
import tvm_ffi
from tvm_ffi import Module

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels._jit import (
    cache_once,
    is_arch_support_pdl,
    lazy_register_class,
    load_jit,
    make_cpp_args,
)


class AllReduceAlgo(enum.Enum):
    ONE_SHOT_PUSH = enum.auto()
    ONE_SHOT_PULL = enum.auto()
    TWO_SHOT_PULL = enum.auto()

    def is_push(self) -> bool:
        return self == AllReduceAlgo.ONE_SHOT_PUSH

    @property
    def algo_name(self) -> str:
        return _ALGO_NAMES[self]


_ALGO_NAMES = {
    AllReduceAlgo.ONE_SHOT_PUSH: "1shot_push",
    AllReduceAlgo.ONE_SHOT_PULL: "1shot_pull",
    AllReduceAlgo.TWO_SHOT_PULL: "2shot_pull",
}

# ``pull_arg`` of the all-reduce kernel: a row of the graph-params pointer
# table selects graph mode; a plain bool selects multicast (True) / eager.
PullArg = Union[torch.Tensor, bool]

if TYPE_CHECKING:
    # (cudaIpcMemHandle bytes, offset-in-allocation) for one device pointer
    IPC_HANDLE_PAIR = Tuple[List[int], int]


def _init_communicator() -> None:
    module = load_jit(
        "communicator",
        cuda_files=["distributed/communicator.cuh"],
        cuda_wrappers=[("register_once", "register_communicator")],
    )
    module.register_once()


@lazy_register_class("sgl.Communicator", _init_communicator)
class Communicator(tvm_ffi.Object):
    """Storage plane of the custom all-reduce: a thin pointer holder.

    All buffers are owned by the caller (symmetric-memory tensor views plus
    a local push counter); this object only validates and records them.
    """

    if TYPE_CHECKING:
        # C++ interface
        rank: int
        world_size: int

        def _config(self, kwargs: dict) -> None: ...

    def __init__(
        self,
        rank: int,
        world_size: int,
        push_workspaces: List[torch.Tensor],
        pull_workspaces: List[torch.Tensor],
        pull_semaphores: List[torch.Tensor],
        push_counter: torch.Tensor,
        pull_mc_workspace: int | None,
    ) -> None:
        """
        :param push_workspaces: per-rank ``[2 * world_size, push_bytes]``
                                uint8 views of symmetric memory.
        :param pull_workspaces: per-rank ``[pull_bytes]`` uint8 views of
                                symmetric memory.
        :param pull_semaphores: per-rank ``[num_pull_blocks, 128]`` uint8
                                views of symmetric memory.
        :param push_counter: local ``[num_push_blocks, 4]`` uint8 tensor.
        :param pull_mc_workspace: multicast address of the pull workspace,
                                  or None when multicast is unavailable.
        """
        self.__ffi_init__(
            rank,
            world_size,
            push_workspaces,
            pull_workspaces,
            pull_semaphores,
            push_counter,
            pull_mc_workspace,
        )

    def config(
        self,
        num_pull_blocks: int | None = None,
        num_multicast_blocks: int | None = None,
    ) -> Communicator:
        kwargs = {}
        if num_pull_blocks is not None:
            kwargs["num_pull_blocks"] = num_pull_blocks
        if num_multicast_blocks is not None:
            kwargs["num_multicast_blocks"] = num_multicast_blocks
        self._config(kwargs)
        return self


def _init_ipc_manager() -> None:
    module = load_jit(
        "cuda_ipc",
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/ipc.cuh"],
        cuda_wrappers=[("register_once", "register_ipc_manager")],
    )
    module.register_once()


@lazy_register_class("sgl.IPCManager", _init_ipc_manager)
class IPCManager(tvm_ffi.Object):
    """Batched cudaIpc handle exchange for CUDA-graph input pointers."""

    if TYPE_CHECKING:
        # C++ interface
        def destroy(self) -> None: ...
        def batch_get_handles(self, ptrs: List[int]) -> List[IPC_HANDLE_PAIR]: ...
        def batch_open_handles(self, handles: List[IPC_HANDLE_PAIR]) -> List[int]: ...

    def __init__(self) -> None:
        self.__ffi_init__()


@cache_once
def get_all_reduce_module(dtype: torch.dtype, world_size: int) -> Module:
    args = make_cpp_args(dtype, world_size, is_arch_support_pdl())
    return load_jit(
        "custom_all_reduce",
        *args,
        cuda_files=["distributed/custom_all_reduce.cuh"],
        cuda_wrappers=[("all_reduce", f"custom_all_reduce<{args}>")],
    )


@debug_kernel_api
def custom_all_reduce(
    comm: Communicator,
    input: torch.Tensor,
    algo: AllReduceAlgo,
    pull_arg: PullArg,
) -> tvm_ffi.Tensor:
    module = get_all_reduce_module(input.dtype, comm.world_size)
    return module.all_reduce(comm, input, algo.algo_name, pull_arg)


@cache_once
def get_fused_parallel_qknorm_module(
    dtype: torch.dtype, world_size: int, q_dim: int, k_dim: int
) -> Module:
    args = make_cpp_args(dtype, world_size, q_dim, k_dim, is_arch_support_pdl())
    cls_name = f"FusedParallelQKNormAcrossHead<{args}>"
    return load_jit(
        "tp_qknorm",
        *args,
        cuda_files=["distributed/tp_qknorm.cuh"],
        cuda_wrappers=[
            ("fused_parallel_qknorm", f"{cls_name}::run"),
            ("get_max_occupancy", f"{cls_name}::get_max_occupancy"),
        ],
    )


def get_fused_parallel_qknorm_max_occupancy(
    dtype: torch.dtype, world_size: int, q_dim: int, k_dim: int
) -> int:
    module = get_fused_parallel_qknorm_module(dtype, world_size, q_dim, k_dim)
    return module.get_max_occupancy()


def fused_parallel_qknorm(
    comm: Communicator,
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    world_size = comm.world_size
    q_dim = q.shape[-1] * world_size
    k_dim = k.shape[-1] * world_size
    module = get_fused_parallel_qknorm_module(q.dtype, world_size, q_dim, k_dim)
    module.fused_parallel_qknorm(comm, q, k, q_weight, k_weight, eps)
