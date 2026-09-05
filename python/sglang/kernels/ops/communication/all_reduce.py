from __future__ import annotations

import enum
from typing import TYPE_CHECKING, List, Tuple

import torch
import tvm_ffi
from tvm_ffi import Module

from sglang.kernels.jit.utils import (
    cache_once,
    empty_sentinel,
    is_arch_support_pdl,
    lazy_register_class,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api


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

if TYPE_CHECKING:
    # (cudaIpcMemHandle bytes, offset-in-allocation) for one device pointer
    IPC_HANDLE_PAIR = Tuple[List[int], int]


@cache_once
def _init_communicator() -> None:
    module = load_jit(
        "communicator",
        cuda_files=["distributed/registry.cuh"],
        cuda_wrappers=[("register_communicator", "register_communicator")],
    )
    module.register_communicator()


@lazy_register_class("sgl.distributed.PushPlane", _init_communicator)
class PushPlane(tvm_ffi.Object):
    """Lamport push plane: a zero-filled symmetric workspace + a local counter.

    All buffers are owned by the caller; this object only validates and
    records them.
    """

    # C++ interface
    if TYPE_CHECKING:
        rank: int
        world_size: int

    def __init__(
        self,
        rank: int,
        world_size: int,
        *,
        workspaces: List[torch.Tensor],
        counter: torch.Tensor,
        mc_workspace: int | None = None,
    ) -> None:
        """
        :param workspaces: per-rank ``[2 * world_size, slot_bytes]`` uint8
                           views of symmetric memory. The local rank's view
                           MUST be zero-filled before first use -- the
                           kernels poll for a pos-zero marker.
        :param counter: local ``[num_blocks, 4]`` uint8 tensor, zero-filled.
        :param mc_workspace: multicast VA of the local workspace, or None.
        """
        self.__ffi_init__(rank, world_size, workspaces, counter, mc_workspace or 0)


@lazy_register_class("sgl.distributed.PullPlane", _init_communicator)
class PullPlane(tvm_ffi.Object):
    """Symmetric per-rank buffers plus the per-block semaphores guarding them.

    Either half may be omitted; the plane then holds a 0-element tensor in its
    place and any kernel needing that half fails with a clear message. The K3
    fused collectives take semaphores only -- they reduce in place on the
    caller's own symmetric input -- while the generic all-reduce takes both,
    since it stages plain tensors through the buffers before reducing.
    """

    # C++ interface
    if TYPE_CHECKING:
        rank: int
        world_size: int

    def __init__(
        self,
        rank: int,
        world_size: int,
        *,
        workspaces: List[torch.Tensor] | None = None,
        semaphores: List[torch.Tensor] | None = None,
        mc_workspace: int | None = None,
        mc_semaphore: int | None = None,
    ) -> None:
        """
        :param workspaces: per-rank ``[num_bytes]`` uint8 views of symmetric
                           memory, or None when the caller brings its own.
        :param semaphores: per-rank ``[num_blocks, 128]`` uint8 views of
                           symmetric memory, zero-filled before first use, or
                           None when the caller never barriers on this plane.
        :param mc_workspace: multicast VA of the local workspace, or None.
        :param mc_semaphore: multicast VA of the local semaphores, or None.
        """
        if workspaces is None:
            device = torch.device("cuda", torch.cuda.current_device())
            sentinel = empty_sentinel(device, torch.uint8).view(-1)
            workspaces = [sentinel for _ in range(world_size)]
        if semaphores is None:
            device = torch.device("cuda", torch.cuda.current_device())
            sentinel = empty_sentinel(device, torch.uint8).view(-1, 128)
            semaphores = [sentinel for _ in range(world_size)]

        mc_workspace = mc_workspace or 0
        mc_semaphore = mc_semaphore or 0
        self.__ffi_init__(
            rank, world_size, workspaces, semaphores, mc_workspace, mc_semaphore
        )


@lazy_register_class("sgl.distributed.Communicator", _init_communicator)
class Communicator(tvm_ffi.Object):
    """The planes shared by every kernel in ``kernels.ops.communication``.

    Pass ``None`` for a plane the owner never uses; kernels that need it then
    fail with a clear message instead of reading a placeholder buffer.
    """

    if TYPE_CHECKING:
        # C++ interface
        def get_rank(self) -> int: ...
        def get_world_size(self) -> int: ...
        def get_push(self) -> PushPlane | None: ...
        def get_pull(self) -> PullPlane | None: ...
        def set_pull_blocks(self, num_blocks: int | None) -> None: ...
        def set_pull_multicast_blocks(self, num_blocks: int | None) -> None: ...

    def __init__(
        self,
        push: PushPlane | None = None,
        pull: PullPlane | None = None,
    ) -> None:
        self.__ffi_init__(push, pull)

    @property
    def rank(self) -> int:
        return self.get_rank()

    @property
    def world_size(self) -> int:
        return self.get_world_size()

    @property
    def push(self) -> PushPlane | None:
        """The push plane, or None for a pull-only communicator."""
        return self.get_push()

    @property
    def pull(self) -> PullPlane | None:
        """The pull plane, or None for a push-only communicator."""
        return self.get_pull()


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
    """Batched cudaIPC handle exchange for CUDA-graph input pointers."""

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
        cuda_wrappers=[("all_reduce", f"AllReduceKernel<{args}>::run")],
    )


@debug_kernel_api
def custom_all_reduce(
    comm: Communicator,
    input: torch.Tensor,
    algo: AllReduceAlgo,
    *,
    graph_params: torch.Tensor | None = None,
    use_multicast: bool = False,
) -> torch.Tensor:
    module = get_all_reduce_module(input.dtype, comm.world_size)
    result = module.all_reduce(comm, input, algo.algo_name, graph_params, use_multicast)
    return torch.from_dlpack(result)


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
