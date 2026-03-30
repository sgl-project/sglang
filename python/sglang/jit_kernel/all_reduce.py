from __future__ import annotations

import enum
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, cast

import torch
import tvm_ffi

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernel_api_logging import debug_kernel_api


class ConfigResult(NamedTuple):
    num_blocks: int
    num_threads: int


class AllReduceAlgo(enum.Enum):
    ONE_SHOT_PUSH = enum.auto()
    ONE_SHOT_PULL = enum.auto()
    TWO_SHOT_PULL = enum.auto()

    def is_push(self) -> bool:
        return self == AllReduceAlgo.ONE_SHOT_PUSH

    @property
    def shot(self) -> int:
        return 2 if self == AllReduceAlgo.TWO_SHOT_PULL else 1


if TYPE_CHECKING:
    CUSTOM_AR_HANDLE = List[int]
    CUSTOM_AR_PAIR = Tuple[int, CUSTOM_AR_HANDLE]

    class CustomAllReduceObj:
        def __init__(
            self,
            rank: int,
            world_size: int,
            pull_buffer_bytes: int,
            push_buffer_bytes: int,
            graph_input_count: int,
            *,
            max_pull_blocks: Optional[int] = None,
            max_push_blocks: Optional[int] = None,
        ) -> None:
            """
            Create a CustomAllReduceObj instance.

            :param rank: The rank of the current process.
            :param world_size: The total number of processes in the group.
            :param pull_buffer_bytes: The size of the buffer (in bytes) used for pull-based all-reduce.
            :param push_buffer_bytes: The size of the buffer (in bytes) used for push-based all-reduce.
            :param graph_input_count: The maximum number of inputs in all CUDA graphs.
            :param max_pull_blocks: The maximum number of thread blocks to launch for pull-based all-reduce.
                                    If None, it will be determined by the implementation.
            :param max_push_blocks: The maximum number of thread blocks to launch for push-based all-reduce.
                                    If None, it will be determined by the implementation.
            """

        @property
        def world_size(self) -> int: ...
        def share_storage(self) -> CUSTOM_AR_HANDLE: ...
        def share_graph_inputs(self) -> List[CUSTOM_AR_PAIR]: ...
        def post_init(self, handles: List[CUSTOM_AR_HANDLE]) -> None: ...
        def register_inputs(self, handles: List[List[CUSTOM_AR_PAIR]]) -> None: ...
        def set_cuda_graph_capture(self, is_capturing: bool) -> None: ...
        def free(self, tp_cpu_group: torch.distributed.ProcessGroup) -> None: ...
        def all_reduce(
            self, input: torch.Tensor, algo: AllReduceAlgo
        ) -> tvm_ffi.Tensor: ...
        def config_pull(
            self, num_blocks: int = -1, num_threads: int = -1
        ) -> ConfigResult:
            """
            Configure the CUDA kernel's grid and block dimensions.
            This provides only the upper bound of the configuration,
            and the actual launch configuration may be determined by implementation.
            Note that push-based all-reduce can not be configured currently.

            :param num_blocks: The maximum number of thread blocks to launch. -1 means no limit.
            :param num_threads: The maximum number of threads per block. -1 means no limit.

            :return: The previous configuration as a ConfigResult named tuple.
            """
            ...


@cache_once
def _jit_custom_all_reduce_pull_module(dtype: torch.dtype, world_size: int):
    args = make_cpp_args(dtype, world_size, is_arch_support_pdl())
    return load_jit(
        "custom_all_reduce",
        *args,
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/custom_all_reduce_pull.cuh"],
        cuda_wrappers=[("all_reduce", f"custom_all_reduce<{args}>")],
    )


@cache_once
def _jit_custom_all_reduce_push_module(dtype: torch.dtype, world_size: int):
    args = make_cpp_args(dtype, world_size, is_arch_support_pdl())
    return load_jit(
        "custom_all_reduce",
        *args,
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/custom_all_reduce_push.cuh"],
        cuda_wrappers=[("all_reduce", f"custom_all_reduce<{args}>")],
    )


@cache_once
def get_custom_all_reduce_cls() -> type[CustomAllReduceObj]:
    module = load_jit(
        "custom_all_reduce_base",
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/custom_all_reduce_base.cuh"],
        cuda_wrappers=[("register_once", "register_custom_all_reduce")],
    )
    module.register_once()
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    NUM_CTA = props.multi_processor_count
    MAX_THREADS = 512

    @tvm_ffi.register_object("sgl.CustomAllReduce")
    class CustomAllReduceObjReal(tvm_ffi.Object):
        __slots__ = ("__dict__",)

        def __init__(
            self,
            rank: int,
            world_size: int,
            pull_buffer_bytes: int,
            push_buffer_bytes: int,
            graph_input_count: int,
            *,
            max_pull_blocks: Optional[int] = None,
            max_push_blocks: Optional[int] = None,
        ) -> None:
            self.__ffi_init__(
                rank,
                world_size,
                NUM_CTA if max_pull_blocks is None else max_pull_blocks,
                NUM_CTA if max_push_blocks is None else max_push_blocks,
                pull_buffer_bytes,
                push_buffer_bytes,
                graph_input_count,
            )
            self._world_size = world_size
            self._pull_config = ConfigResult(NUM_CTA, MAX_THREADS)
            self.configure_pull(*self._pull_config)  # type: ignore

        @property
        def world_size(self) -> int:
            return self._world_size

        @debug_kernel_api
        def all_reduce(
            self,
            input: torch.Tensor,
            algo: AllReduceAlgo,
        ) -> tvm_ffi.Tensor:
            compile_fn = (
                _jit_custom_all_reduce_push_module
                if algo.is_push()
                else _jit_custom_all_reduce_pull_module
            )
            module = compile_fn(input.dtype, self._world_size)
            return module.all_reduce(self, input, algo.shot)

        def config_pull(
            self, num_blocks: int = -1, num_threads: int = -1
        ) -> ConfigResult:
            old_config = self._pull_config
            num_blocks = num_blocks if num_blocks != -1 else old_config.num_blocks
            num_threads = num_threads if num_threads != -1 else old_config.num_threads
            new_config = ConfigResult(num_blocks, num_threads)
            if new_config != old_config:
                result = ConfigResult(*self.configure_pull(*new_config))  # type: ignore
                assert result == self._pull_config
                self._pull_config = new_config
            return old_config

        def free(self, tp_cpu_group: torch.distributed.ProcessGroup) -> None:
            self.free_ipc_handles()  # type: ignore
            torch.distributed.barrier(group=tp_cpu_group)
            self.free_storage()  # type: ignore

    return cast(type["CustomAllReduceObj"], CustomAllReduceObjReal)
