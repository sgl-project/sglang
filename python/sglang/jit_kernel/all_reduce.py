from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, Tuple, cast

import torch
import tvm_ffi

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)


class ConfigResult(NamedTuple):
    num_blocks: int
    num_threads: int


if TYPE_CHECKING:
    CUSTOM_AR_HANDLE = List[int]
    CUSTOM_AR_PAIR = Tuple[int, CUSTOM_AR_HANDLE]

    class CustomAllReduceObj:
        def __init__(
            self,
            rank: int,
            world_size: int,
            buffer_bytes: int,
            graph_input_count: int,
        ) -> None: ...
        @property
        def world_size(self) -> int: ...
        def share_storage(self) -> CUSTOM_AR_HANDLE: ...
        def share_graph_inputs(self) -> List[CUSTOM_AR_PAIR]: ...
        def post_init(self, handles: List[CUSTOM_AR_HANDLE]) -> None: ...
        def register_inputs(self, handles: List[List[CUSTOM_AR_PAIR]]) -> None: ...
        def set_cuda_graph_capture(self, is_capturing: bool) -> None: ...
        def free(self, tp_cpu_group: torch.distributed.ProcessGroup) -> None: ...
        def reset_graph(self) -> None: ...
        def all_reduce(self, input: torch.Tensor, shot: int) -> tvm_ffi.Tensor: ...
        def config(self, num_blocks: int = -1, num_threads: int = -1) -> ConfigResult:
            """
            Configure the CUDA kernel's grid and block dimensions.
            This provides only the upper bound of the configuration,
            and the actual launch configuration may be determined by implementation.

            Args:
                num_blocks: The maximum number of thread blocks to launch. -1 means no limit.
                num_threads: The maximum number of threads per block. -1 means no limit.

            Returns:
                The previous configuration as a ConfigResult named tuple.
            """
            ...


@cache_once
def _jit_custom_all_reduce_module(dtype: torch.dtype, world_size: int):
    args = make_cpp_args(dtype, world_size, is_arch_support_pdl())
    return load_jit(
        "custom_all_reduce",
        *args,
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/custom_all_reduce_impl.cuh"],
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
    MAX_CTA = props.multi_processor_count
    MAX_THREADS = 512

    @tvm_ffi.register_object("sgl.CustomAllReduce")
    class CustomAllReduceObjReal(tvm_ffi.Object):
        def __init__(
            self,
            rank: int,
            world_size: int,
            buffer_bytes: int,
            graph_input_count: int,
        ) -> None:
            args = (rank, world_size, MAX_CTA, buffer_bytes, graph_input_count)
            self.__ffi_init__(*args)
            self._world_size = world_size
            self._config = ConfigResult(MAX_CTA, MAX_THREADS)
            self.configure(*self._config)  # type: ignore

        def all_reduce(self, input: torch.Tensor, shot: int) -> tvm_ffi.Tensor:
            module = _jit_custom_all_reduce_module(input.dtype, self._world_size)
            return module.all_reduce(self, input, shot)

        def config(self, num_blocks: int = -1, num_threads: int = -1) -> ConfigResult:
            old_config = self._config
            num_blocks = num_blocks if num_blocks != -1 else old_config.num_blocks
            num_threads = num_threads if num_threads != -1 else old_config.num_threads
            new_config = ConfigResult(num_blocks, num_threads)
            if new_config != old_config:
                result = ConfigResult(*self.configure(*new_config))  # type: ignore
                assert result == self._config
                self._config = new_config
            return old_config

        def free(self, tp_cpu_group: torch.distributed.ProcessGroup) -> None:
            self.free_ipc_handles()  # type: ignore
            torch.distributed.barrier(group=tp_cpu_group)
            self.free_storage()  # type: ignore

    return cast(type["CustomAllReduceObj"], CustomAllReduceObjReal)
