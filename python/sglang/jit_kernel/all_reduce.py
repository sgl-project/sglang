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
    old_num_blocks: int
    old_num_threads: int


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
        def free(self) -> None: ...
        def reset_graph(self) -> None: ...
        def all_reduce(self, input: torch.Tensor, shot: int) -> tvm_ffi.Tensor: ...
        def config(self, num_blocks: int, num_threads: int) -> ConfigResult: ...


@cache_once
def _jit_custom_all_reduce_module(dtype: torch.dtype, world_size: int):
    args = make_cpp_args(dtype, world_size, is_arch_support_pdl())
    return load_jit(
        "custom_all_reduce",
        *args,
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/custom_all_reduce.cuh"],
        cuda_wrappers=[("all_reduce", f"custom_all_reduce<{args}>")],
    )


@cache_once
def get_custom_all_reduce_cls() -> type[CustomAllReduceObj]:
    module = load_jit(
        "custom_all_reduce_base",
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/custom_all_reduce.cuh"],
        cuda_wrappers=[("register_once", "register_custom_all_reduce<void>")],
    )
    module.register_once()
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    num_sms = props.multi_processor_count  # used as max CTA number

    @tvm_ffi.register_object("sgl.CustomAllReduce")
    class CustomAllReduceObjReal(tvm_ffi.Object):
        def __init__(
            self,
            rank: int,
            world_size: int,
            buffer_bytes: int,
            graph_input_count: int,
        ) -> None:
            args = (rank, world_size, num_sms, buffer_bytes, graph_input_count)
            self.__ffi_init__(*args)
            object.__setattr__(self, "_world_size", world_size)

        def all_reduce(self, input: torch.Tensor, shot: int) -> tvm_ffi.Tensor:
            world_size = object.__getattribute__(self, "_world_size")
            module = _jit_custom_all_reduce_module(input.dtype, world_size)
            return module.all_reduce(self, input, shot)

        def config(self, num_blocks: int, num_threads: int) -> ConfigResult:
            return ConfigResult(*self.configure(num_blocks, num_threads))  # type: ignore

    return cast(type["CustomAllReduceObj"], CustomAllReduceObjReal)
