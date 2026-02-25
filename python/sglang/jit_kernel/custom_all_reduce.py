from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_custom_ar_module() -> Module:
    return load_jit(
        "custom_all_reduce",
        cuda_files=["allreduce/custom_all_reduce.cuh"],
        cuda_wrappers=[
            ("init_custom_ar", "init_custom_ar"),
            ("all_reduce", "all_reduce"),
            ("dispose", "dispose"),
            ("meta_size", "meta_size"),
            ("register_buffer", "register_buffer"),
            ("get_graph_buffer_ipc_meta", "get_graph_buffer_ipc_meta"),
            ("register_graph_buffers", "register_graph_buffers"),
        ],
    )


def init_custom_ar(
    ipc_tensors: List[int],
    rank_data: torch.Tensor,
    rank: int,
    full_nvlink: bool,
) -> int:
    module = _jit_custom_ar_module()
    return module.init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)


def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None:
    module = _jit_custom_ar_module()
    module.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)


def dispose(fa: int) -> None:
    module = _jit_custom_ar_module()
    module.dispose(fa)


def meta_size() -> int:
    module = _jit_custom_ar_module()
    return module.meta_size()


def register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None:
    module = _jit_custom_ar_module()
    module.register_buffer(fa, fake_ipc_ptrs)


def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
    module = _jit_custom_ar_module()
    return module.get_graph_buffer_ipc_meta(fa)


def register_graph_buffers(
    fa: int,
    handles: List[List[int]],
    offsets: List[List[int]],
) -> None:
    module = _jit_custom_ar_module()
    module.register_graph_buffers(fa, handles, offsets)
