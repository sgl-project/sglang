# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/_custom_ops.py
import logging
import os
from typing import List, Tuple

import torch
import torch.library

from sglang.srt.utils import get_bool_env_var, is_hip, is_hpu

logger = logging.getLogger(__name__)
use_vllm_custom_allreduce = get_bool_env_var(
    "USE_VLLM_CUSTOM_ALLREDUCE", default="false"
)

if not is_hpu():
    # ROCm does not use vllm custom allreduce
    if use_vllm_custom_allreduce and not is_hip():
        try:
            import vllm._C
        except ImportError as e:
            logger.warning("Failed to import from vllm._C with %r", e)
    else:
        try:
            import sgl_kernel
        except ImportError as e:
            logger.warning("Failed to import from custom_ar with %r", e)


if not is_hip():
    if use_vllm_custom_allreduce:
        custom_op = torch.ops._C_custom_ar
    else:
        custom_op = sgl_kernel.allreduce

    # custom allreduce
    def init_custom_ar(
        ipc_tensors: List[torch.Tensor],
        rank_data: torch.Tensor,
        rank: int,
        full_nvlink: bool,
    ) -> int:
        return custom_op.init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)

    def all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
    ) -> None:
        custom_op.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)

    def dispose(fa: int) -> None:
        custom_op.dispose(fa)

    def meta_size() -> int:
        return custom_op.meta_size()

    def register_buffer(fa: int, ipc_tensors: List[int]) -> None:
        return custom_op.register_buffer(fa, ipc_tensors)

    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
        return custom_op.get_graph_buffer_ipc_meta(fa)

    def register_graph_buffers(
        fa: int, handles: List[List[int]], offsets: List[List[int]]
    ) -> None:
        custom_op.register_graph_buffers(fa, handles, offsets)

else:
    # ROCM custom allreduce
    if get_bool_env_var("CK_MOE"):
        import aiter.ops.custom_all_reduce as aiter_custom_ar

    def init_custom_ar(
        meta: torch.Tensor,
        rank_data: torch.Tensor,
        handles: List[str],
        offsets: List[int],
        rank: int,
        full_nvlink: bool,
    ) -> int:
        init_func = (
            aiter_custom_ar.init_custom_ar
            if get_bool_env_var("CK_MOE")
            else sgl_kernel.allreduce.init_custom_ar
        )
        return init_func(meta, rank_data, handles, offsets, rank, full_nvlink)

    def all_reduce_reg(fa: int, inp: torch.Tensor, out: torch.Tensor) -> None:
        arr_func = (
            aiter_custom_ar.all_reduce_reg
            if get_bool_env_var("CK_MOE")
            else sgl_kernel.allreduce.all_reduce_reg
        )
        arr_func(fa, inp, out)

    def all_reduce_unreg(
        fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor, out: torch.Tensor
    ) -> None:
        aru_func = (
            aiter_custom_ar.all_reduce_unreg
            if get_bool_env_var("CK_MOE")
            else sgl_kernel.allreduce.all_reduce_unreg
        )
        aru_func(fa, inp, reg_buffer, out)

    def dispose(fa: int) -> None:
        (
            aiter_custom_ar.dispose(fa)
            if get_bool_env_var("CK_MOE")
            else sgl_kernel.allreduce.dispose(fa)
        )

    def meta_size() -> int:
        ms_func = (
            aiter_custom_ar.meta_size
            if get_bool_env_var("CK_MOE")
            else sgl_kernel.allreduce.meta_size
        )
        return ms_func()

    def register_buffer(
        fa: int, t: torch.Tensor, handles: List[str], offsets: List[int]
    ) -> None:
        rb_func = (
            aiter_custom_ar.register_buffer
            if get_bool_env_var("CK_MOE")
            else sgl_kernel.allreduce.register_buffer
        )
        return rb_func(fa, t, handles, offsets)

    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[torch.Tensor, List[int]]:
        ggbim_func = (
            aiter_custom_ar.get_graph_buffer_ipc_meta
            if get_bool_env_var("CK_MOE")
            else sgl_kernel.allreduce.get_graph_buffer_ipc_meta
        )
        return ggbim_func(fa)

    def register_graph_buffers(
        fa: int, handles: List[str], offsets: List[List[int]]
    ) -> None:
        rgb_func = (
            aiter_custom_ar.register_graph_buffers
            if get_bool_env_var("CK_MOE")
            else sgl_kernel.allreduce.register_graph_buffers
        )
        rgb_func(fa, handles, offsets)

    def allocate_meta_buffer(size: int) -> torch.Tensor:
        amb_func = (
            aiter_custom_ar.allocate_meta_buffer
            if get_bool_env_var("CK_MOE")
            else sgl_kernel.allreduce.allocate_meta_buffer
        )
        return amb_func(size)

    def get_meta_buffer_ipc_handle(inp: torch.Tensor) -> torch.Tensor:
        gmbih_func = (
            aiter_custom_ar.get_meta_buffer_ipc_handle
            if get_bool_env_var("CK_MOE")
            else sgl_kernel.allreduce.get_meta_buffer_ipc_handle
        )
        return gmbih_func(inp)