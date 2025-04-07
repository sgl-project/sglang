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

    def init_custom_ar(
        meta: torch.Tensor,
        rank_data: torch.Tensor,
        handles: List[str],
        offsets: List[int],
        rank: int,
        full_nvlink: bool,
    ) -> int:
        return sgl_kernel.allreduce.init_custom_ar(
            meta, rank_data, handles, offsets, rank, full_nvlink
        )

    def all_reduce_reg(fa: int, inp: torch.Tensor, out: torch.Tensor) -> None:
        sgl_kernel.allreduce.all_reduce_reg(fa, inp, out)

    def all_reduce_unreg(
        fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor, out: torch.Tensor
    ) -> None:
        sgl_kernel.allreduce.all_reduce_unreg(fa, inp, reg_buffer, out)

    def dispose(fa: int) -> None:
        sgl_kernel.allreduce.dispose(fa)

    def meta_size() -> int:
        return sgl_kernel.allreduce.meta_size()

    def register_buffer(
        fa: int, t: torch.Tensor, handles: List[str], offsets: List[int]
    ) -> None:
        return sgl_kernel.allreduce.register_buffer(fa, t, handles, offsets)

    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[torch.Tensor, List[int]]:
        return sgl_kernel.allreduce.get_graph_buffer_ipc_meta(fa)

    def register_graph_buffers(
        fa: int, handles: List[str], offsets: List[List[int]]
    ) -> None:
        sgl_kernel.allreduce.register_graph_buffers(fa, handles, offsets)

    def allocate_meta_buffer(size: int) -> torch.Tensor:
        return sgl_kernel.allreduce.allocate_meta_buffer(size)

    def get_meta_buffer_ipc_handle(inp: torch.Tensor) -> torch.Tensor:
        return sgl_kernel.allreduce.get_meta_buffer_ipc_handle(inp)
