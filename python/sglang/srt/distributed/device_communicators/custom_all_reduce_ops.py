# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/_custom_ops.py
import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda, is_gfx95_supported, is_hip, is_musa

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_musa = is_musa()

IS_CUSTOM_AR_AVAILABLE = _is_cuda or _is_hip or _is_musa
IS_QUICK_AR_AVAILABLE = _is_hip

try:
    import sgl_kernel.allreduce as _custom_ar
except ImportError as e:
    if _is_cuda or _is_hip:
        logger.warning("Failed to import from custom_ar with %r", e)
    IS_CUSTOM_AR_AVAILABLE = False
    IS_QUICK_AR_AVAILABLE = False

# region IS_CUSTOM_AR_AVAILABLE

if not IS_CUSTOM_AR_AVAILABLE:
    pass

elif _is_cuda or _is_musa:
    # CUDA custom allreduce

    def init_custom_ar(
        ipc_tensors: List[torch.Tensor],
        rank_data: torch.Tensor,
        rank: int,
        full_nvlink: bool,
    ) -> int:
        return _custom_ar.init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)

    def all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
    ) -> None:
        _custom_ar.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)

    def dispose(fa: int) -> None:
        _custom_ar.dispose(fa)

    def meta_size() -> int:
        return _custom_ar.meta_size()

    def register_buffer(fa: int, ipc_tensors: List[int]) -> None:
        return _custom_ar.register_buffer(fa, ipc_tensors)

    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
        return _custom_ar.get_graph_buffer_ipc_meta(fa)

    def register_graph_buffers(
        fa: int, handles: List[List[int]], offsets: List[List[int]]
    ) -> None:
        _custom_ar.register_graph_buffers(fa, handles, offsets)

elif _is_hip:
    # ROCM custom allreduce

    def init_custom_ar(
        meta: torch.Tensor,
        rank_data: torch.Tensor,
        handles: List[str],
        offsets: List[int],
        rank: int,
        full_nvlink: bool,
    ) -> int:
        return _custom_ar.init_custom_ar(
            meta, rank_data, handles, offsets, rank, full_nvlink
        )

    def all_reduce_reg(fa: int, inp: torch.Tensor, out: torch.Tensor) -> None:
        _custom_ar.all_reduce_reg(fa, inp, out)

    def all_reduce_unreg(
        fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor, out: torch.Tensor
    ) -> None:
        _custom_ar.all_reduce_unreg(fa, inp, reg_buffer, out)

    def deterministic_all_reduce_reg(
        fa: int, inp: torch.Tensor, out: torch.Tensor
    ) -> None:
        _custom_ar.deterministic_all_reduce_reg(fa, inp, out)

    def deterministic_all_reduce_unreg(
        fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor, out: torch.Tensor
    ) -> None:
        _custom_ar.deterministic_all_reduce_unreg(fa, inp, reg_buffer, out)

    def dispose(fa: int) -> None:
        _custom_ar.dispose(fa)

    def meta_size() -> int:
        return _custom_ar.meta_size()

    def register_buffer(
        fa: int, t: torch.Tensor, handles: List[str], offsets: List[int]
    ) -> None:
        return _custom_ar.register_buffer(fa, t, handles, offsets)

    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[torch.Tensor, List[int]]:
        return _custom_ar.get_graph_buffer_ipc_meta(fa)

    def register_graph_buffers(
        fa: int, handles: List[str], offsets: List[List[int]]
    ) -> None:
        _custom_ar.register_graph_buffers(fa, handles, offsets)

    def allocate_meta_buffer(size: int) -> torch.Tensor:
        return _custom_ar.allocate_meta_buffer(size)

    def get_meta_buffer_ipc_handle(inp: torch.Tensor) -> torch.Tensor:
        return _custom_ar.get_meta_buffer_ipc_handle(inp)


# endregion

# region IS_QUICK_AR_AVAILABLE


def _uses_jit_quick_all_reduce() -> bool:
    if not (_is_hip and envs.SGLANG_ROCM_QUICK_ALL_REDUCE_JIT.get()):
        return False
    if not is_gfx95_supported():
        logger.warning(
            "SGLANG_ROCM_QUICK_ALL_REDUCE_JIT is validated on gfx950 only; "
            "using sgl-kernel's quick all-reduce"
        )
        return False
    return True


if not IS_QUICK_AR_AVAILABLE:
    pass

elif _uses_jit_quick_all_reduce():
    from sglang.kernels.ops.communication.quick_all_reduce_hip import (  # noqa: F401
        init_custom_qr,
        qr_all_reduce,
        qr_destroy,
        qr_get_handle,
        qr_max_size,
        qr_open_handles,
    )

elif _is_hip:
    # ROCM custom quick allreduce

    def init_custom_qr(
        rank: int, world_size: int, qr_max_size: Optional[int] = None
    ) -> int:
        return _custom_ar.init_custom_qr(world_size, rank, qr_max_size)

    def qr_get_handle(fa: int) -> torch.Tensor:
        return _custom_ar.qr_get_handle(fa)

    def qr_open_handles(fa: int, handles: list[torch.Tensor]) -> None:
        _custom_ar.qr_open_handles(fa, handles)

    def qr_all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        quant_level: int,
        cast_bf2half: bool,
    ) -> None:
        _custom_ar.qr_all_reduce(fa, inp, out, quant_level, cast_bf2half)

    def qr_destroy(fa: int) -> None:
        _custom_ar.qr_destroy(fa)

    def qr_max_size() -> int:
        return _custom_ar.qr_max_size()


# endregion
