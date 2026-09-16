from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, is_hip_runtime, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_MIN_TOPK = 1
_MAX_TOPK = 4096
# Mirrors CoopMbWorkspace: two uint32 histograms, six uint32 RowState fields,
# one cache line for counters, and 4096 (uint32 index, fp32 score) ties.
_WORKSPACE_BYTES_PER_ROW = 4 * (4096 + 16) + 6 * 4 + 64 + 8 * 4096


@cache_once
def _jit_hip_cooperative_topk_module(topk: int) -> Module:
    if not is_hip_runtime():
        raise RuntimeError("HIP cooperative top-k requires a ROCm runtime")
    if not _topk_width_supported(topk):
        raise RuntimeError(
            f"HIP cooperative top-k requires {_MIN_TOPK} <= topk <= {_MAX_TOPK}, "
            f"got {topk}"
        )
    args = make_cpp_args(topk)
    return load_jit(
        "hip_cooperative_topk",
        *args,
        cuda_files=["dsa/hip_cooperative_topk.cuh"],
        cuda_wrappers=[
            ("hip_cooperative_topk", f"HipCooperativeTopKKernel<{args}>::run")
        ],
    )


def _topk_width_supported(topk: int) -> bool:
    return _MIN_TOPK <= topk <= _MAX_TOPK


@cache_once
def _is_wave64_device(device_index: int) -> bool:
    properties = torch.cuda.get_device_properties(device_index)
    return getattr(properties, "warp_size", 0) == 64


def hip_cooperative_topk_is_available(
    device: torch.device | int | None = None,
) -> bool:
    if not is_hip_runtime() or not torch.cuda.is_available():
        return False
    if isinstance(device, torch.device):
        device_index = device.index
    else:
        device_index = device
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _is_wave64_device(device_index)


def hip_cooperative_topk_supports(
    topk: int, device: torch.device | int | None = None
) -> bool:
    return _topk_width_supported(topk) and hip_cooperative_topk_is_available(device)


@cache_once
def _compute_unit_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _row_split(score: torch.Tensor) -> int:
    override = os.getenv("SGL_DSA_TOPK_ROW_SPLIT")
    if override:
        value = int(override)
        if value >= 0:
            return value

    batch = score.shape[0]
    if batch == 0 or score.stride(0) < 65536:
        return 0
    device_index = score.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    split = min(_compute_unit_count(device_index) // batch, 32)
    return split if split >= 4 else 0


def _run(
    score: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_starts: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
    raw_indices: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> None:
    topk = indices.shape[1]
    if not hip_cooperative_topk_supports(topk, score.device):
        raise RuntimeError(
            "HIP cooperative top-k requires a wave64 ROCm GPU and "
            f"{_MIN_TOPK} <= topk <= {_MAX_TOPK}; got device={score.device}, "
            f"topk={topk}"
        )
    split = _row_split(score)
    workspace = torch.empty(
        score.shape[0] * _WORKSPACE_BYTES_PER_ROW if split else 0,
        dtype=torch.uint8,
        device=score.device,
    )
    module = _jit_hip_cooperative_topk_module(topk)
    module.hip_cooperative_topk(
        score,
        lengths,
        indices,
        row_starts,
        page_table,
        cu_seqlens_q,
        offsets,
        raw_indices,
        page_size,
        workspace,
        split,
    )


def hip_cooperative_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    indices = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    _run(score, lengths, indices, row_starts=row_starts)
    return indices


def hip_cooperative_topk_paged(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
    raw_indices: Optional[torch.Tensor] = None,
) -> None:
    _run(
        score,
        lengths,
        indices,
        page_table=page_table,
        raw_indices=raw_indices,
        page_size=page_size,
    )


def hip_cooperative_topk_page_size_one(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    indices = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    _run(
        score,
        lengths,
        indices,
        row_starts=row_starts,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
    )
    return indices


def hip_cooperative_topk_ragged(
    score: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    indices = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    _run(
        score,
        lengths,
        indices,
        row_starts=row_starts,
        offsets=offsets,
    )
    return indices
