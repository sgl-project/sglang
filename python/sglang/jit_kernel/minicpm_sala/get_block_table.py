from __future__ import annotations

import importlib.util
import pathlib
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Layout constants fixed by the MiniCPM-SALA model configuration. Mirror the
# `constexpr` values in csrc/minicpm_sala/get_block_table.cuh.
_HEAD_GROUP = 2
_SPARSE_BLOCK_SIZE = 64
# Supported compile-time topk values (matches the original VALUE_SPLITS_SWITCH).
_SUPPORTED_TOPK = (96, 128)


@cache_once
def _get_cccl_include_paths() -> list[str]:
    if is_hip_runtime():
        return []
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or spec.origin is None:
        return []
    cccl_root = pathlib.Path(spec.origin).resolve().parent / "data" / "cccl"
    candidates = (
        cccl_root / "libcudacxx" / "include",
        cccl_root / "cub",
        cccl_root / "thrust",
    )
    return [str(path) for path in candidates if path.exists()]


@cache_once
def _jit_get_block_table_module(topk: int) -> Module:
    """Compile and cache the JIT module for a given sparse topk value.

    One module is built per topk value, replacing the original runtime
    ``VALUE_SPLITS_SWITCH(topk, ...)`` dispatch with a compile-time template
    argument ``kSparseTopK``.
    """
    args = make_cpp_args(topk)
    return load_jit(
        f"get_block_table_topk{topk}",
        *args,
        cuda_files=["minicpm_sala/get_block_table.cuh"],
        cuda_wrappers=[
            ("get_block_table_v1", f"minicpm_sala::get_block_table_v1<{args}>"),
            ("get_block_table_v2", f"minicpm_sala::get_block_table_v2<{args}>"),
            ("get_block_table_v3", f"minicpm_sala::get_block_table_v3<{args}>"),
        ],
        extra_include_paths=_get_cccl_include_paths(),
    )


def _run(
    version: int,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    seqlen_q: torch.Tensor,
) -> torch.Tensor:
    if topk_idx.dim() != 3:
        raise RuntimeError(
            f"topk_idx must be 3D [head_group, token_num, topk], got shape {tuple(topk_idx.shape)}"
        )
    token_num = topk_idx.shape[1]
    topk = topk_idx.shape[2]
    if topk not in _SUPPORTED_TOPK:
        raise RuntimeError(
            f"Unsupported topk={topk}. Supported values: {_SUPPORTED_TOPK}"
        )

    out = torch.zeros(
        (token_num, _HEAD_GROUP, topk * _SPARSE_BLOCK_SIZE),
        dtype=torch.int32,
        device=topk_idx.device,
    )
    module = _jit_get_block_table_module(topk)
    getattr(module, f"get_block_table_v{version}")(
        out, topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q
    )
    return out


def get_block_table_v1(
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    seqlen_q: torch.Tensor,
) -> torch.Tensor:
    """Build the sparse block table (prefill, 1 thread per token)."""
    return _run(1, topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q)


def get_block_table_v2(
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    seqlen_q: torch.Tensor,
) -> torch.Tensor:
    """Build the sparse block table (1 thread per (token, head, topk) entry)."""
    return _run(2, topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q)


def get_block_table_v3(
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    seqlen_q: torch.Tensor,
) -> torch.Tensor:
    """Build the sparse block table (decode-optimized, 1 thread per output)."""
    return _run(3, topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q)
