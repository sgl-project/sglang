"""QSA (Qwen4-Exp sparse attention) decode / target-verify attention over an
FP8 E4M3 KV pool.

``qsa_sparse_decode`` reads the K/V rows each query row selected straight from
the KV pool through ``req_to_token`` and runs a split-K online softmax on
tensor cores, so the selected rows are never gathered into a scratch buffer.
Inputs follow the QSA backend's paged path: ``indices`` holds logical token
positions (``-1`` = hole); a column is used when ``0 <= pos < seq_lens[row]``.
Outputs agree with the gather-then-attend path up to bf16 rounding of the
softmax weights; they are deterministic and independent of the batch
composition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HEAD_DIM = 256
_HEADS_PER_TILE = 16
_NUM_SPLITS = 8


@cache_once
def _jit_qsa_sparse_decode_module(stages: int, q_in_smem: bool) -> Module:
    args = make_cpp_args(stages, q_in_smem, is_arch_support_pdl())
    return load_jit(
        "qsa_sparse_decode",
        *args,
        cuda_files=["attention/qsa_sparse_decode.cuh"],
        cuda_wrappers=[("run", f"QsaSparseDecode<{args}>::run")],
    )


def _launch_config(rows: int) -> tuple:
    """(pipeline stages, q fragments in shared memory) per architecture."""
    if current_arch().is_sm_at_least(12):
        return 2, False
    return 3, rows > 32


def qsa_sparse_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    row_req_pool_indices: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Sparse attention of each query row over its selected KV positions.

    Args:
        q: ``[rows, q_heads, 256]`` bf16.
        k_cache, v_cache: ``[slots, kv_heads, 256]`` float8_e4m3fn KV pool
            buffers (unit scale), with ``q_heads / kv_heads <= 16``.
        req_to_token: ``[requests, context]`` int32 position -> slot table.
        row_req_pool_indices: ``[rows]`` int32/int64 request of each row.
        indices: ``[rows, topk]`` int32 logical positions, ``-1`` = hole.
        seq_lens: ``[rows]`` int32 visible length of each row.
        sm_scale: softmax scale.

    Returns:
        ``[rows, q_heads, 256]`` bf16.
    """
    rows, _, head_dim = q.shape
    kv_heads = k_cache.shape[1]
    if head_dim != _HEAD_DIM or k_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "qsa_sparse_decode needs head_dim 256 and a float8_e4m3fn KV cache, "
            f"got head_dim {head_dim} and {k_cache.dtype}"
        )
    partials = rows * kv_heads * _NUM_SPLITS * _HEADS_PER_TILE
    part_acc = torch.empty(partials * head_dim, dtype=torch.float32, device=q.device)
    part_m = torch.empty(partials, dtype=torch.float32, device=q.device)
    part_l = torch.empty(partials, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    stages, q_in_smem = _launch_config(rows)
    _jit_qsa_sparse_decode_module(stages, q_in_smem).run(
        q,
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        req_to_token,
        row_req_pool_indices,
        indices,
        seq_lens,
        part_acc,
        part_m,
        part_l,
        out,
        float(sm_scale),
    )
    return out
