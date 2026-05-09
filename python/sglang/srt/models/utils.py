# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm, fused_inplace_qknorm
from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Re-exported from the canonical home in ``model_loader/auto_loader.py`` so the
# older ``from sglang.srt.models.utils import AutoWeightsLoader, WeightsMapper``
# import path keeps working (``transformers.py`` and out-of-tree code).
from sglang.srt.model_loader.auto_loader import (  # noqa: F401
    AutoWeightsLoader,
    WeightsMapper,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_current_device_stream_fast, is_cuda, is_hip
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.layernorm import RMSNorm

_is_cuda = is_cuda()
_is_hip = is_hip()

WeightsMapping = Mapping[str, Optional[str]]
"""If a key maps to a value of `None`, the corresponding weight is ignored."""


def enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
    """Enable fused set_kv_buffer only on CUDA with bfloat16 KV cache."""
    return (
        _is_cuda
        and hasattr(forward_batch.token_to_kv_pool, "dtype")
        and forward_batch.token_to_kv_pool.dtype == torch.bfloat16
        and not isinstance(forward_batch.token_to_kv_pool, SWAKVPool)
        and not is_prefill_context_parallel_enabled()
    ) or (_is_hip and not is_prefill_context_parallel_enabled())


def create_fused_set_kv_buffer_arg(
    value: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
):
    from sglang.jit_kernel.rope import FusedSetKVBufferArg

    layer_id = layer.layer_id
    token_to_kv_pool = forward_batch.token_to_kv_pool

    k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
    v_buffer = token_to_kv_pool.get_value_buffer(layer_id)

    if not _is_hip:
        assert layer.k_scale is None and layer.v_scale is None, "scale not supported"
        return FusedSetKVBufferArg(
            value=value,
            k_buffer=k_buffer.view(k_buffer.shape[0], -1),
            v_buffer=v_buffer.view(v_buffer.shape[0], -1),
            cache_loc=forward_batch.out_cache_loc,
        )
    else:
        page_size = token_to_kv_pool.page_size
        slot_mapping_swa = (
            token_to_kv_pool.full_to_swa_index_mapping.long()
            if layer.sliding_window_size > 0
            else None
        )
        return {
            "v": value.view(-1, layer.tp_v_head_num, layer.v_head_dim),
            "k_scale": layer.k_scale,
            "v_scale": layer.v_scale,
            "key_cache": k_buffer.view(
                -1, page_size, layer.tp_k_head_num, layer.qk_head_dim
            ),
            "value_cache": v_buffer.view(
                -1, page_size, layer.tp_v_head_num, layer.v_head_dim
            ),
            "slot_mapping": forward_batch.out_cache_loc,
            "swa_slot_mapping": slot_mapping_swa,
        }


def permute_inv(perm: torch.Tensor) -> torch.Tensor:
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv_perm


def compute_cu_seqlens_from_grid_numpy(grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Compute cu_seqlens from grid_thw using NumPy.

    grid_thw: [T, 3] int tensor on CPU.
              columns: [repeat_count, H, W]
    Returns:
        cu_seqlens: 1D int32 tensor on CPU, shape [N + 1]
    """
    assert (
        grid_thw.device.type == "cpu"
    ), "compute_cu_seqlens_from_grid_numpy expects a CPU tensor"
    arr = grid_thw.numpy()

    cu_seqlens = np.repeat(arr[:, 1] * arr[:, 2], arr[:, 0]).cumsum(
        axis=0, dtype=np.int32
    )
    cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
    cu_seqlens = torch.from_numpy(cu_seqlens)
    return cu_seqlens


class RotaryPosMixin:

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        if isinstance(h, torch.Tensor):
            h = int(h.item())
        if isinstance(w, torch.Tensor):
            w = int(w.item())
        if isinstance(spatial_merge_size, torch.Tensor):
            spatial_merge_size = int(spatial_merge_size.item())
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        hpos_ids = hpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))


def _reshape_for_qk_norm(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Reshape a (..., H*D) tensor into (..., H, D) ahead of QK RMSNorm.

    On CUDA with the inductor piecewise-cuda-graph compiler, return a
    stride-preserving view so inductor can fuse this reshape with the
    subsequent RMSNorm (and any upstream/downstream FP8 quant) into a
    single triton kernel -- the original motivation of #21734.

    Everywhere else (ROCm, or CUDA with the eager PCG fallback), use the
    flat 2D reshape that forces a copy when the input is a non-contiguous
    QKV-split stride-trick view. ROCm's RMSNorm kernels assume contiguous
    inputs and fault on strided tensors (root cause of the #21734 revert
    in #23159).
    """
    if (
        _is_cuda
        and get_global_server_args().piecewise_cuda_graph_compiler == "inductor"
    ):
        return x.view(*x.shape[:-1], -1, head_dim)
    return x.reshape(-1, head_dim)


def apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    head_dim: int,
    alt_stream: Optional[torch.cuda.Stream] = None,
    allow_inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply QK normalization for query and key tensors.
    If eligible, we will use JIT fused inplace QK normalization for better performance.

    Args:
        q: Query tensor of shape [batch_size, ...]
        k: Key tensor of shape [batch_size, ...]
        q_norm: RMSNorm layer for query normalization
        k_norm: RMSNorm layer for key normalization
        head_dim: Dimension of each attention head
        alt_stream: Optional alternative CUDA stream for overlapping computation
        allow_inplace: Whether to allow inplace normalization. (True for better performance)

    Returns:
        Tuple of normalized query and key tensors
    """

    batch_size = q.size(0)
    q_eps = q_norm.variance_epsilon
    k_eps = k_norm.variance_epsilon
    if (
        _is_cuda  # TODO(dark): have not tested on ROCm or other backends
        and allow_inplace  # TODO(dark): this can be relaxed if needed
        and (q_eps == k_eps)  # TODO(dark): this can also be relaxed
        and not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
        and get_global_server_args().piecewise_cuda_graph_compiler
        != "inductor"  # let inductor fuse QK norm
        and can_use_fused_inplace_qknorm(head_dim, q.dtype)
    ):
        fused_inplace_qknorm(
            q=q.view(batch_size, -1, head_dim),
            k=k.view(batch_size, -1, head_dim),
            q_weight=q_norm.weight,
            k_weight=k_norm.weight,
            head_dim=head_dim,
            eps=q_eps,
        )
        return q, k

    if alt_stream is not None and get_is_capture_mode():
        current_stream = get_current_device_stream_fast()
        alt_stream.wait_stream(current_stream)
        q_by_head = _reshape_for_qk_norm(q, head_dim)
        q_by_head = q_norm(q_by_head)
        with torch.cuda.stream(alt_stream):
            k_by_head = _reshape_for_qk_norm(k, head_dim)
            k_by_head = k_norm(k_by_head)
        current_stream.wait_stream(alt_stream)
    else:
        q_by_head = _reshape_for_qk_norm(q, head_dim)
        q_by_head = q_norm(q_by_head)
        k_by_head = _reshape_for_qk_norm(k, head_dim)
        k_by_head = k_norm(k_by_head)
    q = q_by_head.view(q.shape)
    k = k_by_head.view(k.shape)
    return q, k


# ---------------------------------------------------------------------------
# Fused QK GemmaRMSNorm Triton kernel
# grid = q_rows (the larger dimension in GQA).  Every block computes Q norm
# for its row; the first k_rows blocks also compute K norm.  No torch.cat,
# no tl.where for weight selection, no output slice.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_qk_gemma_rmsnorm_kernel(
    Q_ptr,
    K_ptr,
    Q_out_ptr,
    K_out_ptr,
    QW_ptr,
    KW_ptr,
    q_stride,
    k_stride,
    k_rows,
    HEAD_DIM: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    EPS: tl.constexpr,
    FP16: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_HD)
    mask = cols < HEAD_DIM
    out_dtype = tl.float16 if FP16 else tl.bfloat16

    # Q norm (every block) — use q_stride to handle non-contiguous input
    q_off = pid * q_stride + cols
    q = tl.load(Q_ptr + q_off, mask=mask, other=0.0).to(tl.float32)
    w_q = tl.load(QW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    q_var = tl.sum(q * q, axis=0) / HEAD_DIM
    q_normed = (q * tl.rsqrt(q_var + EPS) * (w_q + 1.0)).to(out_dtype)
    # output is always contiguous
    q_out_off = pid * HEAD_DIM + cols
    tl.store(Q_out_ptr + q_out_off, q_normed, mask=mask)

    # K norm (first k_rows blocks only) — use k_stride for input
    if pid < k_rows:
        k_off = pid * k_stride + cols
        k = tl.load(K_ptr + k_off, mask=mask, other=0.0).to(tl.float32)
        w_k = tl.load(KW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        k_var = tl.sum(k * k, axis=0) / HEAD_DIM
        k_normed = (k * tl.rsqrt(k_var + EPS) * (w_k + 1.0)).to(out_dtype)
        k_out_off = pid * HEAD_DIM + cols
        tl.store(K_out_ptr + k_out_off, k_normed, mask=mask)


def fused_qk_gemma_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused QK GemmaRMSNorm — single Triton kernel for both q_norm and k_norm.

    grid = q_rows; every block processes its Q row, and the first k_rows
    blocks also process K.  No torch.cat, no slice, no tl.where.
    Passes input strides to the kernel so non-contiguous tensors (e.g. from
    qkv.split()) are read correctly without an extra .contiguous() copy.
    """
    q_flat = q.reshape(-1, head_dim)
    k_flat = k.reshape(-1, head_dim)

    q_rows = q_flat.shape[0]
    k_rows = k_flat.shape[0]

    q_out = torch.empty(q_rows, head_dim, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k_rows, head_dim, dtype=k.dtype, device=k.device)

    BLOCK_HD = triton.next_power_of_2(head_dim)

    _fused_qk_gemma_rmsnorm_kernel[(q_rows,)](
        q_flat,
        k_flat,
        q_out,
        k_out,
        q_weight,
        k_weight,
        q_flat.stride(0),
        k_flat.stride(0),
        k_rows,
        HEAD_DIM=head_dim,
        BLOCK_HD=BLOCK_HD,
        EPS=eps,
        FP16=(q.dtype == torch.float16),
    )

    return q_out, k_out


# Register the inplace op
fused_inplace_qknorm = register_custom_op(fused_inplace_qknorm, mutates_args=["q", "k"])
