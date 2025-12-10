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

from functools import lru_cache

import numpy as np
import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()


if _is_cuda:
    from sgl_kernel import FusedSetKVBufferArg


def enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
    """Enable fused set_kv_buffer only on CUDA with bfloat16 KV cache."""
    return (
        _is_cuda
        and hasattr(forward_batch.token_to_kv_pool, "dtype")
        and forward_batch.token_to_kv_pool.dtype == torch.bfloat16
    )


def create_fused_set_kv_buffer_arg(
    value: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
):
    layer_id = layer.layer_id
    token_to_kv_pool = forward_batch.token_to_kv_pool

    k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
    v_buffer = token_to_kv_pool.get_value_buffer(layer_id)

    return FusedSetKVBufferArg(
        value=value,
        k_buffer=k_buffer.view(k_buffer.shape[0], -1),
        v_buffer=v_buffer.view(v_buffer.shape[0], -1),
        k_scale=layer.k_scale,
        v_scale=layer.v_scale,
        cache_loc=forward_batch.out_cache_loc,
    )


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
