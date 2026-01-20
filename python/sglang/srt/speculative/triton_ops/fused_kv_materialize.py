# Copyright 2023-2024 SGLang Team
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
"""Fused Triton kernel for DFlash KV materialization.

Combines: KV projection (cuBLAS) + RMSNorm + RoPE + KV cache write (Triton).
"""

from typing import List

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_norm_rope_write_kernel(
    kv_ptr,  # [total_ctx, kv_size * 2]
    k_norm_weight_ptr,  # [head_dim]
    cos_sin_cache_ptr,  # [max_pos, rotary_dim]
    positions_ptr,  # [total_ctx]
    cache_loc_ptr,  # [total_ctx]
    k_cache_ptr,  # [total_slots, num_kv_heads, head_dim]
    v_cache_ptr,  # [total_slots, num_kv_heads, head_dim]
    kv_stride_ctx,
    cos_sin_stride_pos,
    k_cache_stride_slot,
    k_cache_stride_head,
    v_cache_stride_slot,
    v_cache_stride_head,
    total_ctx,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    """Fused RMSNorm(K) + RoPE(K) + cache write. Grid: (total_ctx, num_kv_heads)."""
    ctx_id = tl.program_id(0)
    head_id = tl.program_id(1)
    if ctx_id >= total_ctx:
        return

    # Load metadata
    position = tl.load(positions_ptr + ctx_id)
    cache_loc = tl.load(cache_loc_ptr + ctx_id)

    # Compute base pointers
    kv_base = kv_ptr + ctx_id * kv_stride_ctx
    k_base = kv_base + head_id * head_dim
    v_base = kv_base + kv_size + head_id * head_dim
    k_write = (
        k_cache_ptr + cache_loc * k_cache_stride_slot + head_id * k_cache_stride_head
    )
    v_write = (
        v_cache_ptr + cache_loc * v_cache_stride_slot + head_id * v_cache_stride_head
    )

    # Load K and V
    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim
    mask_half = offs < half_rotary_dim

    k_raw = tl.load(k_base + offs, mask=mask_hd, other=0.0).to(tl.float32)
    v_raw = tl.load(v_base + offs, mask=mask_hd, other=0.0)

    # RMSNorm on K
    inv_rms = tl.rsqrt(tl.sum(k_raw * k_raw) / head_dim + eps)
    norm_w = tl.load(k_norm_weight_ptr + offs, mask=mask_hd, other=1.0).to(tl.float32)
    k_normed = k_raw * inv_rms * norm_w

    # RoPE (neox style): k_first, k_second -> rotated
    cos_sin_base = cos_sin_cache_ptr + position * cos_sin_stride_pos
    cos_v = tl.load(cos_sin_base + offs, mask=mask_half, other=1.0).to(tl.float32)
    sin_v = tl.load(
        cos_sin_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)

    # Extract first/second halves of K for rotation
    k_first = tl.where(mask_half, k_normed, 0.0)
    k_second_raw = tl.load(
        k_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)
    norm_w_second = tl.load(
        k_norm_weight_ptr + half_rotary_dim + offs, mask=mask_half, other=1.0
    ).to(tl.float32)
    k_second = k_second_raw * inv_rms * norm_w_second

    # Apply rotation
    k_rot_first = k_first * cos_v - k_second * sin_v
    k_rot_second = k_second * cos_v + k_first * sin_v

    # Store V (no transform)
    tl.store(v_write + offs, v_raw, mask=mask_hd)

    # Store K: rotated halves + pass-through
    tl.store(k_write + offs, k_rot_first.to(v_raw.dtype), mask=mask_half)
    tl.store(
        k_write + half_rotary_dim + offs, k_rot_second.to(v_raw.dtype), mask=mask_half
    )
    mask_pass = (offs >= rotary_dim) & (offs < head_dim)
    tl.store(k_write + offs, k_normed.to(v_raw.dtype), mask=mask_pass)


def _fused_norm_rope_write(
    kv: torch.Tensor,  # [total_ctx, kv_size*2]
    k_norm_weight: torch.Tensor,  # [head_dim]
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim]
    positions: torch.Tensor,  # [total_ctx]
    cache_locs: torch.Tensor,  # [total_ctx]
    k_cache: torch.Tensor,  # [total_slots, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [total_slots, num_kv_heads, head_dim]
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float = 1e-6,
) -> None:
    """Fused RMSNorm + RoPE + cache write for a single layer."""
    total_ctx = kv.shape[0]
    if total_ctx == 0:
        return

    kv_size = num_kv_heads * head_dim
    half_rotary_dim = rotary_dim // 2
    BLOCK_HD = triton.next_power_of_2(head_dim)

    # Ensure int64 for indexing
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if cache_locs.dtype != torch.int64:
        cache_locs = cache_locs.to(torch.int64)

    _fused_norm_rope_write_kernel[(total_ctx, num_kv_heads)](
        kv,
        k_norm_weight,
        cos_sin_cache,
        positions,
        cache_locs,
        k_cache,
        v_cache,
        kv.stride(0),
        cos_sin_cache.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        total_ctx,
        num_kv_heads,
        head_dim,
        kv_size,
        rotary_dim,
        half_rotary_dim,
        eps,
        BLOCK_HD,
    )


class FusedKVMaterializeHelper:
    """Fused KV materialization helper using batched projection.

    Uses torch.einsum for batched KV projection across all layers,
    then a Triton kernel for fused RMSNorm + RoPE + cache write per layer.
    """

    def __init__(
        self,
        layers: List,
        rotary_emb,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.n_layers = len(layers)
        self.device = device

        # Pre-extract and stack weights for batched projection
        kv_weights = []
        self.k_norm_weights = []
        self.eps_values = []

        for layer in layers:
            attn = layer.self_attn
            # Extract KV portion of QKV weight
            qkv_w = attn.qkv_proj.weight
            kv_weight = qkv_w[attn.q_size : attn.q_size + 2 * attn.kv_size]
            kv_weights.append(kv_weight)
            self.k_norm_weights.append(attn.k_norm.weight)
            self.eps_values.append(attn.k_norm.variance_epsilon)

        # Stack for batched einsum: [n_layers, kv_size*2, hidden_size]
        self.batched_kv_weight = torch.stack(kv_weights)

        self.rotary_dim = getattr(rotary_emb, "rotary_dim", head_dim)
        self.is_neox_style = getattr(rotary_emb, "is_neox_style", True)

        if not self.is_neox_style:
            raise NotImplementedError("Only neox-style RoPE is supported.")

    def materialize(
        self,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
        k_cache_buffers: List[torch.Tensor],
        v_cache_buffers: List[torch.Tensor],
    ) -> None:
        """Materialize KV cache for all layers using batched projection."""
        total_ctx = ctx_hidden.shape[0]
        if total_ctx == 0:
            return

        cos_sin_cache = self.rotary_emb.cos_sin_cache

        # Batched KV projection: [n_layers, total_ctx, kv_size*2]
        kv_all = torch.einsum("th,loh->lto", ctx_hidden, self.batched_kv_weight)

        # Per-layer fused norm/RoPE/write
        for layer_id in range(self.n_layers):
            _fused_norm_rope_write(
                kv_all[layer_id],
                self.k_norm_weights[layer_id],
                cos_sin_cache,
                positions,
                cache_locs,
                k_cache_buffers[layer_id],
                v_cache_buffers[layer_id],
                self.num_kv_heads,
                self.head_dim,
                self.rotary_dim,
                self.eps_values[layer_id],
            )
