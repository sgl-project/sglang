# Copyright 2026 SGLang Team
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
"""Storage helpers for TRT-LLM GenMHA's native NVFP4 KV layout.

Packed K/V data stays in SGLang's slot-major NHD pool.  TRT-LLM permits the
outer data strides to be non-contiguous, so the backend can expose an HND view
without copying it.  Scale factors are stricter: the page/token and block-scale
dimensions must be contiguous, and V scales use a four-token interleave.  The
kernels below create and maintain that native scale view while preserving an
optional linear scale view used by FlashInfer's FP8-prefill compatibility path.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


def nvfp4_v_scale_swizzle_indices(
    token_offsets: torch.Tensor, scale_indices: torch.Tensor, scale_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return TRT-LLM's four-token-interleaved V-scale coordinates.

    This small torch reference is intentionally device agnostic so the layout
    contract can be unit-tested without a GPU. ``scale_dim`` is head_dim / 16.
    """
    if scale_dim % 4 != 0:
        raise ValueError(f"NVFP4 scale_dim must be divisible by 4, got {scale_dim}.")
    scale_group = scale_dim // 4
    swizzled_token = (token_offsets // 4) * 4 + scale_indices // scale_group
    swizzled_scale = (scale_indices % scale_group) * 4 + token_offsets % 4
    return swizzled_token, swizzled_scale


@triton.jit
def _store_nvfp4_kv_kernel(
    k_src,
    v_src,
    k_scale_src,
    v_scale_src,
    loc,
    k_dst,
    v_dst,
    k_scale_linear_dst,
    v_scale_linear_dst,
    k_scale_native_dst,
    v_scale_native_dst,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    packed_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    BLOCK_SCALE: tl.constexpr,
    STORE_LINEAR: tl.constexpr,
    STORE_NATIVE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    slot = tl.load(loc + token_idx).to(tl.int64)

    packed_offsets = tl.arange(0, BLOCK_PACKED)
    packed_mask = packed_offsets < packed_dim
    src_packed_base = (token_idx * num_heads + head_idx) * packed_dim
    dst_packed_base = (slot * num_heads + head_idx) * packed_dim
    k_packed = tl.load(k_src + src_packed_base + packed_offsets, mask=packed_mask)
    v_packed = tl.load(v_src + src_packed_base + packed_offsets, mask=packed_mask)
    tl.store(k_dst + dst_packed_base + packed_offsets, k_packed, mask=packed_mask)
    tl.store(v_dst + dst_packed_base + packed_offsets, v_packed, mask=packed_mask)

    scale_offsets = tl.arange(0, BLOCK_SCALE)
    scale_mask = scale_offsets < scale_dim
    src_scale_base = (token_idx * num_heads + head_idx) * scale_dim
    k_scale = tl.load(k_scale_src + src_scale_base + scale_offsets, mask=scale_mask)
    v_scale = tl.load(v_scale_src + src_scale_base + scale_offsets, mask=scale_mask)

    if STORE_LINEAR:
        dst_scale_base = (slot * num_heads + head_idx) * scale_dim
        tl.store(
            k_scale_linear_dst + dst_scale_base + scale_offsets,
            k_scale,
            mask=scale_mask,
        )
        tl.store(
            v_scale_linear_dst + dst_scale_base + scale_offsets,
            v_scale,
            mask=scale_mask,
        )

    if STORE_NATIVE:
        page = slot // page_size
        token_offset = slot % page_size
        native_page_head_base = (page * num_heads + head_idx) * page_size * scale_dim

        k_native_offset = (
            native_page_head_base + token_offset * scale_dim + scale_offsets
        )
        tl.store(k_scale_native_dst + k_native_offset, k_scale, mask=scale_mask)

        scale_group = scale_dim // 4
        swizzled_token = (token_offset // 4) * 4 + scale_offsets // scale_group
        swizzled_scale = (scale_offsets % scale_group) * 4 + token_offset % 4
        v_native_offset = (
            native_page_head_base + swizzled_token * scale_dim + swizzled_scale
        )
        tl.store(v_scale_native_dst + v_native_offset, v_scale, mask=scale_mask)


def store_nvfp4_kv_cache(
    k_src: torch.Tensor,
    v_src: torch.Tensor,
    k_scale_src: torch.Tensor,
    v_scale_src: torch.Tensor,
    loc: torch.Tensor,
    k_dst: torch.Tensor,
    v_dst: torch.Tensor,
    k_scale_linear_dst: Optional[torch.Tensor],
    v_scale_linear_dst: Optional[torch.Tensor],
    k_scale_native_dst: Optional[torch.Tensor],
    v_scale_native_dst: Optional[torch.Tensor],
    page_size: int,
) -> None:
    """Scatter one layer of quantized K/V and both selected scale layouts."""
    if (k_scale_linear_dst is None) != (v_scale_linear_dst is None):
        raise ValueError("Linear NVFP4 K/V scale buffers must be provided together.")
    if (k_scale_native_dst is None) != (v_scale_native_dst is None):
        raise ValueError("Native NVFP4 K/V scale buffers must be provided together.")
    store_linear = k_scale_linear_dst is not None
    store_native = k_scale_native_dst is not None
    if not (store_linear or store_native):
        raise ValueError("At least one NVFP4 scale layout must be selected.")

    num_tokens, num_heads, packed_dim = k_src.shape
    scale_dim = k_scale_src.shape[-1]
    if store_native:
        if page_size % 4 != 0:
            raise ValueError(
                f"Native NVFP4 requires page_size divisible by 4, got {page_size}."
            )
        if scale_dim % 4 != 0:
            raise ValueError(
                "Native NVFP4 requires head_dim divisible by 64; "
                f"got scale_dim={scale_dim} (head_dim={scale_dim * 16})."
            )
    expected_data_shape = (num_tokens, num_heads, packed_dim)
    expected_scale_shape = (num_tokens, num_heads, scale_dim)
    if v_src.shape != expected_data_shape:
        raise ValueError(f"K/V packed shapes differ: {k_src.shape} vs {v_src.shape}.")
    if (
        k_scale_src.shape != expected_scale_shape
        or v_scale_src.shape != expected_scale_shape
    ):
        raise ValueError(
            "Unexpected NVFP4 scale shapes: "
            f"K={k_scale_src.shape}, V={v_scale_src.shape}, expected={expected_scale_shape}."
        )
    if loc.numel() != num_tokens:
        raise ValueError(f"loc has {loc.numel()} entries for {num_tokens} KV rows.")

    # Compile-time-false branches do not dereference these placeholder pointers.
    linear_k = k_scale_linear_dst if store_linear else k_dst
    linear_v = v_scale_linear_dst if store_linear else v_dst
    native_k = k_scale_native_dst if store_native else k_dst
    native_v = v_scale_native_dst if store_native else v_dst
    _store_nvfp4_kv_kernel[(num_tokens, num_heads)](
        k_src,
        v_src,
        k_scale_src,
        v_scale_src,
        loc,
        k_dst,
        v_dst,
        linear_k,
        linear_v,
        native_k,
        native_v,
        num_tokens=num_tokens,
        num_heads=num_heads,
        packed_dim=packed_dim,
        scale_dim=scale_dim,
        page_size=page_size,
        BLOCK_PACKED=triton.next_power_of_2(packed_dim),
        BLOCK_SCALE=triton.next_power_of_2(scale_dim),
        STORE_LINEAR=store_linear,
        STORE_NATIVE=store_native,
        num_warps=4,
    )


@triton.jit
def _move_nvfp4_native_scales_kernel(
    k_scale,
    v_scale,
    tgt_loc,
    src_loc,
    num_heads: tl.constexpr,
    scale_dim: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_SCALE: tl.constexpr,
):
    move_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    target = tl.load(tgt_loc + move_idx).to(tl.int64)
    source = tl.load(src_loc + move_idx).to(tl.int64)
    target_page, target_token = target // page_size, target % page_size
    source_page, source_token = source // page_size, source % page_size

    scale_offsets = tl.arange(0, BLOCK_SCALE)
    mask = scale_offsets < scale_dim
    source_base = (source_page * num_heads + head_idx) * page_size * scale_dim
    target_base = (target_page * num_heads + head_idx) * page_size * scale_dim

    k_values = tl.load(
        k_scale + source_base + source_token * scale_dim + scale_offsets, mask=mask
    )
    tl.store(
        k_scale + target_base + target_token * scale_dim + scale_offsets,
        k_values,
        mask=mask,
    )

    scale_group = scale_dim // 4
    source_swizzled_token = (source_token // 4) * 4 + scale_offsets // scale_group
    source_swizzled_scale = (scale_offsets % scale_group) * 4 + source_token % 4
    target_swizzled_token = (target_token // 4) * 4 + scale_offsets // scale_group
    target_swizzled_scale = (scale_offsets % scale_group) * 4 + target_token % 4
    v_values = tl.load(
        v_scale
        + source_base
        + source_swizzled_token * scale_dim
        + source_swizzled_scale,
        mask=mask,
    )
    tl.store(
        v_scale
        + target_base
        + target_swizzled_token * scale_dim
        + target_swizzled_scale,
        v_values,
        mask=mask,
    )


def move_nvfp4_native_scales(
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
) -> None:
    """Move logical token scale rows between native HND/swizzled slots."""
    if tgt_loc.numel() == 0:
        return
    if k_scale.shape != v_scale.shape or k_scale.ndim != 4:
        raise ValueError(
            f"Expected matching [pages, heads, page, scales] tensors, got "
            f"{k_scale.shape} and {v_scale.shape}."
        )
    _, num_heads, page_size, scale_dim = k_scale.shape
    _move_nvfp4_native_scales_kernel[(tgt_loc.numel(), num_heads)](
        k_scale,
        v_scale,
        tgt_loc,
        src_loc,
        num_heads=num_heads,
        scale_dim=scale_dim,
        page_size=page_size,
        BLOCK_SCALE=triton.next_power_of_2(scale_dim),
        num_warps=1,
    )
