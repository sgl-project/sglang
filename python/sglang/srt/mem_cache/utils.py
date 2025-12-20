# Copyright 2025 SGLang Team
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
"""Common utilities."""

from typing import Any, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs


@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    else:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )

    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )


@triton.jit
def set_mla_kv_scale_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim  # Make sure don't cross the boundary

    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    # Check each offs should read 'nope' or 'rope'
    is_nope = offs < nope_dim
    src_nope = tl.load(
        cache_k_nope_ptr + pid_loc * nope_stride + offs, mask=mask & is_nope, other=0.0
    )
    src_rope = tl.load(
        cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
        mask=mask & ~is_nope,
        other=0.0,
    )

    # Combine nope + rope
    src = src_nope + src_rope
    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_scale_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128  # Keep origin, works for smaller total_dim as well.
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_scale_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )


@triton.jit
def get_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    loc_src_ptr = kv_buffer_ptr + loc * buffer_stride

    nope_offs = tl.arange(0, nope_dim)
    nope_src_ptr = loc_src_ptr + nope_offs
    nope_src = tl.load(nope_src_ptr)

    tl.store(
        cache_k_nope_ptr + pid_loc * nope_stride + nope_offs,
        nope_src,
    )

    rope_offs = tl.arange(0, rope_dim)
    rope_src_ptr = loc_src_ptr + nope_dim + rope_offs
    rope_src = tl.load(rope_src_ptr)
    tl.store(
        cache_k_rope_ptr + pid_loc * rope_stride + rope_offs,
        rope_src,
    )


def get_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    # The source data type will be implicitly converted to the target data type.
    nope_dim = cache_k_nope.shape[-1]  # 512
    rope_dim = cache_k_rope.shape[-1]  # 64
    n_loc = loc.numel()
    grid = (n_loc,)

    get_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
    )


def maybe_init_custom_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Initialize custom memory pool based on environment variable.

    This function can be modified to support more features that require a custom memory pool.

    Args:
        device: The device to allocate memory on

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type)
    """
    enable_custom_mem_pool = (
        True if envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() is not None else False
    )

    if enable_custom_mem_pool:
        # Currently, only mooncake requires a custom mem pool for MNNVL/Barex PD disaggregation
        from sglang.srt.disaggregation.mooncake.utils import (
            init_mooncake_custom_mem_pool,
        )

        return init_mooncake_custom_mem_pool(device)
    else:
        return False, None, None


def set_mla_kv_buffer_fp8_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k: torch.Tensor,
):
    """
    Scatter write FP8-quantized cache_k to kv_buffer using Triton kernel.
    Replaces the expensive aten::index_put_ operation to reduce CPU overhead.

    This function performs: kv_buffer[loc] = cache_k

    Related issues: #15025, #15104

    :param kv_buffer: Target buffer, shape (size + page_size, 1, kv_cache_dim) or (size, kv_cache_dim)
    :param loc: Target token locations, shape (num_tokens,), dtype=int32/int64, must be on GPU
    :param cache_k: Source data, shape (num_tokens, 1, kv_cache_dim) or (num_tokens, kv_cache_dim)
                    Already quantized to FP8 and viewed as uint8.

    Why this avoids index_put_ overhead:
    - PyTorch's index_put_ has ~18us CPU overhead per call
    - This Triton kernel launches once with reduced overhead (~2-3us)
    - For 60 layers, saves ~40ms+ CPU time per decode step
    """
    num_tokens = loc.numel()

    # Validate loc: must be on GPU and int32/int64
    assert loc.is_cuda, f"loc must be on GPU, got device={loc.device}"
    assert loc.dtype in [torch.int32, torch.int64], f"loc must be int32 or int64, got dtype={loc.dtype}"

    # Validate device consistency
    assert kv_buffer.device == cache_k.device, \
        f"kv_buffer and cache_k must be on same device, got {kv_buffer.device} vs {cache_k.device}"

    # Reshape to 2D for Triton kernel
    # kv_buffer: (size + page_size, 1, kv_cache_dim) -> (size + page_size, kv_cache_dim)
    if kv_buffer.ndim == 3:
        assert kv_buffer.shape[1] == 1, f"Expected kv_buffer.shape[1] == 1, got {kv_buffer.shape}"
        kv_buffer_2d = kv_buffer.squeeze(1)
    else:
        kv_buffer_2d = kv_buffer

    # cache_k: (num_tokens, 1, kv_cache_dim) -> (num_tokens, kv_cache_dim)
    if cache_k.ndim == 3:
        assert cache_k.shape[1] == 1, f"Expected cache_k.shape[1] == 1, got {cache_k.shape}"
        cache_k_2d = cache_k.squeeze(1)
    else:
        cache_k_2d = cache_k

    kv_cache_dim = cache_k_2d.shape[1]

    # Get explicit strides for 2D tensors: (stride_dim0, stride_dim1)
    buffer_stride_0, buffer_stride_1 = kv_buffer_2d.stride()
    cache_stride_0, cache_stride_1 = cache_k_2d.stride()

    BLOCK = 128  # Process 128 dimensions per block
    grid = (num_tokens, triton.cdiv(kv_cache_dim, BLOCK))

    set_mla_kv_buffer_fp8_kernel[grid](
        kv_buffer_2d,
        cache_k_2d,
        loc,
        buffer_stride_0,
        buffer_stride_1,
        cache_stride_0,
        cache_stride_1,
        kv_cache_dim,
        BLOCK=BLOCK,
    )


@triton.jit
def set_mla_kv_buffer_fp8_kernel(
    kv_buffer_ptr,
    cache_k_ptr,
    loc_ptr,
    buffer_stride_0: tl.constexpr,
    buffer_stride_1: tl.constexpr,
    cache_stride_0: tl.constexpr,
    cache_stride_1: tl.constexpr,
    kv_cache_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Triton kernel for scatter write operation using explicit 2D strides.

    Grid: (num_tokens, cdiv(kv_cache_dim, BLOCK))
    - Each program handles one token and one block of dimensions

    Memory access pattern (explicit 2D indexing):
    - Read:  cache_k[token_idx, dim_offsets] = cache_k_ptr + token_idx * s0 + dim_offsets * s1
    - Write: kv_buffer[target_loc, dim_offsets] = kv_buffer_ptr + target_loc * s0 + dim_offsets * s1

    This uses explicit stride_0 and stride_1 instead of assuming stride(1)==1.
    """
    token_idx = tl.program_id(0)  # Which token (0 to num_tokens-1)
    dim_block = tl.program_id(1)   # Which dimension block

    # Load target location for this token
    target_loc = tl.load(loc_ptr + token_idx).to(tl.int64)

    # Calculate dimension offsets for this block
    dim_start = dim_block * BLOCK
    dim_offsets = dim_start + tl.arange(0, BLOCK)
    dim_mask = dim_offsets < kv_cache_dim  # Handle last block that may be partial

    # Load from source: cache_k[token_idx, dim_offsets]
    # Use explicit 2D stride: base + row_idx * stride_0 + col_idx * stride_1
    src_offsets = token_idx * cache_stride_0 + dim_offsets * cache_stride_1
    src_data = tl.load(cache_k_ptr + src_offsets, mask=dim_mask, other=0)

    # Store to destination: kv_buffer[target_loc, dim_offsets]
    # Use explicit 2D stride: base + row_idx * stride_0 + col_idx * stride_1
    dst_offsets = target_loc * buffer_stride_0 + dim_offsets * buffer_stride_1
    tl.store(kv_buffer_ptr + dst_offsets, src_data, mask=dim_mask)


def convert_to_bigram_key(tokens: List[int]) -> List[Tuple[int, int]]:
    # EAGLE uses bigram keys in the radix tree since draft sequence is the one-token-shifted version of target
    # [1, 2, 3, 4] -> [(1,2), (2,3), (3,4)]
    if len(tokens) and isinstance(tokens[0], tuple):
        return tokens
    if len(tokens) < 2:
        return []
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
