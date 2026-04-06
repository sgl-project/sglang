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

    # Three-way branch to handle boundary correctly while preserving fast path
    if base + BLOCK <= nope_dim:
        # Fast path: entire block is in nope region
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    elif base >= nope_dim:
        # Fast path: entire block is in rope region
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )
    else:
        # Boundary case: block spans nope/rope boundary (e.g., FP8 with nope_dim=528)
        # Handle each offset individually to avoid negative indexing
        is_nope = offs < nope_dim
        is_rope = (offs >= nope_dim) & (offs < (nope_dim + rope_dim))

        src_nope = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask & is_nope,
            other=0,
        )
        src_rope = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
            mask=mask & is_rope,
            other=0,
        )

        src = tl.where(is_nope, src_nope, src_rope)

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
def set_mla_kv_buffer_fp8_quant_kernel(
    kv_buffer_fp8_ptr,
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
    """Fuse BF16/FP16->FP8 cast with paged KV write."""
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    dst_ptr = kv_buffer_fp8_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
            other=0.0,
        )
    elif base >= nope_dim:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
            other=0.0,
        )
    else:
        is_nope = offs < nope_dim
        src_nope = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask & is_nope,
            other=0.0,
        )
        src_rope = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
            mask=mask & ~is_nope,
            other=0.0,
        )
        src = tl.where(is_nope, src_nope, src_rope)

    # Destination pointer is FP8-typed view; tl.store performs downcast.
    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_triton_fp8_quant(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    fp8_dtype: torch.dtype,
):
    """Fuse BF16/FP16 MLA K quantization with paged KV write."""
    kv_buffer_fp8 = kv_buffer.view(fp8_dtype)

    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_buffer_fp8_quant_kernel[grid](
        kv_buffer_fp8,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer_fp8.stride(0),
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


FP4_BLOCK_SIZE = 32  # MXFP4 standard block size
FP4_NOPE_DIM = 512
FP4_ROPE_DIM = 64
FP4_PACKED_NOPE_BYTES = FP4_NOPE_DIM // 2  # 256
FP4_NUM_SCALE_BLOCKS = FP4_NOPE_DIM // FP4_BLOCK_SIZE  # 16
FP4_ROPE_BYTES = FP4_ROPE_DIM * 2  # 128 (BF16 = 2 bytes per element)
FP4_KV_CACHE_DIM = FP4_PACKED_NOPE_BYTES + FP4_NUM_SCALE_BLOCKS + FP4_ROPE_BYTES  # 400


@triton.jit
def set_mla_kv_buffer_fp4_quant_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_u8_ptr,
    loc_ptr,
    buffer_stride: int,
    nope_stride: int,
    rope_u8_stride: int,
    nope_dim: tl.constexpr,
    rope_bytes: tl.constexpr,
    FP4_BLK: tl.constexpr,
    FB_PER_PROG: tl.constexpr,
    N_NOPE_PROGS: tl.constexpr,
):
    """Fuse BF16 -> FP4 E2M1 quantization with paged KV write.

    Grid: (n_loc, N_NOPE_PROGS + 1).
    Programs 0..N_NOPE_PROGS-1 handle nope quantization.
    Last program copies rope bytes.
    """
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    HALF_BLK: tl.constexpr = FP4_BLK // 2
    packed_nope_bytes: tl.constexpr = nope_dim // 2
    n_fp4_blocks: tl.constexpr = nope_dim // FP4_BLK

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    buf_base = kv_buffer_ptr + loc * buffer_stride

    if pid_blk < N_NOPE_PROGS:
        prog_start = pid_blk * FB_PER_PROG
        nope_base = cache_k_nope_ptr + pid_loc * nope_stride

        for fb_i in range(FB_PER_PROG):
            block_id = prog_start + fb_i
            base_off = block_id * FP4_BLK
            pair_idx = tl.arange(0, HALF_BLK)

            even_offs = base_off + pair_idx * 2
            odd_offs = even_offs + 1
            src_even = tl.load(nope_base + even_offs).to(tl.float32)
            src_odd = tl.load(nope_base + odd_offs).to(tl.float32)

            amax = tl.maximum(tl.max(tl.abs(src_even)), tl.max(tl.abs(src_odd)))
            amax = tl.maximum(amax, 1e-12)
            scale_exp = tl.math.ceil(tl.math.log2(amax / 6.0))
            inv_scale = 1.0 / tl.math.exp2(scale_exp)
            scale_u8 = (scale_exp + 127.0).to(tl.uint8)

            se = src_even * inv_scale
            so = src_odd * inv_scale

            # E2M1 quantize even
            sign_e = (se < 0.0).to(tl.int32) * 8
            ae = tl.abs(se)
            mag_e = (
                (ae >= 0.25).to(tl.int32)
                + (ae >= 0.75).to(tl.int32)
                + (ae >= 1.25).to(tl.int32)
                + (ae >= 1.75).to(tl.int32)
                + (ae >= 2.5).to(tl.int32)
                + (ae >= 3.5).to(tl.int32)
                + (ae >= 5.0).to(tl.int32)
            )
            fp4_e = (sign_e + mag_e).to(tl.uint8)

            # E2M1 quantize odd
            sign_o = (so < 0.0).to(tl.int32) * 8
            ao = tl.abs(so)
            mag_o = (
                (ao >= 0.25).to(tl.int32)
                + (ao >= 0.75).to(tl.int32)
                + (ao >= 1.25).to(tl.int32)
                + (ao >= 1.75).to(tl.int32)
                + (ao >= 2.5).to(tl.int32)
                + (ao >= 3.5).to(tl.int32)
                + (ao >= 5.0).to(tl.int32)
            )
            fp4_o = (sign_o + mag_o).to(tl.uint8)

            packed = (fp4_o << 4) | fp4_e

            tl.store(buf_base + block_id * HALF_BLK + pair_idx, packed)
            tl.store(buf_base + packed_nope_bytes + block_id, scale_u8)
    else:
        # Copy rope bytes (BF16 viewed as uint8)
        ROPE_BLK: tl.constexpr = rope_bytes
        rope_offs = tl.arange(0, ROPE_BLK)
        mask = rope_offs < rope_bytes
        src = tl.load(
            cache_k_rope_u8_ptr + pid_loc * rope_u8_stride + rope_offs, mask=mask
        )
        tl.store(
            buf_base + packed_nope_bytes + n_fp4_blocks + rope_offs, src, mask=mask
        )


def set_mla_kv_buffer_triton_fp4_quant(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    """Fuse BF16 MLA K → FP4 E2M1 quantization with paged KV write."""
    cache_k_nope = cache_k_nope.squeeze(1) if cache_k_nope.ndim == 3 else cache_k_nope
    cache_k_rope = cache_k_rope.squeeze(1) if cache_k_rope.ndim == 3 else cache_k_rope

    cache_k_nope = cache_k_nope.contiguous()
    cache_k_rope = cache_k_rope.contiguous()

    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    rope_bytes = rope_dim * cache_k_rope.element_size()

    assert nope_dim == FP4_NOPE_DIM, f"nope_dim must be {FP4_NOPE_DIM}, got {nope_dim}"
    assert rope_dim == FP4_ROPE_DIM, f"rope_dim must be {FP4_ROPE_DIM}, got {rope_dim}"
    assert nope_dim % FP4_BLOCK_SIZE == 0

    rope_u8 = cache_k_rope.view(torch.uint8)

    FP4_BLK = FP4_BLOCK_SIZE
    n_fp4_blocks = nope_dim // FP4_BLK
    FB_PER_PROG = 4
    N_NOPE_PROGS = n_fp4_blocks // FB_PER_PROG
    assert n_fp4_blocks % FB_PER_PROG == 0

    n_loc = loc.numel()
    grid = (n_loc, N_NOPE_PROGS + 1)

    set_mla_kv_buffer_fp4_quant_kernel[grid](
        kv_buffer,
        cache_k_nope,
        rope_u8,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        rope_u8.stride(0),
        nope_dim,
        rope_bytes,
        FP4_BLK=FP4_BLK,
        FB_PER_PROG=FB_PER_PROG,
        N_NOPE_PROGS=N_NOPE_PROGS,
        num_warps=1,
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


def convert_to_bigram_key(tokens: List[int]) -> List[Tuple[int, int]]:
    # EAGLE uses bigram keys in the radix tree since draft sequence is the one-token-shifted version of target
    # [1, 2, 3, 4] -> [(1,2), (2,3), (3,4)]
    if len(tokens) and isinstance(tokens[0], tuple):
        return tokens
    if len(tokens) < 2:
        return []
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
