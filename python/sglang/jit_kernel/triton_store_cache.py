from typing import Literal

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
_FP8_INFO = torch.finfo(_FP8_DTYPE)


@triton.jit
def _triton_fused_store_flashmla_kernel(
    input_ptr,
    cache_fp8_ptr,
    cache_bf16_ptr,
    cache_u8_ptr,
    indices_ptr,
    N,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    BYTES_PER_PAGE_BF16: tl.constexpr,
    S_OFFSET: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    if token_id >= N:
        return

    loc = tl.load(indices_ptr + token_id).to(tl.int32)
    page = loc // PAGE_SIZE
    slot = loc % PAGE_SIZE

    if tile_id == 7:
        # copy rope part (last 64 dims)
        rope_lane = tl.arange(0, TILE_SIZE)
        rope_vals = tl.load(input_ptr + token_id * 512 + 448 + rope_lane)
        rope_bf16_offset = (
            page * BYTES_PER_PAGE_BF16 + slot * 288 + 224 + rope_lane
        )
        tl.store(cache_bf16_ptr + rope_bf16_offset, rope_vals)
    else:
        # do nope quantization (tile_id < 7)
        tile_lane = tl.arange(0, TILE_SIZE)
        x_bf16 = tl.load(input_ptr + token_id * 512 + tile_id * TILE_SIZE + tile_lane)
        x_fp32 = x_bf16.to(tl.float32)

        abs_max = tl.max(tl.abs(x_fp32))
        scale = tl.maximum(abs_max, EPS) / FP8_MAX

        # cast scale to ue8m0 format
        log2_scale = tl.log2(scale)
        ceil_log2 = tl.math.ceil(log2_scale)
        inv_scale = tl.exp2(-ceil_log2)

        x_fp8 = tl.clamp(x_fp32 * inv_scale, FP8_MIN, FP8_MAX).to(
            cache_fp8_ptr.dtype.element_ty
        )

        nope_offset = (
            page * BYTES_PER_PAGE + slot * 576 + tile_id * TILE_SIZE + tile_lane
        )
        tl.store(cache_fp8_ptr + nope_offset, x_fp8)

        ue8m0 = (ceil_log2.to(tl.int32) + 127).to(tl.uint8)
        scale_offset = page * BYTES_PER_PAGE + S_OFFSET + slot * 8 + tile_id
        tl.store(cache_u8_ptr + scale_offset, ue8m0)


def triton_fused_store_flashmla(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> None:
    """Fused FP8 quantise + paged scatter for the SWA (flashmla) KV cache."""
    N = input.shape[0]
    if N == 0:
        return

    bytes_per_page = cache.shape[1]
    cache_fp8 = cache.view(_FP8_DTYPE)
    cache_bf16 = cache.view(torch.bfloat16)
    indices_i32 = indices.to(torch.int32) if indices.dtype != torch.int32 else indices

    # additional block (tile_id == 7) to handle rope copy
    _triton_fused_store_flashmla_kernel[(N, 8)](
        input,
        cache_fp8,
        cache_bf16,
        cache,
        indices_i32,
        N,
        PAGE_SIZE=page_size,
        BYTES_PER_PAGE=bytes_per_page,
        BYTES_PER_PAGE_BF16=bytes_per_page // 2,
        S_OFFSET=page_size * 576,
        TILE_SIZE=64,
        FP8_MIN=_FP8_INFO.min,
        FP8_MAX=_FP8_INFO.max,
        EPS=1e-8,
    )


@triton.jit
def _triton_fused_store_indexer_kernel(
    input_ptr,
    cache_fp8_ptr,
    cache_f32_ptr,
    indices_ptr,
    N,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    BYTES_PER_PAGE_F32: tl.constexpr,
    SCALE_PAGE_OFFSET_F32: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    token_id = tl.program_id(0)
    if token_id >= N:
        return

    loc = tl.load(indices_ptr + token_id).to(tl.int32)
    page = loc // PAGE_SIZE
    slot = loc % PAGE_SIZE

    lane = tl.arange(0, 128)
    x_fp32 = tl.load(input_ptr + token_id * 128 + lane).to(tl.float32)

    abs_max = tl.max(tl.abs(x_fp32))
    scale = tl.maximum(abs_max, EPS) / FP8_MAX
    inv_scale = 1.0 / scale

    x_fp8 = tl.clamp(x_fp32 * inv_scale, FP8_MIN, FP8_MAX).to(
        cache_fp8_ptr.dtype.element_ty
    )

    fp8_offset = page * BYTES_PER_PAGE + slot * 128 + lane
    tl.store(cache_fp8_ptr + fp8_offset, x_fp8)

    f32_offset = page * BYTES_PER_PAGE_F32 + SCALE_PAGE_OFFSET_F32 + slot
    tl.store(cache_f32_ptr + f32_offset, scale)


def triton_fused_store_indexer(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> None:
    """Fused FP8 quantise + paged scatter for the C4 indexer KV cache."""
    N = input.shape[0]
    if N == 0:
        return

    bytes_per_page = cache.shape[1]
    bytes_per_page_f32 = bytes_per_page // 4
    scale_page_offset_f32 = (128 * page_size) // 4

    cache_fp8 = cache.view(_FP8_DTYPE)
    cache_f32 = cache.view(torch.float32)
    indices_i32 = indices.to(torch.int32) if indices.dtype != torch.int32 else indices

    _triton_fused_store_indexer_kernel[(N,)](
        input,
        cache_fp8,
        cache_f32,
        indices_i32,
        N,
        PAGE_SIZE=page_size,
        BYTES_PER_PAGE=bytes_per_page,
        BYTES_PER_PAGE_F32=bytes_per_page_f32,
        SCALE_PAGE_OFFSET_F32=scale_page_offset_f32,
        FP8_MIN=_FP8_INFO.min,
        FP8_MAX=_FP8_INFO.max,
        EPS=1e-8,
    )


def triton_fused_store_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    type: Literal["flashmla", "indexer"],
) -> None:
    """ROCm dispatch for fused_store_cache()."""
    if type == "flashmla":
        triton_fused_store_flashmla(input, cache, indices, page_size)
    else:
        triton_fused_store_indexer(input, cache, indices, page_size)
