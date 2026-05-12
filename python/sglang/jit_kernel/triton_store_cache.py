from typing import Literal

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
_FP8_INFO = torch.finfo(_FP8_DTYPE)

# DeepSeek-V4 MLA paged FP8 cache layout
_MLA_HEAD_DIM = 512               # full MLA token dim (elements per input row)
_MLA_NOPE_DIM = 448               # nope sub-dim (elements)
_MLA_TILE_SIZE = 64               # FP8 tile width (also rope copy stride)
_MLA_SLOT_BYTES = 576             # bytes per slot in the paged FP8 cache
_MLA_BF16_SLOT_ELEMS = _MLA_SLOT_BYTES // 2     # bf16-view slot stride (elements)
_MLA_BF16_ROPE_OFFSET = _MLA_NOPE_DIM // 2      # bf16-view rope offset (elements)
_MLA_SCALES_PER_TOKEN = 8         # UE8M0 scales per token (7 nope tiles + 1 padding)
_MLA_NUM_TILES = 8                # 7 nope quant tiles + 1 rope copy tile
_MLA_ROPE_TILE_ID = 7             # tile id reserved for the rope copy

# C4 indexer paged FP8 cache layout
_INDEXER_HEAD_DIM = 128

_UE8M0_EXPONENT_BIAS = 127


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
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    SLOT_BYTES: tl.constexpr,
    BF16_SLOT_ELEMS: tl.constexpr,
    BF16_ROPE_OFFSET: tl.constexpr,
    SCALES_PER_TOKEN: tl.constexpr,
    ROPE_TILE_ID: tl.constexpr,
    UE8M0_BIAS: tl.constexpr,
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

    if tile_id == ROPE_TILE_ID:
        rope_lane = tl.arange(0, TILE_SIZE)
        rope_vals = tl.load(input_ptr + token_id * HEAD_DIM + NOPE_DIM + rope_lane)
        rope_bf16_offset = (
            page * BYTES_PER_PAGE_BF16 + slot * BF16_SLOT_ELEMS + BF16_ROPE_OFFSET + rope_lane
        )
        tl.store(cache_bf16_ptr + rope_bf16_offset, rope_vals)
    else:
        tile_lane = tl.arange(0, TILE_SIZE)
        x_bf16 = tl.load(input_ptr + token_id * HEAD_DIM + tile_id * TILE_SIZE + tile_lane)
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
            page * BYTES_PER_PAGE + slot * SLOT_BYTES + tile_id * TILE_SIZE + tile_lane
        )
        tl.store(cache_fp8_ptr + nope_offset, x_fp8)

        ue8m0 = (ceil_log2.to(tl.int32) + UE8M0_BIAS).to(tl.uint8)
        scale_offset = page * BYTES_PER_PAGE + S_OFFSET + slot * SCALES_PER_TOKEN + tile_id
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

    _triton_fused_store_flashmla_kernel[(N, _MLA_NUM_TILES)](
        input,
        cache_fp8,
        cache_bf16,
        cache,
        indices_i32,
        N,
        PAGE_SIZE=page_size,
        BYTES_PER_PAGE=bytes_per_page,
        BYTES_PER_PAGE_BF16=bytes_per_page // 2,
        S_OFFSET=page_size * _MLA_SLOT_BYTES,
        TILE_SIZE=_MLA_TILE_SIZE,
        HEAD_DIM=_MLA_HEAD_DIM,
        NOPE_DIM=_MLA_NOPE_DIM,
        SLOT_BYTES=_MLA_SLOT_BYTES,
        BF16_SLOT_ELEMS=_MLA_BF16_SLOT_ELEMS,
        BF16_ROPE_OFFSET=_MLA_BF16_ROPE_OFFSET,
        SCALES_PER_TOKEN=_MLA_SCALES_PER_TOKEN,
        ROPE_TILE_ID=_MLA_ROPE_TILE_ID,
        UE8M0_BIAS=_UE8M0_EXPONENT_BIAS,
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
    HEAD_DIM: tl.constexpr,
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

    lane = tl.arange(0, HEAD_DIM)
    x_fp32 = tl.load(input_ptr + token_id * HEAD_DIM + lane).to(tl.float32)

    abs_max = tl.max(tl.abs(x_fp32))
    scale = tl.maximum(abs_max, EPS) / FP8_MAX
    inv_scale = 1.0 / scale

    x_fp8 = tl.clamp(x_fp32 * inv_scale, FP8_MIN, FP8_MAX).to(
        cache_fp8_ptr.dtype.element_ty
    )

    fp8_offset = page * BYTES_PER_PAGE + slot * HEAD_DIM + lane
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
    scale_page_offset_f32 = (_INDEXER_HEAD_DIM * page_size) // 4

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
        HEAD_DIM=_INDEXER_HEAD_DIM,
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
