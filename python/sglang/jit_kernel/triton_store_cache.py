"""
Triton fused store-cache kernels for ROCm/HIP.

Fuses FP8 quantization + paged-scatter into a single kernel, replacing the
two-step AMD fallback path in fused_store_cache():

    Kernel 1  quant_to_nope_fp8_rope_bf16_pack_triton()  -> NopeFp8RopeBf16Pack
    Kernel 2  _set_k_and_s_triton_kernel()               -> scatter to paged buf

Two variants mirror the CUDA FusedStoreCache{FlashMLA,Indexer}Kernel in
jit_kernel/csrc/deepseek_v4/store.cuh:

  flashmla  –  input [N, 512] BF16.  Quantises the nope region (dims 0:448)
               to FP8 in 7 tiles of 64 elements each, each tile getting its
               own UE8M0 scale byte.  Copies the rope region (dims 448:512)
               verbatim as BF16.  Scatters both into the paged SWA KV cache.

  indexer   –  input [N, 128] BF16.  Quantises all 128 elements to FP8 with
               a single FP32 scale per token.  Scatters into the paged C4
               indexer KV cache.

Page layouts (matching deepseekv4_memory_pool.py / store.cuh):

  flashmla  bytes_per_page = ceil(584 * page_size / 576) * 576
            ┌─ data region ──────────────────────────────────────────────────┐
            │  page_size slots × 576 bytes each                              │
            │    bytes  [0,   448) : FP8 nope (448 × 1-byte FP8 values)     │
            │    bytes  [448, 576) : BF16 rope (64 × 2-byte BF16 values)    │
            └────────────────────────────────────────────────────────────────┘
            ┌─ scale region ─────────────────────────────────────────────────┐
            │  starts at byte page_size * 576                                │
            │  page_size slots × 8 bytes each                                │
            │    bytes [0, 7) : 7 UE8M0 scale bytes (one per 64-elem tile)  │
            │    byte  [7]    : padding                                       │
            └────────────────────────────────────────────────────────────────┘

  indexer   bytes_per_page = 132 * page_size
            ┌─ data region ──────────────────────────────────────────────────┐
            │  page_size slots × 128 bytes each (128 × 1-byte FP8 values)   │
            └────────────────────────────────────────────────────────────────┘
            ┌─ scale region ─────────────────────────────────────────────────┐
            │  starts at byte page_size * 128                                │
            │  page_size slots × 4 bytes each (1 × FP32 scale)              │
            └────────────────────────────────────────────────────────────────┘

UE8M0 encoding (matching _quant_k_cache_fused_kernel in quant_k_cache_v4.py):
    ceil_log2  = ceil(log2(scale))          # round log2 up to next integer
    ue8m0_byte = (ceil_log2 + 127).to(u8)  # store as IEEE 754 biased exponent
    inv_scale  = 2^(-ceil_log2)            # power-of-2 reciprocal for quantisation

The reader (FlashMLA attention kernel) recovers the scale as:
    inv_scale = 2^(127 - ue8m0_byte)       # = 2^(-ceil_log2)  ✓

FP8 dtype is inferred from the pointer element type (cache_fp8_ptr.dtype.element_ty)
so the same kernel handles both float8_e4m3fn (CUDA) and float8_e4m3fnuz (ROCm/FNUZ)
without any conditional logic.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
_FP8_INFO = torch.finfo(_FP8_DTYPE)


# ---------------------------------------------------------------------------
# flashmla: [N, 512] BF16 → paged SWA KV cache
# ---------------------------------------------------------------------------


@triton.jit
def _triton_fused_store_flashmla_kernel(
    input_ptr,           # [N, 512] BF16
    cache_fp8_ptr,       # same buffer viewed as FP8 (1 elem = 1 byte)
    cache_bf16_ptr,      # same buffer viewed as BF16 (1 elem = 2 bytes)
    cache_u8_ptr,        # same buffer viewed as uint8 (for UE8M0 scale writes)
    indices_ptr,         # [N] int32 – absolute slot index per token
    N,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    BYTES_PER_PAGE_BF16: tl.constexpr,  # BYTES_PER_PAGE // 2
    S_OFFSET: tl.constexpr,             # PAGE_SIZE * 576 (scale region start)
    TILE_SIZE: tl.constexpr,            # 64
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    Grid: (N, 8).  program_id(1) in [0,6] → nope tile; program_id(1)==7 → rope.

    Nope byte layout (slot within page):
      FP8 values : cache[page*BPP + slot*576 + tile*64 + lane]
      UE8M0 scale: cache[page*BPP + S_OFFSET  + slot*8  + tile]

    Rope byte layout (BF16 view):
      BF16 offset = page*(BPP//2) + slot*288 + 224 + lane
        288 = 576//2,  224 = 448//2 (skip nope region)
    """
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    if token_id >= N:
        return

    loc = tl.load(indices_ptr + token_id).to(tl.int32)
    page = loc // PAGE_SIZE
    slot = loc % PAGE_SIZE

    if tile_id == 7:
        # Rope copy: 64 BF16 dims [448:512], no quantisation.
        rope_lane = tl.arange(0, TILE_SIZE)
        rope_vals = tl.load(input_ptr + token_id * 512 + 448 + rope_lane)
        rope_bf16_offset = (
            page * BYTES_PER_PAGE_BF16 + slot * 288 + 224 + rope_lane
        )
        tl.store(cache_bf16_ptr + rope_bf16_offset, rope_vals)

    else:
        # Nope tiles 0-6: quantise 64 BF16 dims to FP8 with UE8M0 scale.
        tile_lane = tl.arange(0, TILE_SIZE)
        x_bf16 = tl.load(input_ptr + token_id * 512 + tile_id * TILE_SIZE + tile_lane)
        x_fp32 = x_bf16.to(tl.float32)

        abs_max = tl.max(tl.abs(x_fp32))
        scale = tl.maximum(abs_max, EPS) / FP8_MAX

        # UE8M0: store biased ceil(log2(scale)) so the reader can recover
        # inv_scale via a single bit-shift (2^(127 - ue8m0_byte)).
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
    """
    Fused quantise + scatter for the SWA (flashmla) KV cache on ROCm.

    Replaces the two-step fallback:
        pack = quant_to_nope_fp8_rope_bf16_pack_triton(input)
        _set_k_and_s_triton(cache, indices, pack, page_size)

    Args:
        input:     [N, 512] BF16 – raw kv vector per token (448 nope + 64 rope)
        cache:     [num_pages, bytes_per_page] uint8 – paged SWA KV buffer
        indices:   [N] int32/int64 – absolute slot index for each token
        page_size: tokens per page; must match the cache's allocation
    """
    N = input.shape[0]
    if N == 0:
        return

    bytes_per_page = cache.shape[1]
    cache_fp8 = cache.view(_FP8_DTYPE)
    cache_bf16 = cache.view(torch.bfloat16)
    indices_i32 = indices.to(torch.int32) if indices.dtype != torch.int32 else indices

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


# ---------------------------------------------------------------------------
# indexer: [N, 128] BF16 → paged C4 indexer KV cache
# ---------------------------------------------------------------------------


@triton.jit
def _triton_fused_store_indexer_kernel(
    input_ptr,        # [N, 128] BF16
    cache_fp8_ptr,    # paged C4 buffer viewed as FP8
    cache_f32_ptr,    # same buffer viewed as float32 (for FP32 scale writes)
    indices_ptr,      # [N] int32 – absolute slot index per token
    N,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,           # 132 * PAGE_SIZE
    BYTES_PER_PAGE_F32: tl.constexpr,       # 33 * PAGE_SIZE
    SCALE_PAGE_OFFSET_F32: tl.constexpr,    # 32 * PAGE_SIZE (scale region in f32 view)
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    Grid: (N,).  One program per token.  FP32 scale (not UE8M0) — one per token.

    Layout:
      FP8 values: cache_fp8[page*BPP + slot*128 + lane]
      FP32 scale: cache_f32[page*BPP_F32 + SCALE_PAGE_OFFSET_F32 + slot]
    """
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
    """
    Fused quantise + scatter for the C4 indexer KV cache on ROCm.

    Replaces the two-step fallback:
        fp8, scale = act_quant(input)
        token_to_kv_pool.set_index_k_scale_buffer(layer_id, loc, fp8, scale)

    Args:
        input:     [N, 128] BF16 – compressed C4 indexer vector per token
        cache:     [num_pages, 132*page_size] uint8 – paged C4 indexer buffer
        indices:   [N] int32/int64 – absolute slot index for each token
        page_size: tokens per page; must match the cache's allocation
    """
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


# ---------------------------------------------------------------------------
# Unified public entry point (mirrors the fused_store_cache() signature)
# ---------------------------------------------------------------------------


def triton_fused_store_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    type: Literal["flashmla", "indexer"],
) -> None:
    """
    ROCm dispatch shim called from fused_store_cache() when is_hip_runtime().

    Routes to triton_fused_store_flashmla (SWA KV cache, 512-dim input) or
    triton_fused_store_indexer (C4 indexer cache, 128-dim input) based on type.
    """
    if type == "flashmla":
        triton_fused_store_flashmla(input, cache, indices, page_size)
    else:
        triton_fused_store_indexer(input, cache, indices, page_size)
