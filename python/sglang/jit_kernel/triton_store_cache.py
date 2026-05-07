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

# Pick the correct FP8 dtype for the current hardware at module load time.
# On ROCm, AMD GPUs use the FNUZ variant (no negative zero, no infinities).
# On CUDA, the standard E4M3 variant is used.
_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
# Pre-fetch the dtype's min/max range so we can pass them as constexprs.
_FP8_INFO = torch.finfo(_FP8_DTYPE)


# ---------------------------------------------------------------------------
# flashmla: [N, 512] BF16 → paged SWA KV cache
# ---------------------------------------------------------------------------


@triton.jit
def _triton_fused_store_flashmla_kernel(
    input_ptr,           # [N, 512] BF16 – the raw kv vector for each token
    cache_fp8_ptr,       # same backing buffer as cache_u8_ptr, viewed as FP8
                         # (1 FP8 element = 1 byte, so element offsets == byte offsets)
    cache_bf16_ptr,      # same backing buffer viewed as BF16
                         # (1 BF16 element = 2 bytes, so byte_offset / 2 = elem offset)
    cache_u8_ptr,        # paged KV buffer as raw uint8 (for UE8M0 scale byte writes)
    indices_ptr,         # [N] int32 – absolute slot index (not page index) per token
    N,                   # total number of tokens in this batch
    PAGE_SIZE: tl.constexpr,           # tokens per page, e.g. 256
    BYTES_PER_PAGE: tl.constexpr,      # total padded bytes per page
    BYTES_PER_PAGE_BF16: tl.constexpr, # BYTES_PER_PAGE // 2 (number of BF16 elements)
    S_OFFSET: tl.constexpr,            # byte offset of scale region within page
                                       # = PAGE_SIZE * 576
    TILE_SIZE: tl.constexpr,           # 64 (elements per nope tile = elements per warp)
    FP8_MIN: tl.constexpr,             # e.g. -448.0 for E4M3FN, -240.0 for FNUZ
    FP8_MAX: tl.constexpr,             # e.g.  448.0 for E4M3FN,  240.0 for FNUZ
    EPS: tl.constexpr,                 # minimum scale denominator (prevents div-by-zero)
):
    """
    Grid: (N, 8).  Axis 0 iterates over tokens; axis 1 over the 8 tile slots:
      program_id(1) in [0, 6] → one of the 7 nope tiles (64 elements each)
      program_id(1) == 7      → rope copy (64 BF16 elements, no quantisation)

    For a given token at absolute slot index `loc`:
      page   = loc // PAGE_SIZE   (which page this token lives in)
      slot   = loc  % PAGE_SIZE   (position within that page)

    Nope byte layout within the page (for a given slot):
      FP8 values : cache[page * BPP + slot * 576 + tile * 64 + lane]
      UE8M0 scale: cache[page * BPP + S_OFFSET  + slot * 8  + tile]

    Rope byte layout (viewed as BF16 to allow native BF16 store):
      BF16 elem index = page * (BPP//2) + slot * 288 + 224 + lane
        288 = 576 // 2   (one slot's 576 bytes = 288 BF16 elements)
        224 = 448 // 2   (skip past the 448-byte nope region = 224 BF16 elements)
    """
    # Each program handles one (token, tile) pair.
    token_id = tl.program_id(0)   # which token in [0, N)
    tile_id = tl.program_id(1)    # which tile in [0, 8)

    # Guard: more programs may be launched than tokens in the last batch.
    if token_id >= N:
        return

    # Load the absolute slot index for this token (one scalar load per program).
    # Cast to int32 because PAGE_SIZE and slot arithmetic fit in 32 bits.
    loc = tl.load(indices_ptr + token_id).to(tl.int32)

    # Decompose slot index into (page, slot-within-page).
    page = loc // PAGE_SIZE
    slot = loc % PAGE_SIZE

    if tile_id == 7:
        # -----------------------------------------------------------------------
        # Tile 7: rope copy — 64 BF16 values starting at input dim 448.
        # No quantisation; copy the values verbatim so the attention kernel
        # can apply rotary position embeddings (RoPE) in BF16 precision.
        # -----------------------------------------------------------------------

        # lane is a vector [0, 1, ..., 63] — one element per rope dim.
        rope_lane = tl.arange(0, TILE_SIZE)

        # Load rope slice: input[token_id, 448:512].
        # Stride along the token axis is 512 (the full kv dim).
        rope_vals = tl.load(input_ptr + token_id * 512 + 448 + rope_lane)

        # Compute the BF16 element offset in the cache.
        # The rope region starts at byte 448 within the slot's 576-byte data block.
        # Divided by 2 → BF16 element 224 within the slot.
        rope_bf16_offset = (
            page * BYTES_PER_PAGE_BF16   # jump to this page in BF16 view
            + slot * 288                  # jump to this slot (576 bytes / 2 = 288 BF16 elems)
            + 224                         # skip past nope region (448 bytes / 2 = 224 BF16 elems)
            + rope_lane                   # individual rope element within [0, 63]
        )
        tl.store(cache_bf16_ptr + rope_bf16_offset, rope_vals)

    else:
        # -----------------------------------------------------------------------
        # Tiles 0–6: nope quantisation — 64 BF16 → FP8 with UE8M0 scale.
        # Each of the 7 tiles covers a non-overlapping 64-element block of
        # the 448-element nope region (tile 0 = dims 0:64, tile 1 = 64:128, …).
        # -----------------------------------------------------------------------

        # lane is a vector [0, 1, ..., 63] — one element per tile element.
        tile_lane = tl.arange(0, TILE_SIZE)

        # Load this tile's 64 BF16 values from the input.
        # Input is laid out as [N, 512], stride = 512 per token.
        x_bf16 = tl.load(
            input_ptr + token_id * 512 + tile_id * TILE_SIZE + tile_lane
        )
        # Upcast to float32 for numerically stable abs-max and quantisation.
        x_fp32 = x_bf16.to(tl.float32)

        # Compute the per-tile absolute maximum.
        # tl.max reduces a vector to a scalar.
        abs_max = tl.max(tl.abs(x_fp32))

        # Compute the scale: scale = max(eps, abs_max) / FP8_MAX.
        # EPS prevents a zero scale when the tile contains all zeros.
        scale = tl.maximum(abs_max, EPS) / FP8_MAX

        # ----- UE8M0 encoding -----
        # UE8M0 is the "unsigned 8-bit mantissa-0" scale format used by FlashMLA.
        # It stores only the biased IEEE 754 exponent (rounded up), so decoding
        # is a single bit-shift rather than a floating-point multiply.
        #
        # Encoding steps (mirrors cast_to_ue8m0 in store.cuh):
        #   1. log2_scale = log2(scale)          e.g. scale=0.5 → log2_scale=-1.0
        #   2. ceil_log2  = ceil(log2_scale)     rounds up to next integer
        #                                        e.g. -1.0 → -1 (exact power of 2)
        #                                             -0.9 →  0 (not exact → rounds up)
        #   3. ue8m0_byte = ceil_log2 + 127      add IEEE 754 bias (127 for float32)
        #                                        so the stored byte is always in [0,255]
        log2_scale = tl.log2(scale)
        ceil_log2 = tl.math.ceil(log2_scale)    # scalar float after tl.max

        # inv_scale = 2^(-ceil_log2): the exact power-of-2 that reverses quantisation.
        # Using exp2 avoids a division and guarantees a power-of-2 result,
        # which is what the reader reconstructs from the UE8M0 byte.
        inv_scale = tl.exp2(-ceil_log2)

        # Multiply, clamp to the FP8 representable range, then cast to FP8.
        # The FP8 dtype is inferred from the cache pointer's element type, so
        # this line works for both float8_e4m3fn (CUDA) and float8_e4m3fnuz (ROCm).
        x_fp8 = tl.clamp(x_fp32 * inv_scale, FP8_MIN, FP8_MAX).to(
            cache_fp8_ptr.dtype.element_ty
        )

        # Write the 64 FP8 values to the cache at the correct byte offset.
        # Layout: page → slot → tile → lane.
        nope_offset = (
            page * BYTES_PER_PAGE   # jump to this page
            + slot * 576            # jump to this slot (each slot = 576 bytes of data)
            + tile_id * TILE_SIZE   # jump to this tile within the 448-byte nope block
            + tile_lane             # individual element within [0, 63]
        )
        tl.store(cache_fp8_ptr + nope_offset, x_fp8)

        # Write the 1-byte UE8M0 scale for this tile to the scale region.
        # (ceil_log2 + 127) fits in uint8 because scale < 1 for typical activations,
        # so ceil_log2 ≤ 0 and the result is ≤ 127.  Large activations shift it up
        # but remain within [0, 255] for all representable FP8 scales.
        ue8m0 = (ceil_log2.to(tl.int32) + 127).to(tl.uint8)

        # Scale region starts at S_OFFSET = PAGE_SIZE * 576 bytes within the page.
        # Each slot gets 8 bytes (7 scale bytes + 1 pad); tile_id selects the byte.
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

    # bytes_per_page is read from the cache shape rather than recomputed so
    # this launcher stays correct for any page_size passed by the allocator.
    bytes_per_page = cache.shape[1]

    # Create typed views of the same underlying memory so the kernel can write
    # FP8, BF16, and uint8 values to the correct byte addresses without any
    # pointer casting inside the Triton program.
    cache_fp8 = cache.view(_FP8_DTYPE)    # 1 FP8 elem = 1 byte → same element strides
    cache_bf16 = cache.view(torch.bfloat16)  # 1 BF16 elem = 2 bytes → halved strides

    # The kernel uses int32 arithmetic throughout; normalise here to avoid a
    # per-program cast inside the hot path.
    indices_i32 = indices.to(torch.int32) if indices.dtype != torch.int32 else indices

    # Launch: N tokens × 8 tile programs per token (7 nope tiles + 1 rope).
    _triton_fused_store_flashmla_kernel[(N, 8)](
        input,
        cache_fp8,
        cache_bf16,
        cache,           # uint8 view — cache itself is already uint8
        indices_i32,
        N,
        PAGE_SIZE=page_size,
        BYTES_PER_PAGE=bytes_per_page,
        BYTES_PER_PAGE_BF16=bytes_per_page // 2,
        S_OFFSET=page_size * 576,   # scale region starts after the data region
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
    input_ptr,        # [N, 128] BF16 – compressed indexer vector per token
    cache_fp8_ptr,    # paged C4 indexer buffer viewed as FP8
    cache_f32_ptr,    # same buffer viewed as float32 (for writing the FP32 scale)
    indices_ptr,      # [N] int32 – absolute slot index per token
    N,
    PAGE_SIZE: tl.constexpr,               # tokens per page
    BYTES_PER_PAGE: tl.constexpr,          # = 132 * PAGE_SIZE
    BYTES_PER_PAGE_F32: tl.constexpr,      # = 33 * PAGE_SIZE (BPP viewed as float32)
    SCALE_PAGE_OFFSET_F32: tl.constexpr,   # = 32 * PAGE_SIZE (float32-element index
                                           #   where the scale region starts within a page)
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    Grid: (N,).  One program per token.

    Unlike flashmla, the indexer uses a simple FP32 scale (not UE8M0) because
    only one scale is needed per token (not one per 64-element tile) and the
    consumer reads it as a standard float32.

    Layout within page for a given slot:
      FP8 values: cache_fp8[page * BPP + slot * 128 + lane]
                    (128 FP8 bytes in the data region)
      FP32 scale: cache_f32[page * BPP_F32 + SCALE_PAGE_OFFSET_F32 + slot]
                    (1 float32 = 4 bytes in the scale region)
    """
    token_id = tl.program_id(0)
    if token_id >= N:
        return

    # Decompose the slot index into (page, slot-within-page).
    loc = tl.load(indices_ptr + token_id).to(tl.int32)
    page = loc // PAGE_SIZE
    slot = loc % PAGE_SIZE

    # Load all 128 BF16 values for this token and upcast to float32.
    lane = tl.arange(0, 128)
    x_bf16 = tl.load(input_ptr + token_id * 128 + lane)
    x_fp32 = x_bf16.to(tl.float32)

    # Compute the per-token scale: scale = max(eps, abs_max) / FP8_MAX.
    # This is standard per-tensor quantisation (one scale covers all 128 dims).
    abs_max = tl.max(tl.abs(x_fp32))
    scale = tl.maximum(abs_max, EPS) / FP8_MAX
    inv_scale = 1.0 / scale   # plain reciprocal (no UE8M0 rounding needed here)

    # Quantise to FP8: multiply by inv_scale, clamp to representable range,
    # then cast to the FP8 dtype inferred from the output pointer.
    x_fp8 = tl.clamp(x_fp32 * inv_scale, FP8_MIN, FP8_MAX).to(
        cache_fp8_ptr.dtype.element_ty
    )

    # Write 128 FP8 values into the data region.
    # Each slot occupies exactly 128 bytes in the data region.
    fp8_offset = page * BYTES_PER_PAGE + slot * 128 + lane
    tl.store(cache_fp8_ptr + fp8_offset, x_fp8)

    # Write the FP32 scale into the scale region.
    # The scale region starts at byte PAGE_SIZE * 128 within the page.
    # Viewed as float32, that byte offset becomes float32-element PAGE_SIZE * 32
    # (i.e., SCALE_PAGE_OFFSET_F32 = 128 * PAGE_SIZE / 4 = 32 * PAGE_SIZE).
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

    bytes_per_page = cache.shape[1]       # = 132 * page_size

    # Derive the float32-view constants used for scale writes.
    # BYTES_PER_PAGE / 4 = 33 * page_size (total f32 elements per page).
    bytes_per_page_f32 = bytes_per_page // 4

    # The scale region starts at byte PAGE_SIZE * 128 within the page.
    # In the float32 view that's element (128 * page_size) // 4 = 32 * page_size.
    scale_page_offset_f32 = (128 * page_size) // 4

    # Create typed views so the kernel can write FP8 and float32 values
    # without pointer casting.
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
