"""References for the DeepSeek-V4 FP8 K cache (V4 layout): quantize, store, dequantize."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    DIM_NOPE,
    DIM_ROPE,
    NOPE_ROPE_BYTES,
    NUM_SCALE_TILES,
    PADDED_SCALE_PER_TOKEN,
    TILE_SIZE,
)
from sglang.kernels.ops.kvcache.dsv4_main_kv_store import NopeFp8RopeBf16Pack
from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


@triton.jit
def _quant_k_cache_fused_kernel(
    k_bf16_ptr,
    k_nope_fp8_ptr,
    k_rope_bf16_ptr,
    scale_k_nope_uint8_ptr,
    k_bf16_stride_0,
    k_nope_fp8_stride_0,
    k_rope_bf16_stride_0,
    scale_stride_0,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    token_id = tl.program_id(0).to(tl.int64)
    tile_id = tl.program_id(1)

    if tile_id == NUM_TILES:
        rope_range = tl.arange(0, TILE_SIZE)
        rope_mask = rope_range < DIM_ROPE

        in_rope_offsets = token_id * k_bf16_stride_0 + DIM_NOPE + rope_range
        rope_data = tl.load(k_bf16_ptr + in_rope_offsets, mask=rope_mask, other=0.0)

        out_rope_offsets = token_id * k_rope_bf16_stride_0 + rope_range
        tl.store(k_rope_bf16_ptr + out_rope_offsets, rope_data, mask=rope_mask)
    else:
        tile_range = tl.arange(0, TILE_SIZE)

        in_tile_offsets = token_id * k_bf16_stride_0 + tile_id * TILE_SIZE + tile_range
        x_bf16 = tl.load(k_bf16_ptr + in_tile_offsets)
        x_fp32 = x_bf16.to(tl.float32)

        abs_x = tl.abs(x_fp32)
        max_abs = tl.max(abs_x)
        max_abs_clamped = tl.maximum(max_abs, EPS)
        scale = max_abs_clamped / FP8_MAX

        log2_scale = tl.log2(scale)
        ceil_log2 = tl.math.ceil(log2_scale)
        scale_pow2_fp32 = tl.exp2(ceil_log2)
        scale_inv = 1.0 / scale_pow2_fp32
        x_scaled = x_fp32 * scale_inv
        x_fp8 = tl.clamp(x_scaled, FP8_MIN, FP8_MAX).to(k_nope_fp8_ptr.dtype.element_ty)

        out_fp8_offsets = (
            token_id * k_nope_fp8_stride_0 + tile_id * TILE_SIZE + tile_range
        )
        tl.store(k_nope_fp8_ptr + out_fp8_offsets, x_fp8)

        exponent = ceil_log2.to(tl.int32)
        scale_uint8 = (exponent + 127).to(tl.uint8)

        out_scale_offset = token_id * scale_stride_0 + tile_id
        tl.store(scale_k_nope_uint8_ptr + out_scale_offset, scale_uint8)


def quant_to_nope_fp8_rope_bf16_pack_triton(
    k_bf16: torch.Tensor,
) -> NopeFp8RopeBf16Pack:
    assert k_bf16.dtype == torch.bfloat16
    num_tokens, hidden_dim = k_bf16.shape
    assert hidden_dim == 512
    dim_nope = 448
    dim_rope = 64
    tile_size = 64
    num_tiles = dim_nope // tile_size

    k_bf16 = k_bf16.contiguous()

    k_nope_fp8 = torch.empty(
        (num_tokens, dim_nope), dtype=fp8_dtype, device=k_bf16.device
    )
    k_rope_bf16 = torch.empty(
        (num_tokens, dim_rope), dtype=torch.bfloat16, device=k_bf16.device
    )
    scale_k_nope_ue8m0 = torch.empty(
        (num_tokens, num_tiles), dtype=torch.uint8, device=k_bf16.device
    )

    fp8_dtype_info = torch.finfo(fp8_dtype)

    grid = (num_tokens, num_tiles + 1)
    _quant_k_cache_fused_kernel[grid](
        k_bf16,
        k_nope_fp8,
        k_rope_bf16,
        scale_k_nope_ue8m0,
        k_bf16.stride(0),
        k_nope_fp8.stride(0),
        k_rope_bf16.stride(0),
        scale_k_nope_ue8m0.stride(0),
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        TILE_SIZE=tile_size,
        NUM_TILES=num_tiles,
        FP8_MIN=fp8_dtype_info.min,
        FP8_MAX=fp8_dtype_info.max,
        EPS=1e-8,
    )

    return NopeFp8RopeBf16Pack(
        k_nope_fp8=k_nope_fp8,
        k_rope_bf16=k_rope_bf16,
        scale_k_nope_ue8m0=scale_k_nope_ue8m0,
    )


def set_k_and_s_torch(
    buf: torch.Tensor,
    loc: torch.Tensor,
    nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    page_size: int,
):
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape

    k_nope, k_rope, scale_k_nope = (
        nope_fp8_rope_bf16_pack.k_nope_fp8,
        nope_fp8_rope_bf16_pack.k_rope_bf16,
        nope_fp8_rope_bf16_pack.scale_k_nope_ue8m0,
    )

    num_tokens_to_write_nope, nope_dim = k_nope.shape
    num_tokens_to_write_rope, rope_dim = k_rope.shape
    num_tokens_to_write_scale, scale_dim = scale_k_nope.shape

    assert (
        num_tokens_to_write
        == num_tokens_to_write_nope
        == num_tokens_to_write_rope
        == num_tokens_to_write_scale
    ), (
        f"{num_tokens_to_write=} {num_tokens_to_write_nope=} {num_tokens_to_write_rope=} {num_tokens_to_write_scale=}"
    )

    assert buf.dtype == torch.uint8
    assert loc.dtype in [
        torch.int64,
        torch.int32,
    ], f"{loc.dtype=}"

    assert k_nope.dtype == fp8_dtype
    assert k_rope.dtype == torch.bfloat16
    assert scale_k_nope.dtype == torch.uint8

    assert buf.is_contiguous()
    assert loc.is_contiguous()
    assert k_nope.is_contiguous()
    assert k_rope.is_contiguous()
    assert scale_k_nope.is_contiguous()

    buf_fp8 = buf.view(fp8_dtype).flatten()
    buf_bf16 = buf.view(torch.bfloat16).flatten()
    buf_scale = buf.view(torch.uint8).flatten()

    loc_page_index = loc // page_size
    loc_token_offset_in_page = loc % page_size

    s_offset_nbytes_in_page = page_size * (nope_dim + rope_dim * 2)

    nope_offset = loc_page_index * buf_numel_per_page + loc_token_offset_in_page * (
        nope_dim + rope_dim * 2
    )

    rope_offset = (
        loc_page_index * buf_numel_per_page // 2
        + (loc_token_offset_in_page * (nope_dim + rope_dim * 2) + nope_dim) // 2
    )

    s_offset = (
        loc_page_index * buf_numel_per_page
        + s_offset_nbytes_in_page
        + loc_token_offset_in_page * (scale_dim + 1)
    )

    for i in range(num_tokens_to_write):
        buf_fp8[nope_offset[i] : nope_offset[i] + nope_dim] = k_nope[i]
        buf_bf16[rope_offset[i] : rope_offset[i] + rope_dim] = k_rope[i]
        buf_scale[s_offset[i] : s_offset[i] + scale_dim] = scale_k_nope[i]


def dequantize_k_cache_paged_ref(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Pure-torch reference for ``dsv4.dequant_k_cache.dequantize_k_cache_paged_dsv4``.

    Decodes the same v4 paged layout with vectorized torch indexing instead of
    a Triton kernel.
    """
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)
    u8 = quant_k_cache.view(torch.uint8)
    bytes_per_page = u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES

    flat_u8 = u8.reshape(-1)
    flat_fp8 = u8.view(fp8_dtype).reshape(-1)
    flat_bf16 = u8.view(torch.bfloat16).reshape(-1)

    loc = page_table_1_flattened.to(torch.int64)
    page_idx = loc // page_size
    in_page = loc % page_size
    page_byte_base = page_idx * bytes_per_page
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + s_offset_bytes + in_page * PADDED_SCALE_PER_TOKEN
    )

    device = quant_k_cache.device
    nope_byte = (
        token_data_base[:, None] + torch.arange(DIM_NOPE, device=device)[None, :]
    )
    nope_fp8 = flat_fp8[nope_byte].to(torch.float32)
    scale_byte = (
        token_scale_base[:, None]
        + torch.arange(NUM_SCALE_TILES, device=device)[None, :]
    )
    scale_u8 = flat_u8[scale_byte].to(torch.int32)
    scale_pow2 = torch.exp2((scale_u8 - 127).to(torch.float32))
    scale_pow2 = torch.where(
        scale_pow2 < (2.0**-126), torch.zeros_like(scale_pow2), scale_pow2
    )
    scale_full = scale_pow2.repeat_interleave(TILE_SIZE, dim=1)
    nope = nope_fp8 * scale_full

    rope_bf16_base = (token_data_base + DIM_NOPE) // 2
    rope_idx = rope_bf16_base[:, None] + torch.arange(DIM_ROPE, device=device)[None, :]
    rope = flat_bf16[rope_idx]

    out = torch.empty(
        (loc.shape[0], 1, DIM_NOPE + DIM_ROPE),
        dtype=torch.bfloat16,
        device=device,
    )
    out[:, 0, :DIM_NOPE] = nope.to(torch.bfloat16)
    out[:, 0, DIM_NOPE:] = rope
    return out
