import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.dsv4.index_buf_accessor import NopeFp8RopeBf16Pack
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

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
    token_id = tl.program_id(0)
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
