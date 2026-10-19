"""Software V4.1 KV cache dequantization for CUDA SM80 (not SM86).

Only byte pointers are used for the packed cache: no native FP8 instructions or
FP8 tensor views are needed. This is a cache-format fallback, not a claim of
end-to-end model support.
"""

import torch
import triton
import triton.language as tl

from .kv_layout import KVLayout


@triton.jit
def _e4m3_to_fp32(byte):
    # E4M3FN: exponent 15 remains finite except for mantissa 7 (NaN).
    code = byte.to(tl.int32)
    exponent = (code >> 3) & 15
    mantissa = code & 7
    normal = ((exponent + 120) << 23) | (mantissa << 20)

    # Normalize the seven nonzero E4M3 subnormals with integer operations.
    # Their magnitudes are mantissa * 2**-9; all are normal FP32 values.
    shift = tl.where(mantissa >= 4, 21, tl.where(mantissa >= 2, 22, 23))
    sub_exponent = tl.where(mantissa >= 4, 120, tl.where(mantissa >= 2, 119, 118))
    subnormal = (sub_exponent << 23) | ((mantissa << shift) & 0x7FFFFF)
    subnormal = tl.where(mantissa == 0, 0, subnormal)
    bits = tl.where(exponent == 0, subnormal, normal)
    bits = tl.where((code & 0x7F) == 0x7F, 0x7FC00000, bits)
    bits = bits | ((code & 0x80) << 24)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _e2m1_to_fp32(code):
    code = code.to(tl.int32)
    exponent = (code >> 1) & 3
    mantissa = code & 1
    normal = ((exponent + 126) << 23) | (mantissa << 22)
    subnormal = tl.where(mantissa == 0, 0, 0x3F000000)
    bits = tl.where(exponent == 0, subnormal, normal)
    bits = bits | ((code & 8) << 28)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _ue8m0_to_fp32(byte):
    code = byte.to(tl.int32)
    # Byte 0 is 2**-127, a nonzero FP32 subnormal; byte 255 is NaN.
    bits = tl.where(code == 0, 0x00400000, code << 23)
    bits = tl.where(code == 255, 0x7FC00000, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _scaled_bf16(value, scale):
    # Do not let compiler FTZ defaults erase UE8M0 byte 0 or tiny products.
    product = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[value, scale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return product.to(tl.bfloat16, fp_downcast_rounding="rtne")


@triton.jit
def _dequantize_k_cache_paged_sm80_kernel(
    output_ptr,
    cache_u8_ptr,
    token_ids_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
    IS_FP4: tl.constexpr,
):
    tl.static_assert(cache_u8_ptr.dtype.element_ty == tl.uint8)
    token_id = tl.program_id(0).to(tl.int64)
    loc = tl.load(token_ids_ptr + token_id).to(tl.int64)
    page_base = (loc // PAGE_SIZE) * BYTES_PER_PAGE
    in_page = loc % PAGE_SIZE
    data_base = page_base + in_page * DATA_BYTES
    scale_base = page_base + S_OFFSET_BYTES + in_page * SCALE_BYTES
    out_base = token_id * output_stride_0.to(tl.int64)

    offsets = tl.arange(0, 512)
    if IS_FP4:
        packed = tl.load(cache_u8_ptr + data_base + offsets // 2)
        code = (packed >> ((offsets & 1) * 4)) & 15
        values = _e2m1_to_fp32(code)
        scale_code = tl.load(cache_u8_ptr + scale_base + offsets // TILE_SIZE)
        scales = _e4m3_to_fp32(scale_code)
    else:
        code = tl.load(cache_u8_ptr + data_base + offsets)
        values = _e4m3_to_fp32(code)
        scale_code = tl.load(cache_u8_ptr + scale_base + offsets // TILE_SIZE)
        scales = _ue8m0_to_fp32(scale_code)
    tl.store(output_ptr + out_base + offsets, _scaled_bf16(values, scales))


def dequantize_k_cache_paged_sm80(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: torch.Tensor | None = None,
    layout: KVLayout | str = KVLayout.V41,
) -> torch.Tensor:
    """Decode V41/V41_FP4 byte storage on SM80, preserving strided output rows.

    Token IDs must address valid cache entries. As with the native kernels, their
    values are not copied to the host or bounds-checked on this hot path.
    """
    layout = KVLayout.parse(layout)
    assert layout in (KVLayout.V41, KVLayout.V41_FP4), layout
    assert quant_k_cache.is_cuda and quant_k_cache.ndim == 2
    assert quant_k_cache.is_contiguous()
    assert torch.version.hip is None
    assert torch.cuda.get_device_capability(quant_k_cache.device) == (8, 0)
    assert page_table_1_flattened.ndim == 1
    assert page_table_1_flattened.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)
    assert page_table_1_flattened.device == quant_k_cache.device
    assert page_size > 0

    # Pools may expose BF16 storage; reinterpret only as bytes, never as FP8.
    cache_u8 = quant_k_cache.view(torch.uint8)
    bytes_per_page = cache_u8.shape[1]
    assert bytes_per_page >= page_size * layout.bytes_per_token, (
        f"{bytes_per_page=} cannot hold {page_size} tokens of {layout}"
    )
    num_tokens = page_table_1_flattened.numel()
    if out is None:
        out = torch.empty(
            (num_tokens, 1, 512), dtype=torch.bfloat16, device=quant_k_cache.device
        )
    else:
        assert out.shape == (num_tokens, 1, 512)
        assert out.dtype == torch.bfloat16
        assert out.device == quant_k_cache.device
        assert out.stride(2) == 1
        assert num_tokens <= 1 or out.stride(0) >= 512
    if num_tokens == 0:
        return out

    _dequantize_k_cache_paged_sm80_kernel[(num_tokens,)](
        out,
        cache_u8.reshape(-1),
        page_table_1_flattened,
        out.stride(0),
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DATA_BYTES=layout.data_bytes,
        SCALE_BYTES=layout.scale_bytes,
        TILE_SIZE=layout.tile_size,
        S_OFFSET_BYTES=layout.scale_offset(page_size),
        IS_FP4=layout is KVLayout.V41_FP4,
    )
    return out
