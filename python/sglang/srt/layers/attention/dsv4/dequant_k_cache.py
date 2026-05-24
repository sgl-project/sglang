from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

# v4 KV cache layout (see dsv4.index_buf_accessor._set_k_and_s_triton_kernel):
#   per-token: 448 fp8 nope + 64 bf16 rope (= 576 contiguous bytes) +
#              7 ue8m0 scales padded to 8 bytes.
#   per-page:  [token 0..P-1 nope+rope (P*576 bytes)] [token 0..P-1 scale (P*8 bytes)]
#              padded up to a multiple of 576.
DIM_NOPE = 448
DIM_ROPE = 64
TILE_SIZE = 64  # one nope scale tile = 64 fp8 values
NUM_SCALE_TILES = DIM_NOPE // TILE_SIZE  # 7
NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2  # 576
PADDED_SCALE_PER_TOKEN = NUM_SCALE_TILES + 1  # 8


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize the DeepSeek v4 paged KV cache for a list of token IDs.

    Args:
        quant_k_cache: (num_pages, bytes_per_page_padded) uint8.
        page_table_1_flattened: (num_tokens,) int — token IDs into the cache.
        page_size: number of tokens per page.
        out: optional (num_tokens, 1, DIM_NOPE + DIM_ROPE) bf16 destination.
            May be a slice of a larger workspace; the kernel uses out.stride(0)
            so contiguous-along-dim-0 slices work.

    Returns:
        (num_tokens, 1, DIM_NOPE + DIM_ROPE) bfloat16.
    """
    assert quant_k_cache.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)

    # The buffer's dtype is whatever the pool exposes (often bf16); the
    # underlying storage is uint8. Reinterpret to byte-space first.
    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_tokens = page_table_1_flattened.shape[0]
    bytes_per_page = quant_k_cache_u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES
    assert DIM_NOPE % 2 == 0 and NOPE_ROPE_BYTES % 2 == 0

    # Three typed views over the same underlying bytes.
    buf_fp8 = quant_k_cache_u8.view(fp8_dtype).reshape(-1)
    buf_bf16 = quant_k_cache_u8.view(torch.bfloat16).reshape(-1)
    buf_uint8 = quant_k_cache_u8.reshape(-1)

    if out is None:
        out = torch.empty(
            (num_tokens, 1, DIM_NOPE + DIM_ROPE),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape == (num_tokens, 1, DIM_NOPE + DIM_ROPE)
        assert out.dtype == torch.bfloat16

    _dequantize_k_cache_paged_kernel[(num_tokens,)](
        out,
        buf_fp8,
        buf_bf16,
        buf_uint8,
        page_table_1_flattened,
        out.stride(0),
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        S_OFFSET_BYTES=s_offset_bytes,
    )
    return out


@triton.jit
def _dequantize_k_cache_paged_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # One program per token: load page_table[token_id] once and emit all
    # NUM_SCALE_TILES nope tiles + rope tail via tl.static_range.
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = tl.load(buf_fp8_ptr + fp8_off).to(tl.float32)

        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))

        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    rope_byte_off = token_data_base + DIM_NOPE
    bf16_off = rope_byte_off // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off)
    tl.store(output_ptr + out_row_base + DIM_NOPE + rope_offs, rope_data)
