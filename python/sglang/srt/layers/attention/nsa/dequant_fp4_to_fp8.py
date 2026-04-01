"""FP4 (E2M1) paged KV cache → FP8 dequantization for NSA.

Vectorized 1D kernel: one program per token processes all 256 packed
nope bytes + 64 rope elements, storing FP8 directly (no BF16 intermediate).

Optimal config determined by bench_fp4_kernels.py on MI355:
  warps=1, stages=0  →  2719 GB/s @ 131k tokens (4.4× over 2D grid)
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.mem_cache.utils import (
    FP4_BLOCK_SIZE,
    FP4_NOPE_DIM,
    FP4_NUM_SCALE_BLOCKS,
    FP4_PACKED_NOPE_BYTES,
    FP4_ROPE_DIM,
)

_is_fp8_fnuz = is_fp8_fnuz()


@triton.jit
def _e2m1_dequant(fp4_code, scale):
    """Decode 4-bit E2M1 nibbles to float32, scaled (vectorized)."""
    mag = (fp4_code & 0x07).to(tl.int32)
    sign_bit = ((fp4_code >> 3) & 1).to(tl.int32)

    exp_field = mag >> 1
    mant_bit = (mag & 1).to(tl.float32)

    is_subnormal = exp_field == 0
    sub_val = 0.5 * mant_bit
    norm_val = tl.math.exp2((exp_field - 1).to(tl.float32)) * (1.0 + 0.5 * mant_bit)

    float_val = tl.where(is_subnormal, sub_val, norm_val) * scale
    return tl.where(sign_bit != 0, -float_val, float_val)


@triton.jit
def _dequant_fp4_to_fp8_paged_kernel(
    output_ptr,
    in_packed_ptr,
    in_scale_ptr,
    in_rope_ptr,
    page_table_ptr,
    output_stride_0: int,
    in_packed_stride_0: int,
    in_scale_stride_0: int,
    in_rope_stride_0: int,
    PACKED_NOPE_BYTES: tl.constexpr,
    NUM_SCALE_BLOCKS: tl.constexpr,
    HALF_BLK: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
):
    """One program per token: load FP4, dequant, store FP8 directly."""
    token_id = tl.program_id(0)

    src_token_id = tl.load(page_table_ptr + token_id)
    src_token_id = tl.maximum(src_token_id, 0)

    pack_offs = tl.arange(0, PACKED_NOPE_BYTES)
    packed = tl.load(in_packed_ptr + src_token_id * in_packed_stride_0 + pack_offs)

    block_ids = pack_offs // HALF_BLK
    scale_uint8 = tl.load(in_scale_ptr + src_token_id * in_scale_stride_0 + block_ids)
    scale = tl.math.exp2(scale_uint8.to(tl.float32) - 127.0)

    low_float = _e2m1_dequant(packed & 0x0F, scale)
    high_float = _e2m1_dequant((packed >> 4) & 0x0F, scale)

    local_offs = pack_offs % HALF_BLK
    block_start = block_ids * SCALE_BLOCK_SIZE
    even_offs = block_start + local_offs * 2
    odd_offs = even_offs + 1

    out_base = output_ptr + token_id * output_stride_0
    if IS_FNUZ:
        tl.store(out_base + even_offs, low_float.to(tl.float8e4b8))
        tl.store(out_base + odd_offs, high_float.to(tl.float8e4b8))
    else:
        tl.store(out_base + even_offs, low_float.to(tl.float8e4nv))
        tl.store(out_base + odd_offs, high_float.to(tl.float8e4nv))

    rope_offs = tl.arange(0, DIM_ROPE)
    rope_data = tl.load(in_rope_ptr + src_token_id * in_rope_stride_0 + rope_offs)
    if IS_FNUZ:
        tl.store(
            out_base + DIM_NOPE + rope_offs, rope_data.to(tl.float32).to(tl.float8e4b8)
        )
    else:
        tl.store(
            out_base + DIM_NOPE + rope_offs, rope_data.to(tl.float32).to(tl.float8e4nv)
        )


FP8_TOTAL_DIM = FP4_NOPE_DIM + FP4_ROPE_DIM  # 576


def dequantize_fp4_to_fp8_paged(
    fp4_buffer: torch.Tensor,
    page_indices: torch.Tensor,
    fp8_dtype: torch.dtype,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather FP4 pages and dequantize to a contiguous FP8 buffer.

    If *output* is provided it must have shape (>=num_tokens, 1, FP8_TOTAL_DIM)
    and the correct FP8 dtype.  The slice [:num_tokens] will be written.
    """
    num_tokens = page_indices.shape[0]
    if num_tokens == 0:
        if output is not None:
            return output[:0]
        return torch.empty(
            (0, 1, FP8_TOTAL_DIM), dtype=fp8_dtype, device=fp4_buffer.device
        )

    if output is not None and output.shape[0] >= num_tokens:
        out = output[:num_tokens]
    else:
        out = torch.empty(
            (num_tokens, 1, FP8_TOTAL_DIM),
            dtype=fp8_dtype,
            device=fp4_buffer.device,
        )

    quant_flat = fp4_buffer.view(-1, fp4_buffer.shape[-1])

    in_packed = quant_flat[:, :FP4_PACKED_NOPE_BYTES]
    in_scale = quant_flat[
        :, FP4_PACKED_NOPE_BYTES : FP4_PACKED_NOPE_BYTES + FP4_NUM_SCALE_BLOCKS
    ]
    in_rope = quant_flat[:, FP4_PACKED_NOPE_BYTES + FP4_NUM_SCALE_BLOCKS :].view(
        torch.bfloat16
    )

    out_flat = out.view(-1, FP8_TOTAL_DIM)

    _dequant_fp4_to_fp8_paged_kernel[(num_tokens,)](
        out_flat,
        in_packed,
        in_scale,
        in_rope,
        page_indices,
        out_flat.stride(0),
        in_packed.stride(0),
        in_scale.stride(0),
        in_rope.stride(0),
        PACKED_NOPE_BYTES=FP4_PACKED_NOPE_BYTES,
        NUM_SCALE_BLOCKS=FP4_NUM_SCALE_BLOCKS,
        HALF_BLK=FP4_BLOCK_SIZE // 2,
        SCALE_BLOCK_SIZE=FP4_BLOCK_SIZE,
        DIM_NOPE=FP4_NOPE_DIM,
        DIM_ROPE=FP4_ROPE_DIM,
        IS_FNUZ=_is_fp8_fnuz,
        num_warps=1,
        num_stages=0,
    )

    return out


def get_fp8_dtype_for_dequant() -> torch.dtype:
    """Return the FP8 dtype matching the TileLang kernel compilation."""
    if _is_fp8_fnuz:
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def dequant_fp4_paged_decode(
    kv_cache_fp4: torch.Tensor,
    page_table_1: torch.Tensor,
    fp8_workspace: torch.Tensor | None = None,
    arange_buf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequant FP4→FP8 for decode (small page_table, CUDA-graph safe).

    Pre-allocated buffers avoid per-step torch.empty / torch.arange:
      fp8_workspace – (max_entries, 1, 576) FP8, reused across steps
      arange_buf    – int32 arange(0..max), e.g. _arange_buf from backend

    Returns (kv_fp8, new_page_table_1).
    """
    assert (
        page_table_1.ndim == 2
    ), f"page_table_1 must be 2D (batch, topk), got shape {page_table_1.shape}"
    batch_size = page_table_1.shape[0]
    topk = page_table_1.shape[1]
    num_entries = batch_size * topk
    flat_indices = page_table_1.reshape(-1)

    fp8_dtype = get_fp8_dtype_for_dequant()
    temp_kv_fp8 = dequantize_fp4_to_fp8_paged(
        kv_cache_fp4,
        flat_indices,
        fp8_dtype,
        output=fp8_workspace,
    )

    if arange_buf is not None and arange_buf.numel() >= num_entries:
        sequential = arange_buf[:num_entries].reshape(batch_size, topk)
    else:
        sequential = torch.arange(
            num_entries, device=page_table_1.device, dtype=page_table_1.dtype
        ).reshape(batch_size, topk)

    valid_mask = page_table_1 >= 0
    new_page_table_1 = torch.where(valid_mask, sequential, page_table_1)

    return temp_kv_fp8, new_page_table_1


def dequant_fp4_paged_extend(
    kv_cache_fp4: torch.Tensor,
    page_table_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequant FP4→FP8 for extend (large page_table, uses unique indices).

    For extend, page_table_1 can be (num_q_tokens, topk) e.g. (8192, 2048)
    = 16M entries.  We deduplicate with torch.unique so we only dequant
    each physical page once, keeping memory bounded.

    Returns (kv_fp8, new_page_table_1).
    """
    fp8_dtype = get_fp8_dtype_for_dequant()
    orig_shape = page_table_1.shape
    flat_indices = page_table_1.reshape(-1)

    valid_mask = flat_indices >= 0
    valid_indices = flat_indices[valid_mask]

    unique_indices, inverse = torch.unique(valid_indices, return_inverse=True)

    temp_kv_fp8 = dequantize_fp4_to_fp8_paged(kv_cache_fp4, unique_indices, fp8_dtype)

    new_flat = torch.full_like(flat_indices, -1)
    new_flat[valid_mask] = inverse.to(flat_indices.dtype)
    new_page_table_1 = new_flat.reshape(orig_shape)

    return temp_kv_fp8, new_page_table_1
