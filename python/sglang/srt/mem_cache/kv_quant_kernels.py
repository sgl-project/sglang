"""
Triton kernels for efficient KV cache quantization (INT4/INT8).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantized_set_kv_int8_kernel(
    input_ptr,
    loc_ptr,
    cache_ptr,
    scales_zeros_ptr,
    num_tokens,
    num_heads,
    head_dim,
    input_stride_token,
    input_stride_head,
    input_stride_dim,
    cache_stride_loc,
    cache_stride_head,
    cache_stride_dim,
    sz_stride_loc,
    sz_stride_head,
    sz_stride_dim,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= num_tokens or head_idx >= num_heads:
        return

    # Load cache location for this token
    cache_loc = tl.load(loc_ptr + token_idx)

    # Load entire head dimension
    dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)
    dim_mask = dim_offsets < head_dim

    input_offset = (
        token_idx * input_stride_token
        + head_idx * input_stride_head
        + dim_offsets * input_stride_dim
    )

    # Load values
    vals = tl.load(input_ptr + input_offset, mask=dim_mask, other=0.0).to(tl.float32)

    # Compute min/max across dimension
    val_min = tl.min(vals, axis=0)
    val_max = tl.max(vals, axis=0)

    # Compute scale and zero
    val_range = tl.maximum(val_max - val_min, 1e-8)
    scale = val_range / 255.0
    zero = -val_min / scale

    # Quantize
    q_vals = (vals / scale + zero + 0.5).to(tl.uint8)

    # Store quantized values directly to cache buffer using location
    cache_offset = (
        cache_loc * cache_stride_loc
        + head_idx * cache_stride_head
        + dim_offsets * cache_stride_dim
    )
    tl.store(cache_ptr + cache_offset, q_vals, mask=dim_mask)

    # Store scale and zero to scales_zeros buffer [cache_loc, head_idx, 0/1]
    sz_offset_base = cache_loc * sz_stride_loc + head_idx * sz_stride_head
    tl.store(scales_zeros_ptr + sz_offset_base + 0 * sz_stride_dim, scale)
    tl.store(scales_zeros_ptr + sz_offset_base + 1 * sz_stride_dim, zero)


@triton.jit
def _quantized_set_kv_int4_kernel(
    input_ptr,
    loc_ptr,
    cache_ptr,
    scales_zeros_ptr,
    num_tokens,
    num_heads,
    head_dim,
    input_stride_token,
    input_stride_head,
    input_stride_dim,
    cache_stride_loc,
    cache_stride_head,
    cache_stride_dim,
    sz_stride_loc,
    sz_stride_head,
    sz_stride_dim,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= num_tokens or head_idx >= num_heads:
        return

    cache_loc = tl.load(loc_ptr + token_idx)

    half_dim = head_dim // 2
    dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)

    dim_mask = dim_offsets < half_dim

    input_offset_base = token_idx * input_stride_token + head_idx * input_stride_head

    # Load first and second halves
    # BUG FIX: Must multiply half_dim by stride to get correct memory offset
    vals1 = tl.load(
        input_ptr + input_offset_base + dim_offsets * input_stride_dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    vals2 = tl.load(
        input_ptr + input_offset_base + (dim_offsets + half_dim) * input_stride_dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    # Compute min/max across both halves
    val_min_1 = tl.min(vals1, axis=0)
    val_min_2 = tl.min(vals2, axis=0)
    val_max_1 = tl.max(vals1, axis=0)
    val_max_2 = tl.max(vals2, axis=0)

    val_min = tl.minimum(val_min_1, val_min_2)
    val_max = tl.maximum(val_max_1, val_max_2)

    # Compute scale and zero
    val_range = tl.maximum(val_max - val_min, 1e-8)
    scale = val_range / 15.0
    zero = -val_min / scale

    # Quantize both halves
    q_vals1 = (vals1 / scale + zero + 0.5).to(tl.uint8)
    q_vals2 = (vals2 / scale + zero + 0.5).to(tl.uint8)

    # Pack: lower nibble from vals1, upper nibble from vals2
    packed = q_vals1 | (q_vals2 << 4)

    # Store packed values directly to cache buffer using location
    cache_offset = (
        cache_loc * cache_stride_loc
        + head_idx * cache_stride_head
        + dim_offsets * cache_stride_dim
    )
    tl.store(cache_ptr + cache_offset, packed, mask=dim_mask)

    # Store scale and zero to scales_zeros buffer [cache_loc, head_idx, 0/1]
    sz_offset_base = cache_loc * sz_stride_loc + head_idx * sz_stride_head
    tl.store(scales_zeros_ptr + sz_offset_base + 0 * sz_stride_dim, scale)
    tl.store(scales_zeros_ptr + sz_offset_base + 1 * sz_stride_dim, zero)


def quantized_set_kv_int8_triton(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    k_cache_buffer: torch.Tensor,
    v_cache_buffer: torch.Tensor,
    k_scales_zeros_buffer: torch.Tensor,
    v_scales_zeros_buffer: torch.Tensor,
):
    """
    Quantize K and V caches to INT8 and write directly to cache buffers using Triton kernels.

    Args:
        cache_k: Input K states [num_tokens, num_heads, head_dim]
        cache_v: Input V states [num_tokens, num_heads, head_dim]
        loc: Cache location indices [num_tokens]
        k_cache_buffer: Output K cache [cache_size, num_heads, head_dim]
        v_cache_buffer: Output V cache [cache_size, num_heads, head_dim]
        k_scales_zeros_buffer: K scales/zeros [cache_size, num_heads, 2]
        v_scales_zeros_buffer: V scales/zeros [cache_size, num_heads, 2]

    Returns:
        None (writes directly to buffers)
    """
    num_tokens, num_heads, head_dim = cache_k.shape

    # Launch kernel
    BLOCK_SIZE_DIM = triton.next_power_of_2(head_dim)
    grid = (num_tokens, num_heads)

    # Quantize K
    _quantized_set_kv_int8_kernel[grid](
        cache_k,
        loc,
        k_cache_buffer,
        k_scales_zeros_buffer,
        num_tokens,
        num_heads,
        head_dim,
        cache_k.stride(0),
        cache_k.stride(1),
        cache_k.stride(2),
        k_cache_buffer.stride(0),
        k_cache_buffer.stride(1),
        k_cache_buffer.stride(2),
        k_scales_zeros_buffer.stride(0),
        k_scales_zeros_buffer.stride(1),
        k_scales_zeros_buffer.stride(2),
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )

    # Quantize V
    _quantized_set_kv_int8_kernel[grid](
        cache_v,
        loc,
        v_cache_buffer,
        v_scales_zeros_buffer,
        num_tokens,
        num_heads,
        head_dim,
        cache_v.stride(0),
        cache_v.stride(1),
        cache_v.stride(2),
        v_cache_buffer.stride(0),
        v_cache_buffer.stride(1),
        v_cache_buffer.stride(2),
        v_scales_zeros_buffer.stride(0),
        v_scales_zeros_buffer.stride(1),
        v_scales_zeros_buffer.stride(2),
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )


def quantized_set_kv_int4_triton(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    k_cache_buffer: torch.Tensor,
    v_cache_buffer: torch.Tensor,
    k_scales_zeros_buffer: torch.Tensor,
    v_scales_zeros_buffer: torch.Tensor,
):
    """
    Quantize K and V caches to INT4 and write directly to cache buffers using Triton kernels.

    Args:
        cache_k: Input K states [num_tokens, num_heads, head_dim] (head_dim must be even)
        cache_v: Input V states [num_tokens, num_heads, head_dim]
        loc: Cache location indices [num_tokens]
        k_cache_buffer: Output K cache [cache_size, num_heads, head_dim//2] (packed)
        v_cache_buffer: Output V cache [cache_size, num_heads, head_dim//2] (packed)
        k_scales_zeros_buffer: K scales/zeros [cache_size, num_heads, 2]
        v_scales_zeros_buffer: V scales/zeros [cache_size, num_heads, 2]

    Returns:
        None (writes directly to buffers)
    """
    num_tokens, num_heads, head_dim = cache_k.shape
    assert head_dim % 2 == 0, f"head_dim must be even for INT4, got {head_dim}"

    # Launch kernel
    BLOCK_SIZE_DIM = triton.next_power_of_2(head_dim // 2)
    grid = (num_tokens, num_heads)

    # Quantize K
    _quantized_set_kv_int4_kernel[grid](
        cache_k,
        loc,
        k_cache_buffer,
        k_scales_zeros_buffer,
        num_tokens,
        num_heads,
        head_dim,
        cache_k.stride(0),
        cache_k.stride(1),
        cache_k.stride(2),
        k_cache_buffer.stride(0),
        k_cache_buffer.stride(1),
        k_cache_buffer.stride(2),
        k_scales_zeros_buffer.stride(0),
        k_scales_zeros_buffer.stride(1),
        k_scales_zeros_buffer.stride(2),
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )

    # Quantize V
    _quantized_set_kv_int4_kernel[grid](
        cache_v,
        loc,
        v_cache_buffer,
        v_scales_zeros_buffer,
        num_tokens,
        num_heads,
        head_dim,
        cache_v.stride(0),
        cache_v.stride(1),
        cache_v.stride(2),
        v_cache_buffer.stride(0),
        v_cache_buffer.stride(1),
        v_cache_buffer.stride(2),
        v_scales_zeros_buffer.stride(0),
        v_scales_zeros_buffer.stride(1),
        v_scales_zeros_buffer.stride(2),
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )


# ============================================================================
# Dequantization Triton Kernels
# ============================================================================


@triton.jit
def _dequantize_kv_int8_kernel(
    quantized_ptr,
    scales_zeros_ptr,
    output_ptr,
    cache_size,
    num_heads,
    head_dim,
    quant_stride_cache,
    quant_stride_head,
    quant_stride_dim,
    sz_stride_cache,
    sz_stride_head,
    sz_stride_dim,
    out_stride_cache,
    out_stride_head,
    out_stride_dim,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    cache_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if cache_idx >= cache_size or head_idx >= num_heads:
        return

    # Load scale and zero
    sz_offset_base = cache_idx * sz_stride_cache + head_idx * sz_stride_head
    scale = tl.load(scales_zeros_ptr + sz_offset_base + 0 * sz_stride_dim)
    zero = tl.load(scales_zeros_ptr + sz_offset_base + 1 * sz_stride_dim)

    # Load entire head dimension
    dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)
    dim_mask = dim_offsets < head_dim

    quant_offset = (
        cache_idx * quant_stride_cache
        + head_idx * quant_stride_head
        + dim_offsets * quant_stride_dim
    )

    # Load quantized values (uint8)
    q_vals = tl.load(quantized_ptr + quant_offset, mask=dim_mask, other=0).to(
        tl.float32
    )

    # Dequantize: x = (q - zeros) * scales
    dequant_vals = (q_vals - zero) * scale

    # Store dequantized values
    out_offset = (
        cache_idx * out_stride_cache
        + head_idx * out_stride_head
        + dim_offsets * out_stride_dim
    )
    tl.store(output_ptr + out_offset, dequant_vals, mask=dim_mask)


@triton.jit
def _dequantize_kv_int4_kernel(
    quantized_ptr,
    scales_zeros_ptr,
    output_ptr,
    cache_size,
    num_heads,
    head_dim,
    quant_stride_cache,
    quant_stride_head,
    quant_stride_dim,
    sz_stride_cache,
    sz_stride_head,
    sz_stride_dim,
    out_stride_cache,
    out_stride_head,
    out_stride_dim,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    cache_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if cache_idx >= cache_size or head_idx >= num_heads:
        return

    # Load scale and zero
    sz_offset_base = cache_idx * sz_stride_cache + head_idx * sz_stride_head
    scale = tl.load(scales_zeros_ptr + sz_offset_base + 0 * sz_stride_dim)
    zero = tl.load(scales_zeros_ptr + sz_offset_base + 1 * sz_stride_dim)

    # Load packed values (half the dimension)
    half_dim = head_dim // 2
    dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)
    dim_mask = dim_offsets < half_dim

    quant_offset = (
        cache_idx * quant_stride_cache
        + head_idx * quant_stride_head
        + dim_offsets * quant_stride_dim
    )

    # Load packed uint8 values
    packed_vals = tl.load(quantized_ptr + quant_offset, mask=dim_mask, other=0)

    # Unpack: lower nibble (bits 0-3) and upper nibble (bits 4-7)
    q_vals1 = (packed_vals & 0x0F).to(tl.float32)
    q_vals2 = ((packed_vals >> 4) & 0x0F).to(tl.float32)

    # Dequantize: x = (q - zeros) * scales
    dequant_vals1 = (q_vals1 - zero) * scale
    dequant_vals2 = (q_vals2 - zero) * scale

    # Store dequantized values (first half)
    out_offset_1 = (
        cache_idx * out_stride_cache
        + head_idx * out_stride_head
        + dim_offsets * out_stride_dim
    )
    tl.store(output_ptr + out_offset_1, dequant_vals1, mask=dim_mask)

    # Store dequantized values (second half)
    out_offset_2 = out_offset_1 + half_dim * out_stride_dim
    tl.store(output_ptr + out_offset_2, dequant_vals2, mask=dim_mask)


def dequantize_kv_int8_triton(
    quantized: torch.Tensor,
    scales_zeros: torch.Tensor,
    head_dim: int,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequantize INT8 KV cache to float using Triton kernel.

    Args:
        quantized: uint8 tensor [cache_size, num_heads, head_dim]
        scales_zeros: float tensor [cache_size, num_heads, 2] where [..., 0]=scale, [..., 1]=zero
        head_dim: dimension of each head
        model_dtype: target dtype for output

    Returns:
        dequantized: float tensor [cache_size, num_heads, head_dim] in model_dtype
    """
    cache_size, num_heads, _ = quantized.shape

    # Allocate output directly in model_dtype
    output = torch.empty(
        (cache_size, num_heads, head_dim), dtype=model_dtype, device=quantized.device
    )

    # Launch kernel
    BLOCK_SIZE_DIM = triton.next_power_of_2(head_dim)
    grid = (cache_size, num_heads)

    _dequantize_kv_int8_kernel[grid](
        quantized,
        scales_zeros,
        output,
        cache_size,
        num_heads,
        head_dim,
        quantized.stride(0),
        quantized.stride(1),
        quantized.stride(2),
        scales_zeros.stride(0),
        scales_zeros.stride(1),
        scales_zeros.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )

    return output


def dequantize_kv_int4_triton(
    quantized: torch.Tensor,
    scales_zeros: torch.Tensor,
    head_dim: int,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequantize INT4 KV cache to float using Triton kernel.

    Args:
        quantized: uint8 tensor (packed) [cache_size, num_heads, head_dim//2]
        scales_zeros: float tensor [cache_size, num_heads, 2] where [..., 0]=scale, [..., 1]=zero
        head_dim: original head dimension (must be even)
        model_dtype: target dtype for output

    Returns:
        dequantized: float tensor [cache_size, num_heads, head_dim] in model_dtype
    """
    cache_size, num_heads, _ = quantized.shape
    assert head_dim % 2 == 0, f"head_dim must be even for INT4, got {head_dim}"

    # Allocate output directly in model_dtype
    output = torch.empty(
        (cache_size, num_heads, head_dim), dtype=model_dtype, device=quantized.device
    )

    # Launch kernel
    BLOCK_SIZE_DIM = triton.next_power_of_2(head_dim // 2)
    grid = (cache_size, num_heads)

    _dequantize_kv_int4_kernel[grid](
        quantized,
        scales_zeros,
        output,
        cache_size,
        num_heads,
        head_dim,
        quantized.stride(0),
        quantized.stride(1),
        quantized.stride(2),
        scales_zeros.stride(0),
        scales_zeros.stride(1),
        scales_zeros.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )

    return output
