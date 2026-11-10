"""Software FP8 helpers for Volta attention kernels.

SM70 can store FP8 bytes, but PTX/Triton cannot use native FP8 operand types on
this architecture. V100 attention kernels therefore receive the cache as raw
uint8 bytes, decode to FP32, and use FP16 tensor-core MMA.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fp8_e4m3fn_to_fp32(raw):
    """Decode raw IEEE-like E4M3FN bytes, including subnormals.

    E4M3FN uses exponent bias 7. Exponent 15 remains finite for mantissas 0-6
    (up to 448); the all-ones mantissa is NaN and is never emitted by SGLang's
    finite KV-cache cast.
    """
    raw = raw.to(tl.int32)
    sign = tl.where((raw & 0x80) != 0, -1.0, 1.0)
    exponent = (raw >> 3) & 0x0F
    mantissa = raw & 0x07
    subnormal = mantissa.to(tl.float32) * (2.0**-9)
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(
        exponent.to(tl.float32) - 7.0
    )
    return sign * tl.where(exponent == 0, subnormal, normal)


@triton.jit
def fp8_e5m2_to_fp32(raw):
    """Decode raw IEEE E5M2 bytes, including subnormals.

    SGLang's SM70 cache writer saturates finite FP16 inputs, so attention does
    not encounter the exponent-31 Inf/NaN encodings in normal operation.
    """
    raw = raw.to(tl.int32)
    sign = tl.where((raw & 0x80) != 0, -1.0, 1.0)
    exponent = (raw >> 2) & 0x1F
    mantissa = raw & 0x03
    subnormal = mantissa.to(tl.float32) * (2.0**-16)
    normal = (1.0 + mantissa.to(tl.float32) * 0.25) * tl.exp2(
        exponent.to(tl.float32) - 15.0
    )
    return sign * tl.where(exponent == 0, subnormal, normal)


@triton.jit
def fp8_sm70_to_fp32(raw, is_e5m2: tl.constexpr):
    if is_e5m2:
        return fp8_e5m2_to_fp32(raw)
    return fp8_e4m3fn_to_fp32(raw)


@triton.jit
def _dequantize_paged_kv_e4m3_kernel(
    k_cache,
    v_cache,
    block_table,
    seq_lens,
    k_out,
    v_out,
    input_page_stride,
    input_token_stride,
    block_table_stride,
    output_page_stride,
    output_token_stride,
    active_pages,
    max_pages,
    page_size: tl.constexpr,
    values_per_token: tl.constexpr,
    block_values: tl.constexpr,
):
    """Materialize logical FP8 pages as contiguous FP16 pages."""
    logical_page_row = tl.program_id(0)
    value_block = tl.program_id(1)
    batch_id = logical_page_row // active_pages
    logical_page = logical_page_row - batch_id * active_pages

    physical_page = tl.load(block_table + batch_id * block_table_stride + logical_page)
    value_offsets = value_block * block_values + tl.arange(0, block_values)
    token_offsets = value_offsets // values_per_token
    values_in_token = value_offsets - token_offsets * values_per_token
    valid_value = value_offsets < page_size * values_per_token
    valid_token = logical_page * page_size + token_offsets < tl.load(
        seq_lens + batch_id
    )
    mask = valid_value & valid_token

    input_offsets = (
        physical_page * input_page_stride
        + token_offsets * input_token_stride
        + values_in_token
    )
    output_page = batch_id * max_pages + logical_page
    output_offsets = (
        output_page * output_page_stride
        + token_offsets * output_token_stride
        + values_in_token
    )

    k_raw = tl.load(k_cache + input_offsets, mask=mask, other=0)
    v_raw = tl.load(v_cache + input_offsets, mask=mask, other=0)
    tl.store(
        k_out + output_offsets,
        fp8_e4m3fn_to_fp32(k_raw),
        mask=mask,
    )
    tl.store(
        v_out + output_offsets,
        fp8_e4m3fn_to_fp32(v_raw),
        mask=mask,
    )


def dequantize_paged_kv_e4m3_sm70(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Dequantize active logical cache pages into reusable FP16 scratch.

    The grouped DFlash verifier reuses every K/V tile across several Q heads.
    Converting once here is cheaper than expanding every cache byte inside
    each GQA attention CTA on SM70.
    """
    if k_cache.dtype != torch.float8_e4m3fn:
        raise TypeError(f"Expected E4M3 K cache, got {k_cache.dtype}.")
    if v_cache.dtype != torch.float8_e4m3fn:
        raise TypeError(f"Expected E4M3 V cache, got {v_cache.dtype}.")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V cache shapes must match.")
    if k_cache.ndim != 4 or k_out.ndim != 4 or v_out.ndim != 4:
        raise ValueError("Paged K/V tensors must use [page, token, head, dim].")
    if k_out.dtype != torch.float16 or v_out.dtype != torch.float16:
        raise TypeError("SM70 verifier scratch must use FP16.")

    batch_size = block_table.shape[0]
    page_size = k_cache.shape[1]
    active_pages = max(1, triton.cdiv(max_seq_len, page_size))
    max_pages = k_out.shape[0] // batch_size
    if active_pages > max_pages:
        raise ValueError(
            f"Active page count {active_pages} exceeds scratch capacity {max_pages}."
        )

    values_per_token = k_cache.shape[2] * k_cache.shape[3]
    block_values = 256
    grid = (
        batch_size * active_pages,
        triton.cdiv(page_size * values_per_token, block_values),
    )
    _dequantize_paged_kv_e4m3_kernel[grid](
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        block_table,
        seq_lens,
        k_out,
        v_out,
        k_cache.stride(0),
        k_cache.stride(1),
        block_table.stride(0),
        k_out.stride(0),
        k_out.stride(1),
        active_pages,
        max_pages,
        page_size=page_size,
        values_per_token=values_per_token,
        block_values=block_values,
        num_warps=4,
    )


@triton.jit
def _store_paged_extend_kv_fp16_kernel(
    k_extend,
    v_extend,
    qo_indptr,
    prefix_lens,
    k_out,
    v_out,
    input_token_stride,
    output_token_stride,
    max_pages,
    page_size: tl.constexpr,
    values_per_token: tl.constexpr,
    block_values: tl.constexpr,
    k_scale,
    v_scale,
):
    batch_id = tl.program_id(0)
    token_offset = tl.program_id(1)
    query_start = tl.load(qo_indptr + batch_id)
    query_end = tl.load(qo_indptr + batch_id + 1)
    input_token = query_start + token_offset

    value_offsets = tl.arange(0, block_values)
    mask = (input_token < query_end) & (value_offsets < values_per_token)
    logical_token = (
        batch_id * max_pages * page_size
        + tl.load(prefix_lens + batch_id)
        + token_offset
    )
    input_offsets = input_token * input_token_stride + value_offsets
    output_offsets = logical_token * output_token_stride + value_offsets

    k = tl.load(k_extend + input_offsets, mask=mask, other=0.0)
    v = tl.load(v_extend + input_offsets, mask=mask, other=0.0)
    # The native attention kernel applies the cache scales to every K/V value.
    # Store fresh extend values in the same pre-scale domain as decoded cache
    # bytes so they retain the canonical mixed-precision extend contract.
    tl.store(k_out + output_offsets, k / k_scale, mask=mask)
    tl.store(v_out + output_offsets, v / v_scale, mask=mask)


def store_paged_extend_kv_fp16_sm70(
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    qo_indptr: torch.Tensor,
    prefix_lens: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    max_extend_len: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Overlay batched fresh K/V into logical FP16 page scratch.

    ``k_out`` and ``v_out`` use ``[batch * max_pages, page, head, dim]``.
    Each batch row owns a fixed ``max_pages``-page region, matching the page
    table produced by the V100 paged-prefill backend.
    """
    if k_extend.dtype != torch.float16 or v_extend.dtype != torch.float16:
        raise TypeError("Extend K/V inputs must be FP16 on SM70.")
    if k_extend.shape != v_extend.shape:
        raise ValueError("Extend K/V input shapes must match.")
    if k_extend.ndim != 3 or k_out.ndim != 4 or v_out.ndim != 4:
        raise ValueError(
            "Expected extend K/V [token, head, dim] and scratch "
            "[batch * page, token, head, dim]."
        )
    if k_out.dtype != torch.float16 or v_out.dtype != torch.float16:
        raise TypeError("SM70 prefill scratch must use FP16.")
    if qo_indptr.ndim != 1 or prefix_lens.ndim != 1:
        raise ValueError("qo_indptr and prefix_lens must be one-dimensional.")
    batch_size = prefix_lens.numel()
    if qo_indptr.numel() != batch_size + 1:
        raise ValueError("qo_indptr must contain batch_size + 1 entries.")
    if k_extend.shape[1:] != k_out.shape[2:]:
        raise ValueError("Extend and scratch K/V head shapes must match.")
    if k_out.shape != v_out.shape or k_out.shape[0] % batch_size != 0:
        raise ValueError("K/V scratch shapes must match and divide by batch size.")
    if k_scale == 0.0 or v_scale == 0.0:
        raise ValueError("K/V cache scales must be non-zero.")

    values_per_token = k_extend.shape[1] * k_extend.shape[2]
    block_values = triton.next_power_of_2(values_per_token)
    max_pages = k_out.shape[0] // batch_size
    _store_paged_extend_kv_fp16_kernel[(batch_size, max_extend_len)](
        k_extend,
        v_extend,
        qo_indptr,
        prefix_lens,
        k_out,
        v_out,
        k_extend.stride(0),
        k_out.stride(1),
        max_pages,
        page_size=k_out.shape[1],
        values_per_token=values_per_token,
        block_values=block_values,
        k_scale=k_scale,
        v_scale=v_scale,
        num_warps=4,
    )


@triton.jit
def _store_linear_verify_kv_fp16_kernel(
    k_extend,
    v_extend,
    prefix_lens,
    k_out,
    v_out,
    input_token_stride,
    output_token_stride,
    num_tokens,
    values_per_token: tl.constexpr,
    block_values: tl.constexpr,
    k_scale,
    v_scale,
):
    token_id = tl.program_id(0)
    value_offsets = tl.arange(0, block_values)
    mask = (token_id < num_tokens) & (value_offsets < values_per_token)
    logical_token = tl.load(prefix_lens) + token_id
    input_offsets = token_id * input_token_stride + value_offsets
    output_offsets = logical_token * output_token_stride + value_offsets

    k = tl.load(k_extend + input_offsets, mask=mask, other=0.0)
    v = tl.load(v_extend + input_offsets, mask=mask, other=0.0)
    tl.store(k_out + output_offsets, k / k_scale, mask=mask)
    tl.store(v_out + output_offsets, v / v_scale, mask=mask)


def store_linear_verify_kv_fp16_sm70(
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    prefix_lens: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Overlay a bs=1 speculative block's fresh FP16 K/V onto scratch.

    SGLang's canonical extend kernel consumes a quantized cached prefix but
    keeps the current extend block in FP16.  Re-reading the newly stored block
    from an FP8 cache changes model tokens, so native linear verification must
    reproduce that mixed-precision contract.
    """
    if k_extend.dtype != torch.float16 or v_extend.dtype != torch.float16:
        raise TypeError("Linear-verify K/V inputs must be FP16 on SM70.")
    if k_extend.shape != v_extend.shape:
        raise ValueError("Linear-verify K/V input shapes must match.")
    if k_extend.ndim != 3 or k_out.ndim != 4 or v_out.ndim != 4:
        raise ValueError(
            "Expected extend K/V [token, head, dim] and scratch "
            "[page, token, head, dim]."
        )
    if k_out.dtype != torch.float16 or v_out.dtype != torch.float16:
        raise TypeError("Linear-verify scratch must be FP16.")
    if prefix_lens.numel() != 1:
        raise ValueError("The SM70 FP8 scratch verifier currently requires bs=1.")
    if k_extend.shape[1:] != k_out.shape[2:]:
        raise ValueError("Extend and scratch K/V head shapes must match.")
    if k_scale == 0.0 or v_scale == 0.0:
        raise ValueError("K/V cache scales must be non-zero.")

    values_per_token = k_extend.shape[1] * k_extend.shape[2]
    block_values = triton.next_power_of_2(values_per_token)
    _store_linear_verify_kv_fp16_kernel[(k_extend.shape[0],)](
        k_extend,
        v_extend,
        prefix_lens,
        k_out,
        v_out,
        k_extend.stride(0),
        k_out.stride(1),
        k_extend.shape[0],
        values_per_token=values_per_token,
        block_values=block_values,
        k_scale=k_scale,
        v_scale=v_scale,
        num_warps=4,
    )
