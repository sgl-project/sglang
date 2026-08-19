from __future__ import annotations

import torch
import triton
import triton.language as tl

_PAGE_SIZE = 64
_HEAD_DIM = 128
_SCALE_BYTES = 4
_PACKED_PAGE_BYTES = _PAGE_SIZE * (_HEAD_DIM + _SCALE_BYTES)
_SCALE_OFFSET_BYTES = _PAGE_SIZE * _HEAD_DIM
_HEAD_TILE = 16


@triton.jit
def _decode_e4m3fn_bytes(value):
    """Decode raw IEEE-like E4M3FN bytes without using a Triton FP8 dtype.

    E4M3FN reserves exponent=15, mantissa=7 for NaN (both signs).  Keeping
    this logic shared by the test kernel and the paged-MQA kernel prevents the
    exhaustive decoder test from validating a separate implementation.
    """

    bits = value.to(tl.uint32)
    sign = (bits >> 7) & 1
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7

    mantissa_f32 = mantissa.to(tl.float32)
    subnormal = mantissa_f32 * (1.0 / 512.0)
    normal = (1.0 + mantissa_f32 * 0.125) * tl.math.exp2(exponent.to(tl.float32) - 7.0)
    magnitude = tl.where(exponent == 0, subnormal, normal)
    # Apply the sign bit with a bitcast so 0x80 remains negative zero.  An
    # arithmetic select/multiply is allowed to canonicalize it to positive zero.
    magnitude_bits = magnitude.to(tl.uint32, bitcast=True)
    decoded = (magnitude_bits | (sign << 31)).to(tl.float32, bitcast=True)

    is_nan = (exponent == 0xF) & (mantissa == 0x7)
    return tl.where(is_nan, float("nan"), decoded)


@triton.jit
def _decode_e4m3fn_kernel(input_ptr, output_ptr, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    encoded = tl.load(input_ptr + offsets, mask=mask, other=0)
    decoded = _decode_e4m3fn_bytes(encoded)
    tl.store(output_ptr + offsets, decoded, mask=mask)


@triton.jit
def _paged_mqa_logits_kernel(
    q_u8_ptr,
    k_u8_ptr,
    k_scale_ptr,
    weights_ptr,
    seq_lens_ptr,
    page_table_ptr,
    logits_ptr,
    num_physical_pages,
    num_logical_pages,
    max_seq_len,
    q_stride_t,
    q_stride_h,
    weight_stride_t,
    weight_stride_h,
    page_table_stride_t,
    page_table_stride_page,
    k_page_stride_bytes,
    k_scale_page_stride,
    logits_stride_t,
    NUM_HEADS: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_idx = tl.program_id(0)
    logical_page = tl.program_id(1)

    token_offsets = tl.arange(0, PAGE_SIZE)
    dim_offsets = tl.arange(0, HEAD_DIM)
    logical_positions = logical_page * PAGE_SIZE + token_offsets

    page_in_table = logical_page < num_logical_pages
    physical_page = tl.load(
        page_table_ptr
        + query_idx * page_table_stride_t
        + logical_page * page_table_stride_page,
        mask=page_in_table,
        other=-1,
    )
    valid_page = (
        page_in_table & (physical_page >= 0) & (physical_page < num_physical_pages)
    )
    safe_page = tl.where(valid_page, physical_page, 0).to(tl.int64)

    k_byte_offsets = (
        safe_page * k_page_stride_bytes
        + token_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )
    k_encoded = tl.load(k_u8_ptr + k_byte_offsets, mask=valid_page, other=0)
    # Every E4M3FN finite value is exactly representable in BF16.  The BF16
    # conversion enables SM80 tensor-core dot while tl.dot accumulates FP32.
    k = _decode_e4m3fn_bytes(k_encoded).to(tl.bfloat16)

    scores = tl.zeros((PAGE_SIZE,), dtype=tl.float32)
    for head_start in tl.static_range(0, NUM_HEADS, HEAD_TILE):
        head_offsets = head_start + tl.arange(0, HEAD_TILE)
        head_mask = head_offsets < NUM_HEADS
        q_encoded = tl.load(
            q_u8_ptr
            + query_idx * q_stride_t
            + head_offsets[:, None] * q_stride_h
            + dim_offsets[None, :],
            mask=head_mask[:, None],
            other=0,
        )
        q = _decode_e4m3fn_bytes(q_encoded).to(tl.bfloat16)
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk = tl.maximum(qk, 0.0, propagate_nan=tl.PropagateNan.ALL)
        head_weights = tl.load(
            weights_ptr + query_idx * weight_stride_t + head_offsets * weight_stride_h,
            mask=head_mask,
            other=0.0,
        )
        scores += tl.sum(qk * head_weights[:, None], axis=0)

    k_scales = tl.load(
        k_scale_ptr + safe_page * k_scale_page_stride + token_offsets,
        mask=valid_page,
        other=0.0,
    )
    seq_len = tl.load(seq_lens_ptr + query_idx)
    valid_tokens = (
        valid_page & (logical_positions < seq_len) & (logical_positions < max_seq_len)
    )
    output = tl.where(valid_tokens, scores * k_scales, 0.0)
    tl.store(
        logits_ptr + query_idx * logits_stride_t + logical_positions,
        output,
        mask=logical_positions < max_seq_len,
    )


def _require_sm80(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError(
            "Triton DSA paged-MQA is CUDA-only and requires NVIDIA SM80; "
            f"got device={device}."
        )
    capability = torch.cuda.get_device_capability(device)
    if capability != (8, 0):
        raise ValueError(
            "Triton DSA paged-MQA currently supports only NVIDIA SM80 "
            f"(compute capability 8.0); got {capability[0]}.{capability[1]}."
        )


def triton_decode_e4m3fn(encoded: torch.Tensor) -> torch.Tensor:
    """Decode raw E4M3FN bytes on device using the runtime kernel's decoder."""

    _require_sm80(encoded.device)
    if encoded.dtype != torch.uint8:
        raise ValueError(
            "triton_decode_e4m3fn expects a torch.uint8 tensor, "
            f"got dtype={encoded.dtype}."
        )
    if not encoded.is_contiguous():
        raise ValueError("triton_decode_e4m3fn expects a contiguous input tensor.")

    output = torch.empty_like(encoded, dtype=torch.float32)
    if encoded.numel() == 0:
        return output
    block = 256
    _decode_e4m3fn_kernel[(triton.cdiv(encoded.numel(), block),)](
        encoded,
        output,
        encoded.numel(),
        BLOCK=block,
        num_warps=4,
    )
    return output


def triton_paged_mqa_logits(
    q_fp8: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Compute DSA paged-MQA logits on NVIDIA A100 with a fixed page grid.

    The query and packed cache are passed to Triton as uint8 views.  FP8 is
    decoded in software, converted to BF16 for SM80 tensor-core dot products,
    and accumulated in FP32.  E4M3FN NaN patterns propagate to the output;
    normal runtime quantization is expected not to produce them.
    """

    _require_sm80(q_fp8.device)
    if q_fp8.ndim != 3 or q_fp8.shape[-1] != _HEAD_DIM:
        raise ValueError(
            "Triton DSA paged-MQA expects q_fp8 shaped [T, H, 128], "
            f"got {tuple(q_fp8.shape)}."
        )
    if q_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "Triton DSA paged-MQA expects torch.float8_e4m3fn queries, "
            f"got dtype={q_fp8.dtype}."
        )
    if weights.shape != q_fp8.shape[:2] or weights.dtype != torch.float32:
        raise ValueError(
            "Triton DSA paged-MQA expects FP32 weights shaped [T, H], "
            f"got shape={tuple(weights.shape)}, dtype={weights.dtype}."
        )
    if seq_lens.ndim != 1 or seq_lens.shape[0] != q_fp8.shape[0]:
        raise ValueError(
            "Triton DSA paged-MQA expects seq_lens shaped [T], "
            f"got {tuple(seq_lens.shape)} for T={q_fp8.shape[0]}."
        )
    if seq_lens.dtype != torch.int32:
        raise ValueError(
            "Triton DSA paged-MQA expects INT32 seq_lens, "
            f"got dtype={seq_lens.dtype}."
        )
    if page_table.ndim != 2 or page_table.shape[0] != q_fp8.shape[0]:
        raise ValueError(
            "Triton DSA paged-MQA expects page_table shaped [T, L], "
            f"got {tuple(page_table.shape)} for T={q_fp8.shape[0]}."
        )
    if page_table.dtype != torch.int32:
        raise ValueError(
            "Triton DSA paged-MQA expects an INT32 page_table, "
            f"got dtype={page_table.dtype}."
        )
    if not isinstance(max_seq_len, int) or isinstance(max_seq_len, bool):
        raise TypeError(
            f"max_seq_len must be a host integer, got {type(max_seq_len).__name__}."
        )
    if max_seq_len < 0:
        raise ValueError(f"max_seq_len must be non-negative, got {max_seq_len}.")

    tensors = {
        "q_fp8": q_fp8,
        "k_cache": k_cache,
        "weights": weights,
        "seq_lens": seq_lens,
        "page_table": page_table,
    }
    for name, tensor in tensors.items():
        if tensor.device != q_fp8.device:
            raise ValueError(
                "Triton DSA paged-MQA expects every input on the query device; "
                f"{name} is on {tensor.device}, query is on {q_fp8.device}."
            )
        if not tensor.is_contiguous():
            raise ValueError(
                f"Triton DSA paged-MQA expects contiguous {name}; "
                f"got strides={tensor.stride()}."
            )

    if k_cache.ndim < 2 or k_cache.shape[0] == 0:
        raise ValueError(
            "Triton DSA paged-MQA expects at least one physical cache page."
        )
    if k_cache.element_size() != 1:
        raise ValueError(
            "Triton DSA paged-MQA expects a byte-addressable packed cache, "
            f"got dtype={k_cache.dtype}."
        )
    page_elements = k_cache[0].numel()
    if page_elements != _PACKED_PAGE_BYTES:
        raise ValueError(
            "Triton DSA paged-MQA expects packed 64-token pages with "
            f"{_PACKED_PAGE_BYTES} bytes, got {page_elements}."
        )

    num_queries, num_heads, _ = q_fp8.shape
    logits = torch.zeros(
        (num_queries, max_seq_len), dtype=torch.float32, device=q_fp8.device
    )
    if num_queries == 0 or max_seq_len == 0:
        return logits

    q_u8 = q_fp8.view(torch.uint8)
    packed_u8 = k_cache.view(torch.uint8).reshape(k_cache.shape[0], -1)
    k_scales = packed_u8[:, _SCALE_OFFSET_BYTES:].view(torch.float32)

    grid = (num_queries, triton.cdiv(max_seq_len, _PAGE_SIZE))
    _paged_mqa_logits_kernel[grid](
        q_u8,
        packed_u8,
        k_scales,
        weights,
        seq_lens,
        page_table,
        logits,
        k_cache.shape[0],
        page_table.shape[1],
        max_seq_len,
        q_u8.stride(0),
        q_u8.stride(1),
        weights.stride(0),
        weights.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        packed_u8.stride(0),
        k_scales.stride(0),
        logits.stride(0),
        NUM_HEADS=num_heads,
        HEAD_TILE=_HEAD_TILE,
        PAGE_SIZE=_PAGE_SIZE,
        HEAD_DIM=_HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )
    return logits


__all__ = ["triton_decode_e4m3fn", "triton_paged_mqa_logits"]
