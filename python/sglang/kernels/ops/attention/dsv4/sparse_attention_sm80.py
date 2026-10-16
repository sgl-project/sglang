"""Isolated V4.1 sparse MLA attention on CUDA SM80, correctness first.

NOT OPTIMIZED: one Triton program per (batch, head), serial over candidates.
This is not runtime dispatch or complete model support; integrating a model
still requires an indexer/compressor and already causal/window-filtered IDs.
Only raw uint8 cache pointers are used, never FP8 pointers or CPU lookup tables.
"""

import math
import numbers

import torch
import triton
import triton.language as tl

from .dequant_k_cache_sm80 import (
    _e2m1_to_fp32,
    _e4m3_to_fp32,
    _scaled_bf16,
    _ue8m0_to_fp32,
)
from .kv_layout import KVLayout


@triton.jit
def _accumulate_candidates(
    q_values,
    maximum,
    denominator,
    numerator,
    cache_ptr,
    indices_ptr,
    length_ptr,
    batch,
    softmax_scale,
    K: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PITCH: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    IS_FP4: tl.constexpr,
):
    tl.static_assert(cache_ptr.dtype.element_ty == tl.uint8)
    offsets = tl.arange(0, 512).to(tl.int64)
    length = tl.full((), K, tl.int64)
    if HAS_LENGTH:
        length = tl.load(length_ptr + batch).to(tl.int64)
    # A negative length masks the entire row, without inspecting it on the host.
    for j in range(K):
        loc = tl.load(
            indices_ptr + batch * K + j, mask=j < length, other=-1
        ).to(tl.int64)
        valid = (loc >= 0) & (j < length)
        # Invalid IDs never form negative/OOB addresses, even for an empty pool.
        safe_loc = tl.where(valid, loc, 0)
        page_base = (safe_loc // PAGE_SIZE) * PITCH
        slot = safe_loc % PAGE_SIZE
        data_base = page_base + slot * DATA_BYTES
        scale_base = page_base + SCALE_OFFSET + slot * SCALE_BYTES
        if IS_FP4:
            packed = tl.load(
                cache_ptr + data_base + offsets // 2, mask=valid, other=0
            )
            code = (packed >> ((offsets & 1) * 4)) & 15
            values = _e2m1_to_fp32(code)
            scale_code = tl.load(
                cache_ptr + scale_base + offsets // TILE_SIZE,
                mask=valid,
                other=0,
            )
            scales = _e4m3_to_fp32(scale_code)
        else:
            code = tl.load(cache_ptr + data_base + offsets, mask=valid, other=0)
            values = _e4m3_to_fp32(code)
            scale_code = tl.load(
                cache_ptr + scale_base + offsets // TILE_SIZE,
                mask=valid,
                other=0,
            )
            scales = _ue8m0_to_fp32(scale_code)
        # Match the standalone decoder: round EACH decoded token to BF16 first.
        kv = _scaled_bf16(values, scales).to(tl.float32)
        score = tl.sum(q_values * kv, 0) * softmax_scale
        score = tl.where(valid, score, float("-inf"))
        next_maximum = tl.maximum(maximum, score)
        # The all-masked prefix must not evaluate -inf - -inf in exp.
        safe_maximum = tl.where(next_maximum == float("-inf"), 0.0, next_maximum)
        old_weight = tl.exp(maximum - safe_maximum)
        weight = tl.exp(score - safe_maximum)
        numerator = numerator * old_weight + kv * weight
        denominator = denominator * old_weight + weight
        maximum = next_maximum
    return maximum, denominator, numerator


@triton.jit
def _sparse_attention_sm80_kernel(
    q_ptr,
    cache_ptr,
    indices_ptr,
    length_ptr,
    sink_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_length_ptr,
    out_ptr,
    lse_ptr,
    softmax_scale,
    H: tl.constexpr,
    K: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PITCH: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    IS_FP4: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    EXTRA_K: tl.constexpr,
    EXTRA_HAS_LENGTH: tl.constexpr,
    EXTRA_PAGE_SIZE: tl.constexpr,
    EXTRA_PITCH: tl.constexpr,
    EXTRA_DATA_BYTES: tl.constexpr,
    EXTRA_SCALE_BYTES: tl.constexpr,
    EXTRA_SCALE_OFFSET: tl.constexpr,
    EXTRA_TILE_SIZE: tl.constexpr,
    EXTRA_IS_FP4: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // H
    head = row % H
    offsets = tl.arange(0, 512).to(tl.int64)
    q_values = tl.load(q_ptr + row * 512 + offsets).to(tl.float32)
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.full((512,), 0.0, tl.float32)
    maximum, denominator, numerator = _accumulate_candidates(
        q_values, maximum, denominator, numerator,
        cache_ptr, indices_ptr, length_ptr, batch, softmax_scale,
        K, HAS_LENGTH, PAGE_SIZE, PITCH, DATA_BYTES, SCALE_BYTES,
        SCALE_OFFSET, TILE_SIZE, IS_FP4,
    )
    if HAS_EXTRA:
        maximum, denominator, numerator = _accumulate_candidates(
            q_values, maximum, denominator, numerator,
            extra_cache_ptr, extra_indices_ptr, extra_length_ptr,
            batch, softmax_scale,
            EXTRA_K, EXTRA_HAS_LENGTH, EXTRA_PAGE_SIZE, EXTRA_PITCH,
            EXTRA_DATA_BYTES, EXTRA_SCALE_BYTES, EXTRA_SCALE_OFFSET,
            EXTRA_TILE_SIZE, EXTRA_IS_FP4,
        )

    has_keys = denominator > 0.0
    safe_maximum = tl.where(has_keys, maximum, 0.0)
    key_lse = safe_maximum + tl.log(tl.where(has_keys, denominator, 1.0))
    tl.store(lse_ptr + row, tl.where(has_keys, key_lse, float("-inf")))

    # Sink contributes only to the output denominator, NEVER to returned LSE.
    sink = tl.full((), float("-inf"), tl.float32)
    if HAS_SINK:
        sink = tl.load(sink_ptr + head)
    output_maximum = tl.maximum(safe_maximum, sink)
    key_factor = tl.exp(safe_maximum - output_maximum)
    sink_factor = tl.exp(sink - output_maximum)
    output_denominator = denominator * key_factor + sink_factor
    safe_denominator = tl.where(output_denominator > 0.0, output_denominator, 1.0)
    result = numerator * key_factor / safe_denominator
    result = tl.where(has_keys, result, 0.0)
    tl.store(out_ptr + row * 512 + offsets, result.to(tl.bfloat16))


def _validate_tensor(name, tensor, shape, dtype, device):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype not in dtype:
        raise TypeError(f"{name} must have dtype in {dtype}, got {tensor.dtype}")
    if tensor.device != device or not tensor.is_cuda:
        raise ValueError(f"{name} must be on {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous (strided output is not supported)")


def _validate_cache(name, cache, indices, length, page_size, layout, batch, device):
    if type(page_size) is not int or page_size <= 0:
        raise ValueError(f"{name} page_size must be a positive Python int")
    layout = KVLayout.parse(layout)
    if layout not in (KVLayout.V41, KVLayout.V41_FP4):
        raise ValueError(f"{name} supports only V41/V41_FP4, got {layout}")
    if not isinstance(cache, torch.Tensor) or cache.ndim != 2:
        raise ValueError(f"{name} must be a raw uint8 tensor [pages, pitch]")
    _validate_tensor(name, cache, cache.shape, (torch.uint8,), device)
    if cache.shape[1] < page_size * layout.bytes_per_token:
        raise ValueError(f"{name} pitch cannot hold page_size tokens and their scales")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 2:
        raise ValueError(f"{name} indices must have shape [B, K]")
    _validate_tensor(
        f"{name} indices", indices, (batch, indices.shape[1]),
        (torch.int32, torch.int64), device,
    )
    if length is not None:
        _validate_tensor(
            f"{name} topk_length", length, (batch,), (torch.int32, torch.int64), device
        )
    return layout


def sparse_attention_sm80(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
    layout: KVLayout | str,
    softmax_scale: float,
    *,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_page_size: int | None = None,
    extra_layout: KVLayout | str | None = None,
    extra_topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend to physical token IDs with V41/V41_FP4 shared key/value vectors.

    Inputs: contiguous BF16 q[B,H,512], raw uint8 cache[pages,pitch], contiguous
    int32/int64 indices[B,K]. Any negative ID is masked (-1 is conventional).
    Optional contiguous int32/int64 topk_length[B] additionally requires j <
    length; negative lengths mask all slots and lengths > K are clipped by K.
    Nonnegative IDs must be < pages * page_size: the CALLER checks upper bounds,
    and supplies causal/window-filtered candidates. Repeats count independently.
    The optional extra cache requires all four extra cache/indices/page_size/
    layout arguments; its format and page size may differ from the main cache.

    q, decoded BF16 KV, and FP32 dot products/scaled scores/accumulators must be
    finite. attn_sink is contiguous FP32[H], finite or -inf (never NaN/+inf).
    Tensor VALUES are not host-checked: no .item(), CPU copy, or synchronization.
    softmax_scale is a finite Python real scalar. Returned lse is FP32[B,H], the
    NATURAL-LOG logsumexp of real valid keys only, excluding the sink. out is
    BF16[B,H,512]; the sink is a zero-value entry in its denominator. All-masked
    rows return exact zero and lse=-inf, including absent sink or sink=-inf.

    Every tensor must be contiguous on the same NVIDIA SM80 device (not SM86).
    B=0, H=0, K=0 and extra-only candidates are supported. Pass BOTH out and lse
    to avoid tensor allocation during CUDA graph capture; warm up compilation
    and device metadata before capture. Output buffers must not overlap inputs
    or each other (caller responsibility). No model dispatch is performed.

    Example::

        out, lse = sparse_attention_sm80(
            q, cache_u8, physical_ids, page_size, KVLayout.V41, 512**-0.5,
            topk_length=lengths, attn_sink=sink, out=out_buffer, lse=lse_buffer,
        )
    """
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or q.shape[2] != 512:
        raise ValueError("q must have shape [B, H, 512]")
    if not q.is_cuda or torch.version.hip is not None:
        raise ValueError("sparse_attention_sm80 requires NVIDIA CUDA SM80")
    device = q.device
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError("sparse_attention_sm80 requires SM80 (not SM86)")
    batch, heads, _ = q.shape
    _validate_tensor("q", q, (batch, heads, 512), (torch.bfloat16,), device)
    if (
        isinstance(softmax_scale, bool)
        or not isinstance(softmax_scale, numbers.Real)
        or not math.isfinite(softmax_scale)
    ):
        raise ValueError("softmax_scale must be a finite Python real scalar")
    softmax_scale = float(softmax_scale)
    layout = _validate_cache(
        "cache", cache, indices, topk_length, page_size, layout, batch, device
    )
    if attn_sink is not None:
        _validate_tensor("attn_sink", attn_sink, (heads,), (torch.float32,), device)

    has_extra = extra_cache is not None
    extra_arguments = (extra_indices, extra_page_size, extra_layout)
    if has_extra:
        if any(argument is None for argument in extra_arguments):
            raise ValueError("extra_cache requires extra_indices/page_size/layout")
        extra_layout = _validate_cache(
            "extra_cache", extra_cache, extra_indices, extra_topk_length,
            extra_page_size, extra_layout, batch, device,
        )
    elif (
        any(argument is not None for argument in extra_arguments)
        or extra_topk_length is not None
    ):
        raise ValueError("extra arguments require extra_cache")

    if out is None:
        out = torch.empty((batch, heads, 512), dtype=torch.bfloat16, device=device)
    else:
        _validate_tensor("out", out, (batch, heads, 512), (torch.bfloat16,), device)
    if lse is None:
        lse = torch.empty((batch, heads), dtype=torch.float32, device=device)
    else:
        _validate_tensor("lse", lse, (batch, heads), (torch.float32,), device)
    if batch == 0 or heads == 0:
        return out, lse

    # Dummy pointers are dead in constexpr branches, and allocate no tensors.
    extra_format = extra_layout if has_extra else layout
    extra_page = extra_page_size if has_extra else page_size
    with torch.cuda.device(device):
        _sparse_attention_sm80_kernel[(batch * heads,)](
            q, cache, indices,
            topk_length if topk_length is not None else indices,
            attn_sink if attn_sink is not None else q,
            extra_cache if has_extra else cache,
            extra_indices if has_extra else indices,
            extra_topk_length if extra_topk_length is not None else indices,
            out, lse, softmax_scale,
            H=heads, K=indices.shape[1], HAS_LENGTH=topk_length is not None,
            HAS_SINK=attn_sink is not None,
            PAGE_SIZE=page_size, PITCH=cache.shape[1],
            DATA_BYTES=layout.data_bytes, SCALE_BYTES=layout.scale_bytes,
            SCALE_OFFSET=layout.scale_offset(page_size), TILE_SIZE=layout.tile_size,
            IS_FP4=layout is KVLayout.V41_FP4,
            HAS_EXTRA=has_extra, EXTRA_K=extra_indices.shape[1] if has_extra else 0,
            EXTRA_HAS_LENGTH=extra_topk_length is not None,
            EXTRA_PAGE_SIZE=extra_page,
            EXTRA_PITCH=extra_cache.shape[1] if has_extra else cache.shape[1],
            EXTRA_DATA_BYTES=extra_format.data_bytes,
            EXTRA_SCALE_BYTES=extra_format.scale_bytes,
            EXTRA_SCALE_OFFSET=extra_format.scale_offset(extra_page),
            EXTRA_TILE_SIZE=extra_format.tile_size,
            EXTRA_IS_FP4=extra_format is KVLayout.V41_FP4,
            num_warps=4, enable_fp_fusion=False,
        )
    return out, lse
