"""Stands in for the stock paged FP8 decode (flash_mla_with_kvcache), whose split-KV schedule (get_mla_metadata) is
planned over the whole batch's cache lengths.

Sparse FlashMLA attention over dequantized selected FP8 KV slots.

FP8 remains the authoritative cache representation. No unquantized fresh-KV
shortcut and no batch-dependent paged decoding reduction are used.
"""

import torch

from sglang.kernels.ops.attention.concat_mla import concat_mla_absorb_q
from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    dequantize_k_cache_paged_dsa,
    dequantize_k_cache_selected,
)


def offset_ragged(indices, offsets):
    """Apply live-row offsets while masking TP/graph padding, without alias writes."""
    if offsets.ndim == 1:
        offsets = offsets[:, None]
    if (
        offsets.ndim != 2
        or offsets.shape[1] != 1
        or offsets.shape[0] > indices.shape[0]
    ):
        raise ValueError(
            "Ragged offsets must describe a prefix of the physical query rows"
        )
    live = offsets.shape[0]
    if live < indices.shape[0]:
        offsets = torch.cat(
            (offsets, offsets.new_zeros((indices.shape[0] - live, 1))), 0
        )
    valid = (indices >= 0) & (
        torch.arange(indices.shape[0], device=indices.device)[:, None] < live
    )
    return torch.where(valid, indices + offsets, -1)


def sparse_from_bf16(q, kv, indices, scale, d_v, topk_length=None, out=None):
    """Sparse BF16 attention in fixed 16-head groups, each in its own 32-head block of a 128-head kernel; FlashMLA's live-head grouping changes rounding, so the grouping is part of the program. ``out`` (rows x heads x d_v, contiguous) receives the result instead of a new tensor."""
    if q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[1] not in (16, 32, 64):
        raise NotImplementedError(
            "FP8-storage TOP attention requires complete 16-head groups (16/32/64 BF16 query heads)"
        )
    # Serving graph padding may have more physical query rows than the live
    # length metadata. Preserve the original backend's full-width fallback.
    if topk_length is not None and topk_length.shape[0] != q.shape[0]:
        topk_length = None
    rows, heads, dim = q.shape
    if rows == 0:
        return q.new_empty((0, heads, d_v))
    if (
        dim == 576
        and q.stride(2) == 1
        and all(stride % 8 == 0 for stride in q.stride()[:2])
        and q.data_ptr() % 16 == 0
    ):
        padded = concat_mla_absorb_q(q[..., :512], q[..., 512:], pad_head_groups=True)
    else:
        padded = q.new_zeros((rows, 4, 32, dim))
        padded[:, : heads // 16, :16] = q.reshape(rows, heads // 16, 16, dim)
        padded = padded.reshape(rows, 128, dim)
    return _attend_groups(padded, kv, indices, scale, d_v, heads, topk_length, out)


def _attend_groups(padded, kv, indices, scale, d_v, heads, topk_length=None, out=None):
    # FlashMLA head128 uses a warp-wide any() for the rescaling decision.
    # Each 16-head group occupies its own 32-head block, followed by zeros,
    # preserving the old decision without repeating query rows or KV indices.
    # Imported here: the interface module loads the FlashMLA wheel at import.
    from sglang.kernels.ops.attention.flash_mla_interface import (
        flash_mla_sparse_fwd,
    )

    value = flash_mla_sparse_fwd(
        padded,
        kv,
        indices.unsqueeze(1),
        scale,
        d_v,
        topk_length=topk_length,
    )[0]
    rows = indices.shape[0]
    groups = value.reshape(rows, 4, 32, d_v)[:, : heads // 16, :16]
    if out is None:
        return groups.reshape(rows, heads, d_v)
    # One copy straight into the caller's rows (no per-chunk tensor to concatenate).
    out.view(rows, heads // 16, 16, d_v).copy_(groups)
    return out


def _validate_fp8(shape, dtype, cache, indices, d_v):
    if (
        dtype != torch.bfloat16
        or len(shape) != 3
        or shape[1] not in (16, 32, 64)
        or shape[2] != 576
        or cache.dtype != torch.float8_e4m3fn
        or cache.shape[-1] != 656
        or indices.dtype != torch.int32
        or indices.shape != (shape[0], 2048)
        or d_v != 512
    ):
        raise NotImplementedError("Unsupported FP8-storage sparse FlashMLA dispatch")


def sparse_from_fp8_split(q_nope, q_rope, cache, indices, scale, d_v):
    """Produce padded queries directly when decode uses independent head groups."""
    if (
        q_nope.ndim != 3
        or q_rope.ndim != 3
        or q_nope.shape[:-1] != q_rope.shape[:-1]
        or q_nope.shape[-1] != 512
        or q_rope.shape[-1] != 64
        or q_nope.dtype != q_rope.dtype
    ):
        raise NotImplementedError("Unsupported split FP8-storage query layout")
    shape = (*q_nope.shape[:-1], 576)
    _validate_fp8(shape, q_nope.dtype, cache, indices, d_v)
    if shape[0] == 0:
        return q_nope.new_empty((0, shape[1], d_v))
    if shape[0] <= 64:
        padded = concat_mla_absorb_q(q_nope, q_rope, pad_head_groups=True)
        gathered, local = dequantize_k_cache_selected(cache, indices.contiguous())
        return _attend_groups(
            padded.reshape(-1, 128, 576), gathered, local, scale, d_v, shape[1]
        )
    return sparse_from_fp8(
        concat_mla_absorb_q(q_nope, q_rope), cache, indices, scale, d_v
    )


# Rows attend independently; this only bounds the 128-head padded query copy (~75 MB per pass at 512 rows) and the
# kernel's per-row buffers.
_ROWS_PER_PASS = 512
# Dequantize the whole cache once when it holds this many times fewer slots than the selections (trainer micro-batches
# and prefill); otherwise gather per 64-row chunk (decode, where the pool can exceed the live selections).
_WHOLE_CACHE_MIN_REUSE = 8


def sparse_from_fp8(q, cache, indices, scale, d_v):
    _validate_fp8(q.shape, q.dtype, cache, indices, d_v)
    if q.shape[0] == 0:
        return q.new_empty((0, q.shape[1], 512))
    slots = cache.numel() // cache.shape[-1]
    if indices.numel() >= _WHOLE_CACHE_MIN_REUSE * slots:
        return _sparse_from_whole_cache(q, cache, indices, scale, d_v, slots)
    out = q.new_empty((q.shape[0], q.shape[1], d_v))
    for start in range(0, q.shape[0], 64):
        qi = q[start : start + 64]
        ix = indices[start : start + 64]
        gathered, local = dequantize_k_cache_selected(cache, ix.contiguous())
        sparse_from_bf16(qi, gathered, local, scale, d_v, out=out[start : start + 64])
    return out


def validate_backend(backend):
    """Admission for TOP FP8 storage in DeepseekSparseAttnBackend."""
    if (
        not backend.dsa_kv_cache_store_fp8
        or backend.device_sm_major not in (9, 10)
        or backend.kv_cache_dim != 656
        or backend.physical_page_size * backend.dsa_index_kpool != 64
        or backend.kv_lora_rank != 512
        or backend.qk_rope_head_dim != 64
        or backend.dsa_index_topk != 2048
        or backend.num_q_heads not in (16, 32, 64)
        or backend.dsa_prefill_impl != "flashmla_sparse"
        # flashmla_kv, the paged FP8 decode, plans its split-KV schedule over the whole batch.
        or backend.dsa_decode_impl != "flashmla_sparse"
    ):
        raise NotImplementedError(
            "[top] FP8 FlashMLA requires SM90/SM100, FP8 KV (packed 656), page 64, top-k 2048, 16/32/64 query heads, sparse prefill and decode"
        )


def roundtrip_cold_prefill_kv(kv_cache):
    """Even cold prefill must read the quantized cache: fresh BF16 k/k_rope
    would change arithmetic on prefix-cache reuse."""
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache
    from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache

    shape = kv_cache.shape
    return dequantize_k_cache(
        quantize_k_cache(kv_cache.reshape(-1, 1, 1, 576))
    ).reshape(shape)


def _sparse_from_whole_cache(q, cache, indices, scale, d_v, slots):
    """Slot i dequantized once into row i, with the selected path's arithmetic; each row then reads the same values in the
    same order. Every slot is distinct, so no ownership pass is needed."""
    every_slot = torch.arange(slots, device=cache.device, dtype=torch.int32)
    kv = dequantize_k_cache_paged_dsa(cache, every_slot)
    return sparse_from_bf16_by_passes(q, kv, indices, scale, d_v)


def sparse_from_bf16_by_passes(q, kv, indices, scale, d_v, topk_length=None):
    """``sparse_from_bf16`` in ``_ROWS_PER_PASS``-row passes. Rows attend independently, so the bytes are the same;
    only the padded query copy and the kernel's per-row buffers are bounded."""
    if q.shape[0] <= _ROWS_PER_PASS:
        return sparse_from_bf16(q, kv, indices, scale, d_v, topk_length)
    # The graph-padding fallback is decided on the whole batch, as one call would.
    if topk_length is not None and topk_length.shape[0] != q.shape[0]:
        topk_length = None
    out = q.new_empty((q.shape[0], q.shape[1], d_v))
    for start in range(0, q.shape[0], _ROWS_PER_PASS):
        rows = slice(start, start + _ROWS_PER_PASS)
        sparse_from_bf16(
            q[rows],
            kv,
            indices[rows],
            scale,
            d_v,
            None if topk_length is None else topk_length[rows],
            out=out[rows],
        )
    return out
