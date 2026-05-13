"""FlashMLA adapter with SM120 fallback.

The FP8 KV cache uses a page-internal layout where NOPE+ROPE data has
stride (nope_dim + rope_dim*2) per token, and scales are stored in a
separate region at the end of each page.  The tensor shape
``(num_pages, page_size, 1, bytes_per_token)`` is just metadata for the
FlashMLA CUDA kernel -- it does NOT mean each token occupies
*bytes_per_token* contiguous bytes.

On SM120 (Blackwell Desktop / RTX PRO 6000) the flash_mla CUDA kernel
is not available, so this module provides a pure-PyTorch fallback that
reads the raw paged buffer with the correct addressing.

When SGLANG_SM120_TRITON_FLASHMLA=1 (default), a fused Triton kernel is
used instead of the PyTorch fallback for significantly better performance.
Set to 0 to fall back to the pure-PyTorch path.
"""
import logging
import os

import torch

from sglang.srt.utils import is_hip, is_xpu
from sglang.srt.utils.common import get_device_sm

logger = logging.getLogger(__name__)

_is_cuda = torch.cuda.is_available() and not is_hip()
# _is_sm120 = _is_cuda and get_device_sm() // 10 == 12
_is_sm120 = True

_is_xpu = is_xpu()

# Page layout constants for DSv4-Flash (MODEL1):
#   nope_dim = 448, rope_dim = 64, quantize_block_size = 64
#   nope_rope_stride = 448 + 64*2 = 576 bytes per token
#   scale_stride = ceil(448/64) + 1 = 8 bytes per token (7 scales + 1 pad)
#   bytes_per_token = 448 + 128 + 8 = 584
#   page_bytes = ceil_div(page_size * 584, 576) * 576

_NOPE_DIM = 448
_ROPE_DIM = 64
_NOPE_ROPE_STRIDE = _NOPE_DIM + _ROPE_DIM * 2  # 576
_TILE_SIZE = 64
_NUM_TILES = _NOPE_DIM // _TILE_SIZE  # 7
_SCALE_STRIDE = _NUM_TILES + 1  # 8 (7 scales + 1 pad)
_D = _NOPE_DIM + _ROPE_DIM  # 512


_GATHER_CHUNK = 16384  # tokens per chunk; ~16k * 1024 B ≈ 16 MiB output per chunk

# Per-chunk peak-memory budget for the sparse decode fallback (MiB).  Read
# once at import time so the forward path doesn't pay an os.environ lookup
# per layer per decode step.
_SM120_SPARSE_CHUNK_MIB = int(os.environ.get("SGLANG_SM120_SPARSE_CHUNK_MIB", "256"))

_use_triton_flashmla = os.environ.get("SGLANG_SM120_TRITON_FLASHMLA", "1") == "1"


def _gather_and_dequant(k_cache, indices, page_size):
    """Gather KV entries from the paged buffer using correct page-internal addressing.

    Args:
        k_cache: (num_pages, page_size, 1, bytes_per_token) float8_e4m3fn
                 Non-contiguous view of the raw page buffer.
        indices: (...) int32/int64, token-level indices. Invalid indices are
                 expected to already be clamped into [0, num_pages*page_size).
        page_size: tokens per page (e.g. 256, 64, 2)

    Returns:
        kv: (..., _D) bfloat16, dequantized KV vectors
    """
    idx_shape = indices.shape
    flat_idx = indices.reshape(-1)  # (N,)
    N = flat_idx.shape[0]
    device = k_cache.device

    page_bytes = k_cache.stride(0)  # actual byte stride between pages
    num_pages = k_cache.shape[0]

    # Flatten the raw byte buffer so we can gather with a single int64 index
    # per byte instead of paying for a full (N, 448) int64 index tensor up
    # front. flat_buf has nelems = num_pages * page_bytes uint8.
    raw_pages = k_cache.as_strided(
        (num_pages, page_bytes),
        (page_bytes, 1),
    ).view(torch.uint8)
    flat_buf = raw_pages.reshape(-1)

    scale_section_offset = page_size * _NOPE_ROPE_STRIDE

    nope_arange = torch.arange(_NOPE_DIM, device=device, dtype=torch.long)
    rope_arange = torch.arange(_ROPE_DIM * 2, device=device, dtype=torch.long)
    scale_arange = torch.arange(_NUM_TILES, device=device, dtype=torch.long)

    result = torch.empty(N, _D, dtype=torch.bfloat16, device=device)

    # Process in chunks to bound peak memory of the int64 advanced-index
    # tensors (which would otherwise be N * 448 * 8 bytes — multiple GB on
    # long-context prefills with large topk).
    for start in range(0, N, _GATHER_CHUNK):
        end = min(start + _GATHER_CHUNK, N)
        chunk = flat_idx[start:end]
        n = end - start

        pages = chunk // page_size
        offsets = chunk % page_size

        # Per-token base byte offset into the flat raw buffer.
        page_base = pages.to(torch.long) * page_bytes  # (n,)
        nope_base = page_base + offsets.to(torch.long) * _NOPE_ROPE_STRIDE  # (n,)

        nope_idx = nope_base.unsqueeze(-1) + nope_arange  # (n, 448)
        rope_idx = nope_base.unsqueeze(-1) + (_NOPE_DIM + rope_arange)  # (n, 128)
        scale_idx = (
            page_base.unsqueeze(-1)
            + scale_section_offset
            + offsets.to(torch.long).unsqueeze(-1) * _SCALE_STRIDE
            + scale_arange
        )  # (n, 7)

        nope_bytes = flat_buf[nope_idx.reshape(-1)].view(n, _NOPE_DIM)
        rope_bytes = flat_buf[rope_idx.reshape(-1)].view(n, _ROPE_DIM * 2)
        scale_bytes = flat_buf[scale_idx.reshape(-1)].view(n, _NUM_TILES)

        nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)  # (n, 448)
        rope_bf16 = rope_bytes.contiguous().view(torch.bfloat16)  # (n, 64)
        scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)  # (n, 7)

        result[start:end, :_NOPE_DIM] = (
            nope_fp8.view(n, _NUM_TILES, _TILE_SIZE).float()
            * scale_e8m0.view(n, _NUM_TILES, 1).float()
        ).view(n, _NOPE_DIM).to(torch.bfloat16)
        result[start:end, _NOPE_DIM:] = rope_bf16

    return result.reshape(*idx_shape, _D)


def _sm120_sparse_decode_fwd(q, k_cache, indices, topk_length, attn_sink,
                              head_dim_v, softmax_scale,
                              extra_k_cache=None, extra_indices=None,
                              extra_topk_length=None):
    B, s_q, H_q, D_qk = q.shape
    num_pages, page_size, H_k, bpt = k_cache.shape
    topk = indices.shape[-1]
    device = q.device

    # FlashMLA kernel treats `index == -1` as invalid; we additionally treat
    # any index outside [0, num_pages*page_size) as invalid because the CUDA
    # tile scheduler would simply never visit those slots, whereas this
    # PyTorch fallback gathers them eagerly.
    max_valid = num_pages * page_size
    invalid_mask = (indices < 0) | (indices >= max_valid)
    safe_indices = indices.clamp(min=0, max=max_valid - 1)
    if topk_length is not None:
        topk_range = torch.arange(topk, device=topk_length.device).view(1, 1, topk)
        invalid_mask = invalid_mask | (topk_range >= topk_length.view(B, 1, 1))

    have_extra = extra_k_cache is not None and extra_indices is not None
    if have_extra:
        extra_topk = extra_indices.shape[-1]
        extra_num_pages, extra_page_size = extra_k_cache.shape[0], extra_k_cache.shape[1]
        extra_max_valid = extra_num_pages * extra_page_size
        extra_invalid = (extra_indices < 0) | (extra_indices >= extra_max_valid)
        extra_safe = extra_indices.clamp(min=0, max=extra_max_valid - 1)
        if extra_topk_length is not None:
            extra_range = torch.arange(extra_topk, device=extra_topk_length.device).view(1, 1, extra_topk)
            extra_invalid = extra_invalid | (extra_range >= extra_topk_length.view(B, 1, 1))
    else:
        extra_topk = 0

    total_topk = topk + extra_topk
    # Flatten the (B, s_q) row dimension so we can chunk easily.
    R = B * s_q  # number of query rows
    q_rows = q.reshape(R, H_q, D_qk)
    safe_indices_rows = safe_indices.reshape(R, topk)
    invalid_rows = invalid_mask.reshape(R, topk)
    if have_extra:
        extra_safe_rows = extra_safe.reshape(R, extra_topk)
        extra_invalid_rows = extra_invalid.reshape(R, extra_topk)

    out_rows = torch.empty(R, H_q, head_dim_v, dtype=torch.bfloat16, device=device)
    lse_rows = torch.empty(R, H_q, dtype=torch.float32, device=device)

    # Bound per-chunk peak memory. Dominant bf16 tensor is gathered KV:
    # chunk * total_topk * _D * 2 bytes; fp32 working set adds ~3x on top.
    # On Intel L0, per-launch overhead is high (~hundreds of us), so prefer
    # fewer/larger chunks. Target 256 MiB peak (override via
    # SGLANG_SM120_SPARSE_CHUNK_MIB at import time).
    bytes_per_row = total_topk * _D * 2
    chunk_rows = max(
        1, min(R, (_SM120_SPARSE_CHUNK_MIB * 1024 * 1024) // max(1, bytes_per_row))
    )

    for start in range(0, R, chunk_rows):
        end = min(start + chunk_rows, R)
        n = end - start

        # Gather KV for this chunk only.
        kv_chunk = _gather_and_dequant(
            k_cache, safe_indices_rows[start:end], page_size
        )  # (n, topk, _D)
        inv_chunk = invalid_rows[start:end]  # (n, topk)
        if have_extra:
            extra_kv_chunk = _gather_and_dequant(
                extra_k_cache, extra_safe_rows[start:end], extra_page_size
            )  # (n, extra_topk, _D)
            kv_chunk = torch.cat([kv_chunk, extra_kv_chunk], dim=1)
            inv_chunk = torch.cat([inv_chunk, extra_invalid_rows[start:end]], dim=1)
            del extra_kv_chunk

        q_chunk = q_rows[start:end].float()  # (n, H_q, D_qk)
        # Scrub NaN from invalid-index dequant so the value reduction is not
        # polluted by 0 * NaN = NaN. Done in-place after the float upcast to
        # avoid a separate allocation; ``scores`` is masked to ``-inf`` below
        # which gives invalid positions exactly zero weight.
        kv_f = kv_chunk.float().nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        kv_d = kv_f.shape[-1]
        if D_qk != kv_d:
            q_chunk = q_chunk[..., :kv_d]

        # scores: (n, H_q, T)
        scores = torch.einsum("nhd,ntd->nht", q_chunk, kv_f) * softmax_scale
        scores.masked_fill_(inv_chunk.unsqueeze(1).expand_as(scores), float("-inf"))

        lse = torch.logsumexp(scores, dim=-1)  # (n, H_q)

        if attn_sink is not None:
            lse_for_out = torch.logsumexp(
                torch.stack([lse, attn_sink.view(1, H_q).expand_as(lse)], dim=0),
                dim=0,
            )
        else:
            lse_for_out = lse.clone()

        lonely = lse == float("-inf")
        lse_for_out[lonely] = float("inf")
        weights = torch.exp(scores - lse_for_out.unsqueeze(-1))
        out_chunk = torch.einsum("nht,ntv->nhv", weights, kv_f[..., :head_dim_v])
        out_chunk[lonely.unsqueeze(-1).expand_as(out_chunk)] = 0.0

        out_rows[start:end] = out_chunk.to(torch.bfloat16)
        lse_rows[start:end] = lse

        del kv_chunk, kv_f, q_chunk, scores, weights, out_chunk, lse, lse_for_out, lonely

    out = out_rows.reshape(B, s_q, H_q, head_dim_v)
    lse = lse_rows.reshape(B, s_q, H_q).permute(0, 2, 1)
    return out, lse


def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if _is_sm120 or _is_xpu:
        q = kwargs["q"]
        k_cache = kwargs["k_cache"]
        indices = kwargs["indices"]
        topk_length = kwargs.get("topk_length")
        attn_sink = kwargs.get("attn_sink")
        head_dim_v = kwargs["head_dim_v"]
        softmax_scale = kwargs.get("softmax_scale")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        extra_k_cache = kwargs.get("extra_k_cache")
        extra_indices = kwargs.get("extra_indices_in_kvcache")
        extra_topk_length = kwargs.get("extra_topk_length")

        if (_is_sm120 or _is_xpu) and _use_triton_flashmla:
            from sglang.srt.layers.attention.flash_mla_sm120_triton import (
                flash_mla_sparse_decode_triton,
            )

            out, lse = flash_mla_sparse_decode_triton(
                q, k_cache, indices, topk_length, attn_sink,
                head_dim_v, softmax_scale,
                extra_k_cache, extra_indices, extra_topk_length,
            )
            return (out, lse)

        out, lse = _sm120_sparse_decode_fwd(
            q, k_cache, indices, topk_length, attn_sink,
            head_dim_v, softmax_scale,
            extra_k_cache, extra_indices, extra_topk_length,
        )
        return (out, lse)

    assert backend == "kernel", f"unsupported backend {backend!r}"
    import flash_mla
    return flash_mla.flash_mla_with_kvcache(**kwargs)
