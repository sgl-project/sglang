"""gfx950 sparse attention over token-interleaved V4.1 caches (KVLayout.V41_NEW / V41_FP4_NEW)
with aiter's opus ``mla_sparse_prefill_v41``: MXFP8 q, QK on scaled FP8 MFMA, the compressed
cache as its prefix source and the SWA cache as its extend source, each addressed by CSR row
lists built from the backend's per-token index lists and lengths."""

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.q_rope_quant_hip import rope_fp8_grid_quantize
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import fp8_grid_quantize

# up to this many query tokens every CSR program sums the preceding lengths itself
_CSR_INLINE_MAX_TOKENS = 4096
# past that, a scan kernel writes each row's offset inside its block of rows and the block
# totals; a CSR program adds the totals of the preceding blocks
_CSR_BLOCK_ROWS = 256
_CSR_NUM_WARPS = 4


@triton.jit
def _csr_block_scan_kernel(
    lengths_ptr, local_ptr, sums_ptr, n, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    """Per block of rows: the exclusive prefix of the clamped lengths inside it, and its total."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(lengths_ptr + offs, mask=offs < n, other=0)
    v = tl.minimum(tl.maximum(v, 0), WIDTH)
    inc = tl.cumsum(v, axis=0)
    tl.store(local_ptr + offs, inc - v, mask=offs < n)
    tl.store(sums_ptr + tl.program_id(0), tl.sum(v))


@triton.jit
def _dense_to_csr_kernel(
    indices_ptr,
    lengths_ptr,
    local_starts_ptr,
    block_sums_ptr,
    out_indices_ptr,
    out_indptr_ptr,
    n,
    INDEX_STRIDE,
    WIDTH: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INLINE_STARTS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCKS_POW2: tl.constexpr,
):
    """Row t: its first min(lengths[t], WIDTH) entries to out_indices[start:], negative
    entries (pad rows) clamped to row 0; out_indptr[t + 1] = start + that count."""
    t = tl.program_id(0)
    if INLINE_STARTS:
        offs_n = tl.arange(0, BLOCK_N)
        prev = tl.load(lengths_ptr + offs_n, mask=offs_n < t, other=0)
        start = tl.sum(tl.minimum(tl.maximum(prev, 0), WIDTH))
    else:
        b = t // BLOCK_ROWS
        offs_b = tl.arange(0, BLOCKS_POW2)
        start = tl.sum(tl.load(block_sums_ptr + offs_b, mask=offs_b < b, other=0))
        start += tl.load(local_starts_ptr + t)
    length = tl.minimum(tl.maximum(tl.load(lengths_ptr + t), 0), WIDTH)
    for c in tl.static_range(0, WIDTH, BLOCK_W):
        offs = c + tl.arange(0, BLOCK_W)
        m = offs < length
        idx = tl.load(
            indices_ptr + t.to(tl.int64) * INDEX_STRIDE + offs, mask=m, other=0
        )
        tl.store(out_indices_ptr + start + offs, tl.maximum(idx, 0), mask=m)
    tl.store(out_indptr_ptr + t + 1, start + length)
    if t == 0:
        tl.store(out_indptr_ptr, 0)


def dense_to_csr(
    indices: torch.Tensor, lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """indices [n, W] int32 and lengths [n] -> (CSR row lists [n * W] int32, indptr [n + 1]
    int32). The row-list buffer keeps its capacity size, which only feeds the kernel's
    split-K choice; rows past indptr[n] are not read."""
    n, width = indices.shape
    assert indices.dtype == torch.int32 and indices.stride(1) == 1
    lengths = lengths.reshape(-1)
    assert lengths.shape[0] == n
    lengths = lengths.to(torch.int32).contiguous()
    out_indices = torch.empty(
        max(n * width, 1), dtype=torch.int32, device=indices.device
    )
    indptr = torch.empty(n + 1, dtype=torch.int32, device=indices.device)
    if n == 0:
        indptr.zero_()
        return out_indices, indptr
    inline = n <= _CSR_INLINE_MAX_TOKENS
    blocks = triton.cdiv(n, _CSR_BLOCK_ROWS)
    if inline:
        local_starts = block_sums = lengths
    else:
        local_starts = torch.empty(n, dtype=torch.int32, device=indices.device)
        block_sums = torch.empty(blocks, dtype=torch.int32, device=indices.device)
        _csr_block_scan_kernel[(blocks,)](
            lengths, local_starts, block_sums, n, WIDTH=width, BLOCK=_CSR_BLOCK_ROWS
        )
    _dense_to_csr_kernel[(n,)](
        indices,
        lengths,
        local_starts,
        block_sums,
        out_indices,
        indptr,
        n,
        indices.stride(0),
        WIDTH=width,
        BLOCK_W=min(512, triton.next_power_of_2(width)),
        BLOCK_N=triton.next_power_of_2(n) if inline else 1,
        INLINE_STARTS=inline,
        BLOCK_ROWS=_CSR_BLOCK_ROWS,
        BLOCKS_POW2=1 if inline else triton.next_power_of_2(blocks),
        num_warps=_CSR_NUM_WARPS,
    )
    return out_indices, indptr


def _rows(cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """A [pages, page_size, 1, bytes] interleaved cache as its (values, scales) row views,
    slot s at row s: fp8 [rows, 512] + ue8m0 [rows, 16], or e2m1 pairs [rows, 256] (uint8)
    + e4m3 [rows, 32] (uint8)."""
    bpt = cache.shape[-1]
    if bpt == KVLayout.V41_NEW.bytes_per_token:
        data = KVLayout.V41_NEW.data_bytes
    elif bpt == KVLayout.V41_FP4_NEW.bytes_per_token:
        data = KVLayout.V41_FP4_NEW.data_bytes
    else:
        raise ValueError(f"not a V4.1 interleaved cache row: {bpt} bytes")
    pages, page_size = cache.shape[0], cache.shape[1]
    raw = cache.view(torch.uint8)
    assert raw.stride(0) == page_size * bpt and raw.stride(1) == bpt, (
        "an interleaved V4.1 page must hold its rows back to back"
    )
    rows = raw.as_strided((pages * page_size, bpt), (bpt, 1))
    values = rows[:, :data]
    if bpt == KVLayout.V41_NEW.bytes_per_token:
        values = values.view(torch.float8_e4m3fn)
    return values, rows[:, data:]


@functools.cache
def aiter_mla_sparse_prefill_v41():
    """aiter's opus ``mla_sparse_prefill_v41`` (aiter #6675 and later); raises ImportError
    on an older aiter."""
    from aiter.ops.mla_sparse_prefill import mla_sparse_prefill_v41

    return mla_sparse_prefill_v41


def opus_v41_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    softmax_scale: float,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    csr_cache: Optional[dict] = None,
    q_rope_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """q [n, h, 512] bf16 over the first topk_length[t] slots of indices[t] in the SWA
    cache, then those of extra_indices[t] in the compressed cache; returns [n, h, 512] bf16.
    csr_cache (one forward's dict) keeps the SWA lists' CSR, which every layer shares.
    q_rope_freqs=(freqs_real, positions): q's rope tail is not yet rotated; the q quant does it."""
    mla_sparse_prefill_v41 = aiter_mla_sparse_prefill_v41()

    n, h, d = q.shape
    assert d == 512 and q.dtype == torch.bfloat16
    if n == 0:
        return torch.empty_like(q)
    if q_rope_freqs is not None:
        q8, q_scale = rope_fp8_grid_quantize(q, *q_rope_freqs)
    else:
        q8, q_scale = fp8_grid_quantize(q.reshape(n * h, d))
    q8 = q8.view(n, h, d)
    q_scale = q_scale.view(n, h, d // 32)
    swa_values, swa_scales = _rows(k_cache)
    key = (
        "opus_swa_csr",
        indices.data_ptr(),
        tuple(indices.shape),
        topk_length.data_ptr(),
    )
    swa_csr = csr_cache.get(key) if csr_cache is not None else None
    if swa_csr is None:
        swa_csr = dense_to_csr(indices.reshape(n, -1), topk_length)
        if csr_cache is not None:
            csr_cache[key] = swa_csr
    swa_idx, swa_ptr = swa_csr
    if extra_k_cache is not None:
        prefix_values, prefix_scales = _rows(extra_k_cache)
        prefix_idx, prefix_ptr = dense_to_csr(
            extra_indices.reshape(n, -1), extra_topk_length
        )
    else:
        prefix_values, prefix_scales = swa_values, swa_scales
        prefix_idx = swa_idx[:1]
        prefix_ptr = torch.zeros(n + 1, dtype=torch.int32, device=q.device)
    return mla_sparse_prefill_v41(
        q8,
        q_scale,
        prefix_values,
        prefix_scales,
        prefix_idx,
        prefix_ptr,
        swa_values,
        swa_scales,
        swa_idx,
        swa_ptr,
        attn_sink.float().contiguous() if attn_sink is not None else None,
        softmax_scale,
    )
