"""Triton-accelerated sparse MLA decode.
    gather + dequant  ->  einsum (QK)  ->  softmax  ->  einsum (PV)

The matmul / softmax math stays in torch to avoid higher complexity.
"""

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# DSv4 KV cache layout
_NOPE_DIM = 448
_ROPE_DIM = 64
_D = _NOPE_DIM + _ROPE_DIM  # 512
_NOPE_ROPE_STRIDE = _NOPE_DIM + _ROPE_DIM * 2  # 576 bytes / token (data section)
_NUM_TILES = _NOPE_DIM // 64  # 7 scale groups per token
_SCALE_STRIDE = _NUM_TILES + 1  # 8 bytes / token (7 scales + 1 pad)


@triton.jit
def _gather_dequant_kernel(
    # Inputs (2D: n_rows x n_cols)
    indices_ptr,                    # [n_rows, n_cols] int32
    indices_row_stride: tl.int64,
    invalid_ptr,                    # [n_rows, n_cols] uint8
    invalid_row_stride: tl.int64,
    # Paged KV (3 typed views of the same uint8 buffer)
    cache_fp8_ptr,                  # float8_e4m3fn view
    cache_uint8_ptr,                # uint8 view (for UE8M0 scale bytes)
    cache_bf16_ptr,                 # bfloat16 view (for ROPE region)
    # Output (3D: n_rows_max x total_cols x _D), pre-allocated workspace.
    # We write rows [0, n_rows) and cols [out_col_offset, out_col_offset+n_cols).
    out_ptr,
    out_row_stride: tl.int64,       # = total_cols * _D, in elements
    out_col_offset: tl.int32,       # column index to start writing at
    # Invalid-mask output (1 byte / col), strided like out's (n_rows, total_cols).
    # Fuses the host-side `inv_chunk.copy_(invalid_mask)` step into the gather.
    inv_out_ptr,
    inv_out_row_stride: tl.int64,
    inv_out_col_offset: tl.int32,
    # Sizes
    n_rows: tl.int32,
    n_cols: tl.int32,
    page_size: tl.int32,
    page_bytes: tl.int64,           # k_cache.stride(0) in bytes
    scale_section_off: tl.int64,    # page_size * 576
    # Constexprs
    NOPE_PAD: tl.constexpr,         # 512 (padded from 448 to power of 2)
    ROPE_DIM: tl.constexpr,         # 64
    NOPE_DIM_RT: tl.int32,          # 448
    OUT_D_STRIDE: tl.constexpr,     # 512 (== _D)
    BLOCK_C: tl.constexpr,          # cols per program
):
    """Gather + dequant BLOCK_C cols of one row into the workspace.

    Grid: (n_rows, ceil(n_cols / BLOCK_C)).  Each program handles BLOCK_C
    consecutive columns within a single output row, writing them at
    ``out[pid_r, out_col_offset + c_offs, :]``.  This lets a single
    workspace serve both the main and extra KV gathers — each launch
    targets a different ``out_col_offset`` slice without copying.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # [BLOCK_C]
    c_mask = c_offs < n_cols

    row_idx_base = pid_r.to(tl.int64) * indices_row_stride
    row_inv_base = pid_r.to(tl.int64) * invalid_row_stride
    raw_indices = tl.load(indices_ptr + row_idx_base + c_offs, mask=c_mask, other=0)
    invalid = tl.load(invalid_ptr + row_inv_base + c_offs, mask=c_mask, other=1)
    valid = (invalid == 0) & c_mask  # [BLOCK_C]

    # Fused: also write the invalid byte into the inv workspace slice so the
    # downstream softmax kernel can read it directly (no torch copy / cat).
    inv_out_addrs = (
        pid_r.to(tl.int64) * inv_out_row_stride
        + (inv_out_col_offset + c_offs).to(tl.int64)
    )
    tl.store(inv_out_ptr + inv_out_addrs, invalid, mask=c_mask)

    page_ids = (raw_indices // page_size).to(tl.int64)
    page_offs = (raw_indices % page_size).to(tl.int64)
    token_data_bases = page_ids * page_bytes + page_offs * 576  # [BLOCK_C]

    # ---- NOPE FP8 gather ----
    nope_offs = tl.arange(0, NOPE_PAD)              # [NOPE_PAD]
    nope_d_mask = nope_offs < NOPE_DIM_RT           # [NOPE_PAD]
    nope_full_mask = valid[:, None] & nope_d_mask[None, :]
    nope_addrs = token_data_bases[:, None] + nope_offs[None, :].to(tl.int64)
    kv_nope_fp8 = tl.load(
        cache_fp8_ptr + nope_addrs, mask=nope_full_mask, other=0.0
    )

    # ---- Scale gather + dequant ----
    # 7 UE8M0 scales / token, one per 64-wide group along NOPE.
    group_ids = (nope_offs // 64).to(tl.int64)      # [NOPE_PAD] in {0..6} for valid d
    scale_bases = page_ids * page_bytes + scale_section_off + page_offs * 8
    scale_addrs = scale_bases[:, None] + group_ids[None, :]
    scale_raw = tl.load(
        cache_uint8_ptr + scale_addrs, mask=nope_full_mask, other=127
    )
    scale_f32 = tl.math.exp2(scale_raw.to(tl.float32) - 127.0)
    # Masked loads already returned 0 for invalid/out-of-bounds, so the
    # bf16 product is 0 there too — no extra tl.where needed.
    kv_nope_bf16 = (kv_nope_fp8.to(tl.float32) * scale_f32).to(tl.bfloat16)

    # ---- ROPE BF16 gather ----
    rope_offs = tl.arange(0, ROPE_DIM)              # [ROPE_DIM]
    rope_byte_bases = token_data_bases + 448        # rope starts after NOPE
    rope_elem_bases = (rope_byte_bases // 2).to(tl.int64)  # bytes -> bf16 elements
    rope_addrs = rope_elem_bases[:, None] + rope_offs[None, :].to(tl.int64)
    kv_rope_bf16 = tl.load(
        cache_bf16_ptr + rope_addrs, mask=valid[:, None], other=0.0
    )

    # ---- Store output into strided workspace slice ----
    # Target element addr: pid_r * out_row_stride + (out_col_offset + c) * _D + d
    out_row_base = pid_r.to(tl.int64) * out_row_stride
    col_bases = out_row_base + (out_col_offset + c_offs).to(tl.int64) * OUT_D_STRIDE
    nope_out_addrs = col_bases[:, None] + nope_offs[None, :].to(tl.int64)
    nope_store_mask = c_mask[:, None] & nope_d_mask[None, :]
    tl.store(out_ptr + nope_out_addrs, kv_nope_bf16, mask=nope_store_mask)
    rope_out_addrs = (
        col_bases[:, None] + NOPE_DIM_RT + rope_offs[None, :].to(tl.int64)
    )
    tl.store(out_ptr + rope_out_addrs, kv_rope_bf16, mask=c_mask[:, None])


def _triton_gather_into(
    k_cache: torch.Tensor,
    indices: torch.Tensor,            # (n_rows, n_cols) int
    invalid_mask: torch.Tensor,       # (n_rows, n_cols) bool/uint8
    page_size: int,
    out_buffer: torch.Tensor,         # (n_rows_max, total_cols, _D) bf16, contig
    out_col_offset: int,              # KV column index to start writing at
    inv_out_buffer: torch.Tensor,     # (n_rows_max, total_cols) uint8, contig
    inv_out_col_offset: int,          # inv  column index to start writing at
) -> None:
    """Gather + dequant into pre-allocated KV and invalid-mask workspaces.

    Writes rows ``[0, n_rows)`` and cols
    ``[out_col_offset, out_col_offset + n_cols)`` of ``out_buffer``, plus the
    invalid byte into the matching slice of ``inv_out_buffer``.  Folding the
    invalid-mask write into this kernel eliminates a separate
    ``inv_chunk.copy_(invalid_mask)`` host call per gather.
    """
    assert indices.dim() == 2 and indices.shape == invalid_mask.shape
    n_rows, n_cols = indices.shape
    assert out_buffer.is_contiguous() and out_buffer.shape[2] == _D
    assert n_rows <= out_buffer.shape[0]
    assert out_col_offset + n_cols <= out_buffer.shape[1]
    assert inv_out_buffer.is_contiguous() and inv_out_buffer.dtype == torch.uint8
    assert n_rows <= inv_out_buffer.shape[0]
    assert inv_out_col_offset + n_cols <= inv_out_buffer.shape[1]

    flat_idx = indices.to(torch.int32).contiguous()
    flat_inv = invalid_mask.to(torch.uint8).contiguous()

    page_bytes = k_cache.stride(0)
    num_pages = k_cache.shape[0]
    total_elems = num_pages * page_bytes
    raw_uint8 = k_cache.as_strided((total_elems,), (1,)).view(torch.uint8)
    raw_fp8 = raw_uint8.view(torch.float8_e4m3fn)
    raw_bf16 = raw_uint8.view(torch.bfloat16)

    BLOCK_C = 16
    grid = (n_rows, triton.cdiv(n_cols, BLOCK_C))
    _gather_dequant_kernel[grid](
        flat_idx,
        flat_idx.stride(0),
        flat_inv,
        flat_inv.stride(0),
        raw_fp8,
        raw_uint8,
        raw_bf16,
        out_buffer,
        out_buffer.stride(0),
        out_col_offset,
        inv_out_buffer,
        inv_out_buffer.stride(0),
        inv_out_col_offset,
        n_rows,
        n_cols,
        page_size,
        int(page_bytes),
        int(page_size * _NOPE_ROPE_STRIDE),
        NOPE_PAD=512,
        ROPE_DIM=_ROPE_DIM,
        NOPE_DIM_RT=_NOPE_DIM,
        OUT_D_STRIDE=_D,
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _softmax_epilogue_kernel(
    scores_ptr,                  # (n, H_q, T) fp32
    scores_n_stride: tl.int64,
    scores_h_stride: tl.int64,
    inv_ptr,                     # (n, T) uint8 (from gather workspace)
    inv_n_stride: tl.int64,
    weights_ptr,                 # (n, H_q, T) bf16
    weights_n_stride: tl.int64,
    weights_h_stride: tl.int64,
    lse_ptr,                     # (n, H_q) fp32 — raw lse (no sink)
    lse_n_stride: tl.int64,
    sink_ptr,                    # (H_q,) fp32 (unused if HAS_SINK==False)
    softmax_scale,
    T: tl.int32,
    HAS_SINK: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Fused scale + mask + logsumexp + (sink merge) + exp -> bf16.

    Grid: ``(n, H_q)``.  Each program handles one (n, h) score row of
    length ``T`` using a 2-pass online softmax over ``BLOCK_T`` tiles,
    then a final pass writing bf16 weights.  Raw lse (without sink) is
    stored to ``lse_ptr`` for the caller's attention lse.

    Lonely rows (all positions masked) get weights=0, so the downstream
    ``out = weights @ v`` is naturally 0 and the caller no longer needs
    a separate ``out[lonely] = 0`` step.
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    s_row = (
        scores_ptr
        + pid_n.to(tl.int64) * scores_n_stride
        + pid_h.to(tl.int64) * scores_h_stride
    )
    w_row = (
        weights_ptr
        + pid_n.to(tl.int64) * weights_n_stride
        + pid_h.to(tl.int64) * weights_h_stride
    )
    inv_row = inv_ptr + pid_n.to(tl.int64) * inv_n_stride

    NEG_INF = float("-inf")

    # ---- Pass 1: online softmax -> lse_raw = log sum_t exp(s_t).
    m = NEG_INF
    l = 0.0
    for t_start in tl.range(0, T, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < T
        s = tl.load(s_row + offs, mask=mask, other=NEG_INF).to(tl.float32)
        inv = tl.load(inv_row + offs, mask=mask, other=1)
        s = s * softmax_scale
        s = tl.where(inv != 0, NEG_INF, s)
        m_new = tl.maximum(m, tl.max(s, axis=0))
        # Guard exp() against -inf - -inf = nan (only matters before first valid).
        alpha = tl.where(m_new == NEG_INF, 0.0, tl.exp(m - m_new))
        beta = tl.where(m_new == NEG_INF, 0.0, tl.sum(tl.exp(s - m_new), axis=0))
        l = l * alpha + beta
        m = m_new

    lonely = l == 0.0
    lse_raw = tl.where(lonely, NEG_INF, m + tl.log(l))

    tl.store(lse_ptr + pid_n.to(tl.int64) * lse_n_stride + pid_h, lse_raw)

    if HAS_SINK:
        sink = tl.load(sink_ptr + pid_h).to(tl.float32)
        m_out = tl.maximum(lse_raw, sink)
        a = tl.where(lse_raw == NEG_INF, 0.0, tl.exp(lse_raw - m_out))
        b = tl.where(m_out == NEG_INF, 0.0, tl.exp(sink - m_out))
        lse_out = m_out + tl.log(a + b)
    else:
        lse_out = lse_raw

    # For lonely rows force lse_safe=0 so exp(-inf - 0) = 0 yields w=0.
    lse_safe = tl.where(lonely, 0.0, lse_out)

    # ---- Pass 2: weights = exp(s - lse_safe), downcast to bf16.
    for t_start in tl.range(0, T, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < T
        s = tl.load(s_row + offs, mask=mask, other=NEG_INF).to(tl.float32)
        inv = tl.load(inv_row + offs, mask=mask, other=1)
        s = s * softmax_scale
        s = tl.where(inv != 0, NEG_INF, s)
        w = tl.exp(s - lse_safe)
        tl.store(w_row + offs, w.to(tl.bfloat16), mask=mask)


def _triton_softmax_epilogue(
    scores: torch.Tensor,           # (n, H_q, T) fp32, contiguous
    inv_chunk_u8: torch.Tensor,     # (n, T) uint8, contiguous
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],  # (H_q,) fp32 or None
    lse_out: torch.Tensor,          # (n, H_q) fp32, contiguous (written in place)
) -> torch.Tensor:
    """Run ``_softmax_epilogue_kernel`` and return the bf16 weights tensor.
    """
    assert scores.is_contiguous() and scores.dtype == torch.float32
    assert inv_chunk_u8.is_contiguous() and inv_chunk_u8.dtype == torch.uint8
    n, H_q, T = scores.shape
    assert inv_chunk_u8.shape == (n, T)
    assert lse_out.shape == (n, H_q) and lse_out.dtype == torch.float32

    weights = torch.empty_like(scores, dtype=torch.bfloat16)

    has_sink = attn_sink is not None
    sink_ptr = attn_sink if has_sink else weights  # dummy when unused

    BLOCK_T = 256
    grid = (n, H_q)
    _softmax_epilogue_kernel[grid](
        scores,
        scores.stride(0),
        scores.stride(1),
        inv_chunk_u8,
        inv_chunk_u8.stride(0),
        weights,
        weights.stride(0),
        weights.stride(1),
        lse_out,
        lse_out.stride(0),
        sink_ptr,
        float(softmax_scale),
        T,
        HAS_SINK=has_sink,
        BLOCK_T=BLOCK_T,
        num_warps=4,
        num_stages=2,
    )
    return weights


def flash_mla_with_kvcache_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    block_table: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    tile_scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode using a fused Triton gather + torch GEMM math.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    B, s_q, H_q, D_qk = q.shape
    num_pages, page_size, H_k, bpt = k_cache.shape
    topk = indices.shape[-1]
    device = q.device

    # Build invalid_mask as a single fused expression (no separate clamp:
    # the Triton gather masks out-of-range loads via `invalid_mask`).
    max_valid = num_pages * page_size
    if topk_length is not None:
        topk_range = torch.arange(topk, device=device).view(1, 1, topk)
        invalid_mask = (
            (indices < 0)
            | (indices >= max_valid)
            | (topk_range >= topk_length.view(B, 1, 1))
        )
    else:
        invalid_mask = (indices < 0) | (indices >= max_valid)

    have_extra = extra_k_cache is not None and extra_indices_in_kvcache is not None
    if have_extra:
        extra_topk = extra_indices_in_kvcache.shape[-1]
        extra_num_pages, extra_page_size = (
            extra_k_cache.shape[0],
            extra_k_cache.shape[1],
        )
        extra_max_valid = extra_num_pages * extra_page_size
        if extra_topk_length is not None:
            extra_range = torch.arange(extra_topk, device=device).view(
                1, 1, extra_topk
            )
            extra_invalid = (
                (extra_indices_in_kvcache < 0)
                | (extra_indices_in_kvcache >= extra_max_valid)
                | (extra_range >= extra_topk_length.view(B, 1, 1))
            )
        else:
            extra_invalid = (extra_indices_in_kvcache < 0) | (extra_indices_in_kvcache >= extra_max_valid)
    else:
        extra_topk = 0

    total_topk = topk + extra_topk
    R = B * s_q
    q_rows = q.reshape(R, H_q, D_qk)
    indices_rows = indices.reshape(R, topk)
    invalid_rows = invalid_mask.reshape(R, topk)
    if have_extra:
        extra_indices_rows = extra_indices_in_kvcache.reshape(R, extra_topk)
        extra_invalid_rows = extra_invalid.reshape(R, extra_topk)

    out_rows = torch.empty(R, H_q, head_dim_v, dtype=torch.bfloat16, device=device)
    lse_rows = torch.empty(R, H_q, dtype=torch.float32, device=device)

    # Chunking bound by the bf16 KV tile + bf16 weights tile.  No fp32
    # KV copy anymore, so the per-row budget is ~ T*(2 bf16 KV bytes).
    # 512 MiB target — for typical workloads this collapses the loop to a
    # single iteration; only the extreme tail (very large B * topk) chunks.
    bytes_per_row = total_topk * _D * 2
    chunk_rows = max(
        1, min(R, (512 * 1024 * 1024) // max(1, bytes_per_row))
    )

    # Pre-allocate the KV tile + invalid-mask workspaces once.  Main and extra
    # gathers both write directly into slices of `kv_workspace` AND
    # `inv_workspace` (the gather kernel emits both), so there's no torch.cat,
    # no host-side `inv.copy_`, and no per-iteration allocation.
    kv_workspace = torch.empty(
        chunk_rows, total_topk, _D, dtype=torch.bfloat16, device=device
    )
    inv_workspace = torch.empty(
        chunk_rows, total_topk, dtype=torch.uint8, device=device
    )

    # Pre-cast Q to bf16 once (the loop is typically 1 iter; this still saves
    # a launch when it isn't).  Slice is a free view when D_qk == _D.
    q_kv = q_rows[..., :_D] if D_qk > _D else q_rows
    q_bf16_all = q_kv.to(torch.bfloat16).contiguous()

    for start in range(0, R, chunk_rows):
        end = min(start + chunk_rows, R)
        n = end - start

        # Views into the pre-allocated workspaces (no allocation, no copy).
        kv_chunk = kv_workspace[:n]    # (n, total_topk, _D) bf16
        inv_chunk = inv_workspace[:n]  # (n, total_topk) uint8

        # Main gather: writes kv_chunk[:, :topk, :] AND inv_chunk[:, :topk].
        _triton_gather_into(
            k_cache,
            indices_rows[start:end],
            invalid_rows[start:end],
            page_size,
            kv_chunk,
            out_col_offset=0,
            inv_out_buffer=inv_chunk,
            inv_out_col_offset=0,
        )

        if have_extra:
            # Extra gather: writes kv_chunk[:, topk:, :] AND inv_chunk[:, topk:].
            _triton_gather_into(
                extra_k_cache,
                extra_indices_rows[start:end],
                extra_invalid_rows[start:end],
                extra_page_size,
                kv_chunk,
                out_col_offset=topk,
                inv_out_buffer=inv_chunk,
                inv_out_col_offset=topk,
            )

        q_chunk = q_bf16_all[start:end]  # (n, H_q, kv_d) bf16

        # QK: bf16 GEMM, upcast to fp32 for softmax math.
        scores = torch.einsum("nhd,ntd->nht", q_chunk, kv_chunk).float()

        # Fused softmax epilogue replaces 6 torch ops (mul_, masked_fill_,
        # logsumexp, sink merge, lse_for_out[lonely]=inf, exp().to(bf16)) with
        # one 2-pass Triton kernel.  Writes raw lse into lse_rows directly;
        # lonely-row weights are zeroed so the downstream PV produces 0.
        weights = _triton_softmax_epilogue(
            scores,
            inv_chunk,
            softmax_scale,
            attn_sink,
            lse_rows[start:end],
        )

        # PV: bf16 @ bf16 -> bf16.  kv_chunk[..., :head_dim_v] is a free view
        # when head_dim_v == _D.
        out_chunk = torch.einsum(
            "nht,ntv->nhv", weights, kv_chunk[..., :head_dim_v]
        )
        out_rows[start:end] = out_chunk

        del q_chunk, scores, weights, out_chunk

    out = out_rows.reshape(B, s_q, H_q, head_dim_v)
    lse = lse_rows.reshape(B, s_q, H_q).permute(0, 2, 1)
    return out, lse
