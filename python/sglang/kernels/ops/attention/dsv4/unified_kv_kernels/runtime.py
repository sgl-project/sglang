"""Runtime glue for the unified_kv backend.

Builds unified_kv-style flat ``kv_indices`` / ``kv_indptr`` from SGLang's already-computed
DSV4 metadata, scatters SWA K into the bf16 ``unified_kv`` ring, and dispatches the
vendored paged decode/prefill kernels.

unified_kv[L] layout (page_size 1, bf16, row-major):
  - rows ``[0, swa_pages)``    = SWA ring (``state_slot * win + pos % win``);
  - rows ``[swa_pages, ...)``  = compressed K (``swa_pages + page_index``), where
    SGLang metadata already encodes the compressed slot id:
      HCA (ratio 128): ``c128_page_indices``      (== phys_block, k_per_block=1)
      CSA (ratio   4): ``c4_sparse_page_indices``  (== phys_block*32 + slot)

Index layout: RAGGED-PACKED. Each token's segment is tightly packed
(``kv_indptr`` is a true prefix sum of per-token valid lengths) so the
attention K-loop scans only real entries. The backing buffer is still
allocated at the fixed worst-case capacity ``N * (win + Wc)`` so its shape is
static across CUDA-graph replay; only ``kv_indptr`` values (and the written
prefix) vary per forward. Compressed valid entries are front-packed in the
``*_page_indices`` rows (the same contract the non-unified_kv flashmla path relies
on via ``topk_length``); the per-token compressed count is recovered from the
``kv_indptr`` delta inside the kernel, so no extra length tensor is threaded.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode_indices import (
    write_v4_paged_decode_indices,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import (
    sparse_attn_v4_paged_prefill,
)


# ---------------------------------------------------------------------------
# SWA ring scatter
# ---------------------------------------------------------------------------
@triton.jit
def _swa_scatter_kernel(
    kv_ptr,  # [T, D] bf16
    state_slot_ptr,  # [T] int
    positions_ptr,  # [T] int
    final_pos_ptr,  # [T] int
    unified_ptr,  # [pages, D] bf16
    n_rows,
    ring_stride,  # SWA ring per-slot stride
    win: tl.constexpr,
    D: tl.constexpr,
    HAS_FINAL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    pos = tl.load(positions_ptr + row)
    if HAS_FINAL:
        fp = tl.load(final_pos_ptr + row)
        if pos <= fp - win:
            return
    s = tl.load(state_slot_ptr + row)
    loc = s * ring_stride + (pos % ring_stride)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    vals = tl.load(kv_ptr + row * D + offs, mask=mask, other=0.0)
    tl.store(unified_ptr + loc * D + offs, vals, mask=mask)


def store_swa_into_unified(
    *,
    kv: torch.Tensor,  # [T, head_dim] bf16
    state_slot: torch.Tensor,  # [T] int
    positions: torch.Tensor,  # [T] int
    unified_kv: torch.Tensor,  # [pages, head_dim] bf16
    win: int,  # SWA attention window length
    ring_stride: int,  # SWA ring stride
    final_pos: Optional[torch.Tensor] = None,  # [T] req's last position
) -> None:
    n_rows, D = kv.shape
    if n_rows == 0:
        return

    has_final = final_pos is not None
    fp_arg = final_pos if has_final else positions
    assert kv.is_contiguous() and kv.dtype == unified_kv.dtype
    assert state_slot.is_contiguous() and positions.is_contiguous()
    assert fp_arg.is_contiguous()
    _swa_scatter_kernel[(n_rows,)](
        kv,
        state_slot,
        positions,
        fp_arg,
        unified_kv,
        n_rows,
        ring_stride,
        win=win,
        D=D,
        HAS_FINAL=has_final,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=8,
    )


@triton.jit
def _scatter_loc_kernel(
    kv_ptr,  # [T, D] bf16
    loc_ptr,  # [T] int (unified row index; <0 => skip)
    unified_ptr,  # [pages, D] bf16
    n_rows,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    loc = tl.load(loc_ptr + row).to(tl.int64)
    if loc < 0:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    vals = tl.load(kv_ptr + row * D + offs, mask=mask, other=0.0)
    tl.store(unified_ptr + loc * D + offs, vals, mask=mask)


def scatter_bf16_into_unified(
    *,
    kv: torch.Tensor,  # [T, head_dim] bf16 (already norm+rope'd)
    loc: torch.Tensor,  # [T] int32/int64 unified ring row; <0 => skip
    unified_kv: torch.Tensor,  # [pages, head_dim] bf16
) -> None:
    """Scatter already-norm+rope'd bf16 K into ``unified_kv[loc]`` (skip loc < 0).

    Companion to ``store_swa_into_unified`` for callers that already hold the
    precomputed ring row index (the DSpark draft: ``get_unified_swa_loc`` for the
    draft forward, or the commit-inject layout for target-hidden injection) and
    need per-row commit masking expressed as ``loc == -1``.
    """
    n_rows, D = kv.shape
    if n_rows == 0:
        return
    assert kv.is_contiguous() and kv.dtype == unified_kv.dtype
    assert loc.is_contiguous()
    assert unified_kv.is_contiguous()
    _scatter_loc_kernel[(n_rows,)](
        kv,
        loc,
        unified_kv,
        n_rows,
        D=D,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=8,
    )


# ---------------------------------------------------------------------------
# Ragged indptr helper (shared by the decode streams + prefill builders)
# ---------------------------------------------------------------------------
def _lengths_to_indptr(lengths: torch.Tensor) -> torch.Tensor:
    """[N] int32 per-token lengths -> [N+1] int32 indptr"""
    return F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))


def decode(
    *,
    q: torch.Tensor,  # [T, H, D] (local heads)
    unified_kv: torch.Tensor,  # [pages, D] bf16
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,  # [H] fp32
    softmax_scale: float,
) -> torch.Tensor:
    return sparse_attn_v4_paged_decode(
        q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale
    )


@triton.jit
def _fill_compress_tail_kernel(
    indices_ptr,  # [*] int32 (out)
    indptr_ptr,  # [N+1] int32
    prefix_len_ptr,  # [N] int
    page_idx_ptr,  # [N, Wc] int
    valid_len_ptr,  # [N] int
    swa_pages,
    Wc: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per token: write valid_len compressed slots (swa_pages+page_idx, -1 for empty) into the stream tail at indptr[t]+prefix_len[t]."""
    t = tl.program_id(0)
    cbase = tl.load(indptr_ptr + t) + tl.load(prefix_len_ptr + t).to(tl.int32)
    nc = tl.load(valid_len_ptr + t).to(tl.int32)
    for off in tl.range(0, Wc, BLOCK):
        j = off + tl.arange(0, BLOCK)
        m = j < nc
        j_clamped = tl.minimum(j, Wc - 1)
        pi = tl.load(page_idx_ptr + t * Wc + j_clamped, mask=m, other=-1).to(tl.int32)
        slot = tl.where(pi >= 0, pi + swa_pages, -1)
        tl.store(indices_ptr + cbase + j, slot, mask=m)


def fill_compress_tail(
    *,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    prefix_len: torch.Tensor,
    page_indices: torch.Tensor,  # [N, Wc] int32
    valid_len: torch.Tensor,
    swa_pages: int,
) -> None:
    N, Wc = page_indices.shape
    if N == 0:
        return
    assert prefix_len.is_contiguous() and page_indices.is_contiguous()
    assert valid_len.is_contiguous()
    _fill_compress_tail_kernel[(N,)](
        indices,
        indptr,
        prefix_len,
        page_indices,
        valid_len,
        swa_pages,
        Wc=Wc,
        BLOCK=min(1024, triton.next_power_of_2(max(Wc, 1))),
        num_warps=4,
    )


def build_decode_streams(
    *,
    state_slot: torch.Tensor,  # [N] int
    positions: torch.Tensor,  # [N] int
    swa_len: torch.Tensor,  # [N] int
    hca_len: torch.Tensor,  # [N] int
    csa_len: torch.Tensor,  # [N] int
    hca_page_indices: torch.Tensor,  # [N, hca_width] int32
    csa_width: int,
    win: int,  # SWA attention window length
    ring_stride: int,  # SWA ring per-slot stride
    swa_pages: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    device = state_slot.device
    N = state_slot.shape[0]
    assert state_slot.is_contiguous() and positions.is_contiguous()
    state_slot = state_slot.to(torch.int32)
    positions = positions.to(torch.int32)
    hca_width = hca_page_indices.shape[1]

    swa_p = _lengths_to_indptr(swa_len)
    hca_p = _lengths_to_indptr(swa_len + hca_len)
    csa_p = _lengths_to_indptr(swa_len + csa_len)

    swa_i = torch.empty(N * win, dtype=torch.int32, device=device)
    hca_i = torch.empty(N * (win + hca_width), dtype=torch.int32, device=device)
    csa_i = torch.empty(N * (win + csa_width), dtype=torch.int32, device=device)

    if N > 0:
        batch_id = torch.arange(N, dtype=torch.int32, device=device)
        write_v4_paged_decode_indices(
            state_slot_per_seq=state_slot,
            batch_id_per_token=batch_id,
            positions=positions,
            swa_indptr=swa_p,
            csa_indptr=csa_p,
            hca_indptr=hca_p,
            swa_indices=swa_i,
            csa_indices=csa_i,
            hca_indices=hca_i,
            T=N,
            win=win,
            ring_stride=ring_stride,
        )
        fill_compress_tail(
            indices=hca_i,
            indptr=hca_p,
            prefix_len=swa_len,
            page_indices=hca_page_indices[:N],
            valid_len=hca_len,
            swa_pages=swa_pages,
        )
    return swa_i, swa_p, hca_i, hca_p, csa_i, csa_p


# ---------------------------------------------------------------------------
# Prefill index builder (ragged-packed: paged prefix + flat extend)
# ---------------------------------------------------------------------------
@triton.jit
def _prefill_lengths_kernel(
    positions_ptr,  # [T] int
    chunk_start_ptr,  # [T] int
    page_idx_ptr,  # [T, Wc] int (front-packed, -1 padded)
    prefix_len_ptr,  # [T] int32 out
    extend_len_ptr,  # [T] int32 out
    win: tl.constexpr,
    Wc: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per token: write extend/prefix segment lengths"""
    t = tl.program_id(0)
    pos = tl.load(positions_ptr + t).to(tl.int32)
    cstart = tl.load(chunk_start_ptr + t).to(tl.int32)
    tpic = pos - cstart
    swa_low = tl.maximum(pos - win + 1, 0)
    extend_count = tl.minimum(tpic + 1, win)
    prefix_swa_count = tl.minimum(tl.maximum(cstart - swa_low, 0), win)
    tl.store(extend_len_ptr + t, extend_count)
    if HAS_COMPRESS:
        nc = 0
        for off in tl.range(0, Wc, BLOCK):
            j = off + tl.arange(0, BLOCK)
            m = j < Wc
            j_clamped = tl.minimum(j, Wc - 1)
            pi = tl.load(page_idx_ptr + t * Wc + j_clamped, mask=m, other=-1)
            nc += tl.sum(tl.where(m & (pi >= 0), 1, 0))
        tl.store(prefix_len_ptr + t, prefix_swa_count + nc)
    else:
        tl.store(prefix_len_ptr + t, prefix_swa_count)


@triton.jit
def _build_prefill_indices_kernel(
    positions_ptr,  # [T] int
    chunk_start_ptr,  # [T] int
    cu_q_ptr,  # [T] int
    state_slot_ptr,  # [T] int
    page_idx_ptr,  # [T, Wc] int (front-packed, -1 padded)
    pre_indptr_ptr,  # [T+1] int32 (prefix stream ragged indptr)
    ext_indptr_ptr,  # [T+1] int32 (extend stream ragged indptr)
    pre_out_ptr,
    ext_out_ptr,
    swa_pages,
    ring_stride,  # SWA ring per-slot stride
    win: tl.constexpr,
    Wc: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per token: write extend rows + prefix (SWA ring slots ++ swa_pages+compressed slots) as two ragged segments"""
    t = tl.program_id(0)
    pos = tl.load(positions_ptr + t).to(tl.int32)
    cstart = tl.load(chunk_start_ptr + t).to(tl.int32)
    cuq = tl.load(cu_q_ptr + t).to(tl.int32)
    s = tl.load(state_slot_ptr + t).to(tl.int32)

    tpic = pos - cstart
    swa_low = tl.maximum(pos - win + 1, 0)
    extend_count = tl.minimum(tpic + 1, win)
    prefix_swa_count = tl.minimum(tl.maximum(cstart - swa_low, 0), win)

    ebase = tl.load(ext_indptr_ptr + t)
    pbase = tl.load(pre_indptr_ptr + t)

    # ---- extend: rows into the current-chunk kv tensor ----
    ext_start = cuq + tpic - extend_count + 1
    for off in tl.range(0, win, BLOCK):
        k = off + tl.arange(0, BLOCK)
        m = k < extend_count
        tl.store(ext_out_ptr + ebase + k, ext_start + k, mask=m)

    # ---- prefix SWA: prior-chunk ring slots (stride = ring_stride) ----
    for off in tl.range(0, win, BLOCK):
        k = off + tl.arange(0, BLOCK)
        m = k < prefix_swa_count
        gp = swa_low + k
        tl.store(pre_out_ptr + pbase + k, s * ring_stride + (gp % ring_stride), mask=m)

    # ---- prefix compressed: swa_pages + front-packed page index ----
    if HAS_COMPRESS:
        nc = tl.load(pre_indptr_ptr + t + 1) - pbase - prefix_swa_count
        cbase = pbase + prefix_swa_count
        for off in tl.range(0, Wc, BLOCK):
            j = off + tl.arange(0, BLOCK)
            m = j < nc
            j_clamped = tl.minimum(j, Wc - 1)
            pi = tl.load(page_idx_ptr + t * Wc + j_clamped, mask=m, other=0).to(
                tl.int32
            )
            tl.store(pre_out_ptr + cbase + j, pi + swa_pages, mask=m)


def build_prefill_indices(
    *,
    compress_ratio: int,
    state_slot: torch.Tensor,  # [T] int (per token)
    positions: torch.Tensor,  # [T] int (per token absolute position)
    chunk_start: torch.Tensor,  # [T] int (absolute start of this chunk for token's seq)
    cu_q: torch.Tensor,  # [T] int (row in extend `kv` of the seq's first chunk token)
    win: int,  # SWA attention window length
    ring_stride: int,  # SWA ring per-slot stride / modulo (win_with_spec)
    swa_pages: int,
    c128_page_indices: Optional[torch.Tensor],
    c4_sparse_page_indices: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ragged prefill indices: prefix (SWA ring + swa_pages + compressed) into unified_kv + extend into current-chunk kv; returns (prefix_indices, prefix_indptr, extend_indices, extend_indptr)."""
    device = state_slot.device
    T = state_slot.shape[0]
    assert positions.is_contiguous() and chunk_start.is_contiguous()
    assert cu_q.is_contiguous() and state_slot.is_contiguous()

    if compress_ratio == 0:
        page_idx = None
    elif compress_ratio == 128:
        assert c128_page_indices is not None
        page_idx = c128_page_indices[:T]
    elif compress_ratio == 4:
        assert c4_sparse_page_indices is not None
        page_idx = c4_sparse_page_indices[:T]
    else:
        raise ValueError(f"bad compress_ratio {compress_ratio}")

    has_compress = page_idx is not None
    if has_compress:
        assert page_idx.is_contiguous()
    Wc = page_idx.shape[1] if has_compress else 0

    block = min(1024, triton.next_power_of_2(max(win, Wc, 1)))
    prefix_len = torch.empty(T, dtype=torch.int32, device=device)
    extend_len = torch.empty(T, dtype=torch.int32, device=device)
    _prefill_lengths_kernel[(T,)](
        positions,
        chunk_start,
        page_idx if has_compress else positions,  # dummy ptr when no compress
        prefix_len,
        extend_len,
        win=win,
        Wc=Wc if has_compress else 1,
        HAS_COMPRESS=has_compress,
        BLOCK=block,
        num_warps=4,
    )
    kv_indptr_prefix = _lengths_to_indptr(prefix_len)
    kv_indptr_extend = _lengths_to_indptr(extend_len)

    kv_indices_prefix = torch.empty(T * (win + Wc), dtype=torch.int32, device=device)
    kv_indices_extend = torch.empty(T * win, dtype=torch.int32, device=device)

    _build_prefill_indices_kernel[(T,)](
        positions,
        chunk_start,
        cu_q,
        state_slot,
        page_idx if has_compress else state_slot,  # dummy ptr when no compress
        kv_indptr_prefix,
        kv_indptr_extend,
        kv_indices_prefix,
        kv_indices_extend,
        swa_pages,
        ring_stride,
        win=win,
        Wc=Wc if has_compress else 1,
        HAS_COMPRESS=has_compress,
        BLOCK=block,
        num_warps=4,
    )
    return kv_indices_prefix, kv_indptr_prefix, kv_indices_extend, kv_indptr_extend


def prefill(
    *,
    q: torch.Tensor,  # [T, H, D]
    unified_kv: torch.Tensor,  # [pages, D]
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_extend: torch.Tensor,  # [T, D] current-chunk K (bf16, norm+rope'd)
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return sparse_attn_v4_paged_prefill(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_extend,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
    )
