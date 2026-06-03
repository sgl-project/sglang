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

from sglang.srt.layers.attention.dsv4.unified_kv_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
)
from sglang.srt.layers.attention.dsv4.unified_kv_kernels.paged_prefill import (
    sparse_attn_v4_paged_prefill,
)
from sglang.srt.layers.attention.dsv4.unified_kv_kernels.store import _scatter_rows


# ---------------------------------------------------------------------------
# SWA ring scatter
# ---------------------------------------------------------------------------
@triton.jit
def _swa_scatter_kernel(
    kv_ptr,  # [T, D] bf16
    state_slot_ptr,  # [T] int32
    positions_ptr,  # [T] int32
    final_pos_ptr,  # [T] int32 (req's last position)
    unified_ptr,  # [pages, D] bf16
    n_rows,
    cs,  # SWA ring per-slot stride / modulo (win_with_spec)
    win: tl.constexpr,
    D: tl.constexpr,
    HAS_FINAL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per token: compute ring slot = state_slot*cs + pos%cs and
    scatter the K row into unified_kv, skipping tokens outside the final SWA window
    (pos <= final_pos - win) so >win prefill chunks don't race on aliased slots."""
    row = tl.program_id(0)
    if row >= n_rows:
        return
    pos = tl.load(positions_ptr + row)
    if HAS_FINAL:
        fp = tl.load(final_pos_ptr + row)
        if pos <= fp - win:
            return
    s = tl.load(state_slot_ptr + row)
    loc = s * cs + (pos % cs)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    vals = tl.load(kv_ptr + row * D + offs, mask=mask, other=0.0)
    tl.store(unified_ptr + loc * D + offs, vals, mask=mask)


def store_swa_into_unified(
    *,
    kv: torch.Tensor,  # [T, head_dim] bf16, already norm+rope'd by the model
    state_slot: torch.Tensor,  # [T] int
    positions: torch.Tensor,  # [T] int
    unified_kv: torch.Tensor,  # [pages, head_dim] bf16
    win: int,  # SWA attention window length (final-window filter)
    cs: int,  # SWA ring stride / modulo (win_with_spec; == win when no spec)
    final_pos: Optional[torch.Tensor] = None,  # [T] req's last position
) -> None:
    """Fused single-Triton-kernel SWA ring scatter (loc compute + window filter +
    row scatter in one launch).

    During prefill a chunk can contain >win tokens of one request; tokens cs apart
    alias to the same ring slot. A parallel scatter of all of them RACES and an older
    token may win, corrupting the ring decode later reads. So we only write each
    request's FINAL window: positions in (final_pos - win, final_pos]. Those are <=win
    <= cs consecutive positions -> distinct ring slots -> race-free. For decode
    (final_pos==pos) every token qualifies.
    """
    kv = kv.to(unified_kv.dtype).contiguous()
    n_rows, D = kv.shape
    if n_rows == 0:
        return
    state_slot = state_slot.to(torch.int32).contiguous()
    positions = positions.to(torch.int32).contiguous()
    has_final = final_pos is not None
    fp_arg = final_pos.to(torch.int32).contiguous() if has_final else positions
    _swa_scatter_kernel[(n_rows,)](
        kv,
        state_slot,
        positions,
        fp_arg,
        unified_kv,
        n_rows,
        cs,
        win=win,
        D=D,
        HAS_FINAL=has_final,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=8,
    )


def store_compressed_into_unified(
    *,
    kv_compressed: torch.Tensor,  # [N, head_dim] (bf16-castable)
    out_loc: torch.Tensor,  # [N] int  == c128_out_loc / c4_out_loc
    unified_kv: torch.Tensor,  # [pages, head_dim] bf16
    swa_pages: int,
    valid: Optional[torch.Tensor] = None,
) -> None:
    """Scatter compressed KV into unified_kv compress region at swa_pages+out_loc."""
    _scatter_rows(
        kv_compressed.to(unified_kv.dtype),
        out_loc.to(torch.int32),
        unified_kv,
        dst_base=swa_pages,
        valid=valid,
    )


# ---------------------------------------------------------------------------
# Ragged indptr helper (shared by the decode streams + prefill builders)
# ---------------------------------------------------------------------------
def _lengths_to_indptr(lengths: torch.Tensor) -> torch.Tensor:
    """[N] int32 per-token lengths -> [N+1] int32 indptr (true prefix sum,
    indptr[0]=0). Two kernels: a single cumsum (kept in int32) + a leading-0
    pad — no host-side cast/clamp/reduce chain."""
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


# ---------------------------------------------------------------------------
# Amortized decode index build (once per forward, unified_kv-style 3 streams).
#
# Per layer only ONE compress_ratio is active, but the SWA prefix is
# layer-invariant and the HCA tail is layer-invariant (block-table derived), so
# the dense (swa) + HCA streams are built ONCE per forward. The CSA tail depends
# on the per-layer indexer top-k, so the csa stream's SWA prefix is built once
# and its compress tail is filled per-c4-layer by ``fill_compress_tail`` (Phase
# C). All three per-token lengths come straight from SGLang metadata (no length
# kernels): swa_len = swa_topk_lengths, hca_len = c128_topk_lengths_clamp1,
# csa_len = c4_sparse_topk_lengths.
# ---------------------------------------------------------------------------
from sglang.srt.layers.attention.dsv4.unified_kv_kernels.paged_decode_indices import (
    write_v4_paged_decode_indices,
)


@triton.jit
def _fill_compress_tail_kernel(
    indices_ptr,  # [*] int32 — ragged stream buffer (in/out)
    indptr_ptr,  # [N+1] int32 — stream indptr
    prefix_len_ptr,  # [N] int32 — per-token SWA-prefix count (tail offset)
    page_idx_ptr,  # [N, Wc] int32 — front-packed compress page ids, -1 padded
    valid_len_ptr,  # [N] int32 — per-token compress entry count
    swa_pages,
    Wc: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per token: write ``valid_len`` compressed slots
    (``swa_pages + page_idx``) into the stream tail at
    ``indices[indptr[t] + prefix_len[t] : + valid_len[t]]``."""
    t = tl.program_id(0)
    cbase = tl.load(indptr_ptr + t) + tl.load(prefix_len_ptr + t)
    nc = tl.load(valid_len_ptr + t)
    for off in tl.range(0, Wc, BLOCK):
        j = off + tl.arange(0, BLOCK)
        m = j < nc
        pi = tl.load(page_idx_ptr + t * Wc + j, mask=m, other=-1)
        # Reserved length (nc) may exceed the actual front-packed valid count
        # (the metadata `*_topk_lengths` are clamped to >=1, so a 0-committed
        # early token reserves one slot whose page_idx is -1). Preserve -1 so
        # the sparse-attn kernel skips it instead of reading swa_pages-1.
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
    """Fill the compress tail of a ragged stream (HCA in Phase B, CSA per c4
    layer in Phase C). ``page_indices`` already encodes the unified compress
    slot (block*k_per_block + slot); we only add ``swa_pages``."""
    N, Wc = page_indices.shape
    if N == 0:
        return
    _fill_compress_tail_kernel[(N,)](
        indices,
        indptr,
        prefix_len.to(torch.int32),
        page_indices.to(torch.int32),
        valid_len.to(torch.int32),
        swa_pages,
        Wc=Wc,
        BLOCK=min(1024, triton.next_power_of_2(max(Wc, 1))),
        num_warps=4,
    )


def build_decode_streams(
    *,
    state_slot: torch.Tensor,  # [N] int — per-token SWA ring slot (req_pool_indices)
    positions: torch.Tensor,  # [N] int — global token position
    swa_len: torch.Tensor,  # [N] int — min(seq_len, win)
    hca_len: torch.Tensor,  # [N] int — committed HCA entries
    csa_len: torch.Tensor,  # [N] int — committed CSA entries (capped to index_topk)
    c128_page_indices: torch.Tensor,  # [N, Wc128] int32 — HCA tail source
    csa_width: int,  # Wc4 — CSA stream tail capacity
    win: int,  # SWA attention window length (per-token prefix count cap)
    cs: int,  # SWA ring per-slot stride / modulo (win_with_spec)
    swa_pages: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Build the three ragged decode index streams ONCE per forward.

    Returns ``(swa_i, swa_p, hca_i, hca_p, csa_i, csa_p)``:
      - swa: SWA ring slots only (ratio-0 layers).
      - hca: SWA prefix + committed HCA tail (ratio-128 layers) — fully built.
      - csa: SWA prefix built; CSA tail RESERVED (sized via ``csa_len``) and
        filled per c4 layer by ``fill_compress_tail`` (Phase C).
    The SWA prefix for all three is written in one launch via the vendored
    ``write_v4_paged_decode_indices`` (ring stride ``cs`` = win_with_spec; the
    per-token prefix count stays ``min(seq_len, win)``).
    """
    device = state_slot.device
    N = state_slot.shape[0]
    state_slot = state_slot.to(torch.int32).contiguous()
    positions = positions.to(torch.int32).contiguous()
    swa_len = swa_len.to(torch.int32)
    hca_len = hca_len.to(torch.int32)
    csa_len = csa_len.to(torch.int32)
    Wc128 = c128_page_indices.shape[1]

    swa_p = _lengths_to_indptr(swa_len)
    hca_p = _lengths_to_indptr(swa_len + hca_len)
    csa_p = _lengths_to_indptr(swa_len + csa_len)

    swa_i = torch.empty(N * win, dtype=torch.int32, device=device)
    hca_i = torch.empty(N * (win + Wc128), dtype=torch.int32, device=device)
    csa_i = torch.empty(N * (win + csa_width), dtype=torch.int32, device=device)

    if N > 0:
        # SWA prefix -> all three streams in one kernel. batch_id = arange(N)
        # (decode is 1 token/seq); ring stride = cs (win_with_spec).
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
            cs=cs,
        )
        # HCA tail is layer-invariant — fill now.
        fill_compress_tail(
            indices=hca_i,
            indptr=hca_p,
            prefix_len=swa_len,
            page_indices=c128_page_indices[:N],
            valid_len=hca_len,
            swa_pages=swa_pages,
        )
    return swa_i, swa_p, hca_i, hca_p, csa_i, csa_p


# ---------------------------------------------------------------------------
# Prefill index builder (ragged-packed: paged prefix + flat extend)
# ---------------------------------------------------------------------------
@triton.jit
def _prefill_lengths_kernel(
    positions_ptr,  # [T] int32
    chunk_start_ptr,  # [T] int32
    page_idx_ptr,  # [T, Wc] int32 (front-packed, -1 padded)
    prefix_len_ptr,  # [T] int32 out
    extend_len_ptr,  # [T] int32 out
    win: tl.constexpr,
    Wc: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per token. Writes prefix_len (= prefix_swa_count + n_comp)
    and extend_len (= extend_count). Folds the prior host clamp/sub/min/max +
    ``(page_idx>=0).sum(1)`` chain into one launch. The per-token formulas match
    ``_build_prefill_indices_kernel`` exactly (so indptr deltas stay consistent)."""
    t = tl.program_id(0)
    pos = tl.load(positions_ptr + t)
    cs = tl.load(chunk_start_ptr + t)
    tpic = pos - cs
    swa_low = tl.maximum(pos - win + 1, 0)
    extend_count = tl.minimum(tpic + 1, win)
    prefix_swa_count = tl.minimum(tl.maximum(cs - swa_low, 0), win)
    tl.store(extend_len_ptr + t, extend_count)
    if HAS_COMPRESS:
        nc = 0
        for off in tl.range(0, Wc, BLOCK):
            j = off + tl.arange(0, BLOCK)
            m = j < Wc
            pi = tl.load(page_idx_ptr + t * Wc + j, mask=m, other=-1)
            nc += tl.sum(tl.where(m & (pi >= 0), 1, 0))
        tl.store(prefix_len_ptr + t, prefix_swa_count + nc)
    else:
        tl.store(prefix_len_ptr + t, prefix_swa_count)


@triton.jit
def _build_prefill_indices_kernel(
    positions_ptr,  # [T] int32
    chunk_start_ptr,  # [T] int32
    cu_q_ptr,  # [T] int32
    state_slot_ptr,  # [T] int32
    page_idx_ptr,  # [T, Wc] int32 (front-packed, -1 padded)
    pre_indptr_ptr,  # [T+1] int32 (ragged prefix sum, prefix stream)
    ext_indptr_ptr,  # [T+1] int32 (ragged prefix sum, extend stream)
    pre_out_ptr,  # [>= pre_indptr[T]] int32
    ext_out_ptr,  # [>= ext_indptr[T]] int32
    swa_pages,
    ring_stride,  # SWA ring per-slot stride / modulo (win_with_spec)
    win: tl.constexpr,
    Wc: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per token. Writes two ragged segments:

      prefix = [SWA prior-chunk ring (prefix_swa_count)] ++
               [swa_pages + compressed slot (nc)]
      extend = [current-chunk kv rows (extend_count)]

    Per-token quantities (mirror the prior torch build):
      token_pos_in_chunk = pos - chunk_start
      swa_low            = max(pos - win + 1, 0)
      extend_count       = min(token_pos_in_chunk + 1, win)
      prefix_swa_count   = min(max(chunk_start - swa_low, 0), win)
    ``nc`` is recovered from the prefix indptr delta minus prefix_swa_count.
    """
    t = tl.program_id(0)
    pos = tl.load(positions_ptr + t)
    cs = tl.load(chunk_start_ptr + t)
    cuq = tl.load(cu_q_ptr + t)
    s = tl.load(state_slot_ptr + t)

    tpic = pos - cs
    swa_low = tl.maximum(pos - win + 1, 0)
    extend_count = tl.minimum(tpic + 1, win)
    prefix_swa_count = tl.minimum(tl.maximum(cs - swa_low, 0), win)

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
            pi = tl.load(page_idx_ptr + t * Wc + j, mask=m, other=0)
            tl.store(pre_out_ptr + cbase + j, pi + swa_pages, mask=m)


def build_prefill_indices(
    *,
    compress_ratio: int,
    state_slot: torch.Tensor,  # [T] int (per token)
    positions: torch.Tensor,  # [T] int (per token absolute position)
    chunk_start: torch.Tensor,  # [T] int (absolute start of this chunk for token's seq)
    cu_q: torch.Tensor,  # [T] int (row in extend `kv` of the seq's first chunk token)
    win: int,  # SWA attention window length
    cs: int,  # SWA ring per-slot stride / modulo (win_with_spec)
    swa_pages: int,
    c128_page_indices: Optional[torch.Tensor],
    c4_sparse_page_indices: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ragged-packed prefill indices in one Triton launch.

    Returns ``(kv_indices_prefix, kv_indptr_prefix, kv_indices_extend,
    kv_indptr_extend)``. Replaces the prior ~10 torch ops (arange/clamp/where/
    cat/reshape/...). Buffers keep the fixed worst-case capacities
    (prefix ``T*(win+Wc)``, extend ``T*win``) for CUDA-graph shape stability;
    the indptrs are true prefix sums so the prefill kernel scans only the real
    per-token entries in each of the two KV sources.
    """
    device = state_slot.device
    T = state_slot.shape[0]
    pos_i = positions.to(torch.int32).contiguous()
    cs_i = chunk_start.to(torch.int32).contiguous()
    cuq_i = cu_q.to(torch.int32).contiguous()
    ss_i = state_slot.to(torch.int32).contiguous()

    if compress_ratio == 0:
        page_idx = None
    elif compress_ratio == 128:
        assert c128_page_indices is not None
        page_idx = c128_page_indices[:T].to(torch.int32).contiguous()
    elif compress_ratio == 4:
        assert c4_sparse_page_indices is not None
        page_idx = c4_sparse_page_indices[:T].to(torch.int32).contiguous()
    else:
        raise ValueError(f"bad compress_ratio {compress_ratio}")

    has_compress = page_idx is not None
    Wc = page_idx.shape[1] if has_compress else 0

    block = min(1024, triton.next_power_of_2(max(win, Wc, 1)))
    # Per-token segment lengths in one kernel, then cumsum -> ragged indptrs.
    prefix_len = torch.empty(T, dtype=torch.int32, device=device)
    extend_len = torch.empty(T, dtype=torch.int32, device=device)
    _prefill_lengths_kernel[(T,)](
        pos_i,
        cs_i,
        page_idx if has_compress else pos_i,  # dummy ptr when no compress
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
        pos_i,
        cs_i,
        cuq_i,
        ss_i,
        page_idx if has_compress else ss_i,  # dummy ptr when no compress
        kv_indptr_prefix,
        kv_indptr_extend,
        kv_indices_prefix,
        kv_indices_extend,
        swa_pages,
        cs,
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
