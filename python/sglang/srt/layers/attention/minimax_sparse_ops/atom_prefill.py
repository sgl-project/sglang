# Copyright 2025 SGLang Team
"""Experimental ATOM-style MiniMax-M3 sparse prefill.

This is intentionally narrow and env-gated. It replaces only the main sparse
attention step after SGLang has already produced ``topk_idx`` with the index
attention path. The implementation builds the sparse page table that ATOM feeds
into AITER's Gluon paged-attention kernel, then treats each prefill query token
as an independent length-1 decode sequence.

DEVIATION from the source port (alexsun07/sglang ``m3-atom-prefill-port``,
commits 1d5282863c + 6f1821cfe0 + 1185d5ca88): the source requires the KV pool
to be allocated with ``--page-size`` equal to the Gluon page size (16 or 64) so
its vectorized_5d K/V buffer is fed to the kernel as a zero-copy view. Here the
main KV pool MUST stay NHD ``[max_slots, 1, head_dim]`` (a pool-wide 5D switch
tanked decode), so instead, per sparse layer and per prefill forward, the
per-request context span (cached prefix + current chunk; the chunk's K/V are
written to the pool by the rope+cache step BEFORE main attention runs) is
gathered from the NHD pool into a persistent scratch buffer laid out as SHUFFLE
5D pages of ``GLUON_PAGE_SIZE`` tokens:

- key scratch:   [num_pages, 1, head_dim // x, GLUON_PAGE_SIZE, x]
- value scratch: [num_pages, 1, GLUON_PAGE_SIZE // x, head_dim, x] (transposed)

with ``x = 16 // dtype.itemsize`` (8 for bf16/fp16), exactly the layouts
``pa_decode_gluon`` asserts. Each request's context occupies a contiguous,
position-ordered run of scratch pages (rounded up to whole 128-token sparse
blocks), so the per-query-token block table is pure arithmetic:
``page_start[req] + block_id * pages_per_block + j`` -- no req_to_token lookup
in the builder (the gather kernel does the slot indirection once).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.attention.aiter_utils import (
    get_recommended_splits,
    pa_decode_gluon,
)

SPARSE_BLOCK_SIZE = 128
# Gluon PS kernel supports kv_block_size in {16, 64}; 64 halves the block-table
# width and page count vs 16 (ATOM ran page 64: 2 pages per sparse block).
GLUON_PAGE_SIZE = 64
PAGES_PER_BLOCK = SPARSE_BLOCK_SIZE // GLUON_PAGE_SIZE
HEAD_DIM = 128
# Scratch grows in 1024-page steps (16 KiB/page/buffer -> 16 MiB granularity).
_SCRATCH_GROW_PAGES = 1024
_PAGE_ELEMS = HEAD_DIM * GLUON_PAGE_SIZE

_max_scratch_pages_cached: Optional[int] = None


def _max_scratch_pages() -> int:
    # Hard cap on the gathered context span per forward (K and V buffers each).
    # Default 512 MiB per buffer at bf16 = 32768 pages = ~2.1M context tokens
    # across the batch; beyond that the entry point raises and the caller falls
    # back to the Triton kernel (which reads the pool in place). Parsed once:
    # this sits on the per-layer hot path and the env is fixed at launch.
    global _max_scratch_pages_cached
    if _max_scratch_pages_cached is None:
        mb = envs.SGLANG_OPT_ATOM_PREFILL_MAX_SCRATCH_MB.get()
        _max_scratch_pages_cached = max(1, (mb * 1024 * 1024) // (_PAGE_ELEMS * 2))
    return _max_scratch_pages_cached


# ---------------------------------------------------------------------------
# NHD pool -> SHUFFLE 5D scratch gather. DEVIATION: no counterpart in the
# source port, which reads the vectorized_5d pool zero-copy.
# ---------------------------------------------------------------------------


@triton.jit
def _gather_nhd_to_shuffle_kernel(
    k_pool_ptr,  # [max_slots, 1, HEAD_DIM] (NHD main pool, this layer)
    v_pool_ptr,  # [max_slots, 1, HEAD_DIM]
    k_out_ptr,  # [num_pages, 1, HEAD_DIM // X, PAGE, X] contiguous
    v_out_ptr,  # [num_pages, 1, PAGE // X, HEAD_DIM, X] contiguous
    req_to_token_ptr,  # [max_reqs, max_kv_len]
    req_pool_indices_ptr,  # [batch]
    seq_lens_ptr,  # [batch] total K length (prefix + current chunk)
    page_req_ptr,  # [total_pages] -> batch index owning this page
    page_pos_ptr,  # [total_pages] -> page index within the request
    stride_k_slot,
    stride_v_slot,
    stride_r2t_b,
    HEAD_DIM_C: tl.constexpr,
    PAGE: tl.constexpr,
    X: tl.constexpr,
):
    pid = tl.program_id(0)
    req = tl.load(page_req_ptr + pid)
    ppos = tl.load(page_pos_ptr + pid)
    pool_id = tl.load(req_pool_indices_ptr + req).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + req)

    off_t = tl.arange(0, PAGE)
    tok = ppos * PAGE + off_t
    tmask = tok < seq_len
    slots = tl.load(
        req_to_token_ptr + pool_id * stride_r2t_b + tok, mask=tmask, other=0
    ).to(tl.int64)

    off_d = tl.arange(0, HEAD_DIM_C)
    # Coalesced NHD reads: each token's head_dim vector is contiguous.
    k_tile = tl.load(
        k_pool_ptr + slots[:, None] * stride_k_slot + off_d[None, :],
        mask=tmask[:, None],
        other=0.0,
    )
    v_tile = tl.load(
        v_pool_ptr + slots[:, None] * stride_v_slot + off_d[None, :],
        mask=tmask[:, None],
        other=0.0,
    )

    page_base = pid.to(tl.int64) * (HEAD_DIM_C * PAGE)
    # K shuffle page [1, D//X, PAGE, X]: elem(d, t) at (d//X)*(PAGE*X) + t*X + d%X
    k_off = (
        (off_d[None, :] // X) * (PAGE * X) + off_t[:, None] * X + (off_d[None, :] % X)
    )
    # V transposed page [1, PAGE//X, D, X]: elem(d, t) at (t//X)*(D*X) + d*X + t%X
    v_off = (
        (off_t[:, None] // X) * (HEAD_DIM_C * X)
        + off_d[None, :] * X
        + (off_t[:, None] % X)
    )
    # Unmasked stores: tiles are zero-padded past seq_len, so tail pages hold
    # zeros rather than a previous layer's stale data (never attended anyway --
    # sparse_ctx stops the walk -- but keeps the buffer well-defined).
    tl.store(k_out_ptr + page_base + k_off, k_tile)
    tl.store(v_out_ptr + page_base + v_off, v_tile)


# Persistent, grow-only scratch: (device, dtype) -> (k_flat, v_flat). One pair
# is shared by all 57 sparse layers (layers run sequentially on one stream) and
# across forwards; ~16 KiB per page per buffer (128 dims x 64 tokens x 2B).
_SCRATCH: dict = {}


def _get_scratch(
    device: torch.device, dtype: torch.dtype, total_pages: int
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, dtype)
    entry = _SCRATCH.get(key)
    needed = total_pages * _PAGE_ELEMS
    if entry is None or entry[0].numel() < needed:
        cap_pages = (
            (total_pages + _SCRATCH_GROW_PAGES - 1) // _SCRATCH_GROW_PAGES
        ) * _SCRATCH_GROW_PAGES
        # This lands after the KV pool has already claimed its budget, so a grow
        # can be what tips the device over. Surface it as a normal unsupported
        # case: the caller catches and runs the Triton kernel, which reads the
        # pool in place and needs no scratch at all.
        try:
            entry = (
                torch.empty(cap_pages * _PAGE_ELEMS, dtype=dtype, device=device),
                torch.empty(cap_pages * _PAGE_ELEMS, dtype=dtype, device=device),
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise ValueError(
                f"atom prefill scratch of {cap_pages} pages "
                f"({cap_pages * _PAGE_ELEMS * dtype.itemsize * 2 >> 20} MiB) "
                "does not fit; lower SGLANG_OPT_ATOM_PREFILL_MAX_SCRATCH_MB or "
                "--mem-fraction-static"
            ) from exc
        _SCRATCH[key] = entry
    return entry


def _gather_context_to_shuffle(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_req: torch.Tensor,
    page_pos: torch.Tensor,
    total_pages: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather this batch's context tokens (NHD pool slots) into SHUFFLE 5D pages."""
    x = 16 // k_cache.dtype.itemsize  # 8 for bf16/fp16
    k_flat, v_flat = _get_scratch(k_cache.device, k_cache.dtype, total_pages)
    _gather_nhd_to_shuffle_kernel[(total_pages,)](
        k_cache,
        v_cache,
        k_flat,
        v_flat,
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_req,
        page_pos,
        k_cache.stride(0),
        v_cache.stride(0),
        req_to_token.stride(0),
        HEAD_DIM_C=HEAD_DIM,
        PAGE=GLUON_PAGE_SIZE,
        X=x,
    )
    k5 = k_flat[: total_pages * _PAGE_ELEMS].view(
        total_pages, 1, HEAD_DIM // x, GLUON_PAGE_SIZE, x
    )
    v5 = v_flat[: total_pages * _PAGE_ELEMS].view(
        total_pages, 1, GLUON_PAGE_SIZE // x, HEAD_DIM, x
    )
    return k5, v5


@triton.jit
def _build_atom_sparse_bt_prefill_kernel(
    topk_ptr,  # [Hkv=1, total_q, topk] int32
    req_id_ptr,  # [total_q] int32
    abs_pos_ptr,  # [total_q] int32
    page_start_ptr,  # [batch] int32: first scratch page of each request
    sparse_bt_ptr,  # [total_q, topk * pages_per_block] int32
    sparse_ctx_ptr,  # [total_q] int32
    max_topk,
    stride_topk_n,
    stride_topk_t,
    stride_sparse_bt_n,
    sparse_block_size: tl.constexpr,
    pages_per_block: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_n = tl.program_id(0)

    req_id = tl.load(req_id_ptr + pid_n)
    base_page = tl.load(page_start_ptr + req_id)
    abs_pos = tl.load(abs_pos_ptr + pid_n)
    causal_len = abs_pos + 1
    self_blk = abs_pos // sparse_block_size

    topk_row = topk_ptr + pid_n * stride_topk_n
    out_row = sparse_bt_ptr + pid_n * stride_sparse_bt_n
    off_t = tl.arange(0, BLOCK_SIZE_T)
    blk = tl.load(topk_row + off_t * stride_topk_t, mask=off_t < max_topk, other=-1)

    # Causal: a query at abs_pos may attend its own block and all earlier ones.
    valid = (blk >= 0) & (blk <= self_blk)
    is_tail = valid & (blk == self_blk)
    is_full = valid & (blk < self_blk)
    n_full = tl.sum(is_full.to(tl.int32), axis=0)
    n_valid = tl.sum(valid.to(tl.int32), axis=0)
    # Pack full blocks first (each contributes a whole sparse_block_size span),
    # the tail/current block last (partial, causal). This keeps the kernel's
    # sequential page walk aligned with sparse_ctx.
    earlier_full = tl.cumsum(is_full.to(tl.int32), axis=0) - is_full.to(tl.int32)
    sparse_slot = tl.where(is_full, earlier_full, n_full)

    # DEVIATION (scratch-page addressing): scratch pages of one request are
    # contiguous and position-ordered (and the span is rounded up to whole
    # sparse blocks), so block -> pages is pure arithmetic; the source port
    # looks up req_to_token at every page boundary instead.
    dst_base = sparse_slot * pages_per_block
    for j in tl.static_range(0, pages_per_block):
        phys_page = base_page + blk * pages_per_block + j
        tl.store(out_row + dst_base + j, phys_page, mask=valid)

    n_used = n_valid * pages_per_block
    off_w = tl.arange(0, BLOCK_SIZE_T * pages_per_block)
    # BLOCK_SIZE_T is next_power_of_2(topk), so off_w overshoots the row whenever
    # topk is not a power of two -- bound the tail fill by the real row width.
    tl.store(
        out_row + off_w,
        tl.zeros_like(off_w),
        mask=(off_w >= n_used) & (off_w < max_topk * pages_per_block),
    )

    # Effective KV length the kernel will walk across the packed pages.
    tail_tokens = causal_len - self_blk * sparse_block_size
    has_tail = tl.sum(is_tail.to(tl.int32), axis=0) > 0
    ctx = n_full * sparse_block_size + tl.where(has_tail, tail_tokens, 0)
    ctx = tl.where(has_tail, ctx, tl.minimum(n_valid * sparse_block_size, causal_len))
    tl.store(sparse_ctx_ptr + pid_n, ctx)


_META_CACHE: Optional[tuple] = None

_BT_OUT_CACHE: Optional[tuple] = None


def _build_atom_prefill_meta(
    cu_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    total_q: int,
):
    global _META_CACHE, _BT_OUT_CACHE
    if _META_CACHE is not None:
        (k_cu, k_prefix, k_seq, k_total_q), cached = _META_CACHE
        if (
            k_cu is cu_seqlens
            and k_prefix is prefix_lens
            and k_seq is seq_lens
            and k_total_q == total_q
        ):
            return cached

    device = cu_seqlens.device
    pos = torch.arange(total_q, dtype=torch.int32, device=device)
    req_id = torch.searchsorted(cu_seqlens[1:].contiguous(), pos, right=True).to(
        torch.int32
    )
    abs_pos = (prefix_lens[req_id] + (pos - cu_seqlens[req_id])).to(torch.int32)

    # Host-side page layout: each request's context span rounds up to whole
    # sparse blocks so every emitted page id stays inside its own span.
    lens = [int(l) for l in seq_lens_cpu.tolist()]
    pages_per_req = [
        ((l + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE) * PAGES_PER_BLOCK
        for l in lens
    ]
    total_pages = int(sum(pages_per_req))
    starts = []
    acc = 0
    for p in pages_per_req:
        starts.append(acc)
        acc += p
    page_start = torch.tensor(starts, dtype=torch.int32, device=device)
    repeats = torch.tensor(pages_per_req, dtype=torch.int64, device=device)
    page_req = torch.repeat_interleave(
        torch.arange(len(pages_per_req), dtype=torch.int32, device=device),
        repeats,
        output_size=total_pages,
    )
    page_pos = (
        torch.arange(total_pages, dtype=torch.int32, device=device)
        - page_start[page_req.long()]
    ).to(torch.int32)

    meta = (req_id, abs_pos, page_start, page_req, page_pos, total_pages)
    _BT_OUT_CACHE = None  # new forward: no block table can be valid anymore
    _META_CACHE = ((cu_seqlens, prefix_lens, seq_lens, total_q), meta)
    return meta


def _build_atom_sparse_bt_prefill(
    topk_idx: torch.Tensor,
    req_id: torch.Tensor,
    abs_pos: torch.Tensor,
    page_start: torch.Tensor,
    sparse_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # topk_idx: [num_kv_heads=1, total_q, topk]
    total_q = topk_idx.shape[1]
    topk = topk_idx.shape[2]

    global _BT_OUT_CACHE
    if _BT_OUT_CACHE is not None:
        (k_topk_idx, k_total_q, k_topk, k_block_size), out_cached = _BT_OUT_CACHE
        if (
            k_topk_idx is topk_idx
            and k_total_q == total_q
            and k_topk == topk
            and k_block_size == sparse_block_size
        ):
            return out_cached

    sparse_bt = torch.empty(
        (total_q, topk * PAGES_PER_BLOCK),
        dtype=torch.int32,
        device=topk_idx.device,
    )
    sparse_ctx = torch.empty((total_q,), dtype=torch.int32, device=topk_idx.device)
    _build_atom_sparse_bt_prefill_kernel[(total_q,)](
        topk_idx,
        req_id,
        abs_pos,
        page_start,
        sparse_bt,
        sparse_ctx,
        topk,
        topk_idx.stride(1),
        topk_idx.stride(2),
        sparse_bt.stride(0),
        sparse_block_size=sparse_block_size,
        pages_per_block=PAGES_PER_BLOCK,
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
    )
    _BT_OUT_CACHE = (
        (topk_idx, total_q, topk, sparse_block_size),
        (sparse_bt, sparse_ctx),
    )
    return sparse_bt, sparse_ctx


def _unit_or_none(scale) -> bool:
    return scale is None or scale == 1.0


def can_use_atom_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    block_size_k: int,
    seq_lens_cpu: Optional[torch.Tensor],
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> bool:
    """Cheap static gate; anything it can't see is caught at runtime and falls back."""
    if (
        pa_decode_gluon is None
        or get_recommended_splits is None
        or sink is not None
        or seq_lens_cpu is None
        or block_size_k != SPARSE_BLOCK_SIZE
    ):
        return False
    # DEVIATION (NHD gather + this tree's fp8 attn-GEMM plumbing): require the
    # NHD main pool [max_slots, num_kv_heads=1, head_dim=128], bf16/fp16 with
    # unit scales (fp8 attn-GEMM / calibrated caches stay on the Triton path),
    # innermost dim contiguous (the gather kernel assumes it). The source port
    # instead requires its vectorized_5d pool with page size in {16, 64}.
    return (
        q.dim() == 3
        and q.shape[-1] == HEAD_DIM
        and k_cache.dim() == 3
        and v_cache.dim() == 3
        and k_cache.shape[1] == 1
        and v_cache.shape[1] == 1
        and k_cache.shape[2] == HEAD_DIM
        and v_cache.shape[2] == HEAD_DIM
        and q.dtype in (torch.bfloat16, torch.float16)
        and k_cache.dtype == q.dtype
        and v_cache.dtype == q.dtype
        and k_cache.stride(2) == 1
        and v_cache.stride(2) == 1
        and _unit_or_none(q_scale)
        and _unit_or_none(k_scale)
        and _unit_or_none(v_scale)
    )


@torch.no_grad()
def atom_gluon_sparse_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    block_size_k: int,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Run ATOM-style Gluon sparse prefill over the NHD KV pool.

    The context span is first gathered into the persistent SHUFFLE 5D scratch
    (see module docstring); ``num_kv_heads == 1`` is enforced by
    ``can_use_atom_prefill``. Raises on unsupported runtime shapes; the caller
    catches and falls back to the Triton sparse kernel.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    q = q.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    # DEVIATION (raise/fallback instead of the source tree's asserts): guard the
    # empty launch and the runtime shapes the static gate cannot see.
    if total_q == 0:
        return torch.empty_like(q)
    if topk_idx.shape[0] != 1 or topk_idx.shape[1] != total_q:
        raise ValueError(
            f"atom prefill expects topk_idx [1, total_q, topk], got {tuple(topk_idx.shape)}"
        )
    batch = cu_seqlens.shape[0] - 1
    if seq_lens_cpu.numel() != batch or seq_lens.shape[0] != batch:
        raise ValueError(
            f"atom prefill batch mismatch: cu_seqlens batch {batch}, "
            f"seq_lens {tuple(seq_lens.shape)}, seq_lens_cpu {seq_lens_cpu.numel()}"
        )

    req_id, abs_pos, page_start, page_req, page_pos, total_pages = (
        _build_atom_prefill_meta(
            cu_seqlens, prefix_lens, seq_lens, seq_lens_cpu, total_q
        )
    )
    max_pages = _max_scratch_pages()
    if total_pages > max_pages:
        raise ValueError(
            f"atom prefill context span too large for scratch: {total_pages} pages "
            f"> cap {max_pages} (raise SGLANG_OPT_ATOM_PREFILL_MAX_SCRATCH_MB)"
        )

    # DEVIATION (NHD pool): rebind k_cache/v_cache to the gathered SHUFFLE 5D
    # scratch views; the source port feeds the pool itself zero-copy. Per layer
    # (never cached): the pool contents differ per layer. Prefix + current
    # chunk are both in the pool at this point (rope+cache runs first).
    k_cache, v_cache = _gather_context_to_shuffle(
        k_cache,
        v_cache,
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_req,
        page_pos,
        total_pages,
    )

    sparse_bt, sparse_ctx = _build_atom_sparse_bt_prefill(
        topk_idx, req_id, abs_pos, page_start, block_size_k
    )

    out = torch.empty_like(q)
    num_seqs = total_q
    ctx_part = 256
    # max_context_partition_num is a PARALLELISM/split knob, not a coverage
    # requirement: the gluon kernel loops over the full context internally, so a
    # single partition still attends the whole sparse_ctx. Verified with a
    # standalone mini-bench -- max_part=1 matches full-coverage output at
    # ctx=2048 and 4096 (rel diff = bf16 noise, no OOB/crash). ATOM sizes it with
    # get_recommended_splits alone. An earlier needed_parts floor
    # (ceil(max_sparse_ctx/256), up to 8) over-split large-ctx chunked-prefill
    # batches, adding ~400us of empty-partition launch overhead per extra split
    # (3433us vs 1013us/launch at num_seqs=16384). Match ATOM: rec_splits only.
    # Escape hatch: SGLANG_M3_PA_NEEDED_PARTS=1 restores the old floor.
    if envs.SGLANG_M3_PA_NEEDED_PARTS.get():
        # Only the escape-hatch path needs the max sparse context, which requires
        # a GPU->CPU sync (.item()). The default path uses get_recommended_splits
        # alone, so skip the sync entirely there (it ran once per sparse layer).
        max_ctx = int(sparse_ctx.max().item()) if sparse_ctx.numel() else 0
        needed_parts = max(1, (max_ctx + ctx_part - 1) // ctx_part)
        max_part_num = max(get_recommended_splits(num_seqs, 1), needed_parts)
    else:
        max_part_num = get_recommended_splits(num_seqs, 1)
    intermediate_shape = (num_seqs, 1, max_part_num, num_q_heads)
    exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device=q.device)
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty(
        (*intermediate_shape, head_dim), dtype=q.dtype, device=q.device
    )

    pa_decode_gluon(
        output=out,
        query=q,
        key_cache=k_cache,
        value_cache=v_cache,
        context_lengths=sparse_ctx,
        block_tables=sparse_bt,
        softmax_scale=sm_scale,
        query_length=1,
        max_context_partition_num=max_part_num,
        context_partition_size=ctx_part,
        compute_type=q.dtype,
        key_scale=None,
        value_scale=None,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        sinks=None,
        sliding_window=0,
        ps=True,
    )
    return out
