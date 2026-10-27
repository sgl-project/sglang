# Copyright 2025 SGLang Team
# Ported from
# https://github.com/alexsun07/sglang/blob/191624ff8a6d8e9fc31b9363d2138c16db66f124/python/sglang/srt/layers/attention/minimax_sparse_ops/atom_prefill.py
"""MiniMax-M3 sparse prefill using AITER's Gluon paged attention.

The main KV pool remains NHD ``[max_slots, 1, head_dim]``. Each request's
context is gathered into persistent SHUFFLE 5D scratch pages:

- key scratch:   [num_pages, 1, head_dim // x, GLUON_PAGE_SIZE, x]
- value scratch: [num_pages, 1, GLUON_PAGE_SIZE // x, head_dim, x] (transposed)

with ``x = 16 // dtype.itemsize``, in q's dtype: a bf16/fp16 pool as is, an fp8 pool
widened and an NVFP4 pool dequantized (both exact), except that an fp8 pool stays fp8 for
the per-query decode kernel, which widens it per tile. Each request occupies a contiguous,
position-ordered page range rounded up to whole sparse blocks.
"""

from __future__ import annotations

import functools
from itertools import accumulate
from types import ModuleType
from typing import Optional, Union

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import Nvfp4Cache
from sglang.kernels.ops.attention.minimax_sparse.prefill.nvfp4_prefill import (
    gather_nvfp4_to_shuffle,
)
from sglang.srt.layers.attention.aiter_utils import (
    get_recommended_splits,
    pa_decode_gluon,
)
from sglang.srt.utils.common import async_h2d

try:
    from . import gluon_prefill_pair_gfx950
except ImportError:  # Gluon without the CDNA4 buffer/MFMA ops
    gluon_prefill_pair_gfx950 = None

SPARSE_BLOCK_SIZE = 128
# The Gluon PS kernel takes kv_block_size 16 or 64; 64 halves the block-table width.
GLUON_PAGE_SIZE = 64
PAGES_PER_BLOCK = SPARSE_BLOCK_SIZE // GLUON_PAGE_SIZE
HEAD_DIM = 128
# aiter's fp8 on gfx950, the only arch that dispatches here (gfx942 would be e4m3fnuz).
_FP8_KV_DTYPE = torch.float8_e4m3fn
# Scratch grows in 1024-page steps (16 KiB/page/buffer -> 16 MiB granularity).
_SCRATCH_GROW_PAGES = 1024
_PAGE_ELEMS = HEAD_DIM * GLUON_PAGE_SIZE

# 512 MiB per buffer at bf16 (~2.1M context tokens per forward); past it an NVFP4 batch
# runs in request groups, others fall back to Triton (reads the pool in place).
_MAX_SCRATCH_PAGES = (512 * 1024 * 1024) // (_PAGE_ELEMS * 2)


class GluonPrefillUnavailableError(RuntimeError):
    """Raised when the scratch cannot hold this forward; the caller falls back to Triton."""


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
    # Unmasked: tiles are zero past seq_len, so no previous layer's data survives.
    tl.store(k_out_ptr + page_base + k_off, k_tile.to(k_out_ptr.dtype.element_ty))
    tl.store(v_out_ptr + page_base + v_off, v_tile.to(v_out_ptr.dtype.element_ty))


# Grow-only device -> (k_bytes, v_bytes) at bf16 width, viewed as each forward's scratch
# dtype (fp8 or bf16); layers and forwards run in stream order and share it.
_SCRATCH: dict = {}


def _get_scratch(
    device: torch.device, dtype: torch.dtype, total_pages: int
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dtype.itemsize <= 2
    entry = _SCRATCH.get(device)
    needed = total_pages * _PAGE_ELEMS * 2
    if entry is None or entry[0].numel() < needed:
        cap_pages = (
            (total_pages + _SCRATCH_GROW_PAGES - 1) // _SCRATCH_GROW_PAGES
        ) * _SCRATCH_GROW_PAGES
        # Grows after the KV pool claimed its budget; report OOM as unsupported so
        # the caller falls back to the scratch-free Triton kernel.
        try:
            entry = (
                torch.empty(
                    cap_pages * _PAGE_ELEMS * 2, dtype=torch.uint8, device=device
                ),
                torch.empty(
                    cap_pages * _PAGE_ELEMS * 2, dtype=torch.uint8, device=device
                ),
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise GluonPrefillUnavailableError(
                f"gluon prefill scratch of {cap_pages} pages "
                f"({cap_pages * _PAGE_ELEMS * 4 >> 20} MiB) "
                "does not fit; lower --mem-fraction-static"
            ) from exc
        _SCRATCH[device] = entry
    return entry[0].view(dtype), entry[1].view(dtype)


def _gather_context_to_shuffle(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_req: torch.Tensor,
    page_pos: torch.Tensor,
    total_pages: int,
    scratch_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather this batch's context tokens (NHD pool slots) into SHUFFLE 5D pages of
    ``scratch_dtype``: the pool's, or the compute dtype an fp8 or NVFP4 pool widens to
    exactly."""
    x = 16 // scratch_dtype.itemsize  # 8 for bf16/fp16, 16 for fp8
    k_flat, v_flat = _get_scratch(k_cache.device, scratch_dtype, total_pages)
    if isinstance(k_cache, Nvfp4Cache):
        gather_nvfp4_to_shuffle(
            k_cache=k_cache,
            v_cache=v_cache,
            k_out=k_flat,
            v_out=v_flat,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            page_req=page_req,
            page_pos=page_pos,
            total_pages=total_pages,
            page=GLUON_PAGE_SIZE,
            x=x,
        )
    else:
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
def _build_gluon_sparse_bt_prefill_kernel(
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
    # Full blocks first, the partial causal tail last, so the walk matches sparse_ctx.
    earlier_full = tl.cumsum(is_full.to(tl.int32), axis=0) - is_full.to(tl.int32)
    sparse_slot = tl.where(is_full, earlier_full, n_full)

    # Scratch pages for each request are contiguous and position-ordered.
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
    ctx_len = n_full * sparse_block_size + tl.where(has_tail, tail_tokens, 0)
    ctx_len = tl.where(
        has_tail, ctx_len, tl.minimum(n_valid * sparse_block_size, causal_len)
    )
    tl.store(sparse_ctx_ptr + pid_n, ctx_len)


_META_CACHE: Optional[tuple] = None

_BT_OUT_CACHE: Optional[tuple] = None


def _scratch_pages_per_req(seq_lens_cpu: torch.Tensor) -> list[int]:
    return [
        ((kv_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE) * PAGES_PER_BLOCK
        for kv_len in seq_lens_cpu.tolist()
    ]


def _build_gluon_prefill_meta(
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
    pages_per_req = _scratch_pages_per_req(seq_lens_cpu)
    total_pages = sum(pages_per_req)
    page_start = async_h2d(
        list(accumulate(pages_per_req, initial=0))[:-1],
        dtype=torch.int32,
        device=device,
    )
    repeats = async_h2d(pages_per_req, dtype=torch.int64, device=device)
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


def _build_gluon_sparse_bt_prefill(
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
    _build_gluon_sparse_bt_prefill_kernel[(total_q,)](
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


@functools.lru_cache(maxsize=None)
def _scale_tensor(value: float, device: torch.device) -> torch.Tensor:
    return torch.full((1,), value, dtype=torch.float32, device=device)


def _kv_scale_tensor(scale: Optional[float], device: torch.device) -> torch.Tensor:
    # the kernel takes per-tensor KV scales only as [1] fp32 tensors
    return _scale_tensor(1.0 if scale is None else float(scale), device)


def can_use_gluon_prefill(
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
    """Cheap static gate; the scratch size is checked per forward."""
    if (
        pa_decode_gluon is None
        or get_recommended_splits is None
        or sink is not None
        or seq_lens_cpu is None
        or block_size_k != SPARSE_BLOCK_SIZE
    ):
        return False
    if isinstance(k_cache, Nvfp4Cache):
        return (
            q.dim() == 3
            and q.shape[-1] == HEAD_DIM
            and q.dtype in (torch.bfloat16, torch.float16)
            and k_cache.num_heads == 1
            and _unit_or_none(q_scale)
            and _unit_or_none(k_scale)
            and _unit_or_none(v_scale)
        )
    # The gather requires a contiguous, single-head NHD pool; an fp8 pool's
    # per-tensor k/v scales pass through to the kernel.
    fp8_kv = k_cache.dtype == _FP8_KV_DTYPE and v_cache.dtype == k_cache.dtype
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
        and (fp8_kv or (k_cache.dtype == q.dtype and v_cache.dtype == q.dtype))
        and k_cache.stride(2) == 1
        and v_cache.stride(2) == 1
        and _unit_or_none(q_scale)
        and (fp8_kv or (_unit_or_none(k_scale) and _unit_or_none(v_scale)))
    )


def _pair_prefill_module(
    q: torch.Tensor, k_cache: Union[torch.Tensor, Nvfp4Cache]
) -> Optional[ModuleType]:
    """The two-queries-per-program kernel module when it serves this forward, else None."""
    if gluon_prefill_pair_gfx950 is None:
        return None
    fp8_pool = not isinstance(k_cache, Nvfp4Cache) and k_cache.dtype == _FP8_KV_DTYPE
    return (
        gluon_prefill_pair_gfx950
        if gluon_prefill_pair_gfx950.can_use_pair_prefill(q, fp8_pool)
        else None
    )


@torch.no_grad()
def _gluon_sparse_prefill_group(
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
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    out_quant: str = "",
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """gluon_sparse_prefill over one request group, its whole context gathered into the
    SHUFFLE scratch; raises GluonPrefillUnavailableError when the scratch cannot hold it."""
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    q = q.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    if total_q == 0:
        return torch.empty_like(q)
    assert topk_idx.shape[0] == 1 and topk_idx.shape[1] == total_q, (
        f"gluon prefill expects topk_idx [1, total_q, topk], got {tuple(topk_idx.shape)}"
    )
    batch = cu_seqlens.shape[0] - 1
    assert seq_lens_cpu.numel() == batch and seq_lens.shape[0] == batch, (
        f"gluon prefill batch mismatch: cu_seqlens batch {batch}, "
        f"seq_lens {tuple(seq_lens.shape)}, seq_lens_cpu {seq_lens_cpu.numel()}"
    )

    req_id, abs_pos, page_start, page_req, page_pos, total_pages = (
        _build_gluon_prefill_meta(
            cu_seqlens, prefix_lens, seq_lens, seq_lens_cpu, total_q
        )
    )
    if total_pages > _MAX_SCRATCH_PAGES:
        raise GluonPrefillUnavailableError(
            f"gluon prefill context span too large for scratch: {total_pages} pages "
            f"> cap {_MAX_SCRATCH_PAGES}"
        )

    nvfp4 = isinstance(k_cache, Nvfp4Cache)
    # fp8 KV needs scale tensors: without them the kernel has no dequant mode.
    if nvfp4 or k_cache.dtype == q.dtype:
        key_scale = value_scale = None
    else:
        key_scale = _kv_scale_tensor(k_scale, q.device)
        value_scale = _kv_scale_tensor(v_scale, q.device)

    # The pair kernel reads bf16 scratch; the decode kernel widens an fp8 one per tile.
    pair = _pair_prefill_module(q, k_cache)
    scratch_dtype = q.dtype if pair is not None or nvfp4 else k_cache.dtype
    # Gather the current layer's prefix and current chunk into SHUFFLE 5D views.
    k_cache, v_cache = _gather_context_to_shuffle(
        k_cache,
        v_cache,
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_req,
        page_pos,
        total_pages,
        scratch_dtype,
    )

    if pair is not None:
        return pair.pair_sparse_prefill(
            q,
            k_cache,
            v_cache,
            topk_idx,
            req_id,
            abs_pos,
            page_start,
            sm_scale,
            1.0 if k_scale is None else float(k_scale),
            1.0 if v_scale is None else float(v_scale),
            out_quant=out_quant,
        )

    sparse_bt, sparse_ctx = _build_gluon_sparse_bt_prefill(
        topk_idx, req_id, abs_pos, page_start, block_size_k
    )

    out = torch.empty_like(q)
    num_seqs = total_q
    ctx_part = 256
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
        key_scale=key_scale,
        value_scale=value_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        sinks=None,
        sliding_window=0,
        ps=True,
    )
    return out


def gluon_sparse_prefill(
    q: torch.Tensor,
    k_cache: Union[torch.Tensor, Nvfp4Cache],
    v_cache: Union[torch.Tensor, Nvfp4Cache],
    topk_idx: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    block_size_k: int,
    sm_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    out_quant: str = "",
    q_lens_cpu: Optional[list[int]] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Run Gluon sparse prefill via the SHUFFLE scratch; raises GluonPrefillUnavailableError
    when the scratch cannot hold the context. An NVFP4 cache past the cap runs consecutive
    request groups of ``q_lens_cpu`` query rows instead, raising only for one oversized
    request. ``out_quant`` may return O quantized for the o_proj, from a single group only."""
    run_group = functools.partial(
        _gluon_sparse_prefill_group,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=req_to_token,
        block_size_k=block_size_k,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    pages = _scratch_pages_per_req(seq_lens_cpu)
    # Only NVFP4 splits an oversized batch: it has no Triton fallback.
    if (
        sum(pages) <= _MAX_SCRATCH_PAGES
        or q_lens_cpu is None
        or not isinstance(k_cache, Nvfp4Cache)
    ):
        return run_group(
            q=q,
            topk_idx=topk_idx,
            req_pool_indices=req_pool_indices,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_quant=out_quant,
        )
    out = torch.empty_like(q)
    start = row = 0
    while start < len(pages):
        end, group_pages = start, 0
        while end < len(pages) and group_pages + pages[end] <= _MAX_SCRATCH_PAGES:
            group_pages += pages[end]
            end += 1
        if end == start:
            raise GluonPrefillUnavailableError(
                f"one request spans {pages[start]} scratch pages > cap {_MAX_SCRATCH_PAGES}"
            )
        rows = sum(q_lens_cpu[start:end])
        out[row : row + rows] = run_group(
            q=q[row : row + rows],
            topk_idx=topk_idx[:, row : row + rows],
            req_pool_indices=req_pool_indices[start:end],
            cu_seqlens=cu_seqlens[start : end + 1] - cu_seqlens[start],
            seq_lens=seq_lens[start:end],
            prefix_lens=prefix_lens[start:end],
            seq_lens_cpu=seq_lens_cpu[start:end],
        )
        start, row = end, row + rows
    return out
