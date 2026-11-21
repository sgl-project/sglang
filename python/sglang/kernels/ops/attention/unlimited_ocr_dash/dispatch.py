"""Device dispatch wrappers for Unlimited-OCR v2 DASH operators.

The Triton entry points in ``attend``, ``route``, ``decode_meta`` and
``page2d_rope`` are registered as CUDA-only Triton kernels; this module
dispatches MPS tensors to ``mps`` / ``metal`` and forwards everything else to
those Triton entry points so the kernel registry stays accurate.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.attention.unlimited_ocr_dash import attend as _attend
from sglang.kernels.ops.attention.unlimited_ocr_dash import decode_meta as _decode_meta
from sglang.kernels.ops.attention.unlimited_ocr_dash import (
    metal,
    mps,
)
from sglang.kernels.ops.attention.unlimited_ocr_dash import page2d_rope as _page2d_rope
from sglang.kernels.ops.attention.unlimited_ocr_dash import route as _route
from sglang.kernels.ops.attention.unlimited_ocr_dash.common import SPARSE_BLOCK_SIZE


def dash_sparse_attn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    blk_bias: Optional[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    local_blocks: int = 0,
    sparse_block_size: int = SPARSE_BLOCK_SIZE,
) -> None:
    if q.device.type == "mps":
        return mps.dash_sparse_attn(
            q,
            k_cache,
            v_cache,
            topk_idx,
            blk_bias,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            num_kv_heads,
            sm_scale,
            output,
            local_blocks=local_blocks,
            sparse_block_size=sparse_block_size,
        )
    return _attend.dash_sparse_attn(
        q,
        k_cache,
        v_cache,
        topk_idx,
        blk_bias,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        sm_scale,
        output,
        local_blocks=local_blocks,
        sparse_block_size=sparse_block_size,
    )


def dash_sparse_attn_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    blk_bias: Optional[torch.Tensor],
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    local_blocks: int = 0,
    sparse_block_size: int = SPARSE_BLOCK_SIZE,
    chunk_width: Optional[int] = None,
    cg_rows: Optional[int] = None,
) -> None:
    if q.device.type == "mps":
        total_q = q.shape[0]
        assert total_q == seq_lens.shape[0] * decode_query_len
        return mps.dash_sparse_attn_decode(
            q,
            k_cache,
            v_cache,
            topk_idx,
            blk_bias,
            block_table,
            seq_lens,
            num_kv_heads,
            sm_scale,
            output,
            decode_query_len,
            local_blocks=local_blocks,
            sparse_block_size=sparse_block_size,
        )
    return _attend.dash_sparse_attn_decode(
        q,
        k_cache,
        v_cache,
        topk_idx,
        blk_bias,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        decode_query_len,
        local_blocks=local_blocks,
        sparse_block_size=sparse_block_size,
        chunk_width=chunk_width,
        cg_rows=cg_rows,
    )


def win_bounds_decode(
    *,
    allow: torch.Tensor,
    seq_lens: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    fit: torch.Tensor,
    max_block: int,
    block_size: int,
    local_blocks: int,
    decode_query_len: int,
    tile: int,
) -> None:
    if allow.device.type == "mps":
        return mps.win_bounds_decode(
            allow=allow,
            seq_lens=seq_lens,
            lo=lo,
            hi=hi,
            fit=fit,
            max_block=max_block,
            block_size=block_size,
            local_blocks=local_blocks,
            decode_query_len=decode_query_len,
            tile=tile,
        )
    return _route.win_bounds_decode(
        allow=allow,
        seq_lens=seq_lens,
        lo=lo,
        hi=hi,
        fit=fit,
        max_block=max_block,
        block_size=block_size,
        local_blocks=local_blocks,
        decode_query_len=decode_query_len,
        tile=tile,
    )


def route_prefill(
    *,
    idx_q: torch.Tensor,
    k_bar: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    row2req: torch.Tensor,
    image_allow: Optional[torch.Tensor],
    out_idx: torch.Tensor,
    out_bias: torch.Tensor,
    max_block: int,
    block_size: int,
    local_blocks: int,
    idx_scale: float,
    prior_floor: float,
    sigma: float,
    topm: int,
    budget: int | None = None,
    q_chunk: int = 2048,
    out_slots: int | None = None,
    score_buf: torch.Tensor | None = None,
):
    if idx_q.device.type == "mps":
        M = _route._check_out_slots(out_slots, topm, out_idx)
        mps.route_prefill(
            idx_q=idx_q,
            k_bar=k_bar,
            block_table=block_table,
            seq_lens=seq_lens,
            cu_seqlens_q=cu_seqlens_q,
            context_lens=context_lens,
            row2req=row2req,
            image_allow=image_allow,
            out_idx=out_idx,
            out_bias=out_bias,
            max_block=max_block,
            block_size=block_size,
            local_blocks=local_blocks,
            idx_scale=idx_scale,
            prior_floor=prior_floor,
            sigma=sigma,
            budget=max(1, min(int(budget or M), M)),
            M=M,
        )
        return M, None, 0
    return _route.route_prefill_triton(
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        context_lens=context_lens,
        row2req=row2req,
        image_allow=image_allow,
        out_idx=out_idx,
        out_bias=out_bias,
        max_block=max_block,
        block_size=block_size,
        local_blocks=local_blocks,
        idx_scale=idx_scale,
        prior_floor=prior_floor,
        sigma=sigma,
        topm=topm,
        budget=budget,
        q_chunk=q_chunk,
        out_slots=out_slots,
        score_buf=score_buf,
    )


def kbar_refresh_tail(
    *,
    kv_cache: torch.Tensor,
    k_bar: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    head_cls: torch.Tensor,
    block_size: int,
    tail_pages: int = 1,
    plan=None,
) -> None:
    if kv_cache.device.type == "mps":
        if plan is None:
            plan = _route.kbar_tail_plan(
                block_table=block_table,
                seq_lens=seq_lens,
                block_size=block_size,
                tail_pages=tail_pages,
                device=k_bar.device,
            )
        return mps.kbar_refresh(
            kv_cache=kv_cache, k_bar=k_bar, head_cls=head_cls, plan=plan
        )
    return _route.kbar_refresh_tail_triton(
        kv_cache=kv_cache,
        k_bar=k_bar,
        block_table=block_table,
        seq_lens=seq_lens,
        head_cls=head_cls,
        block_size=block_size,
        tail_pages=tail_pages,
        plan=plan,
    )


def route_decode(
    *,
    idx_q: torch.Tensor,
    k_bar: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    image_allow: Optional[torch.Tensor],
    out_idx: torch.Tensor,
    out_bias: torch.Tensor,
    max_block: int,
    block_size: int,
    local_blocks: int,
    decode_query_len: int,
    idx_scale: float,
    prior_floor: float,
    sigma: float,
    topm: int,
    score_buf=None,
    budget: int | None = None,
    out_slots: int | None = None,
    entmax_base=None,
    entmax_fit=None,
    entmax_width: int = 0,
    score_buf_win=None,
    bounds=None,
) -> int:
    if idx_q.device.type == "mps":
        dql = max(1, int(decode_query_len))
        M = _route._check_out_slots(out_slots, topm, out_idx)
        return mps.route_decode(
            idx_q=idx_q,
            k_bar=k_bar,
            block_table=block_table,
            seq_lens=seq_lens,
            image_allow=image_allow,
            out_idx=out_idx,
            out_bias=out_bias,
            max_block=max_block,
            block_size=block_size,
            local_blocks=local_blocks,
            decode_query_len=dql,
            idx_scale=idx_scale,
            prior_floor=prior_floor,
            sigma=sigma,
            budget=max(1, min(int(budget or M), M)),
            M=M,
        )
    return _route.route_decode_triton(
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=block_table,
        seq_lens=seq_lens,
        image_allow=image_allow,
        out_idx=out_idx,
        out_bias=out_bias,
        max_block=max_block,
        block_size=block_size,
        local_blocks=local_blocks,
        decode_query_len=decode_query_len,
        idx_scale=idx_scale,
        prior_floor=prior_floor,
        sigma=sigma,
        topm=topm,
        score_buf=score_buf,
        budget=budget,
        out_slots=out_slots,
        entmax_base=entmax_base,
        entmax_fit=entmax_fit,
        entmax_width=entmax_width,
        score_buf_win=score_buf_win,
        bounds=bounds,
    )


def dash_decode_meta(
    *,
    input_ids: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_mapping: torch.Tensor,
    coord_rows: torch.Tensor,
    slot_img_page: torch.Tensor,
    slot_marker_cum: torch.Tensor,
    slot_img_run: torch.Tensor,
    slot_prompt_pages: torch.Tensor,
    coord_marker: torch.Tensor,
    coord_prompt_pages: torch.Tensor,
    coord_page_counts: torch.Tensor,
    image_token_id: int,
    page_token_id: int | None,
    per_page_image_tokens: int,
    num_slots: int,
    max_pages: int,
    block_size: int,
    out_q_page: torch.Tensor | None = None,
    out_tok: torch.Tensor | None = None,
    out_page: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if input_ids.device.type == "mps":
        R = int(seq_lens.shape[0])
        device = input_ids.device
        q_page = (
            out_q_page[:R]
            if out_q_page is not None and out_q_page.shape[0] >= R
            else torch.empty(R, dtype=torch.int32, device=device)
        )
        tok = (
            out_tok[:R]
            if out_tok is not None and out_tok.shape[0] >= R
            else torch.empty(R, dtype=torch.int64, device=device)
        )
        page = (
            out_page[:R]
            if out_page is not None and out_page.shape[0] >= R
            else torch.empty(R, dtype=torch.int64, device=device)
        )
        if R == 0:
            return q_page, (tok, page)
        slots = slot_mapping[:R]
        if slots.dtype != torch.int64:
            slots = slots.to(torch.int64)
        metal.decode_meta(
            input_ids=input_ids[:R],
            block_table=block_table,
            seq_lens=seq_lens,
            slots=slots,
            coord_rows=coord_rows,
            slot_tables=(
                slot_img_page,
                slot_marker_cum,
                slot_img_run,
                slot_prompt_pages,
            ),
            coord_tables=(coord_marker, coord_prompt_pages, coord_page_counts),
            q_page=q_page,
            tok=tok,
            page=page,
            image_token_id=image_token_id,
            page_token_id=page_token_id,
            per_page=per_page_image_tokens,
            num_slots=num_slots,
            max_pages=max_pages,
            block_size=block_size,
        )
        return q_page, (tok, page)
    return _decode_meta.dash_decode_meta(
        input_ids=input_ids,
        block_table=block_table,
        seq_lens=seq_lens,
        slot_mapping=slot_mapping,
        coord_rows=coord_rows,
        slot_img_page=slot_img_page,
        slot_marker_cum=slot_marker_cum,
        slot_img_run=slot_img_run,
        slot_prompt_pages=slot_prompt_pages,
        coord_marker=coord_marker,
        coord_prompt_pages=coord_prompt_pages,
        coord_page_counts=coord_page_counts,
        image_token_id=image_token_id,
        page_token_id=page_token_id,
        per_page_image_tokens=per_page_image_tokens,
        num_slots=num_slots,
        max_pages=max_pages,
        block_size=block_size,
        out_q_page=out_q_page,
        out_tok=out_tok,
        out_page=out_page,
    )


def dash_decode_allow(
    *,
    block_table: torch.Tensor,
    slot_img_page: torch.Tensor,
    q_page: torch.Tensor,
    seq_lens: torch.Tensor,
    max_block: int,
    num_blocks: int,
    block_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if q_page.device.type == "mps":
        R = int(q_page.shape[0])
        W = min(max_block, int(block_table.shape[1]))
        if out is None or out.shape[0] < R or out.shape[1] < max_block:
            out = torch.zeros(R, max_block, dtype=torch.int8, device=q_page.device)
        view = out[:R, :max_block]
        metal.decode_allow(
            block_table=block_table,
            slot_img_page=slot_img_page,
            q_page=q_page,
            seq_lens=seq_lens,
            out=view,
            width=W,
            num_blocks=num_blocks,
            max_block=max_block,
            block_size=block_size,
        )
        return view
    return _decode_meta.dash_decode_allow(
        block_table=block_table,
        slot_img_page=slot_img_page,
        q_page=q_page,
        seq_lens=seq_lens,
        max_block=max_block,
        num_blocks=num_blocks,
        block_size=block_size,
        out=out,
    )


def _mps_rope_pair_ok(x, y, cos, sin, head_dim, t: int) -> bool:
    return (
        x.dtype == torch.bfloat16
        and y.dtype == torch.bfloat16
        and t > 0
        and y.shape[0] == t
        and x.ndim == 2
        and y.ndim == 2
        and x.is_contiguous()
        and y.is_contiguous()
        and cos.shape == (t, head_dim)
        and sin.shape == (t, head_dim)
        and cos.dtype == torch.bfloat16
        and sin.dtype == torch.bfloat16
        and cos.is_contiguous()
        and sin.is_contiguous()
        and head_dim >= 2
        and head_dim % 2 == 0
        and (head_dim & (head_dim - 1)) == 0
        and x.shape[1] % head_dim == 0
        and y.shape[1] % head_dim == 0
        and (x.shape[1] + y.shape[1]) > 0
    )


def apply_page2d_rope_pair(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.device.type == "mps":
        t = x.shape[0]
        if _mps_rope_pair_ok(x, y, cos, sin, head_dim, t):
            return metal.page2d_rope_pair(x, y, cos, sin, head_dim)
        return (
            _page2d_rope.apply_page2d_rope(x, cos, sin, head_dim),
            _page2d_rope.apply_page2d_rope(y, cos, sin, head_dim),
        )
    return _page2d_rope.apply_page2d_rope_pair(x, y, cos, sin, head_dim)
