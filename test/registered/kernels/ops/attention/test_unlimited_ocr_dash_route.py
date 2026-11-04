"""DASH routing kernels (page summaries, block scores, 1.5-entmax top-k with
centered log-prior bias; full-width, window-narrow and tiled decode scans) must
match the PyTorch reference."""

import sys

import pytest
import torch
import triton

from sglang.kernels.ops.attention.unlimited_ocr_dash.route import (
    SCAN_TILE,
    _plan_decode_route,
    _want_tiled,
    kbar_all_plan,
    kbar_refresh_tail_triton,
    prefill_row2req,
    route_decode_triton,
    route_prefill_triton,
    win_bounds_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.unlimited_ocr_dash import (
    block_scores_reference,
    entmax_route_reference,
    kbar_summary_reference,
    win_bounds_reference,
)

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

BS, D, NUM_PAGES = 128, 128, 8192
ROUTER = dict(idx_scale=0.37, prior_floor=1e-2, sigma=1.3)


def _block_table(rows, width, gen):
    # Distinct pages per request: a repeated page would tie its block scores.
    bt = torch.stack(
        [torch.randperm(NUM_PAGES, generator=gen)[:width] for _ in range(rows)]
    )
    return bt.to(torch.int32).cuda()


def _decode_inputs(width, seq_lens, dql, density, gen):
    rows = len(seq_lens) * dql
    idx_q = torch.randn(rows, D, generator=gen).to(torch.bfloat16).cuda()
    k_bar = torch.randn(NUM_PAGES, D, generator=gen).cuda()
    bt = _block_table(len(seq_lens), width, gen)
    allow = torch.rand(rows, width, generator=gen) < density
    return idx_q, k_bar, bt, allow.cuda()


def _decode_reference(idx_q, k_bar, bt, seq, allow, width, dql, budget, out_slots):
    seq_t = torch.tensor(seq, device="cuda")
    row_req = torch.arange(len(seq), device="cuda").repeat_interleave(dql)
    pos = seq_t[row_req] - dql + torch.arange(len(row_req), device="cuda") % dql
    scores = block_scores_reference(
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=bt,
        seq_lens=seq_t,
        row_req=row_req,
        query_pos=pos,
        width=width,
        block_size=BS,
        local_blocks=1,
    )
    return entmax_route_reference(
        scores=scores,
        image_allow=allow,
        budget=budget,
        out_slots=out_slots,
        **ROUTER,
    )


def _assert_routes(out_idx, out_bias, ref_idx, ref_bias):
    M = ref_idx.shape[1]
    assert torch.equal(out_idx[:, :M].cpu(), ref_idx)
    torch.testing.assert_close(out_bias[:, :M].cpu(), ref_bias, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "width,dql,with_img,budget",
    [
        (96, 1, False, 6),
        (200, 2, False, 8),
        (300, 1, True, 6),
        (300, 1, True, 64),  # budget above every candidate count
    ],
)
def test_decode_route_full_width(width, dql, with_img, budget):
    gen = torch.Generator().manual_seed(width + dql)
    # 1 remote block, a block-aligned end, and long contexts.
    seq = [2 * BS + 1, 9 * BS, width * BS - 3, width * BS // 2][: 4 // dql]
    idx_q, k_bar, bt, allow = _decode_inputs(width, seq, dql, 0.3, gen)
    if with_img:
        allow[1] = False  # no image candidate: must fall back to every remote block
    out_slots = 8 if budget <= 8 else 16
    out_idx = torch.full((idx_q.shape[0], 16), -7, dtype=torch.int32, device="cuda")
    out_bias = torch.full((idx_q.shape[0], 16), 9.0, device="cuda")
    M = route_decode_triton(
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=bt,
        seq_lens=torch.tensor(seq, dtype=torch.int32, device="cuda"),
        image_allow=allow.to(torch.int8) if with_img else None,
        out_idx=out_idx,
        out_bias=out_bias,
        max_block=width,
        block_size=BS,
        local_blocks=1,
        decode_query_len=dql,
        topm=16,
        budget=budget,
        out_slots=out_slots,
        **ROUTER,
    )
    assert M == out_slots
    ref = _decode_reference(
        idx_q,
        k_bar,
        bt,
        seq,
        allow if with_img else None,
        width,
        dql,
        min(budget, M),
        M,
    )
    _assert_routes(out_idx, out_bias, *ref)


@pytest.mark.parametrize("width", [1000, 4096, 4097, 6000])
def test_decode_route_windowed_scans(width):
    """Rows whose allowed span fits SCAN_TILE take the narrow window launch; the
    rest take the full-width (W <= 4096) or tiled (W > 4096) scan."""
    gen = torch.Generator().manual_seed(width)
    seq = [2 * BS + 1, width * BS - 3, width * BS, width * BS // 2]
    idx_q, k_bar, bt, allow = _decode_inputs(width, seq, 1, 0.02, gen)
    allow[0] = False
    allow[1, :] = False
    allow[1, 40 : 40 + SCAN_TILE - 8] = True  # fits the narrow window
    rows = idx_q.shape[0]
    seq_t = torch.tensor(seq, dtype=torch.int32, device="cuda")
    lo, hi = (torch.zeros(rows, dtype=torch.int32, device="cuda") for _ in range(2))
    fit = torch.zeros(rows, dtype=torch.int8, device="cuda")
    win_bounds_decode(
        allow=allow,
        seq_lens=seq_t,
        lo=lo,
        hi=hi,
        fit=fit,
        max_block=width,
        block_size=BS,
        local_blocks=1,
        decode_query_len=1,
        tile=SCAN_TILE,
    )
    ref_lo, ref_hi, ref_fit = win_bounds_reference(
        allow=allow.cpu(),
        seq_lens=seq_t.cpu(),
        width=width,
        block_size=BS,
        local_blocks=1,
        decode_query_len=1,
        tile=SCAN_TILE,
    )
    assert torch.equal(lo.cpu(), ref_lo) and torch.equal(hi.cpu(), ref_hi)
    assert torch.equal(fit.cpu(), ref_fit)
    assert fit[1] == 1 and not bool(fit.all())

    score_buf = torch.empty(rows, width, device="cuda")
    window = dict(
        entmax_base=lo,
        entmax_fit=fit,
        entmax_width=SCAN_TILE,
        score_buf_win=torch.empty(rows, SCAN_TILE, device="cuda"),
        bounds=(lo, hi),
    )
    plan = _plan_decode_route(
        nd=rows,
        W=width,
        out_idx=torch.empty(rows, 16, dtype=torch.int32),
        topm=16,
        out_slots=8,
        score_buf=score_buf,
        **window,
    )
    assert plan.tiled == _want_tiled(triton.next_power_of_2(width)) == (width > 4096)
    out_idx = torch.full((rows, 16), -7, dtype=torch.int32, device="cuda")
    out_bias = torch.full((rows, 16), 9.0, device="cuda")
    route_decode_triton(
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=bt,
        seq_lens=seq_t,
        image_allow=allow.to(torch.int8),
        out_idx=out_idx,
        out_bias=out_bias,
        max_block=width,
        block_size=BS,
        local_blocks=1,
        decode_query_len=1,
        topm=16,
        budget=6,
        out_slots=8,
        score_buf=score_buf,
        **window,
        **ROUTER,
    )
    ref = _decode_reference(idx_q, k_bar, bt, seq, allow, width, 1, 6, 8)
    _assert_routes(out_idx, out_bias, *ref)


def test_tiled_router_boundary():
    assert not _want_tiled(8191)
    assert _want_tiled(8192)


@pytest.mark.parametrize("with_img", [False, True])
@pytest.mark.parametrize("q_chunk", [2048, 16])
def test_prefill_route(with_img, q_chunk):
    gen = torch.Generator().manual_seed(int(with_img) + q_chunk)
    width = 300
    q_lens = [7, 33, 2]
    ctx = [0, 2000, 30000]
    seq = [c + n for c, n in zip(ctx, q_lens)]
    total = sum(q_lens)
    idx_q = torch.randn(total, D, generator=gen).to(torch.bfloat16).cuda()
    k_bar = torch.randn(NUM_PAGES, D, generator=gen).cuda()
    bt = _block_table(3, width, gen)
    allow = (torch.rand(total, width, generator=gen) < 0.4).cuda()
    cu = torch.tensor([0] + q_lens).cumsum(0).to(torch.int32).cuda()
    seq_t = torch.tensor(seq, dtype=torch.int32, device="cuda")
    ctx_t = torch.tensor(ctx, dtype=torch.int32, device="cuda")
    out_idx = torch.full((total, 16), -7, dtype=torch.int32, device="cuda")
    out_bias = torch.full((total, 16), 9.0, device="cuda")
    M, _, _ = route_prefill_triton(
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=bt,
        seq_lens=seq_t,
        cu_seqlens_q=cu,
        context_lens=ctx_t,
        row2req=prefill_row2req(cu_seqlens_q=cu, device="cuda"),
        image_allow=allow.to(torch.int8) if with_img else None,
        out_idx=out_idx,
        out_bias=out_bias,
        max_block=width,
        block_size=BS,
        local_blocks=1,
        topm=16,
        budget=6,
        q_chunk=q_chunk,
        out_slots=8,
        **ROUTER,
    )
    row_req = torch.arange(3, device="cuda").repeat_interleave(
        torch.tensor(q_lens, device="cuda")
    )
    pos = torch.cat([c + torch.arange(n) for c, n in zip(ctx, q_lens)]).cuda()
    scores = block_scores_reference(
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=bt,
        seq_lens=seq_t,
        row_req=row_req,
        query_pos=pos,
        width=width,
        block_size=BS,
        local_blocks=1,
    )
    ref = entmax_route_reference(
        scores=scores,
        image_allow=allow if with_img else None,
        budget=6,
        out_slots=M,
        **ROUTER,
    )
    _assert_routes(out_idx, out_bias, *ref)


def test_kbar_refresh_summarizes_planned_pages_only():
    """Zero-token plan entries (pages past the live tail) must not be written."""
    gen = torch.Generator().manual_seed(3)
    kv = torch.randn(NUM_PAGES, BS, D, generator=gen).to(torch.bfloat16).cuda()
    head_cls = torch.randn(D, generator=gen).to(torch.bfloat16).cuda()
    bt = torch.randperm(NUM_PAGES, generator=gen)[:30].view(3, 10).to(torch.int32)
    ctx = torch.tensor([0, BS - 1, 4 * BS], device="cuda")
    seq = (ctx + torch.tensor([5, 2, BS], device="cuda")).to(torch.int32)
    pages, toks = kbar_all_plan(
        block_table=bt.cuda(),
        seq_lens=seq,
        context_lens=ctx,
        block_size=BS,
        max_block=10,
        max_query_len=BS,
        device="cuda",
    )
    assert bool((toks == 0).any()) and bool((toks == BS).any())
    k_bar = torch.full((NUM_PAGES, D), 5.0, device="cuda")
    kbar_refresh_tail_triton(
        kv_cache=kv,
        k_bar=k_bar,
        block_table=bt.cuda(),
        seq_lens=seq,
        head_cls=head_cls,
        block_size=BS,
        plan=(pages, toks),
    )
    live = toks > 0
    ref = kbar_summary_reference(
        kv_cache=kv, pages=pages[live], toks=toks[live], head_cls=head_cls
    )
    torch.testing.assert_close(k_bar[pages[live].long()], ref, atol=1e-4, rtol=1e-4)
    untouched = torch.ones(NUM_PAGES, dtype=torch.bool, device="cuda")
    untouched[pages[live].long()] = False
    assert bool((k_bar[untouched] == 5.0).all())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
