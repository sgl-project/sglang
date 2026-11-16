"""DASH routing and sparse attend on Apple MPS must match the PyTorch reference.

The dispatch wrappers in ``unlimited_ocr_dash.dispatch`` route MPS tensors to
``unlimited_ocr_dash.mps``; these tests call the same entry points the backend
calls.
"""

import copy
import sys

import pytest
import torch

from sglang.kernels.ops.attention.unlimited_ocr_dash import metal
from sglang.kernels.ops.attention.unlimited_ocr_dash.dispatch import (
    apply_page2d_rope_pair,
    dash_sparse_attn,
    dash_sparse_attn_decode,
    kbar_refresh_tail,
    route_decode,
    route_prefill,
    win_bounds_decode,
)
from sglang.kernels.ops.attention.unlimited_ocr_dash.page2d_rope import (
    _rope2d_torch,
)
from sglang.kernels.ops.attention.unlimited_ocr_dash.route import (
    kbar_all_plan,
    prefill_row2req,
)
from sglang.srt.layers.attention.unlimited_dash_ops.decode_meta import (
    decode_allow_fused,
    decode_meta_fused,
)
from sglang.srt.layers.attention.unlimited_dash_ops.page2d_rope import PageCoordState
from sglang.srt.layers.attention.unlimited_dash_ops.page_state import (
    BatchLayout,
    PageIsolationState,
)
from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.kernels.unlimited_ocr_dash import (
    block_scores_reference,
    entmax_route_reference,
    kbar_summary_reference,
    sparse_attn_reference,
    torch_decode_page_meta,
    win_bounds_reference,
)

register_mps_ci(est_time=60, suite="stage-a-unit-test-mps")

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple MPS"
)

DEV = "mps"
BS, D, NUM_PAGES = 32, 128, 4096
ROUTER = dict(idx_scale=0.37, prior_floor=1e-2, sigma=1.3)


@pytest.fixture(params=["metal", "torch"])
def impl(request, monkeypatch):
    """Run each test on the Metal kernels and on the torch fallback."""
    for name in ("sparse_attend", "route", "kbar_refresh"):
        if request.param == "torch":
            monkeypatch.setattr(metal, name, lambda *args, **kwargs: False)
            continue

        def must_run(*args, _kernel=getattr(metal, name), **kwargs):
            # A silent fallback here would test the torch path twice.
            assert _kernel(*args, **kwargs)
            return True

        monkeypatch.setattr(metal, name, must_run)
    return request.param


def _to_dev(t):
    """Move to MPS keeping a stride-0 head expand (``.to`` would materialize it)."""
    return t[:1].to(DEV).expand(t.shape) if t.stride(0) == 0 else t.to(DEV)


def _block_table(rows, width, gen):
    # Distinct pages per request: a repeated page would tie its block scores.
    bt = torch.stack(
        [torch.randperm(NUM_PAGES - 1, generator=gen)[:width] + 1 for _ in range(rows)]
    )
    return bt.to(torch.int32)


def _route_reference(idx_q, k_bar, bt, seq, row_req, pos, allow, width, budget, M):
    scores = block_scores_reference(
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=bt,
        seq_lens=seq,
        row_req=row_req,
        query_pos=pos,
        width=width,
        block_size=BS,
        local_blocks=4,
    )
    return entmax_route_reference(
        scores=scores, image_allow=allow, budget=budget, out_slots=M, **ROUTER
    )


def _assert_routes(out_idx, out_bias, ref_idx, ref_bias):
    M = ref_idx.shape[1]
    assert torch.equal(out_idx[:, :M].cpu(), ref_idx)
    torch.testing.assert_close(out_bias[:, :M].cpu(), ref_bias, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "width,dql,with_img,budget",
    [(40, 1, False, 12), (200, 2, False, 8), (410, 1, True, 12), (300, 1, True, 64)],
)
def test_decode_route(width, dql, with_img, budget, impl):
    gen = torch.Generator().manual_seed(width + dql)
    seq = [5 * BS + 1, 9 * BS, width * BS - 3, width * BS // 2][: 4 // dql]
    rows = len(seq) * dql
    idx_q = torch.randn(rows, D, generator=gen).to(torch.bfloat16)
    k_bar = torch.randn(NUM_PAGES, D, generator=gen)
    bt = _block_table(len(seq), width, gen)
    allow = torch.rand(rows, width, generator=gen) < 0.3
    if with_img:
        allow[1] = False  # no image candidate: falls back to every remote block
    out_slots = 8 if budget <= 8 else 16
    out_idx = torch.full((rows, 16), -7, dtype=torch.int32, device=DEV)
    out_bias = torch.full((rows, 16), 9.0, device=DEV)
    seq_t = torch.tensor(seq, dtype=torch.int32)
    M = route_decode(
        idx_q=idx_q.to(DEV),
        k_bar=k_bar.to(DEV),
        block_table=bt.to(DEV),
        seq_lens=seq_t.to(DEV),
        image_allow=allow.to(torch.int8).to(DEV) if with_img else None,
        out_idx=out_idx,
        out_bias=out_bias,
        max_block=width,
        block_size=BS,
        local_blocks=4,
        decode_query_len=dql,
        topm=16,
        budget=budget,
        out_slots=out_slots,
        **ROUTER,
    )
    assert M == out_slots
    row_req = torch.arange(len(seq)).repeat_interleave(dql)
    pos = seq_t[row_req].long() - dql + torch.arange(rows) % dql
    ref = _route_reference(
        idx_q,
        k_bar,
        bt,
        seq_t,
        row_req,
        pos,
        allow if with_img else None,
        width,
        min(budget, M),
        M,
    )
    _assert_routes(out_idx, out_bias, *ref)


@pytest.mark.parametrize("with_img", [False, True])
def test_prefill_route(with_img, impl):
    gen = torch.Generator().manual_seed(7)
    prefix, q_lens = [0, 40 * BS + 3], [6 * BS + 5, 700]
    seq = [p + n for p, n in zip(prefix, q_lens)]
    width = -(-max(seq) // BS)
    total = sum(q_lens)
    idx_q = torch.randn(total, D, generator=gen).to(torch.bfloat16)
    k_bar = torch.randn(NUM_PAGES, D, generator=gen)
    bt = _block_table(len(seq), width, gen)
    allow = torch.rand(total, width, generator=gen) < 0.4
    cu = torch.tensor([0] + q_lens).cumsum(0).to(torch.int32)
    seq_t = torch.tensor(seq, dtype=torch.int32)
    ctx_t = torch.tensor(prefix, dtype=torch.int32)
    out_idx = torch.full((total, 16), -7, dtype=torch.int32, device=DEV)
    out_bias = torch.full((total, 16), 9.0, device=DEV)
    width_out, _, _ = route_prefill(
        idx_q=idx_q.to(DEV),
        k_bar=k_bar.to(DEV),
        block_table=bt.to(DEV),
        seq_lens=seq_t.to(DEV),
        cu_seqlens_q=cu.to(DEV),
        context_lens=ctx_t.to(DEV),
        row2req=prefill_row2req(cu_seqlens_q=cu.to(DEV), device=DEV),
        image_allow=allow.to(torch.int8).to(DEV) if with_img else None,
        out_idx=out_idx,
        out_bias=out_bias,
        max_block=width,
        block_size=BS,
        local_blocks=4,
        topm=16,
        budget=12,
        out_slots=16,
        **ROUTER,
    )
    assert width_out == 16
    row_req = torch.repeat_interleave(torch.arange(len(seq)), torch.tensor(q_lens))
    pos = torch.cat([p + torch.arange(n) for p, n in zip(prefix, q_lens)])
    ref = _route_reference(
        idx_q,
        k_bar,
        bt,
        seq_t,
        row_req,
        pos,
        allow if with_img else None,
        width,
        12,
        16,
    )
    _assert_routes(out_idx, out_bias, *ref)


def test_kbar_refresh_skips_tokenless_entries(impl):
    gen = torch.Generator().manual_seed(3)
    kv = torch.randn(64, BS, D, generator=gen).to(torch.bfloat16)
    head_cls = torch.randn(D, generator=gen).to(torch.bfloat16)
    # Request 0 ends mid-page; request 1's plan runs past its last page.
    bt = torch.tensor([[5, 9, 11, 0], [20, 21, 0, 0]], dtype=torch.int32)
    seq = torch.tensor([2 * BS + 7, BS + 1], dtype=torch.int32)
    ctx = torch.tensor([BS + 3, 0], dtype=torch.int32)
    k_bar = torch.full((64, D), 3.0, device=DEV)
    plan = kbar_all_plan(
        block_table=bt.to(DEV),
        seq_lens=seq.to(DEV),
        context_lens=ctx.to(DEV),
        block_size=BS,
        max_block=4,
        max_query_len=BS + 4,
        device=DEV,
    )
    kbar_refresh_tail(
        kv_cache=kv.to(DEV),
        k_bar=k_bar,
        block_table=bt.to(DEV),
        seq_lens=seq.to(DEV),
        head_cls=head_cls.to(DEV),
        block_size=BS,
        plan=plan,
    )
    pages, tok = (t.cpu() for t in plan)
    live = tok > 0
    ref = kbar_summary_reference(
        kv_cache=kv, pages=pages[live], toks=tok[live], head_cls=head_cls
    )
    torch.testing.assert_close(
        k_bar.cpu()[pages[live].long()], ref, atol=1e-4, rtol=1e-4
    )
    untouched = torch.ones(64, dtype=torch.bool)
    untouched[pages[live].long()] = False
    if impl == "torch":
        untouched[0] = False  # the padding page absorbs the tokenless entries
    assert bool((k_bar.cpu()[untouched] == 3.0).all())


def test_win_bounds():
    gen = torch.Generator().manual_seed(4)
    width = 300
    seq = [5 * BS + 1, width * BS - 3, width * BS, 2 * BS]
    allow = torch.rand(len(seq), width, generator=gen) < 0.02
    allow[0] = False
    allow[1, :] = False
    allow[1, 40:150] = True
    seq_t = torch.tensor(seq, dtype=torch.int32)
    lo, hi = (torch.zeros(len(seq), dtype=torch.int32, device=DEV) for _ in range(2))
    fit = torch.zeros(len(seq), dtype=torch.int8, device=DEV)
    win_bounds_decode(
        allow=allow.to(DEV),
        seq_lens=seq_t.to(DEV),
        lo=lo,
        hi=hi,
        fit=fit,
        max_block=width,
        block_size=BS,
        local_blocks=4,
        decode_query_len=1,
        tile=128,
    )
    ref_lo, ref_hi, ref_fit = win_bounds_reference(
        allow=allow,
        seq_lens=seq_t,
        width=width,
        block_size=BS,
        local_blocks=4,
        decode_query_len=1,
        tile=128,
    )
    assert torch.equal(lo.cpu(), ref_lo) and torch.equal(hi.cpu(), ref_hi)
    assert torch.equal(fit.cpu(), ref_fit)


NUM_KV_HEADS, NUM_HEADS, HEAD_DIM, TOPK = 2, 4, 128, 6


def _cache(gen):
    shape = (48, BS, NUM_KV_HEADS, HEAD_DIM)
    k = torch.randn(shape, generator=gen).to(torch.bfloat16)
    v = torch.randn(shape, generator=gen).to(torch.bfloat16)
    return k, v


def _select(query_pos, local_blocks, gen, pad_slots=0):
    """Packed remote blocks per (kv head, row), never inside the local window."""
    idx = torch.full((NUM_KV_HEADS, len(query_pos), TOPK + pad_slots), -1)
    for t, pos in enumerate(query_pos):
        n_remote = pos // BS + (1 if local_blocks == 0 else 1 - local_blocks)
        for g in range(NUM_KV_HEADS):
            pool = torch.randperm(max(n_remote, 0), generator=gen)[:TOPK]
            idx[g, t, : pool.numel()] = pool
    return idx.to(torch.int32)


def _reference(q, k, v, idx, bias, bt, query_pos, row_req, local_blocks):
    return sparse_attn_reference(
        q=q,
        k_cache=k,
        v_cache=v,
        topk_idx=idx,
        blk_bias=bias,
        block_table=bt,
        query_pos=query_pos,
        row_req=row_req,
        local_blocks=local_blocks,
        block_size=BS,
        sm_scale=HEAD_DIM**-0.5,
    )


@pytest.mark.parametrize("local_blocks", [0, 1, 4])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("shared", [True, False])
def test_prefill_attend(local_blocks, with_bias, shared, impl):
    """``shared`` is the backend's stride-0 selection over kv heads."""
    gen = torch.Generator().manual_seed(1)
    k, v = _cache(gen)
    prefix, q_lens = [0, 2 * BS - 1, 6 * BS + 5], [3, 4, 2 * BS + 1]
    seq = [p + n for p, n in zip(prefix, q_lens)]
    bt = torch.stack([torch.randperm(47, generator=gen)[:12] + 1 for _ in seq])
    query_pos = [p + j for p, n in zip(prefix, q_lens) for j in range(n)]
    row_req = [r for r, n in enumerate(q_lens) for _ in range(n)]
    idx = _select(query_pos, local_blocks, gen)
    bias = torch.randn(idx.shape, generator=gen) * (idx >= 0) if with_bias else None
    if shared:
        idx = idx[:1].expand_as(idx)
        bias = None if bias is None else bias[:1].expand_as(bias)
    q = torch.randn(len(query_pos), NUM_HEADS, HEAD_DIM, generator=gen).to(
        torch.bfloat16
    )
    cu = torch.tensor([0] + q_lens).cumsum(0).to(torch.int32)
    out = torch.zeros_like(q, device=DEV)
    dash_sparse_attn(
        q.to(DEV),
        k.to(DEV),
        v.to(DEV),
        _to_dev(idx),
        None if bias is None else _to_dev(bias),
        bt.to(torch.int32).to(DEV),
        cu.to(DEV),
        torch.tensor(seq, dtype=torch.int32, device=DEV),
        torch.tensor(prefix, dtype=torch.int32, device=DEV),
        max(q_lens),
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        out,
        local_blocks=local_blocks,
        sparse_block_size=BS,
    )
    ref = _reference(q, k, v, idx, bias, bt, query_pos, row_req, local_blocks)
    torch.testing.assert_close(out.float().cpu(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("decode_query_len", [1, 2])
@pytest.mark.parametrize("local_blocks", [0, 1, 4])
@pytest.mark.parametrize("shared", [True, False])
def test_decode_attend(decode_query_len, local_blocks, shared, impl):
    """``shared`` is the backend's stride-0 selection over kv heads."""
    gen = torch.Generator().manual_seed(2)
    k, v = _cache(gen)
    seq = [max(s, decode_query_len) for s in (1, 2 * BS, 300, BS + 1)]
    bt = torch.stack([torch.randperm(47, generator=gen)[:12] + 1 for _ in seq])
    query_pos = [s - decode_query_len + j for s in seq for j in range(decode_query_len)]
    row_req = [r for r in range(len(seq)) for _ in range(decode_query_len)]
    # Trailing -1 slots keep the buffer wider than any row's selection.
    idx = _select(query_pos, local_blocks, gen, pad_slots=2)
    bias = torch.randn(idx.shape, generator=gen) * (idx >= 0)
    if shared:
        idx, bias = idx[:1].expand_as(idx), bias[:1].expand_as(bias)
    q = torch.randn(len(query_pos), NUM_HEADS, HEAD_DIM, generator=gen).to(
        torch.bfloat16
    )
    out = torch.full_like(q, float("nan"), device=DEV)
    dash_sparse_attn_decode(
        q.to(DEV),
        k.to(DEV),
        v.to(DEV),
        _to_dev(idx),
        _to_dev(bias),
        bt.to(torch.int32).to(DEV),
        torch.tensor(seq, dtype=torch.int32, device=DEV),
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        out,
        decode_query_len,
        local_blocks=local_blocks,
        sparse_block_size=BS,
    )
    ref = _reference(q, k, v, idx, bias, bt, query_pos, row_req, local_blocks)
    torch.testing.assert_close(out.float().cpu(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("prefill", [True, False])
def test_attend_float32_out(prefill):
    """A float32 ``out`` takes the torch path; the bf16 kernel would fill half of it."""
    gen = torch.Generator().manual_seed(3)
    k, v = _cache(gen)
    seq, q_lens = [2 * BS, 300], ([3, BS + 1] if prefill else [1, 1])
    prefix = [s - n for s, n in zip(seq, q_lens)]
    bt = torch.stack([torch.randperm(47, generator=gen)[:12] + 1 for _ in seq])
    query_pos = [p + j for p, n in zip(prefix, q_lens) for j in range(n)]
    row_req = [r for r, n in enumerate(q_lens) for _ in range(n)]
    # Shared selection is the layout the Metal kernel accepts.
    idx = _select(query_pos, 1, gen)[:1].expand(NUM_KV_HEADS, -1, -1)
    bias = (torch.randn(idx.shape[1:], generator=gen) * (idx[0] >= 0)).expand_as(idx)
    q = torch.randn(len(query_pos), NUM_HEADS, HEAD_DIM, generator=gen).to(
        torch.bfloat16
    )
    out = torch.full(q.shape, float("nan"), device=DEV)
    common = (q.to(DEV), k.to(DEV), v.to(DEV), _to_dev(idx), _to_dev(bias))
    common += (bt.to(torch.int32).to(DEV),)
    seq_t = torch.tensor(seq, dtype=torch.int32, device=DEV)
    if prefill:
        cu = torch.tensor([0] + q_lens).cumsum(0).to(torch.int32).to(DEV)
        prefix_t = torch.tensor(prefix, dtype=torch.int32, device=DEV)
        dash_sparse_attn(
            *common,
            cu,
            seq_t,
            prefix_t,
            max(q_lens),
            NUM_KV_HEADS,
            HEAD_DIM**-0.5,
            out,
            local_blocks=1,
            sparse_block_size=BS,
        )
    else:
        dash_sparse_attn_decode(
            *common,
            seq_t,
            NUM_KV_HEADS,
            HEAD_DIM**-0.5,
            out,
            1,
            local_blocks=1,
            sparse_block_size=BS,
        )
    ref = _reference(q, k, v, idx, bias, bt, query_pos, row_req, 1)
    torch.testing.assert_close(out.cpu(), ref, atol=2e-2, rtol=2e-2)


IMG, PAGE, MM_PAD, META_BLOCKS, META_WIDTH = 99, 7, 1_000_000, 64, 12


def _meta_prompt(n_pages, images_per_page, with_marker):
    ids = [1, 2, 3]
    for p in range(n_pages):
        ids += ([PAGE] if with_marker else []) + [MM_PAD + p] * images_per_page + [4, 5]
    return ids


def _prefilled_meta_state(prompts, with_marker):
    """Page / coordinate state after a PyTorch prefill of ``prompts``."""
    cfg = {"image_token_id": IMG, "per_page_image_tokens": 50}
    if with_marker:
        cfg["page_token_id"] = PAGE
    page_state = PageIsolationState(cfg, BS)
    coord_state = PageCoordState(cfg, max_reqs=8, max_pages=16)
    # Block 0 is reserved: never live, so stale columns point at no image page.
    bt = torch.stack(
        [1 + r * META_WIDTH + torch.arange(META_WIDTH) for r in range(len(prompts))]
    )
    bt = bt.to(torch.int32).to(DEV)
    lens = torch.tensor([len(p) for p in prompts], device=DEV)
    q_start = torch.cat([torch.zeros(1, device=DEV), lens.cumsum(0)]).long()
    ids = torch.tensor(sum(prompts, []), device=DEV)
    req = torch.arange(len(prompts), device=DEV).repeat_interleave(lens)
    pos = torch.arange(ids.numel(), device=DEV) - q_start[req]
    slots = bt.long()[req].gather(1, (pos // BS)[:, None]).squeeze(1) * BS + pos % BS
    layout = BatchLayout(
        num_tokens=ids.numel(),
        num_reqs=len(prompts),
        q_start=q_start,
        start_pos=torch.zeros(len(prompts), dtype=torch.int64, device=DEV),
        block_table=bt,
        req_of_token=req,
    )
    page_state.step(ids, layout, slots, META_BLOCKS * BS)
    coord_state.step(ids, layout)
    return page_state, coord_state, bt, lens


@pytest.mark.parametrize("table_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("with_marker", [True, False])
def test_decode_meta_matches_torch_path(with_marker, table_dtype):
    """The Metal decode-metadata kernels advance the same state as the torch path."""
    prompts = [_meta_prompt(2, 60, with_marker), _meta_prompt(3, 40, with_marker)]
    prompts += [[1] * (2 * BS - 1), _meta_prompt(1, 121, with_marker)]
    fused_page, fused_coord, bt, seq = _prefilled_meta_state(prompts, with_marker)
    bt = bt.to(table_dtype)
    torch_page, torch_coord = copy.deepcopy(fused_page), copy.deepcopy(fused_coord)
    gen = torch.Generator().manual_seed(int(with_marker))
    num_slots = META_BLOCKS * BS
    for _ in range(40):
        seq = seq + 1
        pos = seq - 1
        slots = bt.long().gather(1, (pos // BS)[:, None]).squeeze(1) * BS + pos % BS
        choices = torch.tensor([PAGE, IMG, MM_PAD + 3, 4, 5])
        ids = choices[torch.randint(0, 5, (seq.numel(),), generator=gen)].to(DEV)
        q_page, coords = decode_meta_fused(
            input_ids=ids,
            block_table=bt,
            seq_lens=seq,
            slot_mapping=slots,
            page_state=fused_page,
            coord_state=fused_coord,
            num_slots=num_slots,
            block_size=BS,
        )
        allow = decode_allow_fused(
            block_table=bt,
            page_state=fused_page,
            q_page=q_page,
            seq_lens=seq,
            max_block=META_WIDTH + 3,
            num_blocks=META_BLOCKS,
            block_size=BS,
        )
        ref_q_page, ref_allow, ref_coords = torch_decode_page_meta(
            page_state=torch_page,
            coord_state=torch_coord,
            input_ids=ids,
            block_table=bt,
            seq_lens=seq,
            slot_mapping=slots,
            num_slots=num_slots,
            max_block=META_WIDTH + 3,
            num_blocks=META_BLOCKS,
        )
        assert torch.equal(q_page.cpu(), ref_q_page.cpu())
        assert torch.equal(coords[0].cpu(), ref_coords[0].cpu())
        assert torch.equal(coords[1].cpu(), ref_coords[1].cpu())
        assert torch.equal(allow.bool().cpu(), ref_allow.cpu())
        for name in ("_slot_img_page", "_slot_marker_cum", "_slot_img_run"):
            assert torch.equal(
                getattr(fused_page, name).cpu(), getattr(torch_page, name).cpu()
            )
        # The last coordinate row is scratch for masked-out writes.
        live = slice(0, fused_coord.max_reqs)
        for name in ("st_marker", "st_pp", "cnt"):
            assert torch.equal(
                getattr(fused_coord, name)[live].cpu(),
                getattr(torch_coord, name)[live].cpu(),
            )


def test_decode_meta_rejects_table_outside_int32():
    bt = torch.tensor([[1, 2**31]], dtype=torch.int64, device=DEV)
    with pytest.raises(ValueError, match="int32"):
        metal._int32_table(bt)


@pytest.mark.parametrize(
    "t,nx,ny,head_dim", [(1, 10, 10, 128), (37, 10, 1, 128), (5, 2, 3, 64)]
)
def test_page2d_rope_pair_bit_exact(t, nx, ny, head_dim):
    """The Metal pair kernel rounds like the bf16 PyTorch rotation."""
    gen = torch.Generator().manual_seed(t)
    x = torch.randn(t, nx * head_dim, generator=gen).to(torch.bfloat16).to(DEV)
    y = torch.randn(t, ny * head_dim, generator=gen).to(torch.bfloat16).to(DEV)
    ang = torch.rand(t, head_dim, generator=gen) * 6.28
    cos, sin = (f(ang).to(torch.bfloat16).to(DEV) for f in (torch.cos, torch.sin))
    ox, oy = apply_page2d_rope_pair(x, y, cos, sin, head_dim)
    assert torch.equal(ox.cpu(), _rope2d_torch(x.cpu(), cos.cpu(), sin.cpu(), head_dim))
    assert torch.equal(oy.cpu(), _rope2d_torch(y.cpu(), cos.cpu(), sin.cpu(), head_dim))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
