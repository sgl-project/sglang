"""The V4.1 HIP dense prefill indexer on AITER's row-group kernel scores what the schedule kernel
scores, and the tile-gated top-k selects what the stock top-k selects."""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
    topk_transform_paged_hip,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
    LOW_RATIO_PAGE_TABLE_BUCKET,
    aiter_fp4_paged_mqa_logits,
    aiter_has_fp4_rowgroup,
    aiter_has_rowgroup_abs,
    pack_fp4_query_flydsl,
    prepare_fp4_rowgroup_prefill_workspace,
    rowgroup_table_columns,
)
from sglang.kernels.ops.attention.dsv4.topk_tilegate_hip import TILE, TileGateTopK
from sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache_hip import (
    store_fp4_index_k_cache_split,
)
from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
    _score_requests_rowgroup,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=40, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported() and aiter_has_fp4_rowgroup()),
    reason="AITER's row-group FP4 MQA logits kernel (gfx95x, ROCm/aiter#6145).",
)
needs_abs = pytest.mark.skipif(
    not aiter_has_rowgroup_abs(), reason="AITER's row-group |s| epilogue"
)

DEVICE = "cuda"
PAGE = 64
K = 512


def build_prefill_case(extend_lens, contexts, heads=64, seed=0):
    """A ragged prefill step: request b owns extend_lens[b] consecutive rows, row j of it
    sees contexts[b] - (extend_lens[b] - 1 - j) compressed positions (causal)."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    pages_per_req = [-(-c // PAGE) for c in contexts]
    num_pages = sum(pages_per_req) + 1
    payload = torch.zeros((num_pages, 1, 4, PAGE, 16), dtype=torch.uint8, device=DEVICE)
    scale = torch.zeros((num_pages, 1, 4, PAGE), dtype=torch.uint8, device=DEVICE)
    k = torch.randn(num_pages * PAGE, 128, generator=gen, device=DEVICE)
    slots = torch.arange(num_pages * PAGE, device=DEVICE)
    store_fp4_index_k_cache_split(
        k.to(torch.bfloat16), payload, scale, slots, page_size=PAGE, rne=True
    )
    perm = torch.randperm(num_pages, generator=gen, device=DEVICE).to(torch.int32)
    tables = torch.zeros((len(contexts), max(pages_per_req)), dtype=torch.int32)
    start = 0
    for b, n in enumerate(pages_per_req):
        tables[b, :n] = perm[start : start + n].cpu()
        start += n
    counts = torch.tensor(extend_lens)
    page_table = tables.repeat_interleave(counts, dim=0).to(DEVICE)
    row_ends = torch.cat(
        [
            torch.arange(c - n + 1, c + 1).clamp_min(0)
            for n, c in zip(extend_lens, contexts)
        ]
    ).to(device=DEVICE, dtype=torch.int32)
    rows = int(counts.sum())
    q = torch.randn(rows, heads, 128, generator=gen, device=DEVICE)
    q_fp4, q_scale = pack_fp4_query_flydsl(q.to(torch.bfloat16))
    weights = torch.randn(rows, heads, generator=gen, device=DEVICE) * heads**-0.5
    workspace = prepare_fp4_rowgroup_prefill_workspace(
        page_table,
        list(extend_lens),
        page_table_bucket=LOW_RATIO_PAGE_TABLE_BUCKET,
        table_columns=rowgroup_table_columns(1 << 17, 2, LOW_RATIO_PAGE_TABLE_BUCKET),
    )
    return dict(
        q_fp4=q_fp4,
        q_scale=q_scale,
        weights=weights.to(torch.bfloat16),
        k_payload=payload,
        k_scale=scale,
        page_table=page_table,
        row_ends=row_ends,
        extend_lens=list(extend_lens),
        workspace=workspace,
    )


def schedule_logits(case) -> torch.Tensor:
    return aiter_fp4_paged_mqa_logits(
        q_fp4=case["q_fp4"],
        q_scale=case["q_scale"],
        k_payload=case["k_payload"],
        k_scale=case["k_scale"],
        weights=case["weights"],
        page_table=case["page_table"],
        c4_seq_lens=case["row_ends"],
        weight_scale=1.0,
        page_table_bucket=LOW_RATIO_PAGE_TABLE_BUCKET,
        is_decode=False,
    ).clone()


def rowgroup_logits(case, **kw) -> torch.Tensor:
    return _score_requests_rowgroup(
        page8=False,
        q_fp4=case["q_fp4"],
        q_scale=case["q_scale"],
        weights=case["weights"],
        k_payload=case["k_payload"],
        k_scale=case["k_scale"],
        workspace=case["workspace"],
        rows=slice(0, case["page_table"].shape[0]),
        req_lo=0,
        rows_per_request=case["extend_lens"],
        row_ends=case["row_ends"],
        **kw,
    ).clone()


def assert_rows_close(got, ref, ends, rtol):
    for r, end in enumerate(ends.tolist()):
        if end:
            e = ref[r, :end].double()
            err = float((got[r, :end].double() - e).abs().max())
            assert err <= rtol * max(1.0, float(e.abs().max())), (r, err)


@pytest.mark.parametrize("abs_epilogue", [False, pytest.param(True, marks=needs_abs)])
def test_rowgroup_prefill_matches_schedule_kernel(abs_epilogue):
    """8-row tiles with short tails, tiny tiles and an empty request. Row-group sums are
    reordered (2e-6); the |s| epilogue's q' is 8 FP4 pieces (1e-5)."""
    case = build_prefill_case([16, 3, 9, 1, 0, 5], [700, 64, 1000, 33, 50, 6])
    got = rowgroup_logits(case, abs_epilogue=abs_epilogue)
    assert_rows_close(
        got, schedule_logits(case), case["row_ends"], 1e-5 if abs_epilogue else 2e-6
    )


def run_gate(scores, lens, page_table, tile_max, gate):
    """(gated, stock) (page, raw) top-k outputs."""
    outs = []
    for gated in (True, False):
        page, raw = (
            torch.full((scores.shape[0], K), -7, dtype=torch.int32, device=DEVICE)
            for _ in range(2)
        )
        if gated:
            gate(scores, lens, page_table, page, PAGE, raw, tile_max=tile_max)
        else:
            topk_transform_paged_hip(scores, lens, page_table, page, PAGE, raw)
        outs.append((page, raw))
    return outs


def assert_selects_stock_set(scores, lens, page_table, tile_max, gate):
    """Stock's value multiset on every row, its index set where the k-th value is not tied,
    and its page transform."""
    (g_page, g_raw), (_, s_raw) = run_gate(scores, lens, page_table, tile_max, gate)
    for r, n in enumerate(lens.tolist()):
        g, s = g_raw[r][g_raw[r] >= 0], s_raw[r][s_raw[r] >= 0]
        assert g.numel() == s.numel() == min(n, K) == K - int((g_raw[r] == -1).sum()), r
        assert torch.equal(
            scores[r, g.long()].sort().values, scores[r, s.long()].sort().values
        ), r
        if s.numel() and int((scores[r, :n] == scores[r, s.long()].min()).sum()) == 1:
            assert set(g.tolist()) == set(s.tolist()), r
        pos = g_raw[r].long()
        ok = pos >= 0
        want = page_table[r].long()[pos[ok] // PAGE] * PAGE + pos[ok] % PAGE
        assert torch.equal(g_page[r][ok].long(), want) and torch.equal(
            g_page[r][~ok], g_raw[r][~ok]
        ), r


def test_tilegate_topk_selects_the_stock_set():
    """Short, empty and tie-heavy rows (one overflowing the candidate cap), with garbage past
    each row's bound."""
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    width = 64 * 1024
    lens = [0, 5, 17, 511, 513, 2000, 40000, 65536, 65536, 50001, 33]
    rows = len(lens)
    scores = torch.randn(rows, width, generator=gen, device=DEVICE)
    scores[6] = scores[6].round()  # exact ties at the boundary
    scores[7] = 1.0  # every tile max ties: past the cap, the stock fallback
    scores[8, ::97] += 50.0  # survivors over the whole row
    scores[9] = (scores[9] * 4).round() / 4
    lens = torch.tensor(lens, dtype=torch.int32, device=DEVICE)
    col = torch.arange(width, device=DEVICE)
    scores = torch.where(col < lens[:, None], scores, scores.new_tensor(1e30))
    page_table = torch.stack(
        [
            torch.randperm(4 * width // PAGE, generator=gen, device=DEVICE)[
                : width // PAGE
            ]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    gate = TileGateTopK(topk=K, tile_max_elems=rows * width // TILE, device=DEVICE)
    tile_max = gate.tile_max(rows, width)
    tile_max.copy_(scores.view(rows, -1, TILE).amax(dim=2))
    assert_selects_stock_set(scores, lens, page_table, tile_max, gate)


@needs_abs
def test_tilegate_on_kernel_tile_maxima():
    """End to end: the |s| kernel's tile maxima (last partial tiles unmasked) feed the gate."""
    case = build_prefill_case([8, 3, 1], [60000, 33000, 40001], seed=11)
    rows, width = case["page_table"].shape[0], case["workspace"].max_seq_len
    gate = TileGateTopK(topk=K, tile_max_elems=rows * width // TILE, device=DEVICE)
    tile_max = gate.tile_max(rows, width)
    logits = rowgroup_logits(case, abs_epilogue=True, tile_max=tile_max)
    lens = case["row_ends"]
    for r, end in enumerate(lens.tolist()):
        full = end // TILE
        want = logits[r, : full * TILE].view(full, TILE).amax(dim=1)
        assert torch.equal(tile_max[r, :full], want), r
    assert_selects_stock_set(logits, lens, case["page_table"], tile_max, gate)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
