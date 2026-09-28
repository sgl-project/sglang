"""DeepSeek-V4 eager ragged C4 indexer vs the paged reference.

Synthetic MIXED chunk: prefill sequences whose extend rows fill 3,840 rows, plus
20 x 6 verify rows, padded to 4,096 rows as the captured segment sees it. Same fp8
K cache pages, page tables, q and weights for:
  paged  = deep_gemm.fp8_paged_mqa_logits over all 4,096 padded rows + one v2 top-k
  ragged = one batched K gather + deep_gemm.fp8_mqa_logits (compressed logits)
           + v2 top-k over the prefill rows, plus the paged kernel + top-k over
           the verify rows
Prefill-row logits must be bitwise equal over [0, c4len), and the top-k sets
identical on every real row. The same v2 top-k runs on both sides; it is
deterministic only on rows with at most 2048 candidates in the threshold coarse
bin (see test_dsv4_topk_det.py). A last case shrinks the logits memory budget so
the ragged rows run in >= 3 row chunks, and checks they match the single launch.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

deep_gemm = pytest.importorskip("deep_gemm")

from sglang.kernels.ops.attention.dsa.index_buf_accessor import _get_k_and_s_triton
from sglang.kernels.ops.attention.dsv4 import plan_topk_v2, topk_transform_paged_v2
from sglang.srt.layers.attention import mqa_logits_utils
from sglang.srt.layers.attention.dsv4 import indexer as indexer_mod
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    C4IndexerBackendMixin,
    build_ragged_indexer_plan,
)
from sglang.srt.layers.attention.mqa_logits_utils import mqa_logits_row_bytes
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="DeepSeek-V4 indexer kernels need Blackwell (SM100+)",
)


@pytest.fixture(autouse=True)
def _no_server_schedule(monkeypatch):
    # The logits budget reads mem_fraction_static from the server's schedule.
    monkeypatch.setattr(
        mqa_logits_utils,
        "get_schedule",
        lambda: SimpleNamespace(mem_fraction_static=None),
    )


dev = "cuda"
H, D, TOPK, C4PAGE = 64, 128, 512, 64
ROWS_PRE, N_VER, VER_ROWS, PADDED = 3840, 20, 6, 4096


def build_case(g, contexts, ver_contexts):
    """Cache buf, per-row page table [PADDED, max_pages] and c4 lens [PADDED]."""
    n_seq = len(contexts)
    ext = [ROWS_PRE // n_seq] * n_seq
    ext[-1] += ROWS_PRE - sum(ext)
    seqs = list(zip(contexts, ext)) + [(c, VER_ROWS) for c in ver_contexts]
    pages_per = [(c // 4 + C4PAGE - 1) // C4PAGE for c, _ in seqs]
    max_pages = max(pages_per)
    n_pages = sum(pages_per) + 1  # page 0 unused (padding rows point at it)
    # K cache: [n_pages, 64*132] uint8 = 64 tokens x (128 fp8 + 4 B fp32 scale)
    k = (
        (torch.randn(n_pages, C4PAGE, D, generator=g) * 0.5)
        .to(FP8_DTYPE)
        .view(torch.uint8)
    )
    sc = (
        (torch.rand(n_pages, C4PAGE, generator=g) * 0.05 + 0.01)
        .to(torch.float32)
        .view(torch.uint8)
        .view(n_pages, C4PAGE, 4)
    )
    buf = (
        torch.cat([k.reshape(n_pages, -1), sc.reshape(n_pages, -1)], dim=1)
        .contiguous()
        .to(dev)
    )
    assert buf.shape[1] == C4PAGE * 132
    rows = sum(n for _, n in seqs)
    page_table = torch.zeros(PADDED, max_pages, dtype=torch.int32)
    c4 = torch.ones(PADDED, dtype=torch.int32)  # padded rows: length 1
    row = 0
    pg = 1
    for (ctx, n), npg in zip(seqs, pages_per):
        pt = torch.arange(pg, pg + npg, dtype=torch.int32)
        pg += npg
        for i in range(n):
            pos = ctx - n + i
            page_table[row, :npg] = pt
            c4[row] = min(ctx // 4, (pos + 1) // 4)
            row += 1
    assert row == rows
    q = (torch.randn(PADDED, H, D, generator=g) * 0.5).to(FP8_DTYPE)
    w = (torch.rand(PADDED, H, generator=g) + 0.1).to(torch.float32)
    return dict(
        buf=buf,
        page_table=page_table.to(dev),
        c4=c4.to(dev),
        q=q.to(dev),
        w=w.to(dev),
        ext=ext,
        seq_lens=list(contexts),
        real=rows,
        mixed_t=ROWS_PRE,
        max_pages=max_pages,
    )


def paged_path(case, a, b, topk_out):
    pt = case["page_table"][a:b]
    c4l = case["c4"][a:b]
    cache = case["buf"].view(case["buf"].shape[0], C4PAGE, 1, 132)
    meta = deep_gemm.get_paged_mqa_logits_metadata(
        c4l[:, None], C4PAGE, deep_gemm.get_num_sms()
    )
    logits = deep_gemm.fp8_paged_mqa_logits(
        case["q"][a:b].unsqueeze(1),
        cache,
        case["w"][a:b],
        c4l[:, None],
        pt,
        meta,
        case["max_pages"] * C4PAGE,
        False,
    )
    topk_transform_paged_v2(logits, c4l, pt, topk_out, C4PAGE, plan_topk_v2(c4l))
    return logits


def ragged_plan(case):
    return build_ragged_indexer_plan(
        extend_lens=case["ext"],
        seq_lens=case["seq_lens"],
        c4_seq_lens=case["c4"],
        page_table=case["page_table"],
        max_c4_seq_len=case["max_pages"] * C4PAGE,
        c4_page_size=C4PAGE,
        topk=TOPK,
        num_prefill_rows=case["mixed_t"],
        num_rows=case["real"],
    )


def gather_k(case, plan):
    def get_index_k_scale_buffer(
        layer_id, seq_len_tensor, page_indices, seq_len_sum, max_seq_len
    ):
        return _get_k_and_s_triton(
            buf=case["buf"],
            page_indices=page_indices,
            seq_lens=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
            page_size=C4PAGE,
            index_head_dim=D,
        )

    return C4IndexerBackendMixin._gather_nonpaged_index_k(
        c4_indexer=SimpleNamespace(layer_id=0),
        token_to_kv_pool=SimpleNamespace(
            get_index_k_scale_buffer=get_index_k_scale_buffer
        ),
        plan=plan.ragged,
    )


def ragged_logits(case, plan, kv, rows):
    return C4IndexerBackendMixin._nonpaged_mqa_logits(
        q_indexer=case["q"], weights=case["w"], kv=kv, plan=plan.ragged, rows=rows
    )


def ragged_path(case, plan, kv, topk_out):
    C4IndexerBackendMixin._ragged_indexer_topk(
        plan=plan,
        q_indexer=case["q"],
        weights=case["w"],
        kv=kv,
        page_table=case["page_table"],
        page_size=C4PAGE,
        out_page_indices=topk_out,
    )
    for rows in plan.paged_ranges:
        paged_path(case, rows.start, rows.stop, topk_out[rows])


def new_topk_out():
    return torch.full((PADDED, TOPK), -1, dtype=torch.int32, device=dev)


def topk_rows_differ(a, b, rows):
    sa = torch.sort(a[:rows], dim=1).values
    sb = torch.sort(b[:rows], dim=1).values
    return int((sa != sb).any(dim=1).sum())


def logits_rows_mismatch(ref, got, lens):
    w = min(ref.shape[1], got.shape[1])
    cols = torch.arange(w, device=dev, dtype=torch.int32)
    valid = cols[None, :] < lens[:, None]
    n = lens.shape[0]
    mismatch = int((((ref[:n, :w] != got[:n, :w]) & valid).any(dim=1)).sum())
    nan = int((torch.isnan(ref[:n, :w]) & valid).sum())
    return mismatch, nan


def random_verify_contexts(g):
    return [int(x) for x in torch.randint(8192, 40961, (N_VER,), generator=g)]


@pytest.mark.parametrize(
    "contexts",
    [
        [8192, 30720, 61440],
        [8192, 8192, 8192],
        [30720, 30720],
        [61440, 61440],
        [71680],
    ],
    ids=[
        "mixed_8k_30k_60k",
        "uniform_8k_x3",
        "uniform_30k_x2",
        "uniform_60k_x2",
        "uniform_70k_x1",
    ],
)
def test_ragged_matches_paged(contexts):
    g = torch.Generator().manual_seed(7)
    case = build_case(g, contexts, random_verify_contexts(g))
    real, mixed_t = case["real"], case["mixed_t"]
    plan = ragged_plan(case)
    assert plan.ragged is not None and plan.ragged.query_rows == mixed_t
    assert plan.paged_ranges == [slice(mixed_t, real)], plan.paged_ranges

    out_p, out_r = new_topk_out(), new_topk_out()
    lp = paged_path(case, 0, PADDED, out_p)
    kv = gather_k(case, plan)
    ragged_path(case, plan, kv, out_r)
    lr = ragged_logits(case, plan, kv, slice(0, mixed_t))

    mismatch, nan = logits_rows_mismatch(lp, lr, plan.lens)
    assert mismatch == 0 and nan == 0, (mismatch, nan)
    assert topk_rows_differ(out_p, out_r, real) == 0


def test_ragged_row_chunks_match_single_launch(monkeypatch):
    g = torch.Generator().manual_seed(11)
    case = build_case(g, [8192, 30720, 61440], random_verify_contexts(g))
    real = case["real"]
    whole = ragged_plan(case)
    assert whole.ragged.rows_per_chunk is None
    rows = whole.ragged.query_rows
    # A logits budget of a third of the rows, less one: >= 3 chunks.
    budget = mqa_logits_row_bytes(whole.ragged.max_seqlen_k) * (rows // 3 - 1)
    monkeypatch.setattr(indexer_mod, "mqa_logits_needs_budget_check", lambda **_: True)
    monkeypatch.setattr(indexer_mod, "mqa_logits_budget_bytes", lambda **_: budget)
    chunked = ragged_plan(case)
    chunk_rows = chunked.ragged.rows_per_chunk
    assert chunk_rows is not None and len(chunked.topk_plans) >= 3

    kv = gather_k(case, whole)
    lw = ragged_logits(case, whole, kv, slice(0, rows))
    lc = torch.cat(
        [
            ragged_logits(case, chunked, kv, slice(s, min(s + chunk_rows, rows)))
            for s in range(0, rows, chunk_rows)
        ]
    )
    mismatch, nan = logits_rows_mismatch(lw, lc, whole.lens)
    assert mismatch == 0 and nan == 0, (mismatch, nan)

    out_w, out_c = new_topk_out(), new_topk_out()
    ragged_path(case, whole, kv, out_w)
    ragged_path(case, chunked, kv, out_c)
    assert topk_rows_differ(out_w, out_c, real) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
