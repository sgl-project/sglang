"""Context-parallel MSA-v1 decode indexer selects exactly the stock top-k pages.

Run: python test_msa_v1_indexer_decode_cp.py (relaunches under torchrun, 2 GPUs).
Bench: torchrun --nproc-per-node 2 test_msa_v1_indexer_decode_cp.py --bench
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _skip_reason():
    from sglang.srt.layers.attention.minimax_sparse_ops import msa_indexer_decode_cp
    from sglang.srt.layers.attention.minimax_sparse_ops.msa_plan import (
        msa_v1_qmul4_decode_capability,
    )

    if torch.cuda.device_count() < 2:
        return "needs 2 GPUs"
    if msa_indexer_decode_cp._MSA_IMPORT_ERROR is not None:
        return f"MSA score-only entry points are missing: {msa_indexer_decode_cp._MSA_IMPORT_ERROR}"
    available, reason = msa_v1_qmul4_decode_capability()
    return None if available else reason


_SKIP_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))

PAGE = 128
HEADS = 4


def _dist():
    if not dist.is_initialized():
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    group = SimpleNamespace(
        world_size=dist.get_world_size(),
        rank_in_group=dist.get_rank(),
        device_group=dist.group.WORLD,
        barrier=dist.barrier,
    )
    return dist.get_rank(), group


def _inputs(*, batch, query_length, seq_lens, max_pages, tie_pages, seed):
    """Same tensors on every rank; tie_pages > 0 maps logical pages onto that many
    physical pages so equal scores force the page-id tie break."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (
        torch.randn(batch * query_length, HEADS, 128, generator=g, device="cuda") * 0.5
    ).to(torch.float8_e4m3fn)
    phys = tie_pages if tie_pages else batch * max_pages
    packed_k = torch.randint(
        0, 256, (phys, PAGE, 64), dtype=torch.uint8, generator=g, device="cuda"
    )
    k_scale = (torch.rand(phys, PAGE, 8, generator=g, device="cuda") * 0.5 + 0.125).to(
        torch.float8_e4m3fn
    )
    if tie_pages:
        page_table = torch.randint(
            0, phys, (batch, max_pages), generator=g, device="cuda"
        )
    else:
        page_table = torch.randperm(phys, generator=g, device="cuda").view(
            batch, max_pages
        )
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    return q, packed_k, k_scale.view(torch.uint8), page_table.to(torch.int32), lens


def _setup(*, group, batch, query_length, max_pages):
    """The stock plan and a CP plan of the same graph bucket."""
    from sglang.srt.layers.attention.minimax_sparse_ops.msa_indexer_decode_cp import (
        MSAV1IndexerDecodeCP,
    )
    from sglang.srt.layers.attention.minimax_sparse_ops.msa_plan import (
        MSAV1DecodeIndexerPlan,
    )

    heads_local = HEADS // group.world_size
    cp = MSAV1IndexerDecodeCP(
        group=group,
        max_rows=batch * query_length,
        max_pages=max_pages,
        num_local_heads=heads_local,
        device=torch.device("cuda"),
    )
    stock, with_cp = (
        MSAV1DecodeIndexerPlan(
            batch_size=batch,
            query_length=query_length,
            max_pages=max_pages,
            device=torch.device("cuda"),
            use_cuda_graph=True,
            num_index_heads=heads_local,
            indexer_cp=indexer_cp,
        )
        for indexer_cp in (None, cp)
    )
    assert with_cp.indexer_cp_plan is not None
    return stock, with_cp


def _refresh(plans, page_table, final_seq_lens):
    for plan in plans:
        plan.plan(page_table, final_seq_lens)


def _local_q(q, rank, plan):
    hl = plan.num_index_heads
    return q[:, rank * hl : (rank + 1) * hl].contiguous()


def _run_cp(plan, q_local, packed_k, k_scale, out_rows_heads):
    # The backend's layout: the attention reads top-k as [rows, heads, 16].
    plan.run(
        q_local,
        packed_k,
        k_scale,
        torch.ones((), device="cuda"),
        out=out_rows_heads.permute(1, 0, 2),
    )


def _run_both(stock, with_cp, rank, q, packed_k, k_scale):
    q_local = _local_q(q, rank, stock)
    ref = stock.run(q_local, packed_k, k_scale, torch.ones((), device="cuda")).clone()
    out = torch.empty_like(ref).permute(1, 0, 2).contiguous()
    _run_cp(with_cp, q_local, packed_k, k_scale, out)
    return ref, out.permute(1, 0, 2)


CASES = [
    # (batch, query_length, cached lengths, max_pages, tie_pages)
    # Rows around the 16-page top-k cut, a 1-token prefix, and >256-page rows.
    (6, 8, [1, 16 * PAGE - 8, 16 * PAGE - 7, 17 * PAGE, 30001, 65000], 512, 0),
    (3, 1, [5000, 129, 64000], 512, 0),
    # Duplicated physical pages force equal scores, so the page-id tie break decides.
    (5, 16, [20000, 30000, 33333, 40000, 61000], 512, 7),
]


@pytest.mark.parametrize("case", CASES)
def test_cp_topk_matches_stock(case):
    batch, query_length, cached, max_pages, tie_pages = case
    rank, group = _dist()
    plans = _setup(
        group=group, batch=batch, query_length=query_length, max_pages=max_pages
    )
    q, packed_k, k_scale, page_table, lens = _inputs(
        batch=batch,
        query_length=query_length,
        seq_lens=[s + query_length for s in cached],
        max_pages=max_pages,
        tie_pages=tie_pages,
        seed=1234 + batch,
    )
    _refresh(plans, page_table, lens)
    for _ in range(3):
        ref, out = _run_both(*plans, rank, q, packed_k, k_scale)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_cp_topk_matches_stock_under_graph_replay():
    """Graph-captured exchange re-arms correctly across replays and
    out-of-graph length refreshes."""
    batch, query_length, max_pages = 8, 8, 512
    rank, group = _dist()
    stock, with_cp = plans = _setup(
        group=group, batch=batch, query_length=query_length, max_pages=max_pages
    )
    q, packed_k, k_scale, page_table, _ = _inputs(
        batch=batch,
        query_length=query_length,
        seq_lens=[1] * batch,
        max_pages=max_pages,
        tie_pages=0,
        seed=7,
    )
    q_local = _local_q(q, rank, stock)
    gen = torch.Generator().manual_seed(99)

    def random_lens():
        return (
            torch.randint(1, max_pages * PAGE - 64, (batch,), generator=gen)
            .to(torch.int32)
            .cuda()
        )

    _refresh(plans, page_table, random_lens())
    ref_out = torch.empty_like(stock.result)
    cp_out = torch.empty_like(stock.result).permute(1, 0, 2).contiguous()

    def layer():
        ref_out.copy_(
            stock.run(q_local, packed_k, k_scale, torch.ones((), device="cuda"))
        )
        _run_cp(with_cp, q_local, packed_k, k_scale, cp_out)

    layer()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):
            layer()
    for _ in range(6):
        _refresh(plans, page_table, random_lens())
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(cp_out.permute(1, 0, 2), ref_out, rtol=0, atol=0)


def test_peer_cannot_overwrite_query_before_rearm(monkeypatch):
    """Back-to-back calls with only short rows (no score exchange) and one rank
    stalled before its scatter must not lose the peer's next query (hang)."""
    import sglang.srt.layers.attention.minimax_sparse_ops.msa_indexer_decode_cp as m

    batch, query_length, max_pages = 4, 8, 64
    rank, group = _dist()
    plans = _setup(
        group=group, batch=batch, query_length=query_length, max_pages=max_pages
    )
    q, packed_k, k_scale, page_table, lens = _inputs(
        batch=batch,
        query_length=query_length,
        seq_lens=[s + query_length for s in (100, 500, 1000, 1900)],
        max_pages=max_pages,
        tie_pages=0,
        seed=3,
    )
    _refresh(plans, page_table, lens)
    scatter = m.scatter_scores

    def stalled_scatter(**kwargs):
        if rank == 0:
            torch.cuda._sleep(50_000_000)
        scatter(**kwargs)

    monkeypatch.setattr(m, "scatter_scores", stalled_scatter)
    for _ in range(6):
        ref, out = _run_both(*plans, rank, q, packed_k, k_scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def _bench(
    batch=32, query_length=8, context=150_000, max_pages=8192, layers=8, iters=20
):
    rank, group = _dist()
    stock, with_cp = plans = _setup(
        group=group, batch=batch, query_length=query_length, max_pages=max_pages
    )
    pages_per_req = (context + query_length + PAGE - 1) // PAGE
    # One cache per layer so consecutive layers read cold index K.
    caches = []
    for layer in range(layers):
        q, packed_k, k_scale, page_table, lens = _inputs(
            batch=batch,
            query_length=query_length,
            seq_lens=[context + query_length] * batch,
            max_pages=pages_per_req,
            tie_pages=0,
            seed=layer,
        )
        caches.append((packed_k, k_scale))
    full_table = torch.zeros(batch, max_pages, dtype=torch.int32, device="cuda")
    full_table[:, :pages_per_req] = page_table
    _refresh(plans, full_table, lens)
    q_local = _local_q(q, rank, stock)
    one = torch.ones((), device="cuda")
    cp_out = torch.empty_like(stock.result).permute(1, 0, 2).contiguous()

    def stock_step():
        for packed_k, k_scale in caches:
            stock.run(q_local, packed_k, k_scale, one)

    def cp_step():
        for packed_k, k_scale in caches:
            _run_cp(with_cp, q_local, packed_k, k_scale, cp_out)

    results = {}
    for name, fn in (("stock", stock_step), ("cp", cp_step)):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        times = []
        for _ in range(iters):
            dist.barrier()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / layers)
        times.sort()
        results[name] = times[len(times) // 2]
    ref, out = _run_both(*plans, rank, q, *caches[-1])
    same = torch.equal(ref, out)
    print(
        f"[rank{rank}] bs{batch} x {context} ctx x q{query_length}: stock {results['stock']:.1f} us/layer, "
        f"cp {results['cp']:.1f} us/layer, identical={same}",
        flush=True,
    )


if __name__ == "__main__":
    if "--bench" in sys.argv:
        _bench()
        dist.destroy_process_group()
    else:
        multigpu_pytest_main(__name__, __file__, num_gpus=[2])
