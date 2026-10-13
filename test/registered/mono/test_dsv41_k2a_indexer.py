"""DSV4.1 mono K2a stage 1 (K2i, on by default with ``--layer-backend mono``): the vendored indexer against its
references."""

import numpy as np
import pytest
import torch

from sglang.kernels.ops.mono.gfx950.deepseek_v41.indexer import index_plan as ip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _walk_case(rng, n):
    """A step's rows: requests of ``per`` rows (1 decode, 6 verify, or ragged runs) with bounds around a context."""
    kind = rng.integers(3)
    sizes = []
    while sum(sizes) < n:
        sizes.append(1 if kind == 0 else 6 if kind == 1 else int(rng.integers(1, 9)))
    owners, bounds, row = [], [], 0
    for size in sizes:
        size = min(size, n - row)
        ctx = int(
            rng.choice(
                [
                    0,
                    300,
                    512,
                    513,
                    int(rng.integers(0, 4096)),
                    int(rng.integers(0, 1 << 20)),
                ]
            )
        )
        owners += [row] * size
        bounds += [max(ctx - (size - 1 - r), 0) for r in range(size)]
        row += size
    return np.array(bounds, dtype=np.int32), np.array(owners, dtype=np.int32)


@cuda
@pytest.mark.parametrize("seed", range(4))
def test_walk_plan_device_equals_host(seed):
    from sglang.kernels.ops.mono.gfx950.deepseek_v41.indexer.walk_plan import (
        fill_walk_plan_device,
        new_plan,
    )

    rng = np.random.default_rng(seed)
    dev = torch.device("cuda")
    plan = new_plan(dev)
    for _ in range(250):
        n = int(rng.integers(1, 49))
        bounds, owners = _walk_case(rng, n)
        want = np.zeros((plan.shape[0], 4), dtype=np.int32)
        ip.fill_walk_plan(bounds, owners, want)
        plan.fill_(-7)
        fill_walk_plan_device(
            torch.from_numpy(bounds).to(dev), torch.from_numpy(owners).to(dev), plan, n
        )
        got = plan.cpu().numpy()
        assert np.array_equal(got, want), (
            n,
            bounds.tolist(),
            owners.tolist(),
            np.argwhere(got != want)[:4].tolist(),
        )


E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _cdiv(a, b):
    return -(-a // b)


def fp4_case(ctxs, seed, per_req=6, dev="cuda"):
    """Synthetic indexer inputs in SGLang's layouts: ``len(ctxs)`` requests of ``per_req`` rows, request r seeing
    ``ctxs[r]`` compressed rows on its last row (one fewer each row before)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    rows = len(ctxs) * per_req
    ppr = _cdiv(max(ctxs), 64)
    pages = len(ctxs) * ppr + 3
    payload = torch.randint(
        0, 256, (pages, 1, 4, 64, 16), generator=g, device=dev, dtype=torch.uint8
    )
    kscale = torch.randint(
        123, 130, (pages, 1, 4, 64), generator=g, device=dev, dtype=torch.uint8
    )
    perm = torch.randperm(pages, generator=g, device=dev).to(torch.int32)
    table = (
        perm[: len(ctxs) * ppr]
        .view(len(ctxs), ppr)
        .repeat_interleave(per_req, 0)
        .contiguous()
    )
    q = torch.randint(
        0, 256, (rows, 32, 64), generator=g, device=dev, dtype=torch.uint8
    )
    qscale = torch.randint(
        123, 130, (rows, 1, 4, 16, 4), generator=g, device=dev, dtype=torch.uint8
    )
    w = (torch.randn(rows, 32, generator=g, device=dev) * 0.05).to(torch.bfloat16)
    bounds = torch.tensor(
        [max(c - (per_req - 1 - i), 0) for c in ctxs for i in range(per_req)],
        dtype=torch.int32,
        device=dev,
    )
    return dict(q=q, qscale=qscale, w=w, payload=payload, kscale=kscale, table=table, bounds=bounds,
                per_req=per_req)  # fmt: skip


def _unpack(b):
    lut = torch.tensor(E2M1, dtype=torch.float64, device=b.device)
    return torch.stack([lut[(b & 15).long()], lut[(b >> 4).long()]], -1).flatten(-2)


def ref_logits(case, t, cols):
    """fp64 logits of row t at columns ``cols``: sum_h w_h relu(sum_c 2^qe 2^ke q.k)."""
    page = case["table"][t][cols // 64].long()
    slot = cols % 64
    k = _unpack(case["payload"][page, 0, :, slot, :])  # [n, 4, 32]
    ke = case["kscale"][
        page[:, None],
        0,
        torch.arange(4, device=cols.device)[None],
        ((slot % 16) * 4 + slot // 16)[:, None],
    ]
    ks = torch.exp2(ke.double() - 127)  # [n, 4]
    q = _unpack(case["q"][t].view(32, 4, 16))  # [32, 4, 32]
    h = torch.arange(32, device=cols.device)
    qs = torch.exp2(
        case["qscale"][t, 0][:, h % 16, h // 16].double().T - 127
    )  # [32, 4]
    s = torch.einsum("hcd,ncd->hnc", q, k) * qs[:, None, :] * ks[None]
    return (case["w"][t].double()[:, None] * s.sum(-1).clamp_min(0)).sum(0)


def run_k2i(runner, case, kind="full"):
    rows = case["q"].shape[0]
    owners, kmeta = runner.owners(rows, case["per_req"], case["table"].stride(0))
    plan = runner.plan(1, case["bounds"], owners, rows)
    raw = torch.full((rows, 512), -9, dtype=torch.int32, device=case["q"].device)
    page = torch.full_like(raw, -9)
    runner.launch(q_fp4=case["q"], q_scale=case["qscale"], weights=case["w"], k_payload=case["payload"],
                  k_scale=case["kscale"], page_table=case["table"], bounds=case["bounds"], plan=plan, owners=owners,
                  kmeta=kmeta, raw_out=raw, page_out=page, kind=kind)  # fmt: skip
    runner.epoch[0] += 1
    torch.cuda.synchronize()
    return raw, page


def sglang_logits(case):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        aiter_fp4_paged_mqa_logits,
    )

    return aiter_fp4_paged_mqa_logits(
        q_fp4=case["q"], q_scale=case["qscale"], k_payload=case["payload"], k_scale=case["kscale"],
        weights=case["w"], page_table=case["table"], c4_seq_lens=case["bounds"], weight_scale=1.0,
        page_table_bucket=64, is_decode=True,
    )  # fmt: skip


def check_selection(case, logits, raw, page):
    """The selection is exact: every row's 512 best (or every column, in order, when it sees <= 512), ascending, -1
    padded; its pool rows through the page table."""
    for t in range(raw.shape[0]):
        b = int(case["bounds"][t])
        got = raw[t]
        n = min(b, 512)
        assert (got[n:] == -1).all() and (page[t][n:] == -1).all(), t
        ids = got[:n].long()
        assert (ids.diff() > 0).all() and ids.min() >= 0 and ids.max() < b, t
        if b > 512:
            row = logits[t, :b]
            chosen = torch.zeros(b, dtype=torch.bool, device=row.device)
            chosen[ids] = True
            assert row[chosen].min() >= row[~chosen].max(), (
                t,
                float(row[chosen].min()),
                float(row[~chosen].max()),
            )
        want_page = case["table"][t][ids // 64].long() * 64 + ids % 64
        assert torch.equal(page[t][:n].long(), want_page), t


@cuda
@pytest.mark.parametrize(
    "ctxs", [[8192], [300, 8192], [131072], [524288]], ids=["8k", "mix", "128k", "512k"]
)
def test_k2i_logits_and_selection(ctxs):
    from sglang.kernels.ops.mono.gfx950.deepseek_v41.indexer.runner import (
        K2iRunner,
    )

    case = fp4_case(ctxs, seed=len(ctxs) + max(ctxs))
    runner = K2iRunner("cuda", max(ctxs))
    raw, page = run_k2i(runner, case)
    rows = case["q"].shape[0]
    got = runner.logits(rows)
    want = sglang_logits(case)
    g = torch.Generator(device="cuda").manual_seed(1)
    for t in range(rows):
        b = int(case["bounds"][t])
        if b <= 512:
            continue
        a, s = got[t, :b], want[t, :b]
        scale = s.abs().max().clamp_min(1e-30)
        assert torch.isfinite(a).all(), t
        err = float((a - s).abs().max() / scale)
        assert err <= 1e-5, (t, err)
        cols = (
            torch.randperm(b, generator=g, device="cuda")[:4096].int()
            if b > 8192
            else torch.arange(b, device="cuda", dtype=torch.int32)
        )
        ref = ref_logits(case, t, cols)
        err64 = float((a[cols.long()].double() - ref).abs().max() / ref.abs().max())
        assert err64 <= 1e-5, (t, err64)
    check_selection(case, got, raw, page)


def _sets(a, b):
    """Per-row Jaccard of two -1 padded id rows."""
    out = []
    for x, y in zip(a.tolist(), b.tolist()):
        x, y = {v for v in x if v >= 0}, {v for v in y if v >= 0}
        out.append(1.0 if not x and not y else len(x & y) / len(x | y))
    return out


def sglang_select(case, logits, two_level=None):
    """SGLang's decode selection on ``logits``: plain paged top-k, the producer's candidate blocks, or (two_level =
    its CandidateBlocks) the consumer's top-k inside them. -> (raw, page, candidates)."""
    from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
        select_candidate_blocks_hip,
        topk_transform_paged_hip,
        topk_within_candidate_blocks_hip,
    )

    rows = case["q"].shape[0]
    raw = torch.empty(rows, 512, dtype=torch.int32, device="cuda")
    page = torch.empty_like(raw)
    if two_level is not None:
        topk_within_candidate_blocks_hip(logits, case["bounds"], two_level, page_table=case["table"], page_size=64,
                                         page_indices=page, raw_indices=raw)  # fmt: skip
        return raw, page, None
    cands = select_candidate_blocks_hip(
        logits, case["bounds"], topk_blocks=2048, block_size=8
    )
    topk_transform_paged_hip(logits, case["bounds"], case["table"], page, 64, raw)
    return raw, page, cands


@cuda
@pytest.mark.parametrize(
    "ctxs", [[8192, 20000], [131072], [300, 524288]], ids=["8k20k", "128k", "512k"]
)
def test_k2i_vs_sglang_selection(ctxs):
    """(b) the same logits -> the same sets (ties aside); (c) the layer pipeline (FULL, the candidate producer, a
    REINDEX consumer on the producer's plane) vs SGLang's: per-row Jaccard."""
    from sglang.kernels.ops.mono.gfx950.deepseek_v41.indexer.runner import (
        K2iRunner,
    )

    case = fp4_case(ctxs, seed=11 + max(ctxs))
    runner = K2iRunner("cuda", max(ctxs))
    rows = case["q"].shape[0]

    # producer (FULL + blocks)
    raw, page = run_k2i(runner, case, "producer")
    own = runner.logits(rows)[:, : runner.lstride].contiguous()
    s_raw, _, s_cands = sglang_select(case, own)
    assert min(_sets(raw, s_raw)) == 1.0
    assert min(_sets(runner.icout[:rows], s_cands.ids)) == 1.0
    want_raw, _, want_cands = sglang_select(case, sglang_logits(case))
    j_full, j_blk = _sets(raw, want_raw), _sets(runner.icout[:rows], want_cands.ids)

    # consumer: another layer's query on the producer's plane, inside the producer's blocks
    g = torch.Generator(device="cuda").manual_seed(5)
    case2 = dict(case, q=torch.randint(0, 256, case["q"].shape, generator=g, device="cuda", dtype=torch.uint8),
                 w=(torch.randn(rows, 32, generator=g, device="cuda") * 0.05).to(torch.bfloat16))  # fmt: skip
    raw2, page2 = run_k2i(runner, case2, "reindex")
    want_raw2, want_page2, _ = sglang_select(
        case2, sglang_logits(case2), two_level=want_cands
    )
    j_re = _sets(raw2, want_raw2)
    for t in range(rows):
        n = min(int(case["bounds"][t]), 512)
        ids = raw2[t][:n].long()
        assert (ids.diff() > 0).all() and (raw2[t][n:] == -1).all(), t
        assert torch.equal(
            page2[t][:n].long(), case["table"][t][ids // 64].long() * 64 + ids % 64
        ), t
    q = lambda js: (float(np.median(js)), float(np.percentile(js, 1)))  # noqa: E731
    print(f"\n{ctxs}: jaccard full {q(j_full)} blocks {q(j_blk)} reindex {q(j_re)}")
    for js in (j_full, j_blk, j_re):
        med, p1 = q(js)
        assert med == 1.0 and p1 >= 0.99, (med, p1)


def _fake_backend(case, ctx):
    from types import SimpleNamespace

    pool = SimpleNamespace(get_index_k_fp4_payload_buffer=lambda i: case["payload"],
                           get_index_k_fp4_scale_buffer=lambda i: case["kscale"])  # fmt: skip
    meta = SimpleNamespace(
        page_table=case["table"],
        compressed_seq_lens=case["bounds"],
        compressed_page_size=64,
    )
    backend = SimpleNamespace(token_to_kv_pool=pool, forward_metadata=SimpleNamespace(k2i_planned={}),
                              max_context_len=ctx)  # fmt: skip
    return backend, meta


@cuda
@pytest.mark.parametrize(
    "ctxs", [[300, 20000], [131072], [524288, 70000]], ids=["20k", "128k", "512k"]
)
def test_hook_hybrid_vs_sglang(ctxs):
    """K2A=1 as SGLang runs it: K2i's logits into SGLang's top-k and candidate kernels, and a consumer's top-k on
    K2i's compact candidate logits -> per-row Jaccard vs SGLang's own path (median 1.0, p1 >= 0.99)."""
    from types import SimpleNamespace

    from sglang.srt.models.deepseek_common.amd.mono import indexer_hook as hook

    hook._MODE[:] = [1]
    hook._RUNNERS.clear()
    case = fp4_case(ctxs, seed=21 + max(ctxs))
    rows = case["q"].shape[0]
    backend, meta = _fake_backend(case, max(ctxs))
    backend.forward_metadata.k2i_planned = hook.step_plans(
        backend, {1: meta}, SimpleNamespace(batch_size=len(ctxs))
    )
    assert backend.forward_metadata.k2i_planned == {1: (rows, 6)}
    layer = SimpleNamespace(layer_id=0)
    logits = hook.full_logits(
        backend, layer, case["q"], case["qscale"], case["w"], meta, 1
    )
    got_raw, _, got_cands = sglang_select(case, logits)
    want_raw, _, want_cands = sglang_select(case, sglang_logits(case))
    j_full, j_blk = _sets(got_raw, want_raw), _sets(got_cands.ids, want_cands.ids)

    g = torch.Generator(device="cuda").manual_seed(9)
    case2 = dict(case, q=torch.randint(0, 256, case["q"].shape, generator=g, device="cuda", dtype=torch.uint8),
                 w=(torch.randn(rows, 32, generator=g, device="cuda") * 0.05).to(torch.bfloat16))  # fmt: skip
    raw = torch.full((rows, 512), -9, dtype=torch.int32, device="cuda")
    page = torch.full_like(raw, -9)
    assert hook.reindex_select(
        backend,
        layer,
        case2["q"],
        case2["qscale"],
        case2["w"],
        meta,
        1,
        got_cands,
        page,
        raw,
    )
    want_raw2, want_page2, _ = sglang_select(
        case2, sglang_logits(case2), two_level=got_cands
    )
    j_re = _sets(raw, want_raw2)
    for t in range(rows):
        b = int(case["bounds"][t])
        ids = raw[t][raw[t] >= 0].long()
        n = ids.numel()
        assert n == min(b, 512) and (raw[t][n:] == -1).all() and ids.max() < b, (
            t,
            n,
            b,
        )
        assert torch.equal(
            page[t][:n].long(), case["table"][t][ids // 64].long() * 64 + ids % 64
        ), t
    q = lambda js: (float(np.median(js)), float(np.percentile(js, 1)))  # noqa: E731
    print(
        f"\n{ctxs}: hybrid jaccard full {q(j_full)} blocks {q(j_blk)} reindex {q(j_re)}"
    )
    for js in (j_full, j_blk, j_re):
        med, p1 = q(js)
        assert med == 1.0 and p1 >= 0.99, (med, p1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", *sys.argv[1:]]))
