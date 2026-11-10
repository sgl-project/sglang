"""Multi-CTA verify tree sampling: bitwise to the stock tree sampler.

The equality holds against the FlashInfer build whose sampling primitives the kernels reproduce; a failure after a
FlashInfer move means the kernels must be re-derived, not the tolerance widened.
"""

import random

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="checked on SM10x",
)

VOCAB = 248320
TOKENS = 4
ONE_MINUS = 1.0 - 2.0**-24
DISTS = (
    "lm",
    "lm_temp",
    "peaked",
    "narrow",
    "wide",
    "masked",
    "one_finite",
    "const",
    "subnormal",
    "huge",
)


def _logits(name, rows, g, vocab=VOCAB):
    x = torch.randn(rows, vocab, device="cuda", generator=g)
    t = torch.ones(rows, 1, device="cuda")

    def lm(y):
        y = y * 2.0 - 3.0
        base = torch.randperm(vocab, device="cuda", generator=g)[:64]
        idx = (
            base[None, :]
            + torch.randint(0, vocab, (rows, 1), device="cuda", generator=g)
        ) % vocab
        boost = torch.rand(rows, 64, device="cuda", generator=g) * 15 + 3
        return y.scatter(1, idx, y.gather(1, idx) + boost)

    def one_per_row(y, value):
        return y.scatter(
            1, torch.randint(0, vocab, (rows, 1), device="cuda", generator=g), value
        )

    if name == "lm":
        x = lm(x)
    elif name == "lm_temp":
        x = lm(x)
        t = torch.rand(rows, 1, device="cuda", generator=g) * 1.2 + 0.3
    elif name == "peaked":
        x = one_per_row(x, 40.0)
    elif name == "narrow":
        x = x * 1e-3
    elif name == "wide":
        x = x * 40.0
    elif name == "masked":
        x = lm(x).masked_fill(
            torch.rand(rows, vocab, device="cuda", generator=g) < 0.3, float("-inf")
        )
    elif name == "one_finite":
        x = one_per_row(torch.full((rows, vocab), float("-inf"), device="cuda"), 3.0)
    elif name == "const":
        x = torch.zeros(rows, vocab, device="cuda")
    elif name == "subnormal":
        x = one_per_row(
            -80.0 - 24.0 * torch.rand(rows, vocab, device="cuda", generator=g), 0.0
        )
    elif name == "huge":
        x = x * 1e4
    return x.contiguous(), t


def _bits(t):
    return t.contiguous().view(torch.int32) if t.dtype == torch.float32 else t


@pytest.fixture(scope="module")
def mcta():
    from sglang.kernels.ops.speculative import multi_cta_verify_sampling

    return multi_cta_verify_sampling


def _chain(bs, n):
    nxt = torch.arange(1, n + 1, device="cuda")
    nxt[-1] = -1
    return (
        torch.arange(bs * n, device="cuda").view(bs, n),
        nxt.expand(bs, n).contiguous(),
        torch.full((bs, n), -1, device="cuda", dtype=torch.int64),
        [[-1] + list(range(n - 1))] * bs,
    )


def _tree(bs, n, depth, rng):
    """Random trees in SGLang's retrieve layout (first child / next sibling)."""
    nxt = torch.full((bs, n), -1, dtype=torch.int64)
    sib = torch.full((bs, n), -1, dtype=torch.int64)
    parents = []
    for b in range(bs):
        d, last_child, par = [0] * n, {}, [-1] * n
        for i in range(1, n):
            p = rng.choice([j for j in range(i) if d[j] < depth])
            par[i], d[i] = p, d[p] + 1
            if p in last_child:
                sib[b, last_child[p]] = i
            else:
                nxt[b, p] = i
            last_child[p] = i
        parents.append(par)
    return torch.arange(bs * n).view(bs, n).cuda(), nxt.cuda(), sib.cuda(), parents


def _candidates(tp, parents, kind, g):
    bs, n, vocab = tp.shape
    cand = torch.randint(0, vocab, (bs, n), device="cuda", generator=g)
    for b in range(bs):
        for i in range(1, n):
            row = tp[b, parents[b][i]]
            if kind == "argmax":
                cand[b, i] = row.argmax()
            elif kind == "sample":
                cand[b, i] = torch.multinomial(row, 1, generator=g)[0]
            elif kind == "zero":
                z = (row == 0).nonzero()
                if z.numel():
                    cand[b, i] = z[0, 0]
    return cand


def _run_tree(fn, cand, idx, nxt, sib, coins, fcoins, tp, spec, ts=1.0, ta=1.0):
    bs, n = cand.shape
    out = dict(
        predicts=torch.full((bs * n,), -7, dtype=torch.int32, device="cuda"),
        accept_index=torch.full((bs, spec), -7, dtype=torch.int32, device="cuda"),
        accept_token_num=torch.full((bs,), -7, dtype=torch.int32, device="cuda"),
        # An input the sampler reads (relu(q - p)); served as zeros. Its writes are still compared.
        draft_probs=torch.zeros_like(tp),
    )
    fn(
        candidates=cand,
        retrive_index=idx,
        retrive_next_token=nxt,
        retrive_next_sibling=sib,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=fcoins,
        target_probs=tp,
        threshold_single=ts,
        threshold_acc=ta,
        deterministic=True,
        **out,
    )
    return out


def test_tree_sampling_bitwise(mcta):
    from sglang.kernels.ops.speculative.sampling import (
        tree_speculative_sampling_target_only as stock_tree,
    )

    g = torch.Generator(device="cuda").manual_seed(263)
    rng = random.Random(263)
    cases = differing = 0

    def check(*args, **kwargs):
        nonlocal cases, differing
        a = _run_tree(stock_tree, *args, **kwargs)
        b = _run_tree(mcta.tree_speculative_sampling_target_only, *args, **kwargs)
        cases += 1
        differing += sum(int((_bits(a[k]) != _bits(b[k])).sum()) for k in a)
        return b

    for bs in (1, 2, 3, 4, 8, 16):
        for kind in ("argmax", "sample", "random", "zero"):
            for coin_mode in ("uniform", "zero", "one"):
                for dist in ("lm", "masked", "narrow"):
                    logits, _ = _logits(dist, bs * TOKENS, g)
                    tp = torch.softmax(logits, -1).view(bs, TOKENS, VOCAB)
                    idx, nxt, sib, parents = _chain(bs, TOKENS)
                    cand = _candidates(tp, parents, kind, g)
                    if coin_mode == "uniform":
                        coins = torch.rand(bs, TOKENS, device="cuda", generator=g)
                        fcoins = torch.rand(bs, device="cuda", generator=g)
                    else:
                        value = 0.0 if coin_mode == "zero" else ONE_MINUS
                        coins = torch.full((bs, TOKENS), value, device="cuda")
                        fcoins = torch.full((bs,), value, device="cuda")
                    check(cand, idx, nxt, sib, coins, fcoins, tp, TOKENS)
    for bs in (1, 4, 16):
        for ts, ta in ((0.5, 0.9), (0.3, 0.3)):
            logits, _ = _logits("lm", bs * TOKENS, g)
            tp = torch.softmax(logits, -1).view(bs, TOKENS, VOCAB)
            idx, nxt, sib, parents = _chain(bs, TOKENS)
            cand = _candidates(tp, parents, "sample", g)
            coins = torch.rand(bs, TOKENS, device="cuda", generator=g)
            fcoins = torch.rand(bs, device="cuda", generator=g)
            check(cand, idx, nxt, sib, coins, fcoins, tp, TOKENS, ts=ts, ta=ta)
    for bs in (1, 2, 4, 8, 16):
        for rep in range(4):
            n = 8
            logits, _ = _logits("lm" if rep % 2 else "masked", bs * n, g)
            tp = torch.softmax(logits, -1).view(bs, n, VOCAB)
            idx, nxt, sib, parents = _tree(bs, n, 3, rng)
            cand = _candidates(tp, parents, "sample", g)
            coins = torch.rand(bs, n, device="cuda", generator=g)
            check(
                cand,
                idx,
                nxt,
                sib,
                coins,
                torch.rand(bs, device="cuda", generator=g),
                tp,
                4,
            )
    # The final sampling's chunk search on a grid of final coins.
    for dist in ("narrow", "lm", "wide"):
        bs = 16
        logits, _ = _logits(dist, bs * TOKENS, g)
        tp = torch.softmax(logits, -1).view(bs, TOKENS, VOCAB)
        idx, nxt, sib, parents = _chain(bs, TOKENS)
        cand = _candidates(tp, parents, "random", g)
        coins = torch.full((bs, TOKENS), ONE_MINUS, device="cuda")
        for k in range(32):
            fcoins = (
                torch.arange(bs, device="cuda", dtype=torch.float64) * 32 + k
            ) / 512.0
            check(
                cand,
                idx,
                nxt,
                sib,
                coins,
                torch.clamp(fcoins.float(), max=ONE_MINUS),
                tp,
                TOKENS,
            )
    assert cases > 300
    assert differing == 0


def test_graph_replay_equals_stock_eager(mcta):
    from sglang.kernels.ops.speculative.sampling import (
        tree_speculative_sampling_target_only as stock_tree,
    )

    g = torch.Generator(device="cuda").manual_seed(264)
    for bs in (1, 4, 16):
        rows = bs * TOKENS
        logits = torch.zeros(rows, VOCAB, device="cuda")
        coins = torch.zeros(bs, TOKENS, device="cuda")
        fcoins = torch.zeros(bs, device="cuda")
        cand = torch.zeros(bs, TOKENS, dtype=torch.int64, device="cuda")
        idx, nxt, sib, parents = _chain(bs, TOKENS)
        state = {}

        def body():
            tp = torch.softmax(logits, dim=-1).view(bs, TOKENS, VOCAB)
            state["tp"] = tp
            state["out"] = _run_tree(
                mcta.tree_speculative_sampling_target_only,
                cand,
                idx,
                nxt,
                sib,
                coins,
                fcoins,
                tp,
                TOKENS,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        for _ in range(6):
            fresh, _ = _logits("lm", rows, g)
            logits.copy_(fresh)
            tp_ref = torch.softmax(logits, dim=-1).view(bs, TOKENS, VOCAB)
            cand.copy_(_candidates(tp_ref, parents, "sample", g))
            coins.copy_(torch.rand(bs, TOKENS, device="cuda", generator=g))
            fcoins.copy_(torch.rand(bs, device="cuda", generator=g))
            graph.replay()
            ref = _run_tree(
                stock_tree, cand, idx, nxt, sib, coins, fcoins, tp_ref, TOKENS
            )
            assert torch.equal(_bits(state["tp"]), _bits(tp_ref))
            for k in ref:
                assert torch.equal(_bits(ref[k]), _bits(state["out"][k])), (bs, k)
        del graph

    flipped = state["tp"].clone()
    flipped.view(-1).view(torch.int32)[11] ^= 1
    assert int((_bits(state["tp"]) != _bits(flipped)).sum()) == 1
