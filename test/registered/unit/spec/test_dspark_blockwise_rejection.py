"""The block-wise rejection sampler must draw
(accept length, bonus) from the exact chain speculative-sampling law, like the classic
kernel; its draws must be a function of (key, counter) only, so TP ranks and CUDA graph
replays agree; and the accept must route to it only when the noise is given."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.speculative.dspark import blockwise_rejection as bwr
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _info(temps, **overrides):
    fields = dict(
        temperatures=temps,
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _case(*, stride, vocab, device, seed=0, scale=2.0):
    gen = torch.Generator(device=device).manual_seed(seed)
    target = torch.randn(stride, vocab, device=device, generator=gen) * scale
    draft = torch.randn(stride - 1, vocab, device=device, generator=gen) * scale
    # Drafts drawn from the draft law, so some get accepted and some rejected.
    tokens = torch.multinomial(torch.softmax(draft, -1), 1, generator=gen).view(-1)
    root = torch.zeros(1, dtype=torch.int64, device=device)
    return target, draft, torch.cat([root, tokens])


def _expected_law(target, draft, candidates, t_temp, d_temp, one_hot):
    """Exact pmf over (accept length k, bonus token x), shape [gamma + 1, vocab]."""
    gamma, vocab = draft.shape
    p = torch.softmax(target.double() / t_temp, -1)
    if one_hot:
        q = torch.nn.functional.one_hot(candidates[1:], vocab).double()
    else:
        q = torch.softmax(draft.double() / d_temp, -1)
    law = torch.zeros(gamma + 1, vocab, dtype=torch.float64, device=target.device)
    prefix = 1.0
    for i in range(gamma):
        d = candidates[i + 1]
        a = min(1.0, (p[i, d] / q[i, d]).item())
        residual = (p[i] - q[i]).clamp_min(0)
        law[i] = prefix * (1 - a) * residual / residual.sum().clamp_min(1e-300)
        prefix *= a
    law[gamma] = prefix * p[gamma]
    return law


def _chi_square_ok(counts, law, n):
    expected = law.flatten() * n
    keep = expected >= 5
    observed = counts.flatten().double()
    stat = (((observed - expected) ** 2) / expected)[keep].sum().item()
    tail = observed[~keep].sum().item() - expected[~keep].sum().item()
    df = int(keep.sum().item()) - 1
    # About 6 sigma of the chi-square law; the pooled sparse tail within 6 Poisson sigma.
    bound = df + 6 * (2 * df) ** 0.5
    sparse = expected[~keep].sum().item()
    return stat < bound and abs(tail) < 6 * max(sparse, 1.0) ** 0.5 + 5, (stat, bound)


def _tile(target, draft, candidates, rows):
    stride, vocab = target.shape
    return (
        candidates.unsqueeze(0).repeat(rows, 1),
        target.unsqueeze(0).repeat(rows, 1, 1).view(rows * stride, vocab),
        draft.unsqueeze(0).repeat(rows, 1, 1),
    )


def _blockwise_counts(
    target, draft, candidates, *, rows, calls, t_temp, d_temp, one_hot
):
    stride, vocab = target.shape
    device = target.device
    cand, tgt, drf = _tile(target, draft, candidates, rows)
    noise = bwr.make_rejection_noise(1234, device)
    counts = torch.zeros(stride * vocab, dtype=torch.int64, device=device)
    temps_t = torch.full((rows,), t_temp, device=device)
    temps_d = torch.full((rows,), d_temp, device=device)
    for _ in range(calls):
        k, x, _ = bwr.BlockwiseRejection.execute(
            candidates=cand,
            target_logits=tgt,
            draft_logits=None if one_hot else drf,
            target_temperatures=temps_t,
            draft_temperatures=temps_d,
            noise=noise,
        )
        counts += torch.bincount(k.long() * vocab + x, minlength=stride * vocab)
    return counts.view(stride, vocab)


@needs_cuda
@pytest.mark.parametrize(
    "t_temp,d_temp,one_hot,seed",
    [(0.7, 1.3, False, 1), (1.0, 1.0, True, 2)],
)
def test_law_matches_exact(t_temp, d_temp, one_hot, seed):
    stride, vocab, rows, calls = 4, 40, 8192, 64
    target, draft, cand = _case(stride=stride, vocab=vocab, device="cuda", seed=seed)
    law = _expected_law(target, draft, cand, t_temp, d_temp, one_hot)
    n = rows * calls
    kw = dict(rows=rows, calls=calls, t_temp=t_temp, d_temp=d_temp, one_hot=one_hot)
    new = _blockwise_counts(target, draft, cand, **kw)
    ok, detail = _chi_square_ok(new, law, n)
    assert ok, detail


@needs_cuda
def test_graph_replay_equals_eager_and_advances_counter():
    stride, vocab, rows = 6, 129280, 4
    gen = torch.Generator(device="cuda").manual_seed(4)
    target = torch.randn(rows * stride, vocab, device="cuda", generator=gen) * 4
    draft = target.view(rows, stride, vocab)[:, :-1].clone()
    cand = torch.zeros(rows, stride, dtype=torch.int64, device="cuda")
    cand[:, 1:] = draft.argmax(-1)
    temps = torch.full((rows,), 0.8, device="cuda")
    noise = bwr.make_rejection_noise(11, "cuda")

    def run():
        return bwr.BlockwiseRejection.execute(
            candidates=cand,
            target_logits=target,
            draft_logits=draft,
            target_temperatures=temps,
            draft_temperatures=temps,
            noise=noise,
        )

    run()  # compile
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static = run()
    for step in range(3):
        noise.counter.fill_(100 + step)
        ref = [t.clone() for t in run()]
        noise.counter.fill_(100 + step)
        graph.replay()
        torch.cuda.synchronize()
        assert noise.counter.item() == 101 + step
        for x, y in zip(static, ref):
            assert torch.equal(x, y), step


@needs_cuda
def test_cutoff_caps_length_and_bonus_is_the_next_draft():
    stride, vocab, rows = 6, 512, 64
    target = torch.zeros(rows * stride, vocab, device="cuda")
    cand = torch.randint(0, vocab, (rows, stride), device="cuda")
    # Identical draft and target laws: every draft is accepted.
    draft = target.view(rows, stride, vocab)[:, :-1].clone()
    verify_lens = torch.randint(
        1, stride + 1, (rows,), device="cuda", dtype=torch.int32
    )
    k, x, trim = bwr.BlockwiseRejection.execute(
        candidates=cand,
        target_logits=target,
        draft_logits=draft,
        target_temperatures=torch.ones(rows, device="cuda"),
        draft_temperatures=torch.ones(rows, device="cuda"),
        noise=bwr.make_rejection_noise(0, "cuda"),
        cutoff_verify_lens=verify_lens,
    )
    expect_len = torch.minimum(
        torch.full_like(verify_lens, stride - 1), verify_lens - 1
    )
    assert torch.equal(k, expect_len)
    assert torch.equal(trim, stride - 1 - expect_len)
    capped = k < stride - 1
    nxt = cand.gather(1, (k.long() + 1).clamp_max(stride - 1).unsqueeze(1)).squeeze(1)
    assert torch.equal(x[capped], nxt[capped])


def test_routes_to_blockwise_only_with_noise(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dspark_verify.BlockwiseRejection,
        "execute",
        classmethod(lambda cls, **kw: calls.append("blockwise") or (None, None, None)),
    )
    monkeypatch.setattr(
        dspark_verify.AcceptSampling,
        "execute",
        classmethod(lambda cls, **kw: calls.append("classic") or (None, None, None)),
    )
    monkeypatch.setattr(dspark_verify, "_draft_probs", lambda *a: None)
    noise = bwr.make_rejection_noise(0, "cpu")
    block = SimpleNamespace(
        greedy_draft=False, corrected_logits=None, temperatures=None, greedy_mask=None
    )

    def accept(info, **kw):
        dspark_verify.accept_draft_tokens(
            candidates=torch.zeros(1, 3, dtype=torch.int64),
            target_logits=torch.zeros(3, 8),
            draft_block=block,
            sampling_info=info,
            draft_input=None,
            gamma=2,
            verify_num_draft_tokens=3,
            **kw,
        )
        return calls.pop()

    temps = torch.ones(1, 1)
    assert accept(_info(temps)) == "classic"
    assert accept(_info(temps), rejection_noise=noise) == "blockwise"
    assert (
        accept(_info(temps, need_top_p_sampling=True), rejection_noise=noise)
        == "classic"
    )
    assert (
        accept(_info(temps, need_top_k_sampling=True), rejection_noise=noise)
        == "classic"
    )
    assert (
        accept(_info(temps), rejection_noise=noise, block_verification=True)
        == "classic"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
