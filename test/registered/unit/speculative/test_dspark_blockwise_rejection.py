"""The block-wise rejection sampler (--enable-dsv41-low-latency-decode) must draw
(accept length, bonus) from the exact chain speculative-sampling law, like the classic
kernel; its draws must be a function of (key, counter) only, so TP ranks and CUDA graph
replays agree without the accept-site broadcast; and the accept must route to it only
when the noise is given."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dsv41_low_latency.dspark import blockwise_rejection as bwr
from sglang.srt.models.dsv41_low_latency.dspark import sampling_accept_graph as sag
from sglang.srt.models.dsv41_low_latency.dspark.static_epilogue import (
    StaticEpilogue,
    StaticVerifyIO,
)
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.spec_tp_sync import (
    SpecTpSync,
    SpecTpSyncSite,
    parse_spec_tp_sync,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _tp_sync():
    sync = SpecTpSync.__new__(SpecTpSync)
    sync._sites = frozenset()
    return sync


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


def _classic_counts(target, draft, candidates, *, rows, calls, t_temp, d_temp, one_hot):
    stride, vocab = target.shape
    device = target.device
    cand, tgt, drf = _tile(target, draft, candidates, rows)
    counts = torch.zeros(stride * vocab, dtype=torch.int64, device=device)
    block = DraftBlockResult(
        draft_tokens=cand[:, 1:],
        corrected_logits=None if one_hot else drf,
        greedy_mask=torch.zeros(rows, dtype=torch.bool, device=device),
        temperatures=torch.full((rows,), d_temp, device=device),
        greedy_draft=one_hot,
    )
    info = _info(torch.full((rows, 1), t_temp, device=device))
    for _ in range(calls):
        k, x, _ = dspark_verify.accept_draft_tokens(
            candidates=cand,
            target_logits=tgt,
            draft_block=block,
            sampling_info=info,
            draft_input=None,
            gamma=stride - 1,
            verify_num_draft_tokens=stride,
        )
        counts += torch.bincount(k.long() * vocab + x, minlength=stride * vocab)
    return counts.view(stride, vocab)


@needs_cuda
@pytest.mark.parametrize(
    "t_temp,d_temp,one_hot,seed",
    [
        (1.0, 1.0, False, 0),
        (0.7, 1.3, False, 1),
        (1.0, 1.0, True, 2),
        (1.0, 1.0, False, 3),
    ],
)
def test_law_matches_exact_and_classic(t_temp, d_temp, one_hot, seed):
    stride, vocab, rows, calls = 4, 40, 8192, 64
    target, draft, cand = _case(stride=stride, vocab=vocab, device="cuda", seed=seed)
    law = _expected_law(target, draft, cand, t_temp, d_temp, one_hot)
    n = rows * calls
    kw = dict(rows=rows, calls=calls, t_temp=t_temp, d_temp=d_temp, one_hot=one_hot)
    new = _blockwise_counts(target, draft, cand, **kw)
    ok, detail = _chi_square_ok(new, law, n)
    assert ok, ("blockwise", detail)
    old = _classic_counts(target, draft, cand, **kw)
    ok, detail = _chi_square_ok(old, law, n)
    assert ok, ("classic", detail)
    # Power check: the target law alone as the bonus law must be rejected.
    wrong = law.clone()
    wrong[:-1] = law[:-1].sum(-1, keepdim=True) * torch.softmax(
        target[:-1].double() / t_temp, -1
    )
    assert not _chi_square_ok(new, wrong, n)[0]


@needs_cuda
def test_first_token_is_target_distributed():
    """Drafts drawn from q: the first emitted token must follow p (unbiasedness)."""
    stride, vocab, rows = 3, 32, 1 << 16
    gen = torch.Generator(device="cuda").manual_seed(5)
    target = torch.randn(stride, vocab, device="cuda", generator=gen) * 2
    draft = torch.randn(stride - 1, vocab, device="cuda", generator=gen) * 2
    q = torch.softmax(draft, -1)
    noise = bwr.make_rejection_noise(7, "cuda")
    counts = torch.zeros(vocab, dtype=torch.int64, device="cuda")
    ones = torch.ones(rows, device="cuda")
    for _ in range(16):
        drafts = torch.stack(
            [
                torch.multinomial(q[i], rows, replacement=True, generator=gen)
                for i in range(stride - 1)
            ],
            dim=1,
        )
        cand = torch.cat(
            [torch.zeros(rows, 1, dtype=torch.int64, device="cuda"), drafts], 1
        )
        k, x, _ = bwr.BlockwiseRejection.execute(
            candidates=cand,
            target_logits=target.repeat(rows, 1),
            draft_logits=draft.unsqueeze(0).expand(rows, -1, -1).contiguous(),
            target_temperatures=ones,
            draft_temperatures=ones,
            noise=noise,
        )
        first = torch.where(k > 0, cand[:, 1], x)
        counts += torch.bincount(first, minlength=vocab)
    law = torch.softmax(target[0].double(), -1)
    ok, detail = _chi_square_ok(counts, law, counts.sum().item())
    assert ok, detail


@needs_cuda
def test_draws_depend_only_on_key_and_counter():
    stride, vocab, rows = 6, 129280, 8
    gen = torch.Generator(device="cuda").manual_seed(9)
    target = torch.randn(rows * stride, vocab, device="cuda", generator=gen)
    draft = torch.randn(rows, stride - 1, vocab, device="cuda", generator=gen)
    cand = torch.randint(0, vocab, (rows, stride), device="cuda", generator=gen)
    cand[:, 1:] = target.view(rows, stride, vocab)[:, :-1].argmax(-1)
    temps = torch.ones(rows, device="cuda")

    def run(noise):
        return bwr.BlockwiseRejection.execute(
            candidates=cand,
            target_logits=target,
            draft_logits=draft,
            target_temperatures=temps,
            draft_temperatures=temps,
            noise=noise,
        )

    a, b = bwr.make_rejection_noise(3, "cuda"), bwr.make_rejection_noise(3, "cuda")
    first = run(a)
    torch.cuda.manual_seed(12345)  # torch's generator plays no part
    second = run(b)
    for x, y in zip(first, second):
        assert torch.equal(x, y)
    assert a.counter.item() == 1
    outs = [run(a)[1] for _ in range(8)]
    assert len({tuple(o.tolist()) for o in outs}) > 1, (
        "the counter must change the draws"
    )
    assert ((first[1] >= 0) & (first[1] < vocab)).all()
    assert (first[0] <= stride - 1).all() and (first[2] == 0).all()


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
        bwr.BlockwiseRejection,
        "execute",
        classmethod(lambda cls, **kw: calls.append("blockwise") or (None, None, None)),
    )
    monkeypatch.setattr(
        dspark_verify.AcceptSampling,
        "execute",
        classmethod(lambda cls, **kw: calls.append("classic") or (None, None, None)),
    )
    noise = bwr.make_rejection_noise(0, "cpu")
    block = SimpleNamespace(
        greedy_draft=True,
        draft_tokens=torch.zeros(1, 2, dtype=torch.int64),
        corrected_logits=None,
        temperatures=torch.ones(1),
        greedy_mask=None,
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


def test_blockwise_accept_needs_no_sample_site_broadcast(monkeypatch):
    """With the block-wise sampler the eager accept syncs at the greedy site, so the
    SGLANG_SPEC_TP_SYNC=rng preset no longer broadcasts it; without the noise, and
    with top-k / top-p, it keeps the sample site."""
    monkeypatch.setattr(
        dspark_verify, "accept_draft_tokens", lambda **kw: (None, None, None)
    )
    monkeypatch.setattr(dspark_verify, "build_accept_outs", lambda **kw: kw)
    executor = dspark_verify.TargetVerifyExecutor(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(hf_text_config=None)
            )
        ),
        gamma=2,
        verify_num_draft_tokens=3,
        model_runner=None,
        kv_injector=None,
        tp_sync=_tp_sync(),
    )
    # The stock executor has no noise until the switch's install sets it.
    assert executor.rejection_noise is None

    def site(info, noise):
        executor.rejection_noise = noise
        return executor.accept_and_finalize(
            folded_accept=False,
            bs=1,
            verify_ids_2d=None,
            target_logits=None,
            draft_block=None,
            sampling_info=info,
            draft_input=None,
            layout=None,
            prefix_lens=None,
            draft_tokens=None,
        )["sync_site"]

    noise = bwr.make_rejection_noise(0, "cpu")
    temps = torch.ones(1, 1)
    greedy = SpecTpSyncSite.DSPARK_ACCEPT_GREEDY
    sample = SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE
    assert site(_info(temps), noise) == greedy
    assert site(_info(temps, is_any_greedy=True), noise) == greedy
    assert site(_info(temps), None) == sample
    assert site(_info(temps, need_top_p_sampling=True), noise) == sample
    assert site(_info(temps, need_top_k_sampling=True), noise) == sample
    rng = parse_spec_tp_sync("rng")
    assert greedy not in rng and sample in rng


def test_torch_reference_law_on_cpu():
    stride, vocab, rows, calls = 3, 12, 4096, 16
    target, draft, cand = _case(stride=stride, vocab=vocab, device="cpu", seed=21)
    law = _expected_law(target, draft, cand, 1.0, 1.0, False)
    counts = _blockwise_counts(
        target,
        draft,
        cand,
        rows=rows,
        calls=calls,
        t_temp=1.0,
        d_temp=1.0,
        one_hot=False,
    )
    ok, detail = _chi_square_ok(counts, law, rows * calls)
    assert ok, detail


# ---- The in-graph sampling accept (sampling_accept_graph.py) ----


@contextlib.contextmanager
def _single_rank_capture():
    yield SimpleNamespace(stream=torch.cuda.Stream())


def _static(epi):
    """The switch's static epilogue on ``epi``, without Engram or side stream."""
    static = StaticEpilogue(
        epilogue=epi,
        tp_sync=_tp_sync(),
        fused_argmax=False,
        resolve_engram_history=lambda: None,
        context_kv_side_stream=None,
    )
    epi.static_epilogue_override = static
    return static


def _region_case(bs, vocab=4096, seed=0):
    stride = 6
    gen = torch.Generator(device="cuda").manual_seed(seed)
    target = torch.randn(bs * stride, vocab, device="cuda", generator=gen) * 3
    step_logits = torch.randn(8, stride - 1, vocab, device="cuda", generator=gen) * 3
    candidates = torch.randint(0, vocab, (bs, stride), device="cuda", generator=gen)
    io = StaticVerifyIO(
        candidates=candidates,
        logits=target,
        hidden=torch.zeros(bs * stride, 8, device="cuda"),
        seq_lens=torch.randint(100, 200, (bs,), device="cuda", generator=gen),
        req_pool_indices=torch.arange(bs, device="cuda"),
        out_cache_loc=torch.zeros(bs * stride, dtype=torch.int64, device="cuda"),
        positions=torch.zeros(bs * stride, dtype=torch.int64, device="cuda"),
    )
    sampler = SimpleNamespace(
        greedy_draft=False,
        step_logits=step_logits,
        params=SimpleNamespace(
            temperatures=torch.rand(8, device="cuda", generator=gen) + 0.5,
            greedy_mask=torch.zeros(8, dtype=torch.bool, device="cuda"),
        ),
    )
    epi = dspark_verify.DsparkVerifyEpilogue(
        max_bs=8, verify_num_draft_tokens=stride, device="cuda", tp_sync=_tp_sync()
    )
    _static(epi).static_io[bs] = io
    epi.begin_static_step(bs, armed=False)
    return io, sampler, epi


@needs_cuda
@pytest.mark.parametrize("bs", [1, 3, 8])
def test_region_replay_equals_eager_at_same_counter(monkeypatch, bs):
    io, sampler, epi = _region_case(bs, seed=bs)
    noise = bwr.make_rejection_noise(5, "cuda")
    monkeypatch.setattr(sag, "graph_capture", _single_rank_capture)
    graph = sag.DsparkSamplingAcceptGraph(
        epilogue=epi,
        static_epilogue=epi.static_epilogue_override,
        resolve_draft_sampler=lambda: sampler,
        tp_sync=_tp_sync(),
        fused_argmax=False,
        block_verification=False,
        device="cuda",
        rejection_noise=noise,
    )
    eager_noise = bwr.make_rejection_noise(5, "cuda")
    for step in range(4):  # eager warm-up, capture + replay, then replays
        eager_noise.counter.copy_(noise.counter)
        block = DraftBlockResult(
            draft_tokens=io.candidates[:, 1:],
            corrected_logits=sampler.step_logits[:bs],
            greedy_mask=sampler.params.greedy_mask[:bs],
            temperatures=sampler.params.temperatures[:bs],
            greedy_draft=False,
        )
        k_ref, x_ref, _ = dspark_verify.accept_draft_tokens(
            candidates=io.candidates,
            target_logits=io.logits,
            draft_block=block,
            sampling_info=sag.GraphSamplingInfo(
                temperatures=sampler.params.temperatures[:bs]
            ),
            draft_input=None,
            gamma=5,
            verify_num_draft_tokens=6,
            rejection_noise=eager_noise,
        )
        out = graph.run(bs=bs, target_logits=io.logits)
        torch.cuda.synchronize()
        assert torch.equal(out.correct_len, k_ref.to(out.correct_len.dtype)), step
        assert torch.equal(out.bonus, x_ref.to(out.bonus.dtype)), step
        assert torch.equal(noise.counter, eager_noise.counter), step
    assert graph.num_replays == 3


def _num_compiled(kernel, device):
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        pytest.skip("this Triton has no per-device kernel cache to inspect")
    return len(caches[device][0])


@needs_cuda
def test_precompile_covers_served_blockwise_accepts():
    io, sampler, _ = _region_case(8, vocab=3000, seed=5)
    noise = bwr.make_rejection_noise(0, "cuda")
    sag.precompile_sampling_accept(
        capture_bs=[1, 2, 3, 8],
        stride=6,
        target_logits=io.logits,
        draft_sampler=sampler,
        fused_argmax=False,
        block_verification=False,
        rejection_noise=noise,
    )
    dev = torch.cuda.current_device()
    kernels = (
        bwr._stats_kernel,
        bwr._rejection_kernel,
        bwr._resample_kernel,
        bwr._insert_kernel,
    )
    before = [_num_compiled(k, dev) for k in kernels]
    for bs in (1, 3):
        block = DraftBlockResult(
            draft_tokens=io.candidates[:bs, 1:],
            corrected_logits=sampler.step_logits[:bs],
            greedy_mask=sampler.params.greedy_mask[:bs],
            temperatures=sampler.params.temperatures[:bs],
            greedy_draft=False,
        )
        dspark_verify.accept_draft_tokens(
            candidates=io.candidates[:bs],
            target_logits=io.logits[: bs * 6],
            draft_block=block,
            sampling_info=sag.GraphSamplingInfo(
                temperatures=sampler.params.temperatures[:bs].view(bs, 1)
            ),
            draft_input=None,
            gamma=5,
            verify_num_draft_tokens=6,
            rejection_noise=noise,
        )
    torch.cuda.synchronize()
    assert [_num_compiled(k, dev) for k in kernels] == before
