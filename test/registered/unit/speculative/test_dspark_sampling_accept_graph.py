"""--enable-dsv41-low-latency-decode: the captured sampling accept must equal the
eager one at the same generator state, and greedy / ineligible steps never route to it."""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

import contextlib
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dsv41_low_latency.dspark import sampling_accept_graph as sag
from sglang.srt.models.dsv41_low_latency.dspark.static_epilogue import (
    StaticEpilogue,
    StaticVerifyIO,
)
from sglang.srt.models.dsv41_low_latency.dspark.worker import DsparkLowLatency
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.spec_tp_sync import SpecTpSync

STRIDE = 6
GAMMA = STRIDE - 1


def _tp_sync():
    sync = SpecTpSync.__new__(SpecTpSync)
    sync._sites = frozenset()
    return sync


def _sampling_info(**overrides):
    fields = dict(
        is_any_greedy=False,
        is_all_greedy=False,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        has_custom_logit_processor=False,
        acc_linear_penalties=None,
        penalizer_orchestrator=None,
        grammar_mask=None,
        logit_bias=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _low_latency(enabled=True):
    graph = SimpleNamespace(run=lambda bs, target_logits: ("accept", bs))
    return DsparkLowLatency(
        static_epilogue=None,
        sampling_accept_graph=graph if enabled else None,
        resolve_draft_sampler=lambda: None,
        rejection_noise=None,
        capture_bs=[1],
        stride=STRIDE,
        block_verification=False,
    )


def _eligible(
    low_latency, info, *, folded=True, mode="static", grammar=False, sim_bonus=None
):
    out = low_latency.sampling_accept(
        bs=1,
        target_logits=None,
        proposal_folded=folded,
        planner_mode=mode,
        sampling_info=info,
        simulate_bonus_sampling_info=sim_bonus,
        has_grammar=grammar,
    )
    return out is not None


def test_routes_only_all_sampled_static_steps():
    low_latency = _low_latency()
    assert _eligible(low_latency, _sampling_info())
    # Greedy and mixed batches keep the greedy fold / the eager mixed accept.
    assert not _eligible(low_latency, None)
    assert not _eligible(low_latency, _sampling_info(is_any_greedy=True))
    assert not _eligible(low_latency, _sampling_info(need_top_k_sampling=True))
    assert not _eligible(low_latency, _sampling_info(need_top_p_sampling=True))
    assert not _eligible(low_latency, _sampling_info(logit_bias=torch.zeros(1)))
    assert not _eligible(low_latency, _sampling_info(), folded=False)
    assert not _eligible(low_latency, _sampling_info(), grammar=True)
    assert not _eligible(low_latency, _sampling_info(), sim_bonus=_sampling_info())
    assert not _eligible(_low_latency(enabled=False), _sampling_info())
    assert not _eligible(low_latency, _sampling_info(), mode="compact")


def _graph_on_cpu(static_bs=(1, 2, 4, 8)):
    epi = SimpleNamespace(
        static_io={
            bs: SimpleNamespace(logits=torch.zeros(bs * STRIDE, 4)) for bs in static_bs
        },
        read_accept=lambda bs: ("accept", bs),
        run=lambda io, accept, gate: None,
    )
    sampler = SimpleNamespace(
        greedy_draft=False,
        step_logits=torch.zeros(8, GAMMA, 4),
        params=SimpleNamespace(temperatures=torch.ones(8)),
    )
    graph = sag.DsparkSamplingAcceptGraph.__new__(sag.DsparkSamplingAcceptGraph)
    graph._epilogue = graph._static = epi
    graph._resolve_draft_sampler = lambda: sampler
    graph._tp_sync = _tp_sync()
    graph._fused_argmax = False
    graph._block_verification = False
    graph._rejection_noise = None
    graph._gate = torch.ones(1, dtype=torch.int32)
    graph._graphs = {}
    graph._warmed_up = set()
    graph._pool = None
    graph.disabled = False
    graph.num_replays = 0
    return graph, epi, sampler


def test_warmup_capture_replay_sequence():
    graph, epi, _ = _graph_on_cpu()
    events = []
    replays = []

    def capture(key, region):
        events.append(("capture", key[0]))
        g = SimpleNamespace(replay=lambda: replays.append(key[0]))
        graph._graphs[key] = g
        return g

    graph._capture = capture
    epi.run = lambda io, accept, gate: events.append(("eager", io))
    logits = epi.static_io[4].logits
    # bs 3 replays the size-4 verify graph, so it reads that size's buffers.
    for _ in range(3):
        assert graph.run(bs=3, target_logits=logits) == ("accept", 3)
    assert events == [("eager", epi.static_io[4]), ("capture", 4)]
    assert replays == [4, 4] and graph.num_replays == 2


def test_falls_back_to_eager_when_buffers_do_not_match():
    graph, epi, sampler = _graph_on_cpu(static_bs=(1, 2))
    assert graph.run(bs=4, target_logits=epi.static_io[2].logits) is None
    assert graph.run(bs=2, target_logits=torch.zeros(12, 4)) is None
    sampler.step_logits = None
    assert graph.run(bs=2, target_logits=epi.static_io[2].logits) is None
    graph, epi, _ = _graph_on_cpu()
    graph.disabled = True
    assert graph.run(bs=1, target_logits=epi.static_io[1].logits) is None


# ---------------------------------------------------------------- GPU parity


def _gpu_case(*, bs, graph_bs, vocab, seed, same_dist=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    target = torch.randn(graph_bs * STRIDE, vocab, device=dev, generator=gen) * 3
    step_logits = torch.randn(8, GAMMA, vocab, device=dev, generator=gen) * 3
    if same_dist:
        step_logits[:graph_bs] = target.view(graph_bs, STRIDE, vocab)[:, :GAMMA]
    candidates = torch.randint(0, vocab, (graph_bs, STRIDE), device=dev, generator=gen)
    temps = torch.rand(8, device=dev, generator=gen) * 1.5 + 0.25
    io = StaticVerifyIO(
        candidates=candidates,
        logits=target,
        hidden=torch.zeros(graph_bs * STRIDE, 8, device=dev),
        seq_lens=torch.randint(
            100, 200, (graph_bs,), device=dev, dtype=torch.int64, generator=gen
        ),
        req_pool_indices=torch.arange(graph_bs, device=dev),
        out_cache_loc=torch.zeros(graph_bs * STRIDE, dtype=torch.int64, device=dev),
        positions=torch.zeros(graph_bs * STRIDE, dtype=torch.int64, device=dev),
    )
    sampler = SimpleNamespace(
        greedy_draft=False,
        step_logits=step_logits,
        params=SimpleNamespace(
            temperatures=temps, greedy_mask=torch.zeros(8, dtype=torch.bool, device=dev)
        ),
    )
    epi = dspark_verify.DsparkVerifyEpilogue(
        max_bs=8, verify_num_draft_tokens=STRIDE, device=dev, tp_sync=_tp_sync()
    )
    _static(epi).static_io[graph_bs] = io
    epi.begin_static_step(bs, armed=False)
    return io, sampler, epi


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


def _eager_reference(io, sampler, bs):
    """The worker's eager sampling accept on the live rows only."""
    cand = io.candidates[:bs]
    params = sampler.params
    block = DraftBlockResult(
        draft_tokens=cand[:, 1:],
        corrected_logits=sampler.step_logits[:bs],
        greedy_mask=params.greedy_mask[:bs],
        temperatures=params.temperatures[:bs],
        greedy_draft=False,
    )
    info = sag.GraphSamplingInfo(temperatures=params.temperatures[:bs].view(bs, 1))
    correct_len, bonus, trim = dspark_verify.accept_draft_tokens(
        candidates=cand,
        target_logits=io.logits[: bs * STRIDE],
        draft_block=block,
        sampling_info=info,
        draft_input=None,
        gamma=GAMMA,
        verify_num_draft_tokens=STRIDE,
    )
    return dspark_verify.build_accept_outs(
        correct_len=correct_len,
        bonus=bonus,
        cap_trim_lens=trim,
        prefix_lens=io.seq_lens[:bs],
        draft_tokens=cand[:, 1:].contiguous(),
        gamma=GAMMA,
        verify_num_draft_tokens=STRIDE,
        tp_sync=_tp_sync(),
        sync_site=dspark_verify.SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE,
    )


@contextlib.contextmanager
def _single_rank_capture():
    yield SimpleNamespace(stream=torch.cuda.Stream())


def _graph(epi, sampler, monkeypatch):
    monkeypatch.setattr(sag, "graph_capture", _single_rank_capture)
    return sag.DsparkSamplingAcceptGraph(
        epilogue=epi,
        static_epilogue=epi.static_epilogue_override,
        resolve_draft_sampler=lambda: sampler,
        tp_sync=_tp_sync(),
        fused_argmax=False,
        block_verification=False,
        device="cuda",
    )


_FIELDS = ("correct_len", "bonus", "commit_lens", "new_seq_lens", "out_tokens")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("bs", [1, 3, 8])
def test_replay_equals_eager_at_same_seed(monkeypatch, bs):
    # Same batch size as the graph, so the generator draws have the same shapes.
    io, sampler, epi = _gpu_case(bs=bs, graph_bs=bs, vocab=2048, seed=bs)
    graph = _graph(epi, sampler, monkeypatch)
    for step in range(4):  # eager warm-up, capture + replay, then replays
        torch.cuda.manual_seed(1234 + step)
        ref = _eager_reference(io, sampler, bs)
        torch.cuda.manual_seed(1234 + step)
        out = graph.run(bs=bs, target_logits=io.logits)
        torch.cuda.synchronize()
        for name in _FIELDS:
            assert torch.equal(getattr(out, name), getattr(ref, name)), (step, name)
    assert graph.num_replays == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_replay_acceptance_distribution_matches_eager(monkeypatch):
    # bs 3 on a size-4 graph: padded row 3 must not change the live rows' law.
    io, sampler, epi = _gpu_case(bs=3, graph_bs=4, vocab=512, seed=7)
    sampler.params.temperatures.fill_(1.0)
    graph = _graph(epi, sampler, monkeypatch)
    trials = 3000
    hist_graph = torch.zeros(3, STRIDE, device="cuda")
    hist_eager = torch.zeros(3, STRIDE, device="cuda")
    rows = torch.arange(3, device="cuda")
    torch.cuda.manual_seed(99)
    for _ in range(trials):
        out = graph.run(bs=3, target_logits=io.logits)
        hist_graph[rows, out.correct_len] += 1
        ref = _eager_reference(io, sampler, 3)
        hist_eager[rows, ref.correct_len] += 1
    p_graph, p_eager = hist_graph / trials, hist_eager / trials
    # Binomial standard error at n = 3000 is <= 0.0092; allow ~4.5 sigma.
    assert (p_graph - p_eager).abs().max().item() < 0.042, (p_graph, p_eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_identical_draft_and_target_accepts_every_draft(monkeypatch):
    io, sampler, epi = _gpu_case(bs=2, graph_bs=2, vocab=1024, seed=3, same_dist=True)
    graph = _graph(epi, sampler, monkeypatch)
    for _ in range(3):
        out = graph.run(bs=2, target_logits=io.logits)
    torch.cuda.synchronize()
    assert torch.all(out.correct_len == GAMMA)
    assert torch.equal(out.new_seq_lens, io.seq_lens + STRIDE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_greedy_static_epilogue_unchanged():
    """The switch's static epilogue folds the same greedy accept as the stock one:
    AcceptGreedy + build_accept_outs into the epilogue's buffers."""
    io, _, epi = _gpu_case(bs=4, graph_bs=4, vocab=1024, seed=11)
    epi.begin_static_step(4, armed=True)
    gate = epi.inject_gate_buf
    static = epi.static_epilogue_override
    assert not static.run(io, accept=static.accept_greedy, gate=gate)
    correct_len, bonus, trim = dspark_verify.AcceptGreedy.execute(
        candidates=io.candidates,
        target_logits=io.logits,
        verify_num_draft_tokens=STRIDE,
    )
    ref = dspark_verify.build_accept_outs(
        correct_len=correct_len,
        bonus=bonus,
        cap_trim_lens=trim,
        prefix_lens=io.seq_lens,
        draft_tokens=io.candidates[:, 1:].contiguous(),
        gamma=GAMMA,
        verify_num_draft_tokens=STRIDE,
        tp_sync=_tp_sync(),
        sync_site=dspark_verify.SpecTpSyncSite.DSPARK_ACCEPT_GRAPH,
    )
    out = epi.read_accept(4)
    for name in _FIELDS:
        assert torch.equal(getattr(out, name), getattr(ref, name)), name


def _num_compiled(kernel, device):
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        pytest.skip("this Triton has no per-device kernel cache to inspect")
    return len(caches[device][0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_precompile_covers_served_accepts():
    """A served sampling or mixed accept after the warm-up precompile must not JIT
    (the 1-1.5 s serving stalls); vocab width and dtypes come from the served graphs."""
    from sglang.kernels.ops.speculative.dspark import dspark_accept
    from sglang.kernels.ops.speculative.reject_sampling import (
        speculative_sampling_classic_kernel,
    )

    io, sampler, epi = _gpu_case(bs=8, graph_bs=8, vocab=3000, seed=5)
    sag.precompile_sampling_accept(
        capture_bs=[1, 2, 3, 8],
        stride=STRIDE,
        target_logits=io.logits,
        draft_sampler=sampler,
        fused_argmax=False,
        block_verification=False,
    )
    dev = torch.cuda.current_device()
    kernels = (
        speculative_sampling_classic_kernel,
        dspark_accept._gather_two_level_bonus_kernel,
        dspark_accept._mixed_accept_select_kernel,
    )
    before = [_num_compiled(k, dev) for k in kernels]
    for bs in (1, 3):
        _eager_reference(io, sampler, bs)
    torch.cuda.synchronize()
    assert [_num_compiled(k, dev) for k in kernels] == before
