"""SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH: the captured sampling accept must equal the
eager one at the same generator state, and greedy / ineligible steps never route to it."""

import contextlib
import sys
from types import SimpleNamespace

import msgspec
import pytest
import torch

from sglang.srt.speculative.dspark_components import dspark_sampling_accept_graph as sag
from sglang.srt.speculative.dspark_components import (
    dspark_verify,
)
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

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


def _worker(enabled=True, mode="static"):
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker._sampling_accept_graph = object() if enabled else None
    worker._verify_planner = SimpleNamespace(mode_value=mode)
    return worker


def _eligible(worker, info, *, folded=True, grammar=False, sim_bonus=None):
    return worker._sampling_graph_eligible(
        batch=SimpleNamespace(has_grammar=grammar),
        proposal_folded=folded,
        sampling_info=info,
        simulate_bonus_sampling_info=sim_bonus,
    )


def test_routes_only_all_sampled_static_steps():
    worker = _worker()
    assert _eligible(worker, _sampling_info())
    # Greedy and mixed batches keep the greedy fold / the eager mixed accept.
    assert not _eligible(worker, None)
    assert not _eligible(worker, _sampling_info(is_any_greedy=True))
    assert not _eligible(worker, _sampling_info(need_top_k_sampling=True))
    assert not _eligible(worker, _sampling_info(need_top_p_sampling=True))
    assert not _eligible(worker, _sampling_info(logit_bias=torch.zeros(1)))
    assert not _eligible(worker, _sampling_info(), folded=False)
    assert not _eligible(worker, _sampling_info(), grammar=True)
    assert not _eligible(worker, _sampling_info(), sim_bonus=_sampling_info())
    assert not _eligible(_worker(enabled=False), _sampling_info())
    assert not _eligible(_worker(mode="compact"), _sampling_info())


# ---------------------------------------------------------------- GPU parity


def _gpu_case(*, bs, graph_bs, vocab, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    target = torch.randn(graph_bs * STRIDE, vocab, device=dev, generator=gen) * 3
    step_logits = torch.randn(8, GAMMA, vocab, device=dev, generator=gen) * 3
    candidates = torch.randint(0, vocab, (graph_bs, STRIDE), device=dev, generator=gen)
    temps = torch.rand(8, device=dev, generator=gen) * 1.5 + 0.25
    io = dspark_verify.StaticVerifyIO(
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
    epi.static_io[graph_bs] = io
    epi.begin_static_step(bs, armed=False)
    return io, sampler, epi


def _eager_reference(io, sampler, bs, rejection_noise=None):
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
        rejection_noise=rejection_noise,
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


def _graph(epi, sampler, monkeypatch, rejection_noise=None):
    monkeypatch.setattr(sag, "graph_capture", _single_rank_capture)
    return sag.DsparkSamplingAcceptGraph(
        epilogue=epi,
        resolve_draft_sampler=lambda: sampler,
        tp_sync=_tp_sync(),
        fused_argmax=False,
        block_verification=False,
        device="cuda",
        rejection_noise=rejection_noise,
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
        got, want = msgspec.structs.asdict(out), msgspec.structs.asdict(ref)
        for name in _FIELDS:
            assert torch.equal(got[name], want[name]), (step, name)
    assert graph.num_replays == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("bs", [3])
def test_blockwise_replay_equals_eager_at_same_counter(monkeypatch, bs):
    """SGLANG_OPT_DSV41_BLOCKWISE_REJECTION in the region: its draws depend only on (seed,
    device step counter, row, token), so a region replay equals the eager block-wise accept
    at the same counter; two noise objects from one seed advance once per step each."""
    from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
        make_rejection_noise,
    )

    io, sampler, epi = _gpu_case(bs=bs, graph_bs=bs, vocab=2048, seed=100 + bs)
    graph = _graph(epi, sampler, monkeypatch, make_rejection_noise(1234, "cuda"))
    ref_noise = make_rejection_noise(1234, "cuda")
    for step in range(4):  # eager warm-up, capture + replay, then replays
        ref = _eager_reference(io, sampler, bs, rejection_noise=ref_noise)
        out = graph.run(bs=bs, target_logits=io.logits)
        torch.cuda.synchronize()
        got, want = msgspec.structs.asdict(out), msgspec.structs.asdict(ref)
        for name in _FIELDS:
            assert torch.equal(got[name], want[name]), (step, name)
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


def test_region_stages_the_deferred_commit_from_the_joined_projection():
    """The verify graph replayed this step already joined the deferred commit's
    projection (its host flag is spent), so the region must stage from it instead of
    joining again, which would find nothing and fall back to an unprojected inject."""
    calls = []
    commit = SimpleNamespace(
        join_projection=lambda: calls.append("join") or False,
        stage=lambda **k: calls.append(("stage", int(k["commit_lens"].sum()))),
    )
    epi = dspark_verify.DsparkVerifyEpilogue.__new__(dspark_verify.DsparkVerifyEpilogue)
    epi.stride = STRIDE
    epi.verify_lens_buf = torch.full((2,), STRIDE, dtype=torch.int64)
    epi.commit_lens_buf = torch.tensor([3, 4], dtype=torch.int32)
    epi._post_fold = False
    epi.deferred_commit = commit
    epi.commit_ctx = SimpleNamespace(
        resolve_engram_hasher=None,
        resolve_pool=lambda: SimpleNamespace(
            set_swa_key_buffer_radix_fused_norm_rope=1
        ),
        kv_injector=SimpleNamespace(swa_ring_size=None),
    )
    io = SimpleNamespace(
        candidates=torch.zeros(2, STRIDE, dtype=torch.int64),
        req_pool_indices=torch.arange(2),
        out_cache_loc=torch.zeros(2 * STRIDE, dtype=torch.int64),
        positions=torch.zeros(2 * STRIDE, dtype=torch.int64),
    )

    def accept(io, *, verify_lens, gate, gated):
        return torch.minimum(epi.commit_lens_buf, verify_lens.to(torch.int32)) * gate

    gate = torch.ones(1, dtype=torch.int32)
    epi.run_static(io, accept=accept, gate=gate, kv_projected=True)
    assert calls == [("stage", 7)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
