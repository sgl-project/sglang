"""--enable-dsv41-low-latency-decode: the switch's static verify epilogue records the
Engram history commit, gated per row, and reports it only when it was captured, so the
worker skips its eager commit only then; without a history it commits nothing."""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dsv41_low_latency.dspark import static_epilogue as se
from sglang.srt.models.dsv41_low_latency.dspark.worker import DsparkLowLatency
from sglang.srt.speculative.dspark_components import dspark_verify

STRIDE = 6


def _epilogue(resolve):
    tp_sync = SimpleNamespace(sync=lambda site, values: values)
    epi = dspark_verify.DsparkVerifyEpilogue(
        max_bs=2, verify_num_draft_tokens=STRIDE, device="cpu", tp_sync=tp_sync
    )
    epi.begin_static_step(1, armed=True)
    return se.StaticEpilogue(
        epilogue=epi,
        tp_sync=tp_sync,
        fused_argmax=False,
        resolve_engram_history=resolve,
        context_kv_side_stream=None,
    )


def _io(bs):
    return se.StaticVerifyIO(
        candidates=torch.zeros(bs, STRIDE, dtype=torch.int64),
        logits=torch.zeros(bs * STRIDE, 5),
        hidden=torch.zeros(bs * STRIDE, 8),
        seq_lens=torch.ones(bs, dtype=torch.int64),
        req_pool_indices=torch.arange(bs),
        out_cache_loc=torch.zeros(bs * STRIDE, dtype=torch.int64),
        positions=torch.zeros(bs * STRIDE, dtype=torch.int64),
    )


def _accept(io):
    return torch.ones(io.candidates.shape[0], dtype=torch.int32)


def test_commit_passes_live_rows_and_gate(monkeypatch):
    calls = []
    monkeypatch.setattr(se, "engram_commit_history", lambda *a, **k: calls.append(k))
    history = torch.zeros(4, 3, dtype=torch.int32)
    static = _epilogue(lambda: history)
    gate = torch.ones(1, dtype=torch.int32)
    assert static.run(_io(2), accept=_accept, gate=gate)
    assert len(calls) == 1
    # Row 1 is past the live batch of 1: its verify_lens entry is 0.
    assert calls[0]["live_rows"].tolist() == [STRIDE, 0]
    assert calls[0]["gate"] is gate


@pytest.mark.parametrize("capturing", [False, True])
def test_commit_reported_only_under_capture(monkeypatch, capturing):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(se, "engram_commit_history", lambda *a, **k: None)
    static = _epilogue(lambda: torch.zeros(4, 3, dtype=torch.int32))
    monkeypatch.setattr(static, "accept_greedy", _accept)
    io = _io(2)
    fb = SimpleNamespace(
        batch_size=2,
        input_ids=io.candidates.view(-1),
        seq_lens=io.seq_lens,
        req_pool_indices=io.req_pool_indices,
        out_cache_loc=io.out_cache_loc,
        positions=io.positions,
    )
    out = SimpleNamespace(next_token_logits=io.logits, hidden_states=io.hidden)
    static(out, fb)
    # Only a captured commit lets the worker skip its eager one.
    assert static.folds_engram_commit == capturing
    assert (2 in static.static_io) == capturing
    low_latency = DsparkLowLatency(
        static_epilogue=static,
        sampling_accept_graph=None,
        resolve_draft_sampler=lambda: None,
        rejection_noise=None,
        capture_bs=[2],
        stride=STRIDE,
        block_verification=False,
    )
    folded = dict(folded_accept=True, run_compact=False, graph_accept=False)
    assert low_latency.committed_engram(**folded) == capturing
    # An unfolded step replayed the graph with its gate closed: commit eagerly.
    unfolded = dict(folded_accept=False, run_compact=False, graph_accept=False)
    assert not low_latency.committed_engram(**unfolded)


def test_no_history_records_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(se, "engram_commit_history", lambda *a, **k: calls.append(k))
    static = _epilogue(lambda: None)
    assert not static.run(_io(1), accept=_accept, gate=torch.ones(1, dtype=torch.int32))
    assert not calls and not static.folds_engram_commit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gated_kernel_skips_dead_rows_and_closed_gate():
    from sglang.kernels.ops.embeddings.engram_hash import engram_commit_history

    dev = "cuda"
    history = torch.arange(4 * 3, dtype=torch.int32, device=dev).view(4, 3)
    tokens = torch.arange(100, 112, dtype=torch.int64, device=dev).view(2, 6)
    slots = torch.tensor([1, 3], device=dev)
    commit = torch.tensor([2, 4], dtype=torch.int32, device=dev)
    ungated = history.clone()
    engram_commit_history(ungated, tokens, slots, commit)

    gated = history.clone()
    live = torch.tensor([6, 0], device=dev)  # row 1 is a padded capture row
    gate = torch.ones(1, dtype=torch.int32, device=dev)
    engram_commit_history(gated, tokens, slots, commit, live_rows=live, gate=gate)
    assert torch.equal(gated[1], ungated[1])
    assert torch.equal(gated[3], history[3])

    closed = history.clone()
    engram_commit_history(
        closed, tokens, slots, commit, live_rows=torch.full_like(live, 6), gate=gate * 0
    )
    assert torch.equal(closed, history)
