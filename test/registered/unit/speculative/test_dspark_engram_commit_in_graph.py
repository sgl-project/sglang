"""SGLANG_OPT_DSV41_POST_FORWARD_CUDA_GRAPH: the static verify epilogue records the
engram history commit, gated per row, and reports it only when it was captured."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


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


def _commit_reference(history, tokens, slots, commit, live_rows, gate):
    """Anchor + accepted drafts into each live row's history (gate open)."""
    if int(gate[0]) == 0:
        return
    n = history.shape[1]
    for r in range(slots.numel()):
        if int(live_rows[r]) == 0:
            continue
        s = int(slots[r])
        window = torch.cat([history[s], tokens[r].to(history.dtype)])
        c = int(commit[r])
        history[s] = window[c : c + n]


def _epilogue_for_accept(*, fused, bs, stride, history):
    epi = dspark_verify.DsparkVerifyEpilogue.__new__(dspark_verify.DsparkVerifyEpilogue)
    z = lambda *shape, dtype: torch.zeros(*shape, dtype=dtype)
    epi.stride, epi.gamma, epi.max_bs = stride, stride - 1, bs
    epi.verify_lens_buf = torch.full((bs,), stride, dtype=torch.int64)
    epi.verify_lens_buf[-1] = 0  # a padded capture row
    epi.inject_gate_buf = torch.ones(1, dtype=torch.int32)
    epi.correct_len_buf = z(bs, dtype=torch.int64)
    epi.bonus_buf = z(bs, dtype=torch.int64)
    epi.cap_trim_lens_buf = z(bs, dtype=torch.int32)
    epi.commit_lens_buf = z(bs, dtype=torch.int32)
    epi.new_seq_lens_buf = z(bs, dtype=torch.int64)
    epi.out_tokens_buf = z(bs, stride, dtype=torch.int64)
    epi.gated_commit_lens_buf = z(bs, dtype=torch.int32)
    epi.block_accept_lens_buf = torch.full((bs,), -1, dtype=torch.int32)
    epi.simulated_correct_len = None
    epi.simulate_fold = epi._tail_broadcasts = epi._fused_argmax = False
    epi._fused_accept_tail = fused
    epi._accept_tail_bs = (1, 0)
    epi._accept_tail_logged = set()
    epi._tp_sync = SimpleNamespace(sync=lambda site, t: None, enabled=lambda s: False)
    epi.folds_engram_commit = False
    epi.static_io = {}
    epi._post_fold = False
    # No fused-norm pool: the epilogue returns right after the engram commit.
    epi.commit_ctx = SimpleNamespace(
        resolve_engram_hasher=lambda: SimpleNamespace(history=history),
        resolve_pool=lambda: object(),
    )
    return epi


def test_engram_history_fused_accept_tail_matches_unfused(monkeypatch):
    """SGLANG_DSPARK_FUSED_ACCEPT_TAIL returns only gated lengths; the in-graph engram
    commit must still see the ungated ones, so both accepts leave the same history."""
    monkeypatch.setattr(dspark_verify, "engram_commit_history", _commit_reference)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    bs, stride, vocab = 5, 6, 64
    g = torch.Generator().manual_seed(7)
    target = torch.randint(0, vocab, (bs, stride), generator=g)
    logits = torch.randn(bs * stride, vocab, generator=g)
    logits[torch.arange(bs * stride), target.reshape(-1)] += 100.0
    candidates = torch.empty(bs, stride, dtype=torch.int64)
    candidates[:, 0] = torch.randint(0, vocab, (bs,), generator=g)
    candidates[:, 1:] = target[:, :-1]
    for b, stop in enumerate([0, 2, 5, 3, 1]):  # first mismatch column per row
        if stop < stride - 1:
            candidates[b, 1 + stop] = (target[b, stop] + 1) % vocab
    fb = SimpleNamespace(
        batch_size=bs,
        input_ids=candidates.reshape(-1),
        seq_lens=torch.randint(10, 100, (bs,), generator=g),
        req_pool_indices=torch.tensor([4, 0, 6, 2, 1]),
        out_cache_loc=torch.zeros(bs * stride, dtype=torch.int64),
        positions=torch.zeros(bs * stride, dtype=torch.int64),
    )
    out = SimpleNamespace(next_token_logits=logits, hidden_states=None)
    base = torch.randint(0, vocab, (8, 3), generator=g, dtype=torch.int64)
    results = {}
    for fused in (False, True):
        history = base.clone()
        epi = _epilogue_for_accept(fused=fused, bs=bs, stride=stride, history=history)
        epi._static_epilogue(out, fb)
        results[fused] = (history, epi.block_accept_lens_buf.clone())
    assert torch.equal(results[True][0], results[False][0])
    assert not torch.equal(results[False][0], base)
    assert torch.equal(results[True][1], results[False][1])
    assert bool((results[True][1] > 0).all())


@pytest.mark.parametrize("capturing", [False, True])
def test_folds_engram_commit_only_under_capture(monkeypatch, capturing):
    """Only a captured commit lets the worker skip its eager one; a captured static
    epilogue also records its graph IO for the sampling-accept region."""
    monkeypatch.setattr(dspark_verify, "engram_commit_history", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    bs, stride, vocab = 2, 6, 16
    epi = _epilogue_for_accept(
        fused=False, bs=bs, stride=stride, history=torch.zeros(4, 3, dtype=torch.int64)
    )
    fb = SimpleNamespace(
        batch_size=bs,
        input_ids=torch.zeros(bs * stride, dtype=torch.int64),
        seq_lens=torch.full((bs,), 10),
        req_pool_indices=torch.arange(bs),
        out_cache_loc=torch.zeros(bs * stride, dtype=torch.int64),
        positions=torch.zeros(bs * stride, dtype=torch.int64),
    )
    out = SimpleNamespace(
        next_token_logits=torch.randn(bs * stride, vocab), hidden_states=None
    )
    epi._static_epilogue(out, fb)
    assert epi.folds_engram_commit == capturing
    assert (bs in epi.static_io) == capturing


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
