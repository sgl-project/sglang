"""Stateful differential check of compact graph-tail commits and fallback."""

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.speculative.dspark.native_compact import ragged_candidates
from sglang.srt.managers.overlap_utils import FutureRelayBuffers
from sglang.srt.speculative.dspark_components.dspark_verify import DsparkVerifyEpilogue
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def make_tail(bs, device):
    epi = DsparkVerifyEpilogue.__new__(DsparkVerifyEpilogue)
    history = torch.arange((bs + 3) * 3, dtype=torch.int64, device=device).view(-1, 3)
    epi.commit_ctx = SimpleNamespace(
        resolve_engram_hasher=lambda: SimpleNamespace(history=history)
    )
    epi.future_relay = FutureRelayBuffers(req_pool_size=bs + 3, device=device)
    epi.future_relay.new_seq_lens.fill_(-1)
    epi.future_relay.output_tokens.fill_(-1)
    epi.new_seq_lens_buf = torch.zeros(bs, dtype=torch.int64, device=device)
    epi.bonus_buf = torch.zeros_like(epi.new_seq_lens_buf)
    return epi, history


@pytest.mark.parametrize("bs", [3, 8, 32, 128])
def test_tail_multistep_replay_and_fallback(bs):
    device = "cpu" if os.getenv("TRITON_INTERPRET") == "1" else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA or TRITON_INTERPRET=1")
    epi, history = make_tail(bs, device)
    initial = history.clone()
    expected_history = history.cpu().clone()
    expected_seq = torch.full((bs + 3,), -1, dtype=torch.int64)
    expected_bonus = expected_seq.clone()
    ids = torch.zeros(bs * 6, dtype=torch.int64, device=device)
    ptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    lens = torch.zeros(bs, dtype=torch.int64, device=device)
    slots = torch.arange(1, bs + 1, device=device)
    commits = torch.zeros(bs, dtype=torch.int32, device=device)
    gate = torch.zeros(1, dtype=torch.int32, device=device)
    candidates = torch.empty((bs, 6), dtype=torch.int64, device=device)

    def operation():
        ragged_candidates(ids=ids, indptr=ptr, lens=lens, out=candidates)
        epi._commit_engram_history(candidates, slots, commits, lens, gate)
        epi._scatter_future_relay(slots, bs)

    operation()
    graph = None
    if device == "cuda":
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
    history.copy_(initial)
    epi.future_relay.new_seq_lens.fill_(-1)
    epi.future_relay.output_tokens.fill_(-1)
    for step in range(12):
        lengths = (torch.arange(bs) + step) % 7
        current_slots = torch.arange(1, bs + 1).roll(step)
        current_slots[lengths == 0] = 0
        dense = (torch.arange(bs * 6).view(bs, 6) + 100 * step + 1).long()
        mask = torch.arange(6)[None, :] < lengths[:, None]
        current_commits = torch.minimum(lengths, (torch.arange(bs) + step) % 6 + 1)
        prefixes = torch.arange(bs) + 200 + 10 * step
        bonuses = torch.arange(bs) + 5000 + step * 100
        fallback = step % 4 == 2
        slots.copy_(current_slots)
        lens.copy_(lengths)
        ptr.copy_(torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)]))
        ids.zero_()
        ids[: int(lengths.sum())].copy_(dense[mask])
        commits.copy_(current_commits)
        gate.fill_(not fallback)
        epi.new_seq_lens_buf.copy_(prefixes + current_commits)
        epi.bonus_buf.copy_(bonuses)
        if graph is None:
            operation()
        else:
            graph.replay()
        if fallback:
            # The alternate acceptance overwrites the speculative graph relay.
            current_commits = torch.minimum(current_commits, torch.ones_like(lengths))
            commits.copy_(current_commits)
            gate.fill_(1)
            epi._commit_engram_history(candidates, slots, commits, lens, gate)
            epi.new_seq_lens_buf.copy_(prefixes + current_commits)
            epi.bonus_buf.copy_(bonuses + 17)
            epi._scatter_future_relay(slots, bs)
            bonuses += 17
        for row, slot in enumerate(current_slots.tolist()):
            if lengths[row] == 0:
                continue
            count = int(current_commits[row])
            window = torch.cat([expected_history[slot], dense[row]])
            expected_history[slot] = window[count : count + 3]
            expected_seq[slot] = prefixes[row] + count
            expected_bonus[slot] = bonuses[row]
        torch.testing.assert_close(history.cpu(), expected_history, rtol=0, atol=0)
        # Slot zero is the graph's padding sink; only real request slots are read.
        torch.testing.assert_close(
            epi.future_relay.new_seq_lens[1:].cpu(), expected_seq[1:]
        )
        torch.testing.assert_close(
            epi.future_relay.output_tokens[1:].cpu(), expected_bonus[1:]
        )
