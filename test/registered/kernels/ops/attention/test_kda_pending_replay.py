"""Pending KDA state transitions, independent of model loading."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from sglang.kernels.ops.attention.linear.kda_pending_replay import KdaPendingReplay
from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
    commit_kda_replayssm_after_verify,
    commit_kda_replayssm_spec_all_layers,
)


def fixture():
    torch.manual_seed(42)
    state = torch.randn(2, 6, 2, 128, 128, device="cuda") * 0.01
    shape = (2, 6, 2, 16, 128)
    rawv = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    rawk = torch.randn_like(rawv)
    gate = -torch.rand(shape, device="cuda") * 0.1
    beta = torch.rand(shape[:-1], device="cuda")
    slots = torch.tensor([3, 1, -1], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([4, 1, 0], dtype=torch.int32, device="cuda")
    return state, (rawv, rawk, gate, beta), slots, accepted


def fold(state, rings, slots, accepted, **kwargs):
    commit_kda_replayssm_spec_all_layers(
        state,
        *rings,
        slots,
        accepted,
        max_cache_len=16,
        num_k_heads=2,
        null_block_id=-1,
        **kwargs,
    )


def test_selective_flush_padding_and_consumed_layers():
    state, rings, slots, accepted = fixture()
    initial = state.clone()
    expected = state.clone()
    pending = KdaPendingReplay(state)
    pending.stage(*rings, slots, accepted)
    pending.flush(layer=0, indices=slots[:1])
    assert pending.counts[0, 3].count_nonzero().item() == 0
    assert pending.counts[1, 3].count_nonzero().item() > 0
    pending.flush(indices=slots)
    fold(expected, rings, slots, accepted)
    torch.testing.assert_close(state, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        state[:, [0, 2, 4, 5]], initial[:, [0, 2, 4, 5]], rtol=0, atol=0
    )
    pending.flush()
    torch.testing.assert_close(state, expected, rtol=1e-5, atol=1e-6)


def test_fallback_transition_and_captured_stage():
    state, rings, slots, accepted = fixture()
    initial = state.clone()
    expected = state.clone()
    fold(expected, rings, slots, accepted)
    pending = KdaPendingReplay(state)
    calls = []
    original = pending.flush

    def counted_flush(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    pending.flush = counted_flush
    pending.stage(*rings, slots, accepted)
    for batch_size in range(1, 65):
        pending.before_verify(batch_size=batch_size, block_size=8)
    assert not calls
    pending.before_verify(batch_size=65, block_size=8)
    pending.before_verify(batch_size=65, block_size=8)
    assert len(calls) == 1
    torch.testing.assert_close(state, expected, rtol=1e-5, atol=1e-6)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pending.stage(*rings, slots, accepted)
    state.copy_(initial)
    pending.before_verify(batch_size=65, block_size=8)
    state.copy_(initial)
    graph.replay()
    pending.before_verify(batch_size=65, block_size=8)
    torch.testing.assert_close(state, expected, rtol=1e-5, atol=1e-6)


def test_tracking_commits_immediately_and_other_rows_defer():
    state, rings, slots, accepted = fixture()
    expected = state.clone()
    pending = KdaPendingReplay(state)
    track = torch.tensor([5, -1, -1], dtype=torch.int32, device="cuda")
    steps = torch.tensor([1, -1, -1], dtype=torch.int32, device="cuda")
    fold(
        expected,
        rings,
        slots,
        accepted,
        mamba_track_indices=track,
        mamba_steps_to_track=steps,
    )
    spec_state = SimpleNamespace(
        temporal=state,
        replayssm_rawv=rings[0],
        replayssm_rawk=rings[1],
        replayssm_g=rings[2],
        replayssm_beta=rings[3],
        conv=[],
        intermediate_conv_window=[],
    )
    commit_kda_replayssm_after_verify(
        spec_state=spec_state,
        state_batch_indices=slots,
        accept_lens=accepted,
        last_correct_step_indices=accepted - 1,
        mamba_track_indices=track,
        mamba_steps_to_track=steps,
        pending_replay=pending,
    )
    torch.testing.assert_close(
        state[:, [3, 5]], expected[:, [3, 5]], rtol=1e-5, atol=1e-6
    )
    assert pending.counts[:, 1].count_nonzero().item() > 0
    pending.flush()
    torch.testing.assert_close(state, expected, rtol=1e-5, atol=1e-6)


def test_changing_batch_preserves_pending_slot_ownership():
    torch.manual_seed(43)
    state = torch.randn(2, 70, 2, 128, 128, device="cuda") * 0.01
    expected = state.clone()
    pending = KdaPendingReplay(state)
    for batch_size in (31, 17, 33, 48, 64, 16, 1):
        # The same slot can move to a different row or leave the next batch.
        slots = torch.randperm(70, device="cuda")[:batch_size].to(torch.int32)
        accepted = (torch.arange(batch_size, device="cuda") % 9).to(torch.int32)
        pending.before_verify(batch_size=batch_size, block_size=8)
        # Model per-layer consumption before replacing this slot's ring.
        for layer in range(2):
            pending.flush(layer=layer, indices=slots)
        shape = (2, 70, 2, 16, 128)
        rawv = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        rings = (
            rawv,
            torch.randn_like(rawv),
            -torch.rand(shape, device="cuda") * 0.1,
            torch.rand(shape[:-1], device="cuda"),
        )
        pending.stage(*rings, slots, accepted)
        fold(expected, rings, slots, accepted)
    pending.before_verify(batch_size=65, block_size=8)
    torch.testing.assert_close(state, expected, rtol=1e-5, atol=1e-6)
    assert pending.counts.count_nonzero().item() == 0
