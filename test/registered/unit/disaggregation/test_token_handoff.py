import math

import pytest

from sglang.srt.disaggregation.token_handoff import (
    HandoffPhase,
    HandoffProtocolError,
    OutputOwner,
    TokenHandoffState,
    estimate_catch_up,
)


def test_exact_count_commit_transfers_single_output_owner():
    state = TokenHandoffState("req-1", epoch=7, prompt_token_count=8192)
    state.append_tokens(epoch=7, first_output_index=0, token_ids=[11, 12, 13])
    state.mark_prompt_kv_ready(epoch=7)
    state.acknowledge_replay(epoch=7, replayed_count=3)
    state.commit(epoch=7, produced_count=3)

    assert state.phase is HandoffPhase.COMMITTED
    assert state.owner is OutputOwner.DECODE
    assert state.committed_count == 3
    with pytest.raises(HandoffProtocolError, match="terminal"):
        state.append_tokens(epoch=7, first_output_index=3, token_ids=[14])


def test_new_prefill_token_after_replay_ack_forces_decode_to_catch_up_again():
    state = TokenHandoffState("req-race", epoch=1, prompt_token_count=32768)
    state.append_tokens(epoch=1, first_output_index=0, token_ids=[1, 2])
    state.mark_prompt_kv_ready(epoch=1)
    state.acknowledge_replay(epoch=1, replayed_count=2)
    assert state.phase is HandoffPhase.READY_TO_COMMIT

    state.append_tokens(epoch=1, first_output_index=2, token_ids=[3])
    assert state.phase is HandoffPhase.REPLAYING
    with pytest.raises(HandoffProtocolError, match="behind"):
        state.commit(epoch=1, produced_count=2)

    state.acknowledge_replay(epoch=1, replayed_count=3)
    state.commit(epoch=1, produced_count=3)
    assert state.owner is OutputOwner.DECODE


def test_stale_epoch_cannot_advance_or_cancel_handoff():
    state = TokenHandoffState("req-epoch", epoch=9, prompt_token_count=1024)
    with pytest.raises(HandoffProtocolError, match="stale handoff epoch"):
        state.append_tokens(epoch=8, first_output_index=0, token_ids=[1])
    with pytest.raises(HandoffProtocolError, match="stale handoff epoch"):
        state.cancel(epoch=10)

    assert state.phase is HandoffPhase.COPYING_PROMPT_KV
    assert state.owner is OutputOwner.PREFILL


def test_replay_ack_is_monotonic_and_bounded_by_published_log():
    state = TokenHandoffState("req-ack", epoch=2, prompt_token_count=4096)
    state.append_tokens(epoch=2, first_output_index=0, token_ids=[5, 6, 7])
    state.mark_prompt_kv_ready(epoch=2)
    state.acknowledge_replay(epoch=2, replayed_count=2)

    with pytest.raises(HandoffProtocolError, match="monotonic"):
        state.acknowledge_replay(epoch=2, replayed_count=1)
    with pytest.raises(HandoffProtocolError, match="beyond"):
        state.acknowledge_replay(epoch=2, replayed_count=4)


def test_token_append_retry_is_idempotent_but_conflicts_are_rejected():
    state = TokenHandoffState("req-retry", epoch=4, prompt_token_count=2048)
    state.append_tokens(epoch=4, first_output_index=0, token_ids=[31, 32])

    assert state.append_tokens(epoch=4, first_output_index=0, token_ids=[31, 32]) == 2
    with pytest.raises(HandoffProtocolError, match="conflicting"):
        state.append_tokens(epoch=4, first_output_index=0, token_ids=[31, 99])
    with pytest.raises(HandoffProtocolError, match="overlapping"):
        state.append_tokens(epoch=4, first_output_index=1, token_ids=[32, 33])


def test_failure_before_commit_keeps_prefill_as_owner():
    state = TokenHandoffState("req-fail", epoch=3, prompt_token_count=16384)
    state.append_tokens(epoch=3, first_output_index=0, token_ids=[21])
    state.fail(epoch=3, reason="decode replay failed")

    assert state.phase is HandoffPhase.FAILED
    assert state.owner is OutputOwner.PREFILL
    assert state.failure_reason == "decode replay failed"


def test_catch_up_estimate_requires_decode_replay_to_beat_live_decode():
    infeasible = estimate_catch_up(
        remaining_copy_ms=100,
        prefill_decode_tpot_ms=20,
        decode_replay_tokens_per_second=50,
    )
    assert not infeasible.feasible
    assert math.isinf(infeasible.estimated_catch_up_ms)

    feasible = estimate_catch_up(
        remaining_copy_ms=100,
        prefill_decode_tpot_ms=20,
        decode_replay_tokens_per_second=500,
        replay_startup_ms=10,
    )
    assert feasible.feasible
    assert feasible.initial_backlog_tokens == 6
    assert feasible.estimated_bridge_tokens >= feasible.initial_backlog_tokens
    assert feasible.estimated_catch_up_ms > 10
