import math

import pytest

from sglang.srt.disaggregation.token_handoff import (
    BatchedReplayPlan,
    HandoffPhase,
    HandoffProtocolError,
    OutputOwner,
    TokenHandoffState,
    build_batched_replay_plan,
    estimate_catch_up,
)
from sglang.srt.managers.schedule_batch import Req


def test_exact_count_commit_transfers_single_output_owner():
    state = TokenHandoffState("req-1", epoch=7, prompt_token_count=8192)
    state.append_tokens(epoch=7, first_output_index=0, token_ids=[11, 12, 13])
    state.mark_prompt_kv_ready(epoch=7)
    state.acknowledge_replay(epoch=7, replayed_count=3)
    state.seal_token_log(epoch=7)
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
    state.seal_token_log(epoch=1)
    state.commit(epoch=1, produced_count=3)
    assert state.owner is OutputOwner.DECODE


def test_sealed_log_stops_live_producer_and_bounds_final_replay():
    state = TokenHandoffState("req-seal", epoch=5, prompt_token_count=32768)
    state.append_tokens(epoch=5, first_output_index=0, token_ids=[1, 2, 3, 4])
    state.mark_prompt_kv_ready(epoch=5)
    state.acknowledge_replay(epoch=5, replayed_count=2)

    assert state.seal_token_log(epoch=5) == 4
    assert state.phase is HandoffPhase.DRAINING_FINAL_SUFFIX
    with pytest.raises(HandoffProtocolError, match="sealed"):
        state.append_tokens(epoch=5, first_output_index=4, token_ids=[5])

    state.acknowledge_replay(epoch=5, replayed_count=4)
    assert state.phase is HandoffPhase.READY_TO_COMMIT
    state.commit(epoch=5, produced_count=4)
    assert state.owner is OutputOwner.DECODE


def test_commit_requires_a_sealed_output_boundary():
    state = TokenHandoffState("req-unsealed", epoch=6, prompt_token_count=4096)
    state.append_tokens(epoch=6, first_output_index=0, token_ids=[8])
    state.mark_prompt_kv_ready(epoch=6)
    state.acknowledge_replay(epoch=6, replayed_count=1)

    with pytest.raises(HandoffProtocolError, match="must be sealed"):
        state.commit(epoch=6, produced_count=1)


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


def test_batched_replay_plan_teacher_forces_all_but_boundary_token():
    plan = build_batched_replay_plan([21, 22, 23, 24])

    assert plan == BatchedReplayPlan(
        input_token_ids=[21, 22, 23],
        expected_next_token_id=24,
    )
    with pytest.raises(ValueError, match="at least two"):
        build_batched_replay_plan([21])


def test_incremental_detokenizer_can_resume_at_cross_worker_output_boundary():
    req = type("ReqStub", (), {})()
    req.origin_input_ids_unpadded = [10, 11, 12, 13, 14, 15]
    req.output_ids = [21, 22, 23, 24]

    decode_ids, read_offset = Req.prime_incremental_detokenize_at_output_offset(req, 2)

    # Five tokens of surrounding context end exactly at the Prefill-owned
    # boundary; the Decode-owned suffix remains available for first emission.
    assert decode_ids == [13, 14, 15, 21, 22, 23, 24]
    assert read_offset == 5
    assert req.cur_decode_ids_len == 4
