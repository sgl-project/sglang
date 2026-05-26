"""Edge cases — retract / preempt × scheduler-iteration timing.

Covers A.7 series from the expansion plan and fan-out variations
(retract at every chunk position, retract + immediate re-admit,
multi-victim, retract during decode, retract-then-abort).
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase


def _script_force_retract_at_chunk_0(t: ScriptedRuntime):
    # Force retract before first chunk completes.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield  # admission
    yield  # first chunk
    t.force_retract(r)
    yield
    assert r.kv_pages == 0


def _script_force_retract_at_chunk_mid(t: ScriptedRuntime):
    # Force retract in the middle of chunked extend.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.chunks_done >= 2 and h.is_chunking)
    t.force_retract(r)
    yield
    assert r.kv_pages == 0


def _script_force_retract_at_last_chunk(t: ScriptedRuntime):
    # Force retract during the last chunk's admit.
    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
    yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
    t.force_retract(r)
    yield
    assert r.kv_pages == 0
    assert r.pending_middle_outputs == 0


def _script_force_retract_then_readmit(t: ScriptedRuntime):
    # Force retract; next yield re-admits and completes.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    t.force_retract(r)
    yield
    # Continue: scheduler should re-admit on its own.
    yield from run_until_finished(r)
    assert r.finished


def _script_retract_one_admit_one(t: ScriptedRuntime):
    # Force retract r1 + simultaneously admit r2.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r1, lambda h: h.is_chunking)
    r2 = t.start_req(prompt_len=8, max_new_tokens=2)
    t.force_retract(r1)
    yield from run_until_finished(r2)
    assert r2.finished
    yield from run_until_finished(r1)
    assert r1.finished


def _script_priority_preempt_chunked_victim(t: ScriptedRuntime):
    # Submit chunked low-priority, then high-priority new req.
    # Victim chunked must be moved from active to waiting.
    r_low = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low"
    )
    yield from run_until(r_low, lambda h: h.is_chunking)
    # NEW API NEEDED: t.force_preempt(victim, by) — explicit priority preempt
    # trigger; auto-triggered when high-prio req can't admit otherwise.
    r_high = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high"
    )
    t.force_preempt(victim_rid=r_low.rid, by_rid=r_high.rid)
    yield
    assert r_low.status == "waiting"

    yield from run_until_all_finished([r_low, r_high])


def _script_preempt_five_victims(t: ScriptedRuntime):
    # 5 chunked reqs preempted simultaneously.
    victims = [
        t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2) for _ in range(5)
    ]
    yield from run_until(victims[0], lambda h: h.is_chunking)

    r_high = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high"
    )
    for v in victims:
        t.force_preempt(victim_rid=v.rid, by_rid=r_high.rid)
    yield
    for v in victims:
        assert v.kv_pages == 0

    yield from run_until_all_finished(victims + [r_high])


def _script_retract_during_decode(t: ScriptedRuntime):
    # Retract during pure decode (no chunked); ensure clean.
    r = t.start_req(prompt_len=8, max_new_tokens=32)
    yield from run_until(r, lambda h: h.status == "running")
    t.force_retract(r)
    yield
    yield from run_until_finished(r)
    assert r.finished


def _script_retract_then_abort_idempotent(t: ScriptedRuntime):
    # Retract + abort same step; final state stable.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    t.force_retract(r)
    t.abort(r)
    yield
    assert r.kv_pages == 0


def _script_disagg_retract_resets_send_state(t: ScriptedRuntime):
    # disagg path: retract must reset send-side state (414efd4a27).
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    t.force_retract(r)
    yield
    # NEW API NEEDED: r.disagg_send_state — current state of the
    # send-side machine (None when not disagg or after reset).
    assert r.disagg_send_state in (None, "idle")
    yield from run_until_finished(r)


def _script_retract_chunked_resume_in_waiting(t: ScriptedRuntime):
    # Chunked-resume already in waiting → force retract on it.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    yield from run_until(r, lambda h: h.status == "waiting")
    t.force_retract(r)
    yield
    assert r.kv_pages == 0
    assert r.row_idx is None


def _script_two_retracts_same_yield(t: ScriptedRuntime):
    # Two reqs force_retracted in the same yield step.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r1, lambda h: h.is_chunking)
    t.force_retract(r1)
    t.force_retract(r2)
    yield
    assert r1.kv_pages == 0
    yield from run_until_all_finished([r1, r2])


def _script_priority_preempt_release_invariant(t: ScriptedRuntime):
    # Each preemption releases the victim's kv_pages strictly to 0.
    r_low = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low"
    )
    yield from run_until(r_low, lambda h: h.is_chunking)
    pages_before = r_low.kv_pages
    assert pages_before > 0

    r_high = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high"
    )
    t.force_preempt(victim_rid=r_low.rid, by_rid=r_high.rid)
    yield
    assert r_low.kv_pages == 0


def _script_retract_then_re_chunk(t: ScriptedRuntime):
    # Retract a mid-chunk req; verify subsequent re-chunk completes
    # without prefix_indices residue.
    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
    yield from run_until(r, lambda h: h.chunks_done >= 1)
    t.force_retract(r)
    # Explicit yield so the retract side effect happens-before the
    # subsequent run_until_finished observes any state.
    yield
    yield from run_until_finished(r)
    assert r.finished


class TestEdgeRetractPreemptTiming(CustomTestCase):
    def test_force_retract_at_chunk_0(self):
        execute_scripted_runtime(
            _script_force_retract_at_chunk_0,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_force_retract_at_chunk_mid(self):
        execute_scripted_runtime(
            _script_force_retract_at_chunk_mid,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_force_retract_at_last_chunk(self):
        execute_scripted_runtime(
            _script_force_retract_at_last_chunk,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_force_retract_then_readmit(self):
        execute_scripted_runtime(
            _script_force_retract_then_readmit,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_retract_one_admit_one(self):
        execute_scripted_runtime(
            _script_retract_one_admit_one,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_priority_preempt_chunked_victim(self):
        execute_scripted_runtime(
            _script_priority_preempt_chunked_victim,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )

    def test_preempt_five_victims(self):
        execute_scripted_runtime(
            _script_preempt_five_victims,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )

    def test_retract_during_decode(self):
        execute_scripted_runtime(
            _script_retract_during_decode,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_retract_then_abort_idempotent(self):
        execute_scripted_runtime(
            _script_retract_then_abort_idempotent,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_disagg_retract_resets_send_state(self):
        execute_scripted_runtime(
            _script_disagg_retract_resets_send_state,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_retract_chunked_resume_in_waiting(self):
        execute_scripted_runtime(
            _script_retract_chunked_resume_in_waiting,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_two_retracts_same_yield(self):
        execute_scripted_runtime(
            _script_two_retracts_same_yield,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_priority_preempt_release_invariant(self):
        execute_scripted_runtime(
            _script_priority_preempt_release_invariant,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )

    def test_retract_then_re_chunk(self):
        execute_scripted_runtime(
            _script_retract_then_re_chunk,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
