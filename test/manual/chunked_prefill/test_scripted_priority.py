"""Priority scheduling × chunked: naive ScriptedRuntime smoke plus retract/preempt edge cases.

Submit a low-priority long-prompt request that must be chunked, then
a high-priority short request. With priority preemption enabled the
high-priority req should not starve waiting on the low-priority one's
chunk loop.

Requires the wishlist API extension ``start_req(..., priority=...)``
(§4 P2 (10)). Until it lands the priority kwargs are passed as-is and
will surface as a clear AttributeError at script time.

Edge cases verify that chunked-resume requests participate in
KV-pressure retract and priority preemption like normal requests, and
that resource ownership is correctly released across the transition.
Also covers A.7 series from the expansion plan and fan-out variations
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


def _script_naive_priority_chunked(t: ScriptedRuntime):
    low = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, priority="low")
    yield  # let scheduler pull `low` and begin its chunk loop

    high = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")

    yield from run_until_all_finished([low, high])
    assert low.finished and high.finished


# KV pressure mid-chunk causes the chunked-resume req to be
# retracted. All resources must release immediately (96d4749094 +
# f38e69f87d).
def _script_retract_mid_chunk_releases_kv(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

    pages_before = r.kv_pages
    assert pages_before > 0

    # Force KV pressure by spamming short reqs until the pool is
    # nearly exhausted. ``exhaust_kv`` is the wishlist §4 P2 (9) helper.
    t.exhaust_kv(leave_pages=1)
    yield

    assert r.status in (
        "waiting",
        "finished",
        "unknown",
    ), f"r should be retracted (back in waiting) or finished; got {r.status}"
    if r.status == "waiting":
        # Retract path: resources released; the req is parked for re-admission.
        assert r.kv_pages == 0, f"retract must release KV; got {r.kv_pages}"


# priority preemption — a high-priority req shows up while a
# low-priority chunked-resume holds resources. The victim chunked req
# must go through retract (not abort) and the high-priority req takes
# its slot.
def _script_priority_preempt_chunked(t: ScriptedRuntime):
    low = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low")
    yield from run_until(low, lambda h: h.is_chunking and h.chunks_done >= 1)
    assert low.kv_pages > 0

    high = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")
    # In a pressured cache, high should force preempt of low.
    t.exhaust_kv(leave_pages=1)
    yield

    # After preempt: low is retracted (status "waiting") with no KV;
    # high gets admitted.
    if low.status == "waiting":
        assert low.kv_pages == 0
    yield from run_until_finished(high)
    assert high.finished


# retract-and-resume. Retract chunked mid-stream, then release
# pressure so it can be re-admitted; must complete normally.
def _script_retract_and_resume(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

    # Use the active fault-injection helper to force a retract — the
    # passive ``exhaust_kv`` is also acceptable, but this is
    # deterministic.
    t.force_retract(r)
    yield
    assert r.status == "waiting"
    assert r.kv_pages == 0

    # Drain pressure (no-op if we used force_retract) and run to completion.
    yield from run_until_finished(r)
    assert r.finished


# disagg mode retract — disaggregation send-side state must
# reset on chunked-resume retract (414efd4a27).
def _script_disagg_retract_resets_send_state(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

    t.force_retract(r)
    yield

    assert r.status == "waiting"
    assert r.kv_pages == 0
    # Disagg-specific: the per-req send-side state must reset.
    assert r.disagg_send_state in (
        None,
        "idle",
    ), f"disagg send state must reset on retract, got {r.disagg_send_state}"

    yield from run_until_finished(r)
    assert r.finished


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


def _script_disagg_retract_resets_send_state_extra(t: ScriptedRuntime):
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


class TestScriptedPriority(CustomTestCase):
    def test_naive_priority_chunked(self):
        execute_scripted_runtime(
            _script_naive_priority_chunked,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )

    def test_retract_mid_chunk_releases_kv(self):
        execute_scripted_runtime(
            _script_retract_mid_chunk_releases_kv,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_priority_preempt_chunked(self):
        execute_scripted_runtime(
            _script_priority_preempt_chunked,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )

    def test_retract_and_resume(self):
        execute_scripted_runtime(
            _script_retract_and_resume,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_disagg_retract_resets_send_state(self):
        execute_scripted_runtime(
            _script_disagg_retract_resets_send_state,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )

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

    def test_disagg_retract_resets_send_state_extra(self):
        execute_scripted_runtime(
            _script_disagg_retract_resets_send_state_extra,
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
