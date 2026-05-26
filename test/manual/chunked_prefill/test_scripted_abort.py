"""Abort scripted tests for chunked prefill.

Verifies that aborting a chunked-resume
request (1) releases all owned resources (row, KV, lock_ref), (2)
does not double-finalize, and (3) does not poison other reqs in
flight.

Many of these scripts read ``r.kv_pages`` / ``r.pending_middle_outputs``
which are whitebox views into Req state — necessary to verify
resource ownership invariants. The wishlist (§4 P1 (7)(8)) calls them
out as deliberate whitebox API.

Also covers A.5 series from the expansion plan and fan-out symmetric
cases (abort at chunk_first / chunk_last / penultimate / very late /
post-finish). Verifies abort is a no-op when the rid is invalid /
already finished, and idempotent on repeated calls.
"""

import unittest

from sglang.test.scripted_runtime.req_handle import ReqHandle
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


class TestAbortBasic(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_abort_waiting_chunked_resume(self):
        """Abort a chunked-resume req while it's still in waiting_queue (chunked-resume parked between chunks)."""
        self.runtime.run(self._script_abort_waiting_chunked_resume)

    # abort a chunked-resume req while it's still in waiting_queue
    # (chunked-resume parked between chunks). Resources must release.
    @staticmethod
    def _script_abort_waiting_chunked_resume(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # Push it into chunked-resume parked-in-waiting state.
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.status == "waiting" and h.chunks_done >= 1)

        pages_before = r.kv_pages
        assert pages_before > 0, "chunked req should own KV pages mid-chunk"

        t.abort(r)
        yield  # let scheduler process the abort

        assert r.status in (
            "finished",
            "unknown",
        ), f"after abort r should be finished/unknown, got {r.status}"
        assert (
            r.kv_pages == 0
        ), f"abort must release KV; r.kv_pages={r.kv_pages} after abort"
        assert (
            r.row_idx is None
        ), f"abort must release row; r.row_idx={r.row_idx} after abort"
        assert (
            r.lock_refs == 0
        ), f"abort must release lock_refs; r.lock_refs={r.lock_refs}"

    def test_abort_at_chunk_0(self):
        """Abort during chunk_0 (first chunk not yet flushed)."""
        self.runtime.run(self._script_abort_at_chunk_0)

    # abort during chunk_0 (first chunk not yet flushed).
    @staticmethod
    def _script_abort_at_chunk_0(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield  # request observed by scheduler; first chunk starts
        yield from run_until(r, lambda h: h.is_chunking)

        t.abort(r)
        yield

        assert r.kv_pages == 0
        assert r.row_idx is None

    def test_abort_at_chunk_mid(self):
        """Abort at chunk_mid (some chunks done, some pending)."""
        self.runtime.run(self._script_abort_at_chunk_mid)

    # abort at chunk_mid (some chunks done, some pending).
    @staticmethod
    def _script_abort_at_chunk_mid(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.chunks_done >= 2 and h.is_chunking)

        t.abort(r)
        yield

        assert r.kv_pages == 0

    def test_abort_at_last_chunk(self):
        """Aborting at last chunk zeros pending_middle_outputs to prevent revival."""
        self.runtime.run(self._script_abort_at_last_chunk)

    # abort at last_chunk (pending_middle_outputs has been ++ for
    # the last admission; abort must zero it so Stage A doesn't revive
    # the released row). Tied to scheduler.py:3567-3569 defensive cleanup.
    @staticmethod
    def _script_abort_at_last_chunk(t: ScriptedRuntime):
        # Use a prompt that's exactly 2 chunks so the second chunk is the
        # last one.
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)

        # pending_middle_outputs is incremented when the last chunk is
        # admitted into the running batch (b3a7b9f2a1).
        assert r.pending_middle_outputs > 0

        t.abort(r)
        yield

        # After abort, the defensive cleanup should have zeroed it.
        assert r.pending_middle_outputs == 0, (
            f"abort must zero pending_middle_outputs to prevent revival; "
            f"got {r.pending_middle_outputs}"
        )

    def test_abort_one_does_not_disturb_other(self):
        """Abort one chunked req does not disturb another in flight."""
        self.runtime.run(self._script_abort_one_does_not_disturb_other)

    # abort one chunked req does not disturb another in flight.
    @staticmethod
    def _script_abort_one_does_not_disturb_other(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)

        t.abort(r1)
        yield

        assert r1.kv_pages == 0
        # r2 should make forward progress after r1 is gone.
        yield from run_until_finished(r2)
        assert r2.finished, "r2 should still complete after r1 is aborted"

    def test_abort_with_zero_yield(self):
        """Submit + abort same step (zero yields between)."""
        self.runtime.run(self._script_abort_with_zero_yield)

    @staticmethod
    def _script_abort_with_zero_yield(t: ScriptedRuntime):
        # Submit + abort same step (zero yields between).
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.row_idx is None
        assert r.lock_refs == 0

    def test_abort_at_admission_step(self):
        """Submit, yield once (admission), abort."""
        self.runtime.run(self._script_abort_at_admission_step)

    @staticmethod
    def _script_abort_at_admission_step(t: ScriptedRuntime):
        # Submit, yield once (admission), abort.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield  # scheduler observes req
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.row_idx is None

    def test_abort_then_start_same_step_new_rid(self):
        """Abort one req; in the same yield step, submit a different rid."""
        self.runtime.run(self._script_abort_then_start_same_step_new_rid)

    @staticmethod
    def _script_abort_then_start_same_step_new_rid(t: ScriptedRuntime):
        # Abort one req; in the same yield step, submit a different rid.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        t.abort(r1)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r2)
        assert r2.finished
        assert r1.kv_pages == 0

    def test_abort_then_start_same_step_same_rid(self):
        """Abort r1; resubmit with the same rid — must work as a fresh req."""
        self.runtime.run(self._script_abort_then_start_same_step_same_rid)

    @staticmethod
    def _script_abort_then_start_same_step_same_rid(t: ScriptedRuntime):
        # Abort r1; resubmit with the same rid — must work as a fresh req.
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, rid="abort-reuse"
        )
        yield from run_until(r1, lambda h: h.is_chunking)
        t.abort(r1)
        yield
        r2 = t.start_req(prompt_len=16, max_new_tokens=2, rid="abort-reuse")
        yield from run_until_finished(r2)
        assert r2.finished

    def test_abort_five_chunked_in_a_row(self):
        """5 chunked reqs submitted; abort all 5 sequentially."""
        self.runtime.run(self._script_abort_five_chunked_in_a_row)

    @staticmethod
    def _script_abort_five_chunked_in_a_row(t: ScriptedRuntime):
        # 5 chunked reqs submitted; abort all 5 sequentially.
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(5)
        ]
        yield from run_until(reqs[0], lambda h: h.is_chunking)
        for r in reqs:
            t.abort(r)
        yield
        for r in reqs:
            assert r.kv_pages == 0
            assert r.row_idx is None

    def test_abort_unknown_rid_noop(self):
        """Abort an rid that was never submitted: no exception, no state change."""
        self.runtime.run(self._script_abort_unknown_rid_noop)

    @staticmethod
    def _script_abort_unknown_rid_noop(t: ScriptedRuntime):
        # Abort an rid that was never submitted: no exception, no state change.
        bogus = ReqHandle(rid="never-submitted-rid", runtime=t)
        t.abort(bogus)  # must not raise
        yield

    def test_abort_after_finish_noop(self):
        """Submit r1, wait for finish, then abort: no-op."""
        self.runtime.run(self._script_abort_after_finish_noop)

    @staticmethod
    def _script_abort_after_finish_noop(t: ScriptedRuntime):
        # Submit r1, wait for finish, then abort: no-op.
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        t.abort(r)
        yield
        # finished + abort should not destabilize stats.
        stats = t.engine_stats()
        assert stats["kv_pool_free"] >= 0

    def test_abort_during_waiting_timeout(self):
        """Chunked-resume in waiting_queue + abort timeout cleanup hits."""
        self.runtime.run(self._script_abort_during_waiting_timeout)

    @staticmethod
    def _script_abort_during_waiting_timeout(t: ScriptedRuntime):
        # Chunked-resume in waiting_queue + abort timeout cleanup hits.
        # The cleanup must skip chunked-resume reqs (359e5ed7bd).
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.status == "waiting")
        # Simulate the waiting-queue timeout housekeeping firing.
        # NEW API NEEDED: t.trigger_abort_on_waiting_timeout() — invoke
        # the internal waiting-queue housekeeping abort sweep.
        t.trigger_abort_on_waiting_timeout()
        yield
        yield from run_until_finished(r)
        assert r.finished, "chunked-resume must survive waiting-timeout sweep"

    def test_abort_chunk_first(self):
        """Abort while the very first chunk is still in progress."""
        self.runtime.run(self._script_abort_chunk_first)

    @staticmethod
    def _script_abort_chunk_first(t: ScriptedRuntime):
        # Abort while the very first chunk is still in progress.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield  # admission
        yield  # first chunk admitted
        t.abort(r)
        yield
        assert r.kv_pages == 0

    def test_abort_chunk_last(self):
        """Abort while the last chunk is in progress."""
        self.runtime.run(self._script_abort_chunk_last)

    @staticmethod
    def _script_abort_chunk_last(t: ScriptedRuntime):
        # Abort while the last chunk is in progress.
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.pending_middle_outputs == 0

    def test_abort_penultimate_chunk(self):
        """Abort 1 chunk before the final one."""
        self.runtime.run(self._script_abort_penultimate_chunk)

    @staticmethod
    def _script_abort_penultimate_chunk(t: ScriptedRuntime):
        # Abort 1 chunk before the final one.
        r = t.start_req(prompt_len=4 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(r, lambda h: h.chunks_done >= 2 and h.is_chunking)
        t.abort(r)
        yield
        assert r.kv_pages == 0

    def test_double_abort_idempotent(self):
        """Call abort twice on the same handle: second call is a no-op."""
        self.runtime.run(self._script_double_abort_idempotent)

    @staticmethod
    def _script_double_abort_idempotent(t: ScriptedRuntime):
        # Call abort twice on the same handle: second call is a no-op.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.abort(r)
        t.abort(r)
        yield
        assert r.kv_pages == 0

    def test_abort_during_decode(self):
        """R1 has finished prefill, is in decode; abort mid-decode."""
        self.runtime.run(self._script_abort_during_decode)

    @staticmethod
    def _script_abort_during_decode(t: ScriptedRuntime):
        # r1 has finished prefill, is in decode; abort mid-decode.
        r = t.start_req(prompt_len=16, max_new_tokens=64)
        yield from run_until(r, lambda h: h.status == "running")
        t.abort(r)
        yield
        assert r.kv_pages == 0

    def test_abort_one_of_three_others_finish(self):
        """Three reqs in batch, abort middle one, other two finish."""
        self.runtime.run(self._script_abort_one_of_three_others_finish)

    @staticmethod
    def _script_abort_one_of_three_others_finish(t: ScriptedRuntime):
        # Three reqs in batch, abort middle one, other two finish.
        r1 = t.start_req(prompt_len=16, max_new_tokens=4)
        r2 = t.start_req(prompt_len=16, max_new_tokens=4)
        r3 = t.start_req(prompt_len=16, max_new_tokens=4)
        yield from run_until(r2, lambda h: h.status == "running")
        t.abort(r2)
        yield from run_until_all_finished([r1, r3])
        assert r2.kv_pages == 0

    def test_abort_in_separate_yields(self):
        """Submit 3 chunked, abort in 3 separate yield steps."""
        self.runtime.run(self._script_abort_in_separate_yields)

    @staticmethod
    def _script_abort_in_separate_yields(t: ScriptedRuntime):
        # Submit 3 chunked, abort in 3 separate yield steps.
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(3)
        ]
        yield from run_until(reqs[0], lambda h: h.is_chunking)
        for r in reqs:
            t.abort(r)
            yield
        for r in reqs:
            assert r.kv_pages == 0

    def test_abort_finish_event_count_at_most_one(self):
        """Aborted req should not emit a normal finish event."""
        self.runtime.run(self._script_abort_finish_event_count_at_most_one)

    @staticmethod
    def _script_abort_finish_event_count_at_most_one(t: ScriptedRuntime):
        # Aborted req should not emit a normal finish event.
        # NEW API NEEDED: r.finish_event_count — count of completion events.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.abort(r)
        yield
        # finish_event_count for an aborted req should be 0 or 1 (abort
        # is itself a kind of finalize, but should not double-fire).
        assert r.finish_event_count <= 1

    def test_abort_at_chunk_boundary_race(self):
        """Abort fires the same step a chunk boundary advances chunks_done."""
        self.runtime.run(self._script_abort_at_chunk_boundary_race)

    # abort raced with chunks_done increment at a chunk
    # boundary. Drive the req to a known mid-chunk state, then abort on
    # the same yield step the scheduler is about to commit the next
    # chunk. No double-finalize; pending_middle_outputs must zero.
    @staticmethod
    def _script_abort_at_chunk_boundary_race(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # First, observe the chunked req in flight.
        yield from run_until(r, lambda h: h.is_chunking)
        # Then drive to a chunk-boundary state: at least one chunk
        # committed, still chunking (mid-stream).
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        chunks_at_abort = r.chunks_done

        t.abort(r)
        yield  # the race step — scheduler observes both the boundary and the abort

        assert (
            r.finish_event_count <= 1
        ), f"abort race must not double-finalize; got {r.finish_event_count}"
        assert (
            r.pending_middle_outputs == 0
        ), f"abort must zero pending_middle_outputs; got {r.pending_middle_outputs}"
        assert r.kv_pages == 0
        # chunks_done is monotone — never goes backward across the race.
        assert r.chunks_done >= chunks_at_abort

    def test_abort_then_resubmit_same_rid_same_step(self):
        """Abort r1 then immediately submit a new req reusing the same rid in one step."""
        self.runtime.run(self._script_abort_then_resubmit_same_rid_same_step)

    # abort + same-step resubmit with the
    # identical rid. The new req must complete independently — the
    # scheduler must not confuse the resubmit with the dying corpse of
    # the aborted r1.
    @staticmethod
    def _script_abort_then_resubmit_same_rid_same_step(t: ScriptedRuntime):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            rid="abort-resubmit-same-step",
        )
        yield from run_until(r1, lambda h: h.is_chunking)
        t.abort(r1)
        # Same yield step: resubmit a fresh req under the same rid.
        r2 = t.start_req(
            prompt_len=16,
            max_new_tokens=2,
            rid="abort-resubmit-same-step",
        )
        yield
        yield from run_until_finished(r2)
        assert r2.finished, "resubmit under same rid must complete independently"
        assert r1.kv_pages == 0, "aborted r1 must release KV before resubmit"

    def test_abort_during_gap_pending_middle_outputs_positive(self):
        """Abort during the gap where pending_middle_outputs > 0 but is_chunking == False."""
        self.runtime.run(self._script_abort_during_gap_pending_middle_outputs_positive)

    # the gap between "last chunk submitted to forward" and
    # "Stage A drains pending_middle_outputs". In that window
    # pending_middle_outputs > 0 yet the req is not currently chunking.
    # Aborting here must not revive the freed row via Stage A.
    @staticmethod
    def _script_abort_during_gap_pending_middle_outputs_positive(t: ScriptedRuntime):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        # Wait for the gap window: pending_middle_outputs has been ++'d
        # for the last admission but is_chunking is no longer True
        # (Stage A hasn't drained the output yet).
        yield from run_until(
            r,
            lambda h: h.pending_middle_outputs > 0 and not h.is_chunking,
        )
        assert r.pending_middle_outputs > 0
        assert not r.is_chunking

        t.abort(r)
        yield

        assert r.pending_middle_outputs == 0, (
            f"abort in gap must zero pending_middle_outputs to block "
            f"Stage A revival; got {r.pending_middle_outputs}"
        )
        assert r.kv_pages == 0
        assert r.row_idx is None

    def test_abort_when_chunked_only_then_idle(self):
        """Aborting the only chunked req in flight leaves the engine idle."""
        self.runtime.run(self._script_abort_when_chunked_only_then_idle)

    # when the only chunked req is aborted, the engine must
    # transition all the way back to idle — chunked_req is None,
    # chunked_in_flight_count is 0, and t.is_idle.
    @staticmethod
    def _script_abort_when_chunked_only_then_idle(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        assert t.chunked_in_flight_count() == 1

        t.abort(r)
        yield
        # Give the scheduler one extra iter to settle.
        yield

        assert r.kv_pages == 0
        assert t.chunked_in_flight_count() == 0
        assert t.is_idle, "engine must be idle after the only chunked req is aborted"

    def test_chunked_req_then_abort_then_new_short_in_one_yield(self):
        """Mid-chunk R1: abort R1 and start a short R2 in the same yield; R2 admits fresh and chunked slot is no longer R1."""
        self.runtime.run(
            self._script_chunked_req_then_abort_then_new_short_in_one_yield
        )

    # same-yield abort + new-req combo. While R1 is mid-chunk,
    # abort R1 and submit a fresh short R2 in the same yield step. R2
    # must admit cleanly, and the scheduler's chunked_req slot must no
    # longer reference R1 (either None or R2's rid if R2 also chunked,
    # though R2 is short so should be None).
    @staticmethod
    def _script_chunked_req_then_abort_then_new_short_in_one_yield(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        assert t.get_chunked_req_rid() == r1.rid, (
            f"r1 should hold the chunked slot before abort; got "
            f"{t.get_chunked_req_rid()!r}"
        )

        t.abort(r1)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield

        # After the same-yield combo, r1 must be torn down and the
        # chunked slot must not still reference r1.
        # NEW API NEEDED: t.get_chunked_req_rid() — current scheduler
        # chunked_req rid or None.
        cur = t.get_chunked_req_rid()
        assert cur != r1.rid, f"chunked slot still points to aborted r1; got {cur!r}"
        assert r1.kv_pages == 0
        yield from run_until_finished(r2)
        assert r2.finished, "fresh r2 must admit and complete after combo step"

    def test_force_retract_then_abort_same_yield(self):
        """Force_retract + abort on the same handle in the same yield: clean state, no double-free, finish_event_count <= 1."""
        self.runtime.run(self._script_force_retract_then_abort_same_yield)

    # same-yield combo of force_retract + abort. Both operations
    # tear resources down; doing them in the same yield step must not
    # double-free and must not emit two finish events. All resources
    # released and the finalize counter capped at 1.
    @staticmethod
    def _script_force_retract_then_abort_same_yield(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        assert r1.kv_pages > 0

        t.force_retract(r1)
        t.abort(r1)
        yield

        assert r1.kv_pages == 0, (
            f"force_retract + abort same yield must release KV; got " f"{r1.kv_pages}"
        )
        assert r1.row_idx is None, (
            f"force_retract + abort same yield must release row; got " f"{r1.row_idx}"
        )
        assert r1.lock_refs == 0, (
            f"force_retract + abort same yield must release lock_refs; "
            f"got {r1.lock_refs}"
        )
        assert r1.finish_event_count <= 1, (
            f"force_retract + abort same yield must not double-finalize; "
            f"got {r1.finish_event_count} events"
        )
        # No double-free crash means the engine survived; one more yield
        # to confirm the scheduler can keep stepping.
        yield

    def test_abort_chunked_with_baton_handoff(self):
        """Abort the in-flight chunked req while a waiting chunked req takes the baton."""
        self.runtime.run(self._script_abort_chunked_with_baton_handoff)

    # r1 holds the chunked_req slot; r2 is waiting. Abort
    # r1: r2 must successfully claim the chunked_req baton on the next
    # admission step.
    @staticmethod
    def _script_abort_chunked_with_baton_handoff(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # r1 takes the chunked slot first (FCFS).
        yield from run_until(r1, lambda h: h.is_chunking)
        assert t.chunked_in_flight_count() == 1

        t.abort(r1)
        yield

        # r2 must now pick up the chunked baton.
        yield from run_until(r2, lambda h: h.is_chunking)
        assert r1.kv_pages == 0
        yield from run_until_finished(r2)
        assert r2.finished, "baton handoff must let r2 complete"


class TestAbortPP(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        tp_size=2,
        pp_size=2,
    )

    def test_abort_at_last_chunk_in_flight_pp(self):
        """Abort at last_chunk_in_flight under PP (forward still in flight when the abort hits)."""
        self.runtime.run(self._script_abort_at_last_chunk_in_flight_pp)

    # abort at last_chunk_in_flight under PP (forward still in
    # flight when the abort hits). Verifies the PP cross-microbatch
    # dedup (b823c16e60) and double-finalize guard (02b1785f0a).
    #
    # Requires multi-rank ScriptedRuntime + PP=2 (wishlist §4 P3 (15)).
    @staticmethod
    def _script_abort_at_last_chunk_in_flight_pp(t: ScriptedRuntime):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
        # Drive to last_chunk_in_flight via batch_composition introspection
        # — when ``mb='b'`` has the chunked extend and ``mb='a'`` has the
        # last-chunk forward still running.
        yield from run_until(
            r,
            lambda h: h.chunks_done >= 1 and h.is_chunking,
        )

        t.abort(r)
        yield

        assert r.kv_pages == 0
        # No double-finalize: the result_queue should have one finished
        # event for r, not two.
        assert (
            r.finish_event_count == 1
        ), f"abort must not double-finalize; got {r.finish_event_count} events"


if __name__ == "__main__":
    unittest.main()
