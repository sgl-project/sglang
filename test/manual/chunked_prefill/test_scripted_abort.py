import unittest

from sglang.srt.environ import envs
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    SMALL_KV_POOL_BALLAST_MAX_NEW_TOKENS,
    SMALL_KV_POOL_BALLAST_PROMPT_LEN,
    SMALL_KV_POOL_MAX_TOTAL_TOKENS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


def _drain_until_released(t: ScriptedContext, *handles: ScriptedReqHandle):
    # Under the overlap scheduler an abort is injected at the top of the next
    # get_next_batch_to_run, but the actual KV/row/lock release lands a couple of
    # forward steps later -- the in-flight forward's result must drain first. Step
    # the loop until every aborted handle is fully released (KV pages, req-pool row,
    # AND the last_node lock ref, which can linger one extra step past the row)
    # instead of asserting after a single yield.
    for _ in range(12):
        if all(
            h.kv_pages == 0
            and h.lock_refs == 0
            and (h.req is None or h.req.req_pool_idx is None)
            for h in handles
        ):
            return
        yield


class TestAbortBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_abort_waiting_chunked_resume(self):
        self.server.execute_script(self._script_abort_waiting_chunked_resume)

    @staticmethod
    def _script_abort_waiting_chunked_resume(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=100
        )
        yield from run_until(r, lambda h: h.is_chunking)

        pages_before = r.kv_pages
        assert pages_before > 0, "chunked req should own KV pages mid-chunk"

        t.abort(r)
        yield from _drain_until_released(t, r)

        assert r.status in (
            "finished",
            "unknown",
        ), f"after abort r should be finished/unknown, got {r.status}"
        assert (
            r.kv_pages == 0
        ), f"abort must release KV; r.kv_pages={r.kv_pages} after abort"
        assert (
            r.req is None or r.req.req_pool_idx is None
        ), f"abort must release row; r.req={r.req} after abort"
        assert (
            r.lock_refs == 0
        ), f"abort must release lock_refs; r.lock_refs={r.lock_refs}"

    def test_abort_at_chunk_0(self):
        self.server.execute_script(self._script_abort_at_chunk_0)

    @staticmethod
    def _script_abort_at_chunk_0(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=110
        )
        yield
        yield from run_until(r, lambda h: h.is_chunking)

        t.abort(r)
        yield from _drain_until_released(t, r)

        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None

    def test_abort_at_chunk_mid(self):
        self.server.execute_script(self._script_abort_at_chunk_mid)

    @staticmethod
    def _script_abort_at_chunk_mid(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=120
        )
        yield from run_until(r, lambda h: h.chunks_done >= 2 and h.is_chunking)

        t.abort(r)
        yield from _drain_until_released(t, r)

        assert r.kv_pages == 0

    def test_abort_one_does_not_disturb_other(self):
        self.server.execute_script(self._script_abort_one_does_not_disturb_other)

    @staticmethod
    def _script_abort_one_does_not_disturb_other(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=130
        )
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=131
        )
        yield from run_until(r1, lambda h: h.is_chunking)

        t.abort(r1)
        yield from _drain_until_released(t, r1)

        assert r1.kv_pages == 0
        yield from run_until_finished(r2)
        assert r2.finished, "r2 should still complete after r1 is aborted"

    def test_abort_with_zero_yield(self):
        self.server.execute_script(self._script_abort_with_zero_yield)

    @staticmethod
    def _script_abort_with_zero_yield(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        t.abort(r)
        yield from _drain_until_released(t, r)
        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None
        assert r.lock_refs == 0

    def test_abort_at_admission_step(self):
        self.server.execute_script(self._script_abort_at_admission_step)

    @staticmethod
    def _script_abort_at_admission_step(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield
        t.abort(r)
        yield from _drain_until_released(t, r)
        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None

    def test_abort_then_start_same_step_new_rid(self):
        self.server.execute_script(self._script_abort_then_start_same_step_new_rid)

    @staticmethod
    def _script_abort_then_start_same_step_new_rid(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=140
        )
        yield from run_until(r1, lambda h: h.is_chunking)
        t.abort(r1)
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=141
        )
        yield from run_until_finished(r2)
        assert r2.finished
        assert r1.kv_pages == 0

    def test_abort_then_start_same_step_same_rid(self):
        self.server.execute_script(self._script_abort_then_start_same_step_same_rid)

    @staticmethod
    def _script_abort_then_start_same_step_same_rid(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            rid="abort-reuse",
            prompt_token=150,
        )
        yield from run_until(r1, lambda h: h.is_chunking)
        t.abort(r1)
        yield
        r2 = t.start_req(prompt_len=16, max_new_tokens=2, rid="abort-reuse")
        yield from run_until_finished(r2)
        assert r2.finished

    def test_abort_five_chunked_in_a_row(self):
        self.server.execute_script(self._script_abort_five_chunked_in_a_row)

    @staticmethod
    def _script_abort_five_chunked_in_a_row(t: ScriptedContext):
        reqs = [
            t.start_req(
                prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=160 + i
            )
            for i in range(5)
        ]
        yield from run_until(reqs[0], lambda h: h.is_chunking)
        for r in reqs:
            t.abort(r)
        yield from _drain_until_released(t, *reqs)
        for r in reqs:
            assert r.kv_pages == 0
            assert r.req is None or r.req.req_pool_idx is None

    def test_abort_unknown_rid_noop(self):
        self.server.execute_script(self._script_abort_unknown_rid_noop)

    @staticmethod
    def _script_abort_unknown_rid_noop(t: ScriptedContext):
        bogus = ScriptedReqHandle(rid="never-submitted-rid", context=t)
        # Unknown rid: TokenizerManager drops the abort without forwarding it to
        # the scheduler, so no AbortReq transits the recv socket; fire-and-forget.
        t.abort(bogus, await_arrival=False)
        yield
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        # The finished req's KV/row/lock release lands a couple of steps after the
        # finished flag flips under overlap; drain before asserting full release.
        for _ in range(12):
            if r.kv_pages == 0 and r.lock_refs == 0:
                break
            yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_abort_after_finish_noop(self):
        self.server.execute_script(self._script_abort_after_finish_noop)

    @staticmethod
    def _script_abort_after_finish_noop(t: ScriptedContext):
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished

        # run_until_finished returns the step `finished` flips; the finished req's
        # KV release and any lazy radix bookkeeping land a couple of forward steps
        # later. Drain to a fully-idle, stable pool before asserting full release
        # and before snapshotting the baseline, so the abort-is-noop comparison is
        # not corrupted by background release that was still in flight.
        for _ in range(12):
            if t.is_fully_idle:
                break
            yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        kv_pool_free_before = t.engine_stats()["kv_pool_free"]

        # The req is already finished, so TokenizerManager drops the abort without
        # forwarding it to the scheduler; fire-and-forget (no AbortReq ever
        # transits the recv socket to await).
        t.abort(r, await_arrival=False)
        yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        # at-most-one finish is enforced by the engine (output_streamer: assert not req.finished_output)
        kv_pool_free_after = t.engine_stats()["kv_pool_free"]
        assert kv_pool_free_after == kv_pool_free_before, (
            f"abort-after-finish must not move KV pool; "
            f"before={kv_pool_free_before} after={kv_pool_free_after}"
        )

    def test_abort_chunk_last(self):
        self.server.execute_script(self._script_abort_chunk_last)

    @staticmethod
    def _script_abort_chunk_last(t: ScriptedContext):
        r = t.start_req(
            prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4, prompt_token=170
        )
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        t.abort(r)
        yield from _drain_until_released(t, r)
        assert r.kv_pages == 0
        assert r.req is None or r.req.inflight_middle_chunks == 0

    def test_abort_penultimate_chunk(self):
        self.server.execute_script(self._script_abort_penultimate_chunk)

    @staticmethod
    def _script_abort_penultimate_chunk(t: ScriptedContext):
        r = t.start_req(
            prompt_len=4 * DEFAULT_CHUNK_SIZE, max_new_tokens=2, prompt_token=180
        )
        yield from run_until(r, lambda h: h.chunks_done >= 2 and h.is_chunking)
        t.abort(r)
        yield from _drain_until_released(t, r)
        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None
        assert r.lock_refs == 0

    def test_double_abort_idempotent(self):
        self.server.execute_script(self._script_double_abort_idempotent)

    @staticmethod
    def _script_double_abort_idempotent(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=190
        )
        yield from run_until(r, lambda h: h.is_chunking)
        t.abort(r)
        t.abort(r)
        yield from _drain_until_released(t, r)
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        # at-most-one finish is enforced by the engine (output_streamer: assert not req.finished_output)

    def test_abort_during_decode(self):
        self.server.execute_script(self._script_abort_during_decode)

    @staticmethod
    def _script_abort_during_decode(t: ScriptedContext):
        r = t.start_req(prompt_len=16, max_new_tokens=64)
        yield from run_until(r, lambda h: h.status == "running")
        assert r.kv_pages > 0, "decode req must own KV before abort"
        t.abort(r)
        yield from _drain_until_released(t, r)
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        # at-most-one finish is enforced by the engine (output_streamer: assert not req.finished_output)

    def test_abort_one_of_three_others_finish(self):
        self.server.execute_script(self._script_abort_one_of_three_others_finish)

    @staticmethod
    def _script_abort_one_of_three_others_finish(t: ScriptedContext):
        r1 = t.start_req(prompt_len=16, max_new_tokens=4)
        r2 = t.start_req(prompt_len=16, max_new_tokens=4)
        r3 = t.start_req(prompt_len=16, max_new_tokens=4)
        yield from run_until(r2, lambda h: h.status == "running")
        t.abort(r2)
        yield from run_until_all_finished([r1, r3])
        assert r2.kv_pages == 0

    def test_abort_in_separate_yields(self):
        self.server.execute_script(self._script_abort_in_separate_yields)

    @staticmethod
    def _script_abort_in_separate_yields(t: ScriptedContext):
        reqs = [
            t.start_req(
                prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=200 + i
            )
            for i in range(3)
        ]
        yield from run_until(reqs[0], lambda h: h.is_chunking)
        for r in reqs:
            t.abort(r)
            yield
        yield from _drain_until_released(t, *reqs)
        for r in reqs:
            assert r.kv_pages == 0

    def test_abort_at_chunk_boundary_race(self):
        self.server.execute_script(self._script_abort_at_chunk_boundary_race)

    @staticmethod
    def _script_abort_at_chunk_boundary_race(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=210
        )
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)

        t.abort(r)
        yield from _drain_until_released(t, r)

        # at-most-one finish is enforced by the engine (output_streamer: assert not req.finished_output)
        assert r.kv_pages == 0

        # abort at the chunk boundary must not revive: subsequent steps produce
        # no new chunk and the req does not reappear as a live chunked req.
        chunks_after_abort = r.chunks_done
        yield
        yield
        assert not r.is_chunking, "aborted req must not resume chunking"
        assert r.chunks_done == chunks_after_abort, (
            f"aborted req revived and ran another chunk; "
            f"chunks_done went {chunks_after_abort} -> {r.chunks_done}"
        )
        assert r.req is None or r.req.req_pool_idx is None

    def test_abort_mid_chunk_no_extra_radix_node(self):
        self.server.execute_script(self._script_abort_mid_chunk_no_extra_radix_node)

    @staticmethod
    def _script_abort_mid_chunk_no_extra_radix_node(
        t: ScriptedContext,
    ):
        """Aborting a chunked req mid-chunk leaves no locked radix node and does not revive it."""
        # NOTE: this intentionally does NOT exercise the skipped-stash branch
        # (`if self._chunked_req_scheduled_last_iter: stash_chunked_request(...)`
        # in get_next_batch_to_run). On a non-SWA engine that flag is only False
        # when chunked_req is None, so a held chunked_req is always stashed; the
        # "chunked req parked in waiting_queue with the flag False" inter-chunk gap
        # is hybrid-SWA-specific and unreachable here.
        #
        # The reachable, abort-specific invariant is NOT "no new radix node": a
        # held chunked_req is stashed via cache_unfinished_req at the top of every
        # get_next_batch_to_run (stash_chunked_request), so its partial prefix is
        # legitimately cached and lock-protected WHILE it chunks -- that caching is
        # normal prefill behavior, not an abort artifact, and one more such stash
        # can land between the abort being injected and taking effect. What the
        # abort must guarantee is that once the req is released it leaves NO radix
        # node still LOCKED on its behalf (the chunk's protective lock is dropped),
        # and that it does not revive.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=220
        )
        yield from run_until(r, lambda h: h.is_chunking)

        t.abort(r)
        yield from _drain_until_released(t, r)

        # The abort releases the req and does not revive it. The abort lands
        # when the req exits the chunked state, so chunks_done may legitimately
        # advance between injection and landing; once RELEASED it must freeze.
        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None
        chunks_after_release = r.chunks_done
        for _ in range(4):
            yield
        assert r.chunks_done == chunks_after_release, (
            f"aborted mid-chunk req revived after release; chunks_done went "
            f"{chunks_after_release} -> {r.chunks_done}"
        )

        # No radix node remains locked once the aborted chunked req is released:
        # the protective lock the in-flight chunk held while stashed is dropped.
        lock_refs_after = t.get_all_node_lock_refs()
        assert all(ref == 0 for ref in lock_refs_after.values()), (
            f"abort mid-chunk left a locked radix node behind; "
            f"node lock_refs={lock_refs_after!r}"
        )

    def test_abort_then_resubmit_same_rid_same_step(self):
        self.server.execute_script(self._script_abort_then_resubmit_same_rid_same_step)

    @staticmethod
    def _script_abort_then_resubmit_same_rid_same_step(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            rid="abort-resubmit-same-step",
            prompt_token=230,
        )
        yield from run_until(r1, lambda h: h.is_chunking)
        t.abort(r1)
        r2 = t.start_req(
            prompt_len=16,
            max_new_tokens=2,
            rid="abort-resubmit-same-step",
        )
        yield
        yield from run_until_finished(r2)
        assert r2.finished, "resubmit under same rid must complete independently"
        assert r1.kv_pages == 0, "aborted r1 must release KV before resubmit"
        assert r1.req is None or r1.req.req_pool_idx is None
        assert r1.lock_refs == 0

    def test_abort_during_gap_inflight_middle_chunks_positive(self):
        self.server.execute_script(
            self._script_abort_during_gap_inflight_middle_chunks_positive
        )

    @staticmethod
    def _script_abort_during_gap_inflight_middle_chunks_positive(t: ScriptedContext):
        # The "gap" state (inflight_middle_chunks > 0 AND not is_chunking) is
        # unreachable on v1: the in-flight latch bumps and clears in the same
        # transition as the chunked_req slot. Abort while the req is mid-flight in a
        # middle chunk -- inflight_middle_chunks > 0 holds during chunking once at
        # least one chunk has run.
        r = t.start_req(
            prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2, prompt_token=240
        )
        yield from run_until(
            r,
            lambda h: h.is_chunking and h.chunks_done >= 1,
        )
        assert r.req.inflight_middle_chunks > 0

        t.abort(r)
        yield from _drain_until_released(t, r)

        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None

        # abort during the gap must block Stage A revival: the req stays out of
        # the chunked slot across subsequent steps.
        assert not r.is_chunking, "aborted gap req must not re-enter chunking"
        yield
        assert not r.is_chunking, "aborted gap req must stay out of chunking"

        if r.req is not None:
            assert (
                r.req.inflight_middle_chunks == 0
            ), f"inflight_middle_chunks not cleared; got {r.req.inflight_middle_chunks}"

    def test_abort_when_chunked_only_then_idle(self):
        self.server.execute_script(self._script_abort_when_chunked_only_then_idle)

    @staticmethod
    def _script_abort_when_chunked_only_then_idle(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=250
        )
        yield from run_until(r, lambda h: h.is_chunking)
        assert (1 if t.scheduler.chunked_req is not None else 0) == 1

        t.abort(r)
        yield from _drain_until_released(t, r)

        # The handle's KV/row/lock are released by the drain above, but the
        # scheduler-wide idle state (chunked slot cleared, waiting_queue and
        # running_batch drained) can settle one extra step later under overlap.
        # Step until idle instead of asserting immediately after release.
        for _ in range(12):
            if t.scheduler.chunked_req is None and t.is_idle:
                break
            yield

        assert r.kv_pages == 0
        assert (1 if t.scheduler.chunked_req is not None else 0) == 0
        assert t.is_idle, "engine must be idle after the only chunked req is aborted"

    def test_chunked_req_then_abort_then_new_short_in_one_yield(self):
        self.server.execute_script(
            self._script_chunked_req_then_abort_then_new_short_in_one_yield
        )

    @staticmethod
    def _script_chunked_req_then_abort_then_new_short_in_one_yield(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=260
        )
        yield from run_until(r1, lambda h: h.is_chunking)
        assert (
            t.scheduler.chunked_req.rid if t.scheduler.chunked_req is not None else None
        ) == r1.rid, (
            f"r1 should hold the chunked slot before abort; got "
            f"{(t.scheduler.chunked_req.rid if t.scheduler.chunked_req is not None else None)!r}"
        )

        t.abort(r1)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from _drain_until_released(t, r1)

        cur = (
            t.scheduler.chunked_req.rid if t.scheduler.chunked_req is not None else None
        )
        assert cur != r1.rid, f"chunked slot still points to aborted r1; got {cur!r}"
        assert r1.kv_pages == 0
        yield from run_until_finished(r2)
        assert r2.finished, "fresh r2 must admit and complete after combo step"

    def test_force_retract_then_abort_same_yield(self):
        self.server.execute_script(self._script_force_retract_then_abort_same_yield)

    @staticmethod
    def _script_force_retract_then_abort_same_yield(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=270
        )
        yield from run_until(r1, lambda h: h.is_chunking)
        assert r1.kv_pages > 0

        t.pause_generation(mode="retract")
        t.abort(r1)
        yield from _drain_until_released(t, r1)

        assert r1.kv_pages == 0, (
            f"force_retract + abort same yield must release KV; got " f"{r1.kv_pages}"
        )
        assert r1.req is None or r1.req.req_pool_idx is None, (
            f"force_retract + abort same yield must release row; got " f"{r1.req}"
        )
        assert r1.lock_refs == 0, (
            f"force_retract + abort same yield must release lock_refs; "
            f"got {r1.lock_refs}"
        )
        # at-most-one finish is enforced by the engine (output_streamer: assert not req.finished_output)
        yield
        t.continue_generation()

    def test_abort_chunked_with_baton_handoff(self):
        self.server.execute_script(self._script_abort_chunked_with_baton_handoff)

    @staticmethod
    def _script_abort_chunked_with_baton_handoff(t: ScriptedContext):
        # Distinct prompt_token so r2 does not hit r1's partially-cached prefix
        # after r1 is aborted -- it must re-chunk from scratch to take the baton.
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=280
        )
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=281
        )
        yield from run_until(r1, lambda h: h.is_chunking)
        assert (1 if t.scheduler.chunked_req is not None else 0) == 1

        t.abort(r1)
        yield from _drain_until_released(t, r1)

        yield from run_until(r2, lambda h: h.is_chunking)
        assert r1.kv_pages == 0
        assert r1.req is None or r1.req.req_pool_idx is None
        assert r1.lock_refs == 0
        yield from run_until_finished(r2)
        assert r2.finished, "baton handoff must let r2 complete"
        assert r2.lock_refs == 0


class TestAbortPP(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        pp_size=2,
    )

    def test_abort_at_last_chunk_in_flight_pp(self):
        self.server.execute_script(self._script_abort_at_last_chunk_in_flight_pp)

    @staticmethod
    def _script_abort_at_last_chunk_in_flight_pp(t: ScriptedContext):
        r = t.start_req(
            prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4, prompt_token=290
        )
        yield from run_until(
            r,
            lambda h: h.chunks_done >= 1 and h.is_chunking,
        )

        t.abort(r)
        yield from _drain_until_released(t, r)

        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None
        assert r.lock_refs == 0
        assert r.finished


class TestAbortSmallPool(ScriptedTestCase):
    # A capped KV pool lets two real ballast decode reqs create genuine,
    # engine-resolvable admission pressure; raw exhaust_kv pages have no backing
    # Req and repeatedly wedged the engine (unkillable teardown hangs).
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        max_total_tokens=SMALL_KV_POOL_MAX_TOTAL_TOKENS,
    )

    def test_waiting_timeout_sweep_aborts_pressured_waiting_req(self):
        self.server.execute_script(
            self._script_waiting_timeout_sweep_aborts_pressured_waiting_req
        )

    @staticmethod
    def _script_waiting_timeout_sweep_aborts_pressured_waiting_req(t: ScriptedContext):
        # NOTE: source-true behavior of the waiting-timeout sweep. The scheduler
        # runs _abort_on_waiting_timeout() at the top of get_next_batch_to_run on
        # every NON-paused iteration (only when SGLANG_REQ_WAITING_TIMEOUT > 0).
        # It scans the WHOLE waiting_queue and aborts every req whose
        # wait_queue_entry_time is older than the deadline -- there is NO
        # chunked-resume immunity guard, so a parked chunked-resume req (which
        # also lives in waiting_queue carrying its original entry_time) is swept
        # out exactly like any other waiting req. This test drives that sweep
        # through the REAL event loop -- no pause, no scheduler-private call -- by
        # holding a req in waiting_queue under real KV pressure and letting the
        # loop's own sweep abort it.
        #
        # The timeout-abort path only notifies the tokenizer and drops the req
        # from waiting_queue; it does NOT itself release KV/row (the req under
        # test never held any, since it never got admitted). So the observable
        # consequence asserted here is removal from waiting_queue + aborted status.

        # Real admission pressure, engine-native: two long-lived ballast decode
        # reqs fill the small pool's rem_total (held tokens + clipped decode
        # reservations), so the 16-token newcomer cannot admit -- yet the loop
        # keeps running get_next_batch_to_run (and thus the sweep) every step.
        b1 = t.start_req(
            prompt_len=SMALL_KV_POOL_BALLAST_PROMPT_LEN,
            max_new_tokens=SMALL_KV_POOL_BALLAST_MAX_NEW_TOKENS,
            ignore_eos=True,
            prompt_token=300,
        )
        b2 = t.start_req(
            prompt_len=SMALL_KV_POOL_BALLAST_PROMPT_LEN,
            max_new_tokens=SMALL_KV_POOL_BALLAST_MAX_NEW_TOKENS,
            ignore_eos=True,
            prompt_token=301,
        )
        yield from run_until(b1, lambda h: h.status == "running")
        yield from run_until(b2, lambda h: h.status == "running")

        r = t.start_req(prompt_len=16, max_new_tokens=2)

        def waiting_rids():
            return {req.rid for req in t.scheduler.waiting_queue}

        # The req parks in waiting_queue with a real wait_queue_entry_time set by
        # _add_request_to_queue; it cannot be admitted because there is no free KV.
        yield from run_until(r, lambda h: r.rid in waiting_rids())
        assert r.kv_pages == 0, "pressured waiting req must not own KV before admission"

        # Enable the timeout with a value tiny enough that, after the req has been
        # queued and a couple of real-time yields pass, entry_time is already past
        # perf_counter() - timeout_s. The override stays active across the yields
        # below (the script frame is suspended inside the with-block), so the
        # scheduler reads it when it runs the sweep at the top of
        # get_next_batch_to_run on each non-paused iteration.
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1e-6):
            # Drive the REAL loop until the scheduler's own sweep removes the req
            # from waiting_queue.
            for _ in range(DEFAULT_MAX_STEPS):
                if r.rid not in waiting_rids():
                    break
                yield
            else:
                raise AssertionError(
                    f"waiting-timeout sweep never removed the req from "
                    f"waiting_queue after {DEFAULT_MAX_STEPS} steps; "
                    f"waiting_rids={waiting_rids()!r}"
                )

        assert r.rid not in waiting_rids(), (
            f"the loop's waiting-timeout sweep must drop the timed-out waiting "
            f"req from waiting_queue; got {waiting_rids()!r}"
        )
        assert r.status in ("finished", "unknown"), (
            f"swept-out req must be aborted (gone from every live scheduler "
            f"structure); got status={r.status!r}"
        )
        assert r.kv_pages == 0, "timeout-abort of an unadmitted req owns no KV"

        # Clean up the ballasts so teardown starts from a drained engine.
        t.abort(b1)
        t.abort(b2)
        yield from _drain_until_released(t, b1, b2)


if __name__ == "__main__":
    unittest.main()
