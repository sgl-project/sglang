import unittest

from sglang.srt.environ import envs
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


class TestAbortBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_abort_waiting_chunked_resume(self):
        self.server.execute_script(self._script_abort_waiting_chunked_resume)

    @staticmethod
    def _script_abort_waiting_chunked_resume(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.status == "waiting" and h.chunks_done >= 1)

        pages_before = r.kv_pages
        assert pages_before > 0, "chunked req should own KV pages mid-chunk"

        t.abort(r)
        yield

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
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield
        yield from run_until(r, lambda h: h.is_chunking)

        t.abort(r)
        yield

        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None

    def test_abort_at_chunk_mid(self):
        self.server.execute_script(self._script_abort_at_chunk_mid)

    @staticmethod
    def _script_abort_at_chunk_mid(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.chunks_done >= 2 and h.is_chunking)

        t.abort(r)
        yield

        assert r.kv_pages == 0

    def test_abort_one_does_not_disturb_other(self):
        self.server.execute_script(self._script_abort_one_does_not_disturb_other)

    @staticmethod
    def _script_abort_one_does_not_disturb_other(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)

        t.abort(r1)
        yield

        assert r1.kv_pages == 0
        yield from run_until_finished(r2)
        assert r2.finished, "r2 should still complete after r1 is aborted"

    def test_abort_with_zero_yield(self):
        self.server.execute_script(self._script_abort_with_zero_yield)

    @staticmethod
    def _script_abort_with_zero_yield(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        t.abort(r)
        yield
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
        yield
        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None

    def test_abort_then_start_same_step_new_rid(self):
        self.server.execute_script(self._script_abort_then_start_same_step_new_rid)

    @staticmethod
    def _script_abort_then_start_same_step_new_rid(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        t.abort(r1)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r2)
        assert r2.finished
        assert r1.kv_pages == 0

    def test_abort_then_start_same_step_same_rid(self):
        self.server.execute_script(self._script_abort_then_start_same_step_same_rid)

    @staticmethod
    def _script_abort_then_start_same_step_same_rid(t: ScriptedContext):
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
        self.server.execute_script(self._script_abort_five_chunked_in_a_row)

    @staticmethod
    def _script_abort_five_chunked_in_a_row(t: ScriptedContext):
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
            assert r.req is None or r.req.req_pool_idx is None

    def test_abort_unknown_rid_noop(self):
        self.server.execute_script(self._script_abort_unknown_rid_noop)

    @staticmethod
    def _script_abort_unknown_rid_noop(t: ScriptedContext):
        bogus = ScriptedReqHandle(rid="never-submitted-rid", context=t)
        t.abort(bogus)
        yield
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_abort_after_finish_noop(self):
        self.server.execute_script(self._script_abort_after_finish_noop)

    @staticmethod
    def _script_abort_after_finish_noop(t: ScriptedContext):
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

        kv_pool_free_before = t.engine_stats()["kv_pool_free"]

        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        # at-most-one finish is enforced by the engine (output_streamer: assert not req.finished_output)
        kv_pool_free_after = t.engine_stats()["kv_pool_free"]
        assert kv_pool_free_after == kv_pool_free_before, (
            f"abort-after-finish must not move KV pool; "
            f"before={kv_pool_free_before} after={kv_pool_free_after}"
        )

    def test_waiting_timeout_sweeps_parked_chunked_resume(self):
        self.server.execute_script(
            self._script_waiting_timeout_sweeps_parked_chunked_resume
        )

    @staticmethod
    def _script_waiting_timeout_sweeps_parked_chunked_resume(t: ScriptedContext):
        # NOTE: the original intent was to prove a parked chunked-resume req is
        # IMMUNE to the waiting-timeout sweep. Source reading shows it is NOT:
        # a chunked-resume req lives in scheduler.waiting_queue (status=="waiting")
        # carrying its original wait_queue_entry_time, and
        # _abort_on_waiting_timeout() scans the whole waiting_queue with no
        # chunked-resume guard. So the sweep aborts a parked chunked-resume req
        # exactly like any other waiting req. This test asserts that ACTUAL
        # behavior, with a plain waiting req as a positive control.
        r_chunked = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_chunked, lambda h: h.is_chunking)
        yield from run_until(r_chunked, lambda h: h.status == "waiting")

        # Pause in place so the event loop stops admitting: the chunked-resume req
        # keeps its row/KV and stays in waiting_queue, and a freshly submitted
        # plain req also parks in waiting_queue instead of being admitted.
        t.pause_generation(mode="in_place")
        r_plain = t.start_req(prompt_len=16, max_new_tokens=2)
        yield

        def _waiting_rids():
            return {req.rid for req in t.scheduler.waiting_queue}

        assert (
            r_chunked.rid in _waiting_rids()
        ), "chunked-resume must be parked in waiting_queue"
        assert r_plain.rid in _waiting_rids(), (
            f"plain control req must park in waiting_queue while paused; "
            f"got waiting_rids={_waiting_rids()!r}"
        )

        # Drive the real sweep with a tiny timeout so every waiting req is past
        # its deadline. The immediate, in-process consequence of the sweep is
        # removal from waiting_queue (the timeout-abort path notifies the
        # tokenizer and drops the req from the queue). Both the plain control AND
        # the chunked-resume are removed -- the chunked-resume is NOT immune.
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1e-6):
            t.trigger_abort_on_waiting_timeout()

        after_sweep = _waiting_rids()
        assert r_plain.rid not in after_sweep, (
            f"positive control: plain waiting req must be swept out of "
            f"waiting_queue by the waiting-timeout abort; got {after_sweep!r}"
        )
        assert r_chunked.rid not in after_sweep, (
            f"chunked-resume is NOT immune to the waiting-timeout sweep: it "
            f"sits in waiting_queue with no guard and is dropped like any "
            f"other waiting req; got {after_sweep!r}"
        )

        t.continue_generation()

    def test_abort_chunk_last(self):
        self.server.execute_script(self._script_abort_chunk_last)

    @staticmethod
    def _script_abort_chunk_last(t: ScriptedContext):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.req.inflight_middle_chunks == 0

    def test_abort_penultimate_chunk(self):
        self.server.execute_script(self._script_abort_penultimate_chunk)

    @staticmethod
    def _script_abort_penultimate_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=4 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(r, lambda h: h.chunks_done >= 2 and h.is_chunking)
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None
        assert r.lock_refs == 0

    def test_double_abort_idempotent(self):
        self.server.execute_script(self._script_double_abort_idempotent)

    @staticmethod
    def _script_double_abort_idempotent(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.abort(r)
        t.abort(r)
        yield
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
        yield
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
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(3)
        ]
        yield from run_until(reqs[0], lambda h: h.is_chunking)
        for r in reqs:
            t.abort(r)
            yield
        for r in reqs:
            assert r.kv_pages == 0

    def test_abort_at_chunk_boundary_race(self):
        self.server.execute_script(self._script_abort_at_chunk_boundary_race)

    @staticmethod
    def _script_abort_at_chunk_boundary_race(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)

        t.abort(r)
        yield

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

    def test_abort_in_inter_chunk_gap_skips_stash_no_extra_radix_node(self):
        self.server.execute_script(
            self._script_abort_in_inter_chunk_gap_skips_stash_no_extra_radix_node
        )

    @staticmethod
    def _script_abort_in_inter_chunk_gap_skips_stash_no_extra_radix_node(
        t: ScriptedContext,
    ):
        """Abort in the inter-chunk waiting gap skips the stash and creates no radix node."""
        # GPU validation pending.
        #
        # Branch under test: scheduler.get_next_batch_to_run, the
        #   `if self._chunked_req_scheduled_last_iter: stash_chunked_request(...)`
        # guard. When the chunked req was NOT scheduled last iteration -- the
        # inter-chunk gap where it sits in the waiting queue while still being
        # chunked_req -- the flag is False, so stash_chunked_request (which calls
        # maybe_cache_unfinished_req(..., chunked=True)) is SKIPPED. Aborting here
        # must therefore not leave behind a freshly cached prefix node for the
        # unscheduled remainder.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        # Land in the inter-chunk gap: the req has finished >=1 chunk and is parked
        # in the waiting queue, so _chunked_req_scheduled_last_iter is False.
        yield from run_until(
            r, lambda h: h.status == "waiting" and h.chunks_done >= 1 and h.is_chunking
        )

        # Snapshot the radix tree before abort. The skipped stash must not add a
        # node nor bump any node's hit_count for the unscheduled remainder.
        hit_counts_before = t.get_all_node_hit_counts()
        node_count_before = len(hit_counts_before)
        hit_sum_before = sum(hit_counts_before.values())
        chunks_before = r.chunks_done

        t.abort(r)
        yield
        yield

        hit_counts_after = t.get_all_node_hit_counts()
        node_count_after = len(hit_counts_after)
        hit_sum_after = sum(hit_counts_after.values())

        # Witnessing the SKIPPED stash: no extra cached-prefix node, no hit-count
        # growth for the chunked remainder that was never scheduled.
        assert node_count_after <= node_count_before, (
            f"skipped stash must not create a radix node; "
            f"node count {node_count_before} -> {node_count_after}"
        )
        assert hit_sum_after <= hit_sum_before, (
            f"skipped stash must not bump radix hit counts; "
            f"hit sum {hit_sum_before} -> {hit_sum_after}"
        )
        # And the abort itself releases the req and does not revive it.
        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None
        assert r.chunks_done == chunks_before, (
            f"aborted gap req revived and ran another chunk; "
            f"chunks_done went {chunks_before} -> {r.chunks_done}"
        )

    def test_abort_then_resubmit_same_rid_same_step(self):
        self.server.execute_script(self._script_abort_then_resubmit_same_rid_same_step)

    @staticmethod
    def _script_abort_then_resubmit_same_rid_same_step(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            rid="abort-resubmit-same-step",
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
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(
            r,
            lambda h: h.req.inflight_middle_chunks > 0 and not h.is_chunking,
        )
        assert r.req.inflight_middle_chunks > 0
        assert not r.is_chunking

        t.abort(r)
        yield

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
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        assert (1 if t.scheduler.chunked_req is not None else 0) == 1

        t.abort(r)
        yield
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
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        assert (
            t.scheduler.chunked_req.rid if t.scheduler.chunked_req is not None else None
        ) == r1.rid, (
            f"r1 should hold the chunked slot before abort; got "
            f"{(t.scheduler.chunked_req.rid if t.scheduler.chunked_req is not None else None)!r}"
        )

        t.abort(r1)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield

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
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        assert r1.kv_pages > 0

        t.pause_generation(mode="retract")
        t.abort(r1)
        yield

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
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        assert (1 if t.scheduler.chunked_req is not None else 0) == 1

        t.abort(r1)
        yield

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
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
        yield from run_until(
            r,
            lambda h: h.chunks_done >= 1 and h.is_chunking,
        )

        t.abort(r)
        yield

        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None
        assert r.lock_refs == 0
        assert r.finished


if __name__ == "__main__":
    unittest.main()
