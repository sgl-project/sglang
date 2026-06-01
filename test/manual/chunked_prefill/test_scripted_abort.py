import unittest

from sglang.srt.managers.io_struct import AbortReq
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

    def test_abort_at_last_chunk(self):
        self.server.execute_script(self._script_abort_at_last_chunk)

    @staticmethod
    def _script_abort_at_last_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)

        assert r.req.inflight_middle_chunks > 0

        t.abort(r)
        yield

        assert r.req.inflight_middle_chunks == 0, (
            f"abort must zero inflight_middle_chunks to prevent revival; "
            f"got {r.req.inflight_middle_chunks}"
        )

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
        bogus = ScriptedReqHandle(
            rid="never-submitted-rid", scheduler_hook=t._scheduler_hook
        )
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

        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert r.finish_event_count <= 1, (
            f"abort-after-finish should not emit a second finish event; "
            f"got {r.finish_event_count}"
        )
        stats = t.engine_stats()
        assert stats["kv_pool_free"] >= 0

    def test_abort_during_waiting_timeout(self):
        self.server.execute_script(self._script_abort_during_waiting_timeout)

    @staticmethod
    def _script_abort_during_waiting_timeout(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.status == "waiting")
        t.trigger_abort_on_waiting_timeout()
        yield
        yield from run_until_finished(r)
        assert r.finished, "chunked-resume must survive waiting-timeout sweep"
        assert len(r.req.output_ids) == 2
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_abort_chunk_first(self):
        self.server.execute_script(self._script_abort_chunk_first)

    @staticmethod
    def _script_abort_chunk_first(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield
        yield
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert r.finish_event_count <= 1

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
        assert (
            r.finish_event_count <= 1
        ), f"double abort must not double-finalize; got {r.finish_event_count}"

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
        assert r.finish_event_count <= 1

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

    def test_abort_finish_event_count_at_most_one(self):
        self.server.execute_script(self._script_abort_finish_event_count_at_most_one)

    @staticmethod
    def _script_abort_finish_event_count_at_most_one(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.abort(r)
        yield
        assert r.finish_event_count <= 1

    def test_abort_at_chunk_boundary_race(self):
        self.server.execute_script(self._script_abort_at_chunk_boundary_race)

    @staticmethod
    def _script_abort_at_chunk_boundary_race(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        chunks_at_abort = r.chunks_done

        t.abort(r)
        yield

        assert (
            r.finish_event_count <= 1
        ), f"abort race must not double-finalize; got {r.finish_event_count}"
        assert (
            r.req.inflight_middle_chunks == 0
        ), f"abort must zero inflight_middle_chunks; got {r.req.inflight_middle_chunks}"
        assert r.kv_pages == 0
        assert r.chunks_done >= chunks_at_abort

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

        assert r.req.inflight_middle_chunks == 0, (
            f"abort in gap must zero inflight_middle_chunks to block "
            f"Stage A revival; got {r.req.inflight_middle_chunks}"
        )
        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None

    def test_abort_when_chunked_only_then_idle(self):
        self.server.execute_script(self._script_abort_when_chunked_only_then_idle)

    @staticmethod
    def _script_abort_when_chunked_only_then_idle(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        assert (1 if t._scheduler.chunked_req is not None else 0) == 1

        t.abort(r)
        yield
        yield

        assert r.kv_pages == 0
        assert (1 if t._scheduler.chunked_req is not None else 0) == 0
        assert t.is_idle, "engine must be idle after the only chunked req is aborted"

    def test_chunked_req_then_abort_then_new_short_in_one_yield(self):
        self.server.execute_script(
            self._script_chunked_req_then_abort_then_new_short_in_one_yield
        )

    @staticmethod
    def _script_chunked_req_then_abort_then_new_short_in_one_yield(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        assert (t._scheduler.chunked_req.rid if t._scheduler.chunked_req is not None else None) == r1.rid, (
            f"r1 should hold the chunked slot before abort; got "
            f"{(t._scheduler.chunked_req.rid if t._scheduler.chunked_req is not None else None)!r}"
        )

        t.abort(r1)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield

        cur = (t._scheduler.chunked_req.rid if t._scheduler.chunked_req is not None else None)
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

        t.force_retract(r1)
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
        assert r1.finish_event_count <= 1, (
            f"force_retract + abort same yield must not double-finalize; "
            f"got {r1.finish_event_count} events"
        )
        yield

    def test_abort_chunked_with_baton_handoff(self):
        self.server.execute_script(self._script_abort_chunked_with_baton_handoff)

    @staticmethod
    def _script_abort_chunked_with_baton_handoff(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        assert (1 if t._scheduler.chunked_req is not None else 0) == 1

        t.abort(r1)
        yield

        yield from run_until(r2, lambda h: h.is_chunking)
        assert r1.kv_pages == 0
        assert r1.req is None or r1.req.req_pool_idx is None
        assert r1.lock_refs == 0
        yield from run_until_finished(r2)
        assert r2.finished, "baton handoff must let r2 complete"
        assert r2.lock_refs == 0


class TestAbortDualQueueInvariant(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_dual_queue_abort_no_double_release_invariant(self):
        self.server.execute_script(
            self._script_dual_queue_abort_no_double_release_invariant
        )

    @staticmethod
    def _script_dual_queue_abort_no_double_release_invariant(t: ScriptedContext):
        s = t._scheduler
        allocator = s.token_to_kv_pool_allocator

        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        live_req = s.chunked_req
        if live_req is None or live_req.rid != r.rid:
            for candidate in s.running_batch.reqs:
                if candidate.rid == r.rid:
                    live_req = candidate
                    break
        assert (
            live_req is not None and live_req.rid == r.rid
        ), "could not locate live Req object to drive W3 dual-queue scenario"

        if live_req not in s.waiting_queue:
            s.waiting_queue.append(live_req)
        if live_req not in s.running_batch.reqs:
            s.running_batch.reqs.append(live_req)

        waiting_count_before: int = sum(1 for q in s.waiting_queue if q.rid == r.rid)
        batch_count_before: int = sum(1 for q in s.running_batch.reqs if q.rid == r.rid)
        assert waiting_count_before == 1, (
            f"dual-queue setup failed: waiting_queue should contain rid "
            f"exactly once, got {waiting_count_before}"
        )
        assert batch_count_before == 1, (
            f"dual-queue setup failed: running_batch.reqs should contain "
            f"rid exactly once, got {batch_count_before}"
        )
        free_before: int = allocator.available_size()

        s.abort_request(AbortReq(rid=r.rid))

        waiting_count_after: int = sum(1 for q in s.waiting_queue if q.rid == r.rid)
        assert waiting_count_after == waiting_count_before, (
            f"W3 dedup gate broken: waiting_queue rid count went from "
            f"{waiting_count_before} to {waiting_count_after} after "
            f"abort_request; gate at scheduler.abort_request must skip "
            f"waiting_queue removal when rid is also in batch"
        )

        free_after_abort: int = allocator.available_size()
        assert free_after_abort == free_before, (
            f"abort_request triggered a KV release on a dual-queue rid; "
            f"free_before={free_before}, free_after={free_after_abort}; "
            f"the dedup gate must skip the waiting_queue release branch"
        )
        yield


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
        assert (
            r.finish_event_count == 1
        ), f"abort must not double-finalize; got {r.finish_event_count} events"


if __name__ == "__main__":
    unittest.main()
