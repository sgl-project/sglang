
import time
import unittest

from sglang.srt.environ import envs
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


class TestPriorityBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_retract_mid_chunk_releases_kv(self):
        self.server.execute_script(self._script_retract_mid_chunk_releases_kv)

    @staticmethod
    def _script_retract_mid_chunk_releases_kv(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        pages_before = r.kv_pages
        assert pages_before > 0

        t.exhaust_kv(leave_pages=1)
        yield

        assert r.status in (
            "waiting",
            "finished",
            "unknown",
        ), f"r should be retracted (back in waiting) or finished; got {r.status}"
        if r.status == "waiting":
            assert r.kv_pages == 0, f"retract must release KV; got {r.kv_pages}"

    def test_retract_and_resume(self):
        self.server.execute_script(self._script_retract_and_resume)

    @staticmethod
    def _script_retract_and_resume(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        t.force_retract(r)
        yield
        assert r.status == "waiting"
        assert r.kv_pages == 0

        yield from run_until_finished(r)
        assert r.finished

    def test_force_retract_at_chunk_0(self):
        self.server.execute_script(self._script_force_retract_at_chunk_0)

    @staticmethod
    def _script_force_retract_at_chunk_0(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield
        yield
        t.force_retract(r)
        yield
        assert r.kv_pages == 0

    def test_force_retract_at_chunk_mid(self):
        self.server.execute_script(self._script_force_retract_at_chunk_mid)

    @staticmethod
    def _script_force_retract_at_chunk_mid(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.chunks_done >= 2 and h.is_chunking)
        t.force_retract(r)
        yield
        assert r.kv_pages == 0

    def test_force_retract_at_last_chunk(self):
        self.server.execute_script(self._script_force_retract_at_last_chunk)

    @staticmethod
    def _script_force_retract_at_last_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        t.force_retract(r)
        yield
        assert r.kv_pages == 0
        assert r.pending_middle_outputs == 0

    def test_force_retract_then_readmit(self):
        self.server.execute_script(self._script_force_retract_then_readmit)

    @staticmethod
    def _script_force_retract_then_readmit(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.force_retract(r)
        yield
        assert r.kv_pages == 0, "retract must release KV before re-admission"
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_retract_one_admit_one(self):
        self.server.execute_script(self._script_retract_one_admit_one)

    @staticmethod
    def _script_retract_one_admit_one(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        r2 = t.start_req(prompt_len=8, max_new_tokens=2)
        t.force_retract(r1)
        yield from run_until_finished(r2)
        assert r2.finished
        assert r2.kv_pages == 0
        yield from run_until_finished(r1)
        assert r1.finished
        assert r1.kv_pages == 0
        assert r1.lock_refs == 0

    def test_retract_during_decode(self):
        self.server.execute_script(self._script_retract_during_decode)

    @staticmethod
    def _script_retract_during_decode(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=32)
        yield from run_until(r, lambda h: h.status == "running")
        assert r.kv_pages > 0, "decode-state req must own KV before retract"
        t.force_retract(r)
        yield
        assert r.kv_pages == 0, f"retract must release KV; got {r.kv_pages}"
        yield from run_until_finished(r)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_retract_then_abort_idempotent(self):
        self.server.execute_script(self._script_retract_then_abort_idempotent)

    @staticmethod
    def _script_retract_then_abort_idempotent(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.force_retract(r)
        t.abort(r)
        yield
        assert r.kv_pages == 0

    def test_disagg_retract_resets_send_state_extra(self):
        self.server.execute_script(self._script_disagg_retract_resets_send_state_extra)

    @staticmethod
    def _script_disagg_retract_resets_send_state_extra(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.force_retract(r)
        yield
        assert r.disagg_send_state in (None, "idle")
        yield from run_until_finished(r)
        assert r.finished

    def test_retract_chunked_resume_in_waiting(self):
        self.server.execute_script(self._script_retract_chunked_resume_in_waiting)

    @staticmethod
    def _script_retract_chunked_resume_in_waiting(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.status == "waiting")
        t.force_retract(r)
        yield
        assert r.kv_pages == 0
        assert r.row_idx is None
        assert r.status in ("waiting", "finished")

    def test_two_retracts_same_yield(self):
        self.server.execute_script(self._script_two_retracts_same_yield)

    @staticmethod
    def _script_two_retracts_same_yield(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        t.force_retract(r1)
        t.force_retract(r2)
        yield
        assert r1.kv_pages == 0
        assert r2.kv_pages == 0
        yield from run_until_all_finished([r1, r2])
        assert r1.finished and r2.finished
        assert r1.lock_refs == 0
        assert r2.lock_refs == 0

    def test_retract_then_re_chunk(self):
        self.server.execute_script(self._script_retract_then_re_chunk)

    @staticmethod
    def _script_retract_then_re_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(r, lambda h: h.chunks_done >= 1)
        t.force_retract(r)
        yield
        assert r.kv_pages == 0, "retract must release KV"
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.lock_refs == 0
        assert r.kv_pages == 0

    def test_watchdog_skips_chunked_resume_invariant(self):
        self.server.execute_script(self._script_watchdog_skips_chunked_resume_invariant)

    @staticmethod
    def _script_watchdog_skips_chunked_resume_invariant(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        s = t._scheduler
        req = t._find_req_by_rid(r.rid)
        assert req is not None
        assert s.chunked_req is req, (
            f"setup expected s.chunked_req to point at r; got "
            f"s.chunked_req={s.chunked_req!r}, r={req!r}"
        )
        assert req.inflight_middle_chunks > 0, (
            f"setup expected inflight_middle_chunks > 0, got "
            f"{req.inflight_middle_chunks}"
        )

        was_in_queue = req in s.waiting_queue
        if not was_in_queue:
            s.waiting_queue.append(req)
        original_entry_time = req.time_stats.wait_queue_entry_time
        req.time_stats.wait_queue_entry_time = 1.0
        try:
            with envs.SGLANG_REQ_WAITING_TIMEOUT.override(0.5):
                deadline = time.perf_counter() - 0.5
                assert (
                    0 < req.time_stats.wait_queue_entry_time < deadline
                ), "setup did not backdate entry_time past the watchdog deadline"
                s._abort_on_waiting_timeout()

            assert req in s.waiting_queue, (
                "watchdog incorrectly aborted a chunked-resume req: "
                "r was removed from waiting_queue despite inflight_middle_chunks > 0"
            )
            assert s.chunked_req is req, (
                f"chunked_req slot must still hold r after watchdog "
                f"skip, got s.chunked_req={s.chunked_req!r}"
            )
            assert req.finished_reason is None, (
                f"chunked-resume req must not be marked finished by "
                f"watchdog abort, got finished_reason={req.finished_reason!r}"
            )
        finally:
            if not was_in_queue and req in s.waiting_queue:
                s.waiting_queue.remove(req)
            req.time_stats.wait_queue_entry_time = original_entry_time

        yield from run_until_finished(r, max_steps=DEFAULT_MAX_STEPS * 2)
        assert r.finished


class TestPriorityPriority(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_naive_priority_chunked(self):
        self.server.execute_script(self._script_naive_priority_chunked)

    @staticmethod
    def _script_naive_priority_chunked(t: ScriptedContext):
        low = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, priority="low"
        )
        yield

        high = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")

        yield from run_until_all_finished([low, high])
        assert low.finished and high.finished

    def test_priority_preempt_chunked(self):
        self.server.execute_script(self._script_priority_preempt_chunked)

    @staticmethod
    def _script_priority_preempt_chunked(t: ScriptedContext):
        low = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low"
        )
        yield from run_until(low, lambda h: h.is_chunking and h.chunks_done >= 1)
        assert low.kv_pages > 0

        high = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")
        t.exhaust_kv(leave_pages=1)
        yield

        if low.status == "waiting":
            assert low.kv_pages == 0
        yield from run_until_finished(high)
        assert high.finished

    def test_priority_preempt_chunked_victim(self):
        self.server.execute_script(self._script_priority_preempt_chunked_victim)

    @staticmethod
    def _script_priority_preempt_chunked_victim(t: ScriptedContext):
        r_low = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low"
        )
        yield from run_until(r_low, lambda h: h.is_chunking)
        r_high = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high"
        )
        t.force_preempt(victim_rid=r_low.rid, by_rid=r_high.rid)
        yield
        assert r_low.status == "waiting"

        yield from run_until_all_finished([r_low, r_high])

    def test_preempt_five_victims(self):
        self.server.execute_script(self._script_preempt_five_victims)

    @staticmethod
    def _script_preempt_five_victims(t: ScriptedContext):
        victims = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(5)
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

    def test_priority_preempt_release_invariant(self):
        self.server.execute_script(self._script_priority_preempt_release_invariant)

    @staticmethod
    def _script_priority_preempt_release_invariant(t: ScriptedContext):
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

    def test_priority_preempt_with_chunked_admission_same_yield(self):
        self.server.execute_script(
            self._script_priority_preempt_with_chunked_admission_same_yield
        )

    @staticmethod
    def _script_priority_preempt_with_chunked_admission_same_yield(
        t: ScriptedContext,
    ):
        r3 = t.start_req(
            prompt_len=16,
            max_new_tokens=32,
            priority="low",
        )
        yield from run_until(r3, lambda h: h.status == "running")
        assert r3.kv_pages > 0

        r1 = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            priority="high",
        )
        t.force_preempt(victim_rid=r3.rid, by_rid=r1.rid)
        yield

        assert (
            r3.kv_pages == 0
        ), f"preempted low-priority r3 must release KV; got {r3.kv_pages}"

        yield from run_until(r2, lambda h: h.is_chunking)
        assert r2.chunks_done >= 1, (
            f"long chunked r2 should start advancing chunks_done; got "
            f"{r2.chunks_done}"
        )

        yield from run_until_all_finished([r1, r2, r3], max_steps=800)
        assert r1.finished and r2.finished and r3.finished


class TestPriorityDisagg(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disaggregation_mode="prefill",
    )

    def test_disagg_retract_resets_send_state(self):
        self.server.execute_script(self._script_disagg_retract_resets_send_state)

    @staticmethod
    def _script_disagg_retract_resets_send_state(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        t.force_retract(r)
        yield

        assert r.status == "waiting"
        assert r.kv_pages == 0
        assert r.disagg_send_state in (
            None,
            "idle",
        ), f"disagg send state must reset on retract, got {r.disagg_send_state}"

        yield from run_until_finished(r)
        assert r.finished


if __name__ == "__main__":
    unittest.main()
