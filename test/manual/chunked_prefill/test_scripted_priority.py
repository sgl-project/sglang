import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
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
        ), f"r should be retracted (back in waiting) or finished; got {r.status}"
        if r.status == "waiting":
            assert r.kv_pages == 0, f"retract must release KV; got {r.kv_pages}"
        else:
            assert r.kv_pages == 0, f"finished req must release KV; got {r.kv_pages}"

    def test_retract_and_resume(self):
        self.server.execute_script(self._script_retract_and_resume)

    @staticmethod
    def _script_retract_and_resume(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        t.pause_generation(mode="retract")
        yield
        assert r.status == "waiting"
        assert r.kv_pages == 0

        t.continue_generation()
        yield from run_until_finished(r)
        assert r.finished

    def test_force_retract_at_chunk_0(self):
        self.server.execute_script(self._script_force_retract_at_chunk_0)

    @staticmethod
    def _script_force_retract_at_chunk_0(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done <= 1)
        t.pause_generation(mode="retract")
        yield
        assert r.kv_pages == 0
        t.continue_generation()
        yield from run_until_finished(r, max_steps=800)
        assert r.finished

    def test_force_retract_at_chunk_mid(self):
        self.server.execute_script(self._script_force_retract_at_chunk_mid)

    @staticmethod
    def _script_force_retract_at_chunk_mid(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.chunks_done >= 2 and h.is_chunking)
        t.pause_generation(mode="retract")
        yield
        assert r.kv_pages == 0
        t.continue_generation()
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.lock_refs == 0

    def test_force_retract_at_last_chunk(self):
        self.server.execute_script(self._script_force_retract_at_last_chunk)

    @staticmethod
    def _script_force_retract_at_last_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        t.pause_generation(mode="retract")
        yield
        assert r.kv_pages == 0
        t.continue_generation()
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_force_retract_then_readmit(self):
        self.server.execute_script(self._script_force_retract_then_readmit)

    @staticmethod
    def _script_force_retract_then_readmit(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.pause_generation(mode="retract")
        yield
        assert r.kv_pages == 0, "retract must release KV before re-admission"
        t.continue_generation()
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
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
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
        t.pause_generation(mode="retract")
        yield
        assert r.kv_pages == 0, f"retract must release KV; got {r.kv_pages}"
        t.continue_generation()
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
        t.pause_generation(mode="retract")
        t.abort(r)
        # retract + abort on the same yield must idempotently release every
        # resource; the aborted req is dropped from the waiting queue, not
        # resumed to a normal finish. Drain the overlap lag, then assert release.
        for _ in range(12):
            if (
                r.kv_pages == 0
                and r.lock_refs == 0
                and (r.req is None or r.req.req_pool_idx is None)
            ):
                break
            yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert r.req is None or r.req.req_pool_idx is None
        t.continue_generation()
        yield
        # the aborted req must not revive after generation resumes
        assert r.kv_pages == 0 and r.lock_refs == 0

    def test_retract_chunked_resume_in_waiting(self):
        self.server.execute_script(self._script_retract_chunked_resume_in_waiting)

    @staticmethod
    def _script_retract_chunked_resume_in_waiting(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.status == "waiting")
        t.pause_generation(mode="retract")
        yield
        assert r.kv_pages == 0
        assert r.status == "waiting"
        t.continue_generation()
        yield from run_until_finished(r, max_steps=800)
        assert r.finished

    def test_two_retracts_same_yield(self):
        self.server.execute_script(self._script_two_retracts_same_yield)

    @staticmethod
    def _script_two_retracts_same_yield(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        t.pause_generation(mode="retract")
        yield
        assert r1.kv_pages == 0
        assert r2.kv_pages == 0
        t.continue_generation()
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
        t.pause_generation(mode="retract")
        yield
        assert r.kv_pages == 0, "retract must release KV"
        t.continue_generation()
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.lock_refs == 0
        assert r.kv_pages == 0


class TestPriorityPriority(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_naive_priority_chunked(self):
        self.server.execute_script(self._script_naive_priority_chunked)

    @staticmethod
    def _script_naive_priority_chunked(t: ScriptedContext):
        low = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, priority=0)
        yield

        high = t.start_req(prompt_len=8, max_new_tokens=2, priority=10)

        yield from run_until_finished(high)
        assert high.finished
        assert not low.finished
        assert low.is_chunking

        yield from run_until_all_finished([low, high])
        assert low.finished and high.finished

    def test_priority_preempt_chunked(self):
        self.server.execute_script(self._script_priority_preempt_chunked)

    @staticmethod
    def _script_priority_preempt_chunked(t: ScriptedContext):
        low = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority=0)
        yield from run_until(low, lambda h: h.is_chunking and h.chunks_done >= 1)
        assert low.kv_pages > 0

        high = t.start_req(prompt_len=8, max_new_tokens=2, priority=10)
        t.exhaust_kv(leave_pages=1)
        yield from run_until(low, lambda h: h.status == "waiting")

        assert low.status == "waiting"
        assert low.kv_pages == 0
        yield from run_until_finished(high)
        assert high.finished

    def test_priority_preempt_release_invariant(self):
        self.server.execute_script(self._script_priority_preempt_release_invariant)

    @staticmethod
    def _script_priority_preempt_release_invariant(t: ScriptedContext):
        r_low = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority=0
        )
        yield from run_until(r_low, lambda h: h.is_chunking and h.chunks_done >= 1)
        pages_before = r_low.kv_pages
        assert pages_before > 0

        r_high = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority=10
        )
        t.exhaust_kv(leave_pages=1)
        yield from run_until(r_low, lambda h: h.status == "waiting")
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
            priority=0,
        )
        yield from run_until(r3, lambda h: h.status == "running")
        assert r3.kv_pages > 0

        r1 = t.start_req(prompt_len=8, max_new_tokens=2, priority=10)
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            priority=10,
        )
        t.exhaust_kv(leave_pages=1)
        yield from run_until(r3, lambda h: h.status == "waiting")

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


if __name__ == "__main__":
    unittest.main()
