import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    BALLAST_MAX_NEW_TOKENS,
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

        t.pause_generation(mode="retract")
        yield

        assert (
            r.status == "waiting"
        ), f"force-retracted chunked req must be back in waiting; got {r.status}"
        assert r.kv_pages == 0, f"retract must release KV; got {r.kv_pages}"

        t.continue_generation()
        yield from run_until_finished(r)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

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
        assert r.kv_pages == 0 and r.lock_refs == 0

    def test_retract_chunked_resume_in_waiting(self):
        self.server.execute_script(self._script_retract_chunked_resume_in_waiting)

    @staticmethod
    def _script_retract_chunked_resume_in_waiting(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
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
        yield from run_until(low, lambda h: h.is_chunking)

        high = t.start_req(prompt_len=8, max_new_tokens=2, priority=10)

        yield from run_until_finished(high)
        assert high.finished
        assert not low.finished

        yield from run_until_all_finished([low, high])
        assert low.finished and high.finished


class TestPriorityPreempt(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
        max_running_requests=1,
        priority_scheduling_preemption_threshold=0,
    )

    def test_priority_preempt_decode_victim_to_waiting(self):
        self.server.execute_script(
            self._script_priority_preempt_decode_victim_to_waiting
        )

    @staticmethod
    def _script_priority_preempt_decode_victim_to_waiting(t: ScriptedContext):
        low = t.start_req(
            prompt_len=8,
            max_new_tokens=BALLAST_MAX_NEW_TOKENS,
            priority=0,
            ignore_eos=True,
        )
        yield from run_until(low, lambda h: h.status == "running")
        assert low.kv_pages > 0

        high = t.start_req(prompt_len=8, max_new_tokens=2, priority=10)
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
            prompt_len=8,
            max_new_tokens=BALLAST_MAX_NEW_TOKENS,
            priority=0,
            ignore_eos=True,
        )
        yield from run_until(r_low, lambda h: h.status == "running")
        pages_before = r_low.kv_pages
        assert pages_before > 0

        r_high = t.start_req(prompt_len=8, max_new_tokens=2, priority=10)
        yield from run_until(r_low, lambda h: h.status == "waiting")
        assert r_low.kv_pages == 0


if __name__ == "__main__":
    unittest.main()
