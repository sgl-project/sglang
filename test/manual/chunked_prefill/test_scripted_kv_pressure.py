import unittest

from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    exhaust_row_pool,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


class TestKVPressureBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_kv_almost_empty_then_abort(self):
        self.server.execute_script(self._script_kv_almost_empty_then_abort)

    @staticmethod
    def _script_kv_almost_empty_then_abort(t: ScriptedContext):
        t.exhaust_kv(leave_pages=1)
        yield

        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.abort(r)
        yield

        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert r.req.req_pool_idx is None
        stats = t.engine_stats()
        assert stats["kv_pool_free"] >= 1

    def test_kv_full_chunked_new_req_retracts(self):
        self.server.execute_script(self._script_kv_full_chunked_new_req_retracts)

    @staticmethod
    def _script_kv_full_chunked_new_req_retracts(t: ScriptedContext):
        baseline = t.engine_stats()["kv_pool_free"]
        t.exhaust_kv(leave_pages=2)
        yield

        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(2000):
            if r.finished or (
                r.req is not None and isinstance(r.req.finished_reason, FINISH_ABORT)
            ):
                break
            yield
        assert r.finished or (
            r.req is not None and isinstance(r.req.finished_reason, FINISH_ABORT)
        )
        assert r.kv_pages == 0
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"KV not released after OOM/retract path: baseline={baseline}, "
            f"final={final}"
        )

    def test_kv_full_chunked_plus_decode_retract(self):
        self.server.execute_script(self._script_kv_full_chunked_plus_decode_retract)

    @staticmethod
    def _script_kv_full_chunked_plus_decode_retract(t: ScriptedContext):
        baseline = t.engine_stats()["kv_pool_free"]
        r_long = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r_short = t.start_req(prompt_len=8, max_new_tokens=8)
        yield from run_until(r_long, lambda h: h.is_chunking)

        t.exhaust_kv(leave_pages=1)
        yield

        for _ in range(2000):
            if r_long.finished and r_short.finished:
                break
            yield
        assert r_long.finished and r_short.finished
        assert r_long.kv_pages == 0
        assert r_short.kv_pages == 0
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"KV not fully released after pressure: baseline={baseline}, "
            f"final={final}"
        )

    def test_row_pool_tight_admits_after_release(self):
        self.server.execute_script(self._script_row_pool_tight_admits_after_release)

    @staticmethod
    def _script_row_pool_tight_admits_after_release(t: ScriptedContext):
        baseline_rows_used = (
            t._scheduler.req_to_token_pool.size
            - t._scheduler.req_to_token_pool.available_size()
        )
        yield from exhaust_row_pool(t, leave_rows=2)

        reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(5)]
        yield from run_until_all_finished(reqs, max_steps=2000)
        for r in reqs:
            assert r.finished, f"req {r.rid} did not finish"
            assert r.kv_pages == 0, (
                f"row-pool pressure must not leave KV held: rid={r.rid}, "
                f"kv_pages={r.kv_pages}"
            )
        final_rows_used = (
            t._scheduler.req_to_token_pool.size
            - t._scheduler.req_to_token_pool.available_size()
        )
        assert final_rows_used <= baseline_rows_used, (
            f"row pool leak after admit-after-release: baseline used="
            f"{baseline_rows_used}, final used={final_rows_used}"
        )

    def test_lock_refs_tight_concurrent_prefix(self):
        self.server.execute_script(self._script_lock_refs_tight_concurrent_prefix)

    @staticmethod
    def _script_lock_refs_tight_concurrent_prefix(t: ScriptedContext):
        baseline_lock_refs = t.get_all_node_lock_refs()
        t.exhaust_lock_refs(leave_refs=4)
        yield
        r_warm = t.start_req(prompt_len=128, max_new_tokens=2)
        yield from run_until_finished(r_warm)
        assert r_warm.finished
        assert r_warm.lock_refs == 0
        reqs = [t.start_req(prompt_len=128, max_new_tokens=2) for _ in range(8)]
        yield from run_until_all_finished(reqs)
        for r in reqs:
            assert r.finished
            assert (
                r.lock_refs == 0
            ), f"req {r.rid} leaked {r.lock_refs} lock_refs after finish"
        final_lock_refs = t.get_all_node_lock_refs()
        assert final_lock_refs <= baseline_lock_refs, (
            f"global lock_refs leaked from baseline={baseline_lock_refs} "
            f"to final={final_lock_refs}"
        )

    def test_kv_at_one_page_chunked_completes(self):
        self.server.execute_script(self._script_kv_at_one_page_chunked_completes)

    @staticmethod
    def _script_kv_at_one_page_chunked_completes(t: ScriptedContext):
        t.exhaust_kv(leave_pages=4)
        yield
        r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 1, max_new_tokens=1)
        yield from run_until(r, lambda h: h.finished, max_steps=2000)
        assert r.finished, "tiny chunked req must complete under tight KV"
        assert r.kv_pages == 0, (
            f"tight-KV chunked completion must release all pages; got "
            f"kv_pages={r.kv_pages}"
        )
        assert r.lock_refs == 0

    def test_kv_recovery_after_full(self):
        self.server.execute_script(self._script_kv_recovery_after_full)

    @staticmethod
    def _script_kv_recovery_after_full(t: ScriptedContext):
        baseline = t.engine_stats()["kv_pool_free"]
        t.exhaust_kv(leave_pages=1)
        yield

        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until(r, lambda h: h.finished, max_steps=3000)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"KV pool failed to recover post-pressure: baseline={baseline}, "
            f"final={final}"
        )

    def test_kv_pressure_with_retract_resume(self):
        self.server.execute_script(self._script_kv_pressure_with_retract_resume)

    @staticmethod
    def _script_kv_pressure_with_retract_resume(t: ScriptedContext):
        baseline = t.engine_stats()["kv_pool_free"]
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        chunks_before_retract = r.chunks_done
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        for _ in range(2000):
            if r.finished:
                break
            yield
        assert r.finished
        assert r.kv_pages == 0
        assert r.chunks_done >= chunks_before_retract, (
            f"chunks_done regressed across retract+resume: "
            f"before={chunks_before_retract}, final={r.chunks_done}"
        )
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"KV pool failed to recover after retract+resume: "
            f"baseline={baseline}, final={final}"
        )

    def test_cumulative_alloc_does_not_grow_unbounded(self):
        self.server.execute_script(
            self._script_cumulative_alloc_does_not_grow_unbounded
        )

    @staticmethod
    def _script_cumulative_alloc_does_not_grow_unbounded(t: ScriptedContext):
        baseline = t.engine_stats()["kv_pool_free"]
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(50)]
        yield from run_until_all_finished(reqs)
        for r in reqs:
            assert r.finished
            assert r.cumulative_kv_alloc_bytes >= 0
            assert r.kv_pages == 0, f"req {r.rid} kept {r.kv_pages} pages after finish"
        final = t.engine_stats()["kv_pool_free"]
        assert (
            final >= baseline
        ), f"50-req batch leaked KV: baseline={baseline}, final={final}"

    def test_engine_stats_pool_invariant(self):
        self.server.execute_script(self._script_engine_stats_pool_invariant)

    @staticmethod
    def _script_engine_stats_pool_invariant(t: ScriptedContext):
        before = t.engine_stats()
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(10)]
        yield from run_until_all_finished(reqs)
        after = t.engine_stats()
        assert after["kv_pool_free"] >= before["kv_pool_free"]
        assert after["row_pool_free"] >= before["row_pool_free"]

    def test_kv_pressure_with_radix_evict(self):
        self.server.execute_script(self._script_kv_pressure_with_radix_evict)

    @staticmethod
    def _script_kv_pressure_with_radix_evict(t: ScriptedContext):
        baseline = t.engine_stats()["kv_pool_free"]
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        assert r1.finished
        assert r1.kv_pages == 0

        t.exhaust_kv(leave_pages=1)
        yield

        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(2000):
            if r2.finished:
                break
            yield
        assert r2.finished
        assert r2.kv_pages == 0
        assert r2.chunks_done >= 2, (
            f"r2 must re-chunk after radix eviction; got chunks_done="
            f"{r2.chunks_done}"
        )
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"KV did not recover after evict + re-chunk: baseline={baseline}, "
            f"final={final}"
        )

    def test_chunked_retract_resume_kv_recovers_exactly(self):
        self.server.execute_script(
            self._script_chunked_retract_resume_kv_recovers_exactly
        )

    @staticmethod
    def _script_chunked_retract_resume_kv_recovers_exactly(t: ScriptedContext):
        baseline = t.engine_stats()["kv_pool_free"]
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.kv_pages == 0
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"kv_pool_free did not recover after chunked retract+resume: "
            f"baseline={baseline}, final={final}"
        )

    def test_chunked_retract_at_chunk_first_mid_last(self):
        self.server.execute_script(self._script_chunked_retract_at_chunk_first_mid_last)

    @staticmethod
    def _script_chunked_retract_at_chunk_first_mid_last(t: ScriptedContext):
        expected_chunks = VERY_LONG_PROMPT_LEN // DEFAULT_CHUNK_SIZE
        mid_chunk = expected_chunks // 2
        last_minus_one = max(1, expected_chunks - 1)

        r_first = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_first, lambda h: h.is_chunking)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r_first, max_steps=2000)
        assert r_first.finished
        assert r_first.kv_pages == 0

        r_mid = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_mid, lambda h: h.chunks_done >= mid_chunk)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r_mid, max_steps=2000)
        assert r_mid.finished
        assert r_mid.kv_pages == 0

        r_last = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_last, lambda h: h.chunks_done >= last_minus_one)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r_last, max_steps=2000)
        assert r_last.finished
        assert r_last.kv_pages == 0

    def test_flush_cache_during_chunked_in_flight(self):
        self.server.execute_script(self._script_flush_cache_during_chunked_in_flight)

    @staticmethod
    def _script_flush_cache_during_chunked_in_flight(t: ScriptedContext):
        r_warm = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1)
        yield from run_until_finished(r_warm)
        assert r_warm.finished

        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.flush_cache()
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.kv_pages == 0

    def test_chunked_oscillation_three_force_retracts(self):
        self.server.execute_script(
            self._script_chunked_oscillation_three_force_retracts
        )

    @staticmethod
    def _script_chunked_oscillation_three_force_retracts(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        chunks_at_first = r.chunks_done
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until(
            r,
            lambda h: h.is_chunking and h.chunks_done >= chunks_at_first,
            max_steps=800,
        )
        chunks_after_first_resume = r.chunks_done
        assert chunks_after_first_resume >= chunks_at_first, (
            f"chunks_done regressed across retract: "
            f"before={chunks_at_first}, after={chunks_after_first_resume}"
        )

        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until(
            r,
            lambda h: h.is_chunking and h.chunks_done >= chunks_after_first_resume,
            max_steps=800,
        )
        chunks_after_second_resume = r.chunks_done
        assert chunks_after_second_resume >= chunks_after_first_resume

        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.chunks_done >= chunks_after_second_resume


class TestKVPressurePriority(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_priority_preempt_multiple_chunked(self):
        self.server.execute_script(self._script_priority_preempt_multiple_chunked)

    @staticmethod
    def _script_priority_preempt_multiple_chunked(t: ScriptedContext):
        baseline = t.engine_stats()["kv_pool_free"]
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)

        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high"
        )
        for _ in range(DEFAULT_MAX_STEPS * 4):
            assert not (r1.is_chunking and r2.is_chunking), (
                f"two reqs cannot share the chunked slot; "
                f"r1.is_chunking={r1.is_chunking}, r2.is_chunking={r2.is_chunking}"
            )
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished and r2.finished
        assert r1.kv_pages == 0
        assert r2.kv_pages == 0
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"KV pool not fully released after preemption: "
            f"baseline={baseline}, final={final}"
        )


class TestKVPressurePageSize(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        page_size=16,
    )

    def test_strict_mem_check_handles_chunked_tail(self):
        self.server.execute_script(self._script_strict_mem_check_handles_chunked_tail)

    @staticmethod
    def _script_strict_mem_check_handles_chunked_tail(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN + 17, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0


if __name__ == "__main__":
    unittest.main()
