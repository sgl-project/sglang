import unittest

from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    BALLAST_MAX_NEW_TOKENS,
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

        # A retractable decode peer: it holds KV that the scheduler's decode-OOM
        # path can actually retract to admit/continue the chunked req. With only
        # the raw-allocator exhauster (no Req) there is nothing to retract, so
        # the chunked req's next-chunk alloc_for_extend hard-OOMs and crashes
        # the scheduler.
        r_peer = t.start_req(
            prompt_len=8, max_new_tokens=BALLAST_MAX_NEW_TOKENS, ignore_eos=True
        )
        yield from run_until(r_peer, lambda h: h.status == "running")
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
            t.scheduler.req_to_token_pool.size
            - t.scheduler.req_to_token_pool.available_size()
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
            t.scheduler.req_to_token_pool.size
            - t.scheduler.req_to_token_pool.available_size()
        )
        assert final_rows_used <= baseline_rows_used, (
            f"row pool leak after admit-after-release: baseline used="
            f"{baseline_rows_used}, final used={final_rows_used}"
        )

    def test_lock_refs_tight_concurrent_prefix(self):
        self.server.execute_script(self._script_lock_refs_tight_concurrent_prefix)

    @staticmethod
    def _script_lock_refs_tight_concurrent_prefix(t: ScriptedContext):
        """A long chunked req sharing a pinned warm prefix finishes and leaks no lock_refs."""
        # GPU validation pending.
        #
        # Warm a shared prefix: a short req that finishes leaves cached,
        # evictable prefix nodes in the radix tree. Use a distinct prompt_token
        # so the long req below provably hits THIS prefix (not stray nodes).
        warm_token = 7
        warm_len = DEFAULT_CHUNK_SIZE
        r_warm = t.start_req(
            prompt_len=warm_len, max_new_tokens=1, prompt_token=warm_token
        )
        yield from run_until_finished(r_warm)
        assert r_warm.finished
        assert r_warm.lock_refs == 0

        # Baseline AFTER warming: this is the global lock_ref state the reset
        # path must restore to. The warm prefix nodes are present and evictable
        # (lock_ref == 0) at this point.
        baseline_lock_refs = t.get_all_node_lock_refs()

        # Pin most evictable nodes via the real inc_lock_ref path, leaving only a
        # few unlocked. The KV is still present, just protected/non-evictable.
        t.exhaust_lock_refs(leave_refs=1)
        yield
        pinned_lock_refs = t.get_all_node_lock_refs()
        assert any(
            pinned_lock_refs.get(node_id, 0) > baseline_lock_refs.get(node_id, 0)
            for node_id in pinned_lock_refs
        ), "exhaust_lock_refs(leave_refs=1) must pin at least one warm-prefix node"

        # Run a long chunked req that shares the warm prefix (same prompt_token,
        # longer prompt). It must really chunk while the cache is pinned.
        r_long = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            prompt_token=warm_token,
        )
        yield from run_until(r_long, lambda h: h.is_chunking)
        yield from run_until_finished(r_long, max_steps=2000)
        assert r_long.finished
        # Even with the warm prefix hit, the remaining new tokens span multiple
        # chunks: (VERY_LONG_PROMPT_LEN - warm_len) / DEFAULT_CHUNK_SIZE == 7.
        assert r_long.chunks_done >= 2, (
            f"long req must really chunk under pinned cache; got chunks_done="
            f"{r_long.chunks_done}"
        )
        # The finished req must hold no locks of its own.
        assert (
            r_long.lock_refs == 0
        ), f"req {r_long.rid} leaked {r_long.lock_refs} lock_refs after finish"

        # Release the exhauster's locks (mirrors the reset path) and confirm the
        # global lock_ref state returns EXACTLY to the post-warm baseline: no
        # leaked locks from either the exhauster or the chunked req.
        t._release_exhausted_pools()
        final_lock_refs = t.get_all_node_lock_refs()
        for node_id, baseline in baseline_lock_refs.items():
            assert final_lock_refs.get(node_id, 0) == baseline, (
                f"node {node_id} lock_ref leaked: baseline={baseline}, "
                f"final={final_lock_refs.get(node_id, 0)}"
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
        # Force real forward progress past the retract point: the resumed req must
        # re-chunk and run a strictly later chunk than it had before the retract,
        # not merely fail to regress (which a finished-or-aborted req satisfies
        # vacuously). Only then do we drive it to completion.
        yield from run_until(
            r,
            lambda h: h.chunks_done > chunks_before_retract,
            max_steps=2000,
        )
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.kv_pages == 0
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"KV pool failed to recover after retract+resume: "
            f"baseline={baseline}, final={final}"
        )

    def test_chunked_batch_recovers_pools_to_steady_state(self):
        self.server.execute_script(
            self._script_chunked_batch_recovers_pools_to_steady_state
        )

    @staticmethod
    def _script_chunked_batch_recovers_pools_to_steady_state(t: ScriptedContext):
        before = t.engine_stats()
        # prompt_len must exceed chunk_size so each req actually walks the chunked
        # prefill path; prompt_len=16 (< chunk size) never chunks.
        reqs = [
            t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 1, max_new_tokens=2)
            for _ in range(50)
        ]
        yield from run_until_all_finished(reqs, max_steps=2000)
        for r in reqs:
            assert r.finished
            assert r.kv_pages == 0, f"req {r.rid} kept {r.kv_pages} pages after finish"
        after = t.engine_stats()
        assert after["kv_pool_free"] >= before["kv_pool_free"], (
            f"50 chunked reqs leaked KV: baseline={before['kv_pool_free']}, "
            f"final={after['kv_pool_free']}"
        )
        assert after["req_pool_free"] >= before["req_pool_free"], (
            f"50 chunked reqs leaked req-pool rows: "
            f"baseline={before['req_pool_free']}, final={after['req_pool_free']}"
        )

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
        # Distinct prompt_token per req: with identical prompts r2 would hit
        # r1's cached prefix and never genuinely chunk, so the preemption under
        # test would not occur.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=11)
        yield from run_until(r1, lambda h: h.is_chunking)

        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority=10, prompt_token=12
        )
        # Probe BOTH handles every step (no short-circuit) so the faster req is
        # registered before recycle.
        done = {r1.rid: False, r2.rid: False}
        for _ in range(DEFAULT_MAX_STEPS * 4):
            assert not (r1.is_chunking and r2.is_chunking), (
                f"two reqs cannot share the chunked slot; "
                f"r1.is_chunking={r1.is_chunking}, r2.is_chunking={r2.is_chunking}"
            )
            done[r1.rid] = done[r1.rid] or r1.finished
            done[r2.rid] = done[r2.rid] or r2.finished
            if all(done.values()):
                break
            yield
        assert done[r1.rid] and done[r2.rid]
        assert r1.kv_pages == 0
        assert r2.kv_pages == 0
        # The finished prompts legitimately stay committed in the radix tree
        # (cached != leaked); drain to idle and flush before the leak comparison.
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
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
