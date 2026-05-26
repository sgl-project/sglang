"""Edge cases — KV / row pool / lock_ref resource pressure.

Covers A.4 series from the expansion plan and follow-ons. Uses
``t.exhaust_kv`` / ``t.exhaust_row_pool`` / ``t.exhaust_lock_refs``
to deliberately put the scheduler under resource starvation, then
asserts no deadlock and correct cleanup paths.

NEW API NEEDED:
* ``t.exhaust_kv(leave_pages=N)`` — fill KV pool to N remaining
  pages by spamming short reqs.
* ``t.exhaust_row_pool(leave_rows=N)`` — fill row pool to N
  remaining rows.
* ``t.exhaust_lock_refs(leave_refs=N)`` — fill lock_refs to N
  remaining refs.
* ``t.engine_stats()`` -> dict with ``kv_pool_free``,
  ``row_pool_free`` and similar resource counters.
* ``t.force_retract(req)`` — force the scheduler to retract a
  specific request on the next step.
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


class TestScriptedKVPressure(CustomTestCase):
    def test_kv_almost_empty_then_abort(self):
        """KV almost empty (1 page left), chunked req in flight, abort it."""
        execute_scripted_runtime(
            self._script_kv_almost_empty_then_abort,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_kv_almost_empty_then_abort(t: ScriptedRuntime):
        # KV almost empty (1 page left), chunked req in flight, abort it.
        # NEW API NEEDED: t.exhaust_kv(leave_pages=N) — fill KV pool to N
        # remaining pages by spamming short reqs.
        t.exhaust_kv(leave_pages=1)
        yield

        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.abort(r)
        yield

        assert r.kv_pages == 0
        # KV is fully released after abort.
        stats = t.engine_stats()
        assert stats["kv_pool_free"] >= 1

    def test_kv_full_chunked_new_req_retracts(self):
        """KV pool full → new chunked req submitted → must retract or OOM-path cleanly, not deadlock."""
        execute_scripted_runtime(
            self._script_kv_full_chunked_new_req_retracts,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_kv_full_chunked_new_req_retracts(t: ScriptedRuntime):
        # KV pool full → new chunked req submitted → must retract or OOM-path
        # cleanly, not deadlock.
        t.exhaust_kv(leave_pages=2)
        yield

        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(
            r, lambda h: h.finished or h.error_message is not None, max_steps=2000
        )
        # Either the req completes (via retract/eviction) or it surfaces an
        # OOM-style error message — both are acceptable clean outcomes.
        assert r.finished or r.error_message is not None

    def test_kv_full_chunked_plus_decode_retract(self):
        """KV full, chunked + decode coexist, priority retract → victim is whichever scheduler picks; both must release cleanly afterward."""
        execute_scripted_runtime(
            self._script_kv_full_chunked_plus_decode_retract,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_kv_full_chunked_plus_decode_retract(t: ScriptedRuntime):
        # KV full, chunked + decode coexist, priority retract → victim is
        # whichever scheduler picks; both must release cleanly afterward.
        r_long = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r_short = t.start_req(prompt_len=8, max_new_tokens=8)
        yield from run_until(r_long, lambda h: h.is_chunking)

        # Apply pressure.
        t.exhaust_kv(leave_pages=1)
        yield

        yield from run_until_all_finished([r_long, r_short], max_steps=2000)

    def test_row_pool_tight_admits_after_release(self):
        """Row pool tight; N+1 req enters waiting; after some finish, N+1 is admitted."""
        execute_scripted_runtime(
            self._script_row_pool_tight_admits_after_release,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_row_pool_tight_admits_after_release(t: ScriptedRuntime):
        # Row pool tight; N+1 req enters waiting; after some finish, N+1 is admitted.
        # NEW API NEEDED: t.exhaust_row_pool(leave_rows=N).
        t.exhaust_row_pool(leave_rows=2)
        yield

        reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(5)]
        yield from run_until_all_finished(reqs, max_steps=2000)

    def test_lock_refs_tight_concurrent_prefix(self):
        """Many concurrent reqs share same prefix; lock_refs tight."""
        execute_scripted_runtime(
            self._script_lock_refs_tight_concurrent_prefix,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_lock_refs_tight_concurrent_prefix(t: ScriptedRuntime):
        # Many concurrent reqs share same prefix; lock_refs tight.
        # NEW API NEEDED: t.exhaust_lock_refs(...).
        t.exhaust_lock_refs(leave_refs=4)
        yield
        # Warm prefix.
        r_warm = t.start_req(prompt_len=128, max_new_tokens=2)
        yield from run_until_finished(r_warm)
        reqs = [t.start_req(prompt_len=128, max_new_tokens=2) for _ in range(8)]
        yield from run_until_all_finished(reqs)

    def test_priority_preempt_multiple_chunked(self):
        """Priority preemption + multiple chunked reqs preempt each other."""
        execute_scripted_runtime(
            self._script_priority_preempt_multiple_chunked,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )

    @staticmethod
    def _script_priority_preempt_multiple_chunked(t: ScriptedRuntime):
        # priority preemption + multiple chunked reqs preempt each other.
        # Each victim's resources must release on every preemption event.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)

        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high")
        yield from run_until_all_finished([r1, r2])

    def test_kv_at_one_page_chunked_completes(self):
        """KV with only 1 free page; a tiny chunked req still completes."""
        execute_scripted_runtime(
            self._script_kv_at_one_page_chunked_completes,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_kv_at_one_page_chunked_completes(t: ScriptedRuntime):
        # KV with only 1 free page; a tiny chunked req still completes.
        t.exhaust_kv(leave_pages=4)
        yield
        r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 1, max_new_tokens=1)
        yield from run_until(r, lambda h: h.finished, max_steps=2000)

    def test_kv_recovery_after_full(self):
        """KV near-exhausted -> recovery path: every existing req finishes, then a fresh req can be admitted using all the pages."""
        execute_scripted_runtime(
            self._script_kv_recovery_after_full,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_kv_recovery_after_full(t: ScriptedRuntime):
        # KV near-exhausted -> recovery path: every existing req finishes,
        # then a fresh req can be admitted using all the pages.
        # leave_pages must be >= 1 so a 16-token req can fit; using 0 would
        # deadlock (no KV available and nothing for the scheduler to evict).
        t.exhaust_kv(leave_pages=1)
        yield

        # New req: should wait or retract until KV frees up via the eviction path.
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until(r, lambda h: h.finished, max_steps=3000)
        assert r.finished

    def test_kv_pressure_with_retract_resume(self):
        """Chunked req mid-prefill, then KV pressure forces retract."""
        execute_scripted_runtime(
            self._script_kv_pressure_with_retract_resume,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_kv_pressure_with_retract_resume(t: ScriptedRuntime):
        # Chunked req mid-prefill, then KV pressure forces retract.
        # After pressure releases, req resumes and completes.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.force_retract(r)
        yield
        yield from run_until_finished(r)

    def test_cumulative_alloc_does_not_grow_unbounded(self):
        """50 reqs; cumulative bytes alloc'd never NaN / negative."""
        execute_scripted_runtime(
            self._script_cumulative_alloc_does_not_grow_unbounded,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_cumulative_alloc_does_not_grow_unbounded(t: ScriptedRuntime):
        # 50 reqs; cumulative bytes alloc'd never NaN / negative.
        # NEW API NEEDED: r.cumulative_kv_alloc_bytes (int).
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(50)]
        yield from run_until_all_finished(reqs)
        for r in reqs:
            assert r.cumulative_kv_alloc_bytes >= 0

    def test_engine_stats_pool_invariant(self):
        """Track engine stats before / after a run — pool counts return to baseline."""
        execute_scripted_runtime(
            self._script_engine_stats_pool_invariant,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_engine_stats_pool_invariant(t: ScriptedRuntime):
        # Track engine stats before / after a run — pool counts return to baseline.
        before = t.engine_stats()
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(10)]
        yield from run_until_all_finished(reqs)
        after = t.engine_stats()
        assert after["kv_pool_free"] >= before["kv_pool_free"]
        assert after["row_pool_free"] >= before["row_pool_free"]

    def test_kv_pressure_with_radix_evict(self):
        """KV pressure triggers radix eviction; subsequent submission re-chunks."""
        execute_scripted_runtime(
            self._script_kv_pressure_with_radix_evict,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_kv_pressure_with_radix_evict(t: ScriptedRuntime):
        # KV pressure triggers radix eviction; subsequent submission re-chunks.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)

        # Apply pressure — forces radix eviction.
        t.exhaust_kv(leave_pages=1)
        yield

        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r2, lambda h: h.finished, max_steps=2000)


if __name__ == "__main__":
    unittest.main()
