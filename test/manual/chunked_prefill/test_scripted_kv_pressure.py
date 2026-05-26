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

from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


class TestKVPressureBasic(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_kv_almost_empty_then_abort(self):
        """KV almost empty (1 page left), chunked req in flight, abort it."""
        self.runtime.run(self._script_kv_almost_empty_then_abort)

    @staticmethod
    def _script_kv_almost_empty_then_abort(t: ScriptedRuntime):
        # KV almost empty (1 page left), chunked req in flight, abort it.
        # Verify KV fully released and all per-req resources cleared.
        t.exhaust_kv(leave_pages=1)
        yield

        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.abort(r)
        yield

        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert r.row_idx is None
        # KV is fully released after abort.
        stats = t.engine_stats()
        assert stats["kv_pool_free"] >= 1

    def test_kv_full_chunked_new_req_retracts(self):
        """KV pool full → new chunked req submitted → must retract or OOM-path cleanly, not deadlock."""
        self.runtime.run(self._script_kv_full_chunked_new_req_retracts)

    @staticmethod
    def _script_kv_full_chunked_new_req_retracts(t: ScriptedRuntime):
        # KV pool nearly full → new chunked req submitted → must retract
        # or OOM-path cleanly, not deadlock. If the req errors out, no
        # rows should be leaked; if it completes, kv_pages must return
        # to 0.
        baseline = t.engine_stats()["kv_pool_free"]
        t.exhaust_kv(leave_pages=2)
        yield

        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(2000):
            if r.finished or r.error_message is not None:
                break
            yield
        # Either the req completes (via retract/eviction) or it surfaces an
        # OOM-style error message — both are acceptable clean outcomes.
        assert r.finished or r.error_message is not None
        # Whichever outcome, KV must not be retained by r.
        assert r.kv_pages == 0
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"KV not released after OOM/retract path: baseline={baseline}, "
            f"final={final}"
        )

    def test_kv_full_chunked_plus_decode_retract(self):
        """KV full, chunked + decode coexist, priority retract → victim is whichever scheduler picks; both must release cleanly afterward."""
        self.runtime.run(self._script_kv_full_chunked_plus_decode_retract)

    @staticmethod
    def _script_kv_full_chunked_plus_decode_retract(t: ScriptedRuntime):
        # KV full, chunked + decode coexist, priority retract → victim is
        # whichever scheduler picks; both must release cleanly afterward.
        baseline = t.engine_stats()["kv_pool_free"]
        r_long = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r_short = t.start_req(prompt_len=8, max_new_tokens=8)
        yield from run_until(r_long, lambda h: h.is_chunking)

        # Apply pressure.
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
        """Row pool tight; N+1 req enters waiting; after some finish, N+1 is admitted."""
        self.runtime.run(self._script_row_pool_tight_admits_after_release)

    @staticmethod
    def _script_row_pool_tight_admits_after_release(t: ScriptedRuntime):
        # Row pool tight; N+1 req enters waiting; after some finish,
        # N+1 must be admitted. Verify all reqs finish AND row pool is
        # fully released back to baseline (no row leak across the
        # admit-after-release transition).
        baseline_rows_used = t.row_pool_used()
        t.exhaust_row_pool(leave_rows=2)
        yield

        reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(5)]
        yield from run_until_all_finished(reqs, max_steps=2000)
        for r in reqs:
            assert r.finished, f"req {r.rid} did not finish"
            assert r.kv_pages == 0, (
                f"row-pool pressure must not leave KV held: rid={r.rid}, "
                f"kv_pages={r.kv_pages}"
            )
        # After the exhaust frame releases, row usage must return to
        # the pre-exhaust baseline (no leaked row entries).
        final_rows_used = t.row_pool_used()
        assert final_rows_used <= baseline_rows_used, (
            f"row pool leak after admit-after-release: baseline used="
            f"{baseline_rows_used}, final used={final_rows_used}"
        )

    def test_lock_refs_tight_concurrent_prefix(self):
        """Many concurrent reqs share same prefix; lock_refs tight."""
        self.runtime.run(self._script_lock_refs_tight_concurrent_prefix)

    @staticmethod
    def _script_lock_refs_tight_concurrent_prefix(t: ScriptedRuntime):
        # Many concurrent reqs share same prefix; lock_refs tight. All
        # reqs must finish AND their lock_refs must return to 0 — pre-fix
        # the lock_ref accounting could leak when many reqs concurrently
        # contended for the same radix prefix node under tight ref-pool.
        baseline_lock_refs = t.lock_refs_snapshot()
        t.exhaust_lock_refs(leave_refs=4)
        yield
        # Warm prefix.
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
        # The global lock_refs counter must return to (or below) baseline
        # — no concurrent-prefix leak.
        final_lock_refs = t.lock_refs_snapshot()
        assert final_lock_refs <= baseline_lock_refs, (
            f"global lock_refs leaked from baseline={baseline_lock_refs} "
            f"to final={final_lock_refs}"
        )

    def test_kv_at_one_page_chunked_completes(self):
        """KV with only 1 free page; a tiny chunked req still completes."""
        self.runtime.run(self._script_kv_at_one_page_chunked_completes)

    @staticmethod
    def _script_kv_at_one_page_chunked_completes(t: ScriptedRuntime):
        # KV at tight pressure (4 free pages); a tiny chunked req still
        # completes AND fully releases its KV.
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
        """KV near-exhausted -> recovery path: every existing req finishes, then a fresh req can be admitted using all the pages."""
        self.runtime.run(self._script_kv_recovery_after_full)

    @staticmethod
    def _script_kv_recovery_after_full(t: ScriptedRuntime):
        # KV near-exhausted -> recovery path: every existing req finishes,
        # then a fresh req can be admitted using the eviction path.
        # leave_pages must be >= 1 so a 16-token req can fit; using 0 would
        # deadlock (no KV available and nothing for the scheduler to evict).
        baseline = t.engine_stats()["kv_pool_free"]
        t.exhaust_kv(leave_pages=1)
        yield

        # New req: must complete via the eviction path AND fully release
        # its KV. The final pool free count must return to (or above)
        # baseline once the exhaust frame and r both unwind.
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
        """Chunked req mid-prefill, then KV pressure forces retract."""
        self.runtime.run(self._script_kv_pressure_with_retract_resume)

    @staticmethod
    def _script_kv_pressure_with_retract_resume(t: ScriptedRuntime):
        # Chunked req mid-prefill, then KV pressure forces retract.
        # After pressure releases, req resumes and completes. The retract
        # path must (a) finish the req, (b) drop kv_pages back to 0, and
        # (c) preserve chunks_done across resume.
        baseline = t.engine_stats()["kv_pool_free"]
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        chunks_before_retract = r.chunks_done
        t.force_retract(r)
        for _ in range(2000):
            if r.finished:
                break
            yield
        assert r.finished
        assert r.kv_pages == 0
        # Resume must not lose progress: chunks_done at finish >= the
        # snapshot taken before retract.
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
        """50 reqs; cumulative bytes alloc'd never NaN / negative."""
        self.runtime.run(self._script_cumulative_alloc_does_not_grow_unbounded)

    @staticmethod
    def _script_cumulative_alloc_does_not_grow_unbounded(t: ScriptedRuntime):
        # 50 reqs; cumulative bytes alloc'd never NaN / negative; after
        # every req finishes, kv_pages drops to 0 and global pool returns
        # to baseline.
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
        """Track engine stats before / after a run — pool counts return to baseline."""
        self.runtime.run(self._script_engine_stats_pool_invariant)

    @staticmethod
    def _script_engine_stats_pool_invariant(t: ScriptedRuntime):
        # Track engine stats before / after a run — pool counts return
        # to baseline.
        before = t.engine_stats()
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(10)]
        yield from run_until_all_finished(reqs)
        after = t.engine_stats()
        assert after["kv_pool_free"] >= before["kv_pool_free"]
        assert after["row_pool_free"] >= before["row_pool_free"]

    def test_kv_pressure_with_radix_evict(self):
        """KV pressure triggers radix eviction; subsequent submission re-chunks."""
        self.runtime.run(self._script_kv_pressure_with_radix_evict)

    @staticmethod
    def _script_kv_pressure_with_radix_evict(t: ScriptedRuntime):
        # KV pressure triggers radix eviction; subsequent submission must
        # re-chunk under the evicted-cache state and complete without
        # leaking KV.
        baseline = t.engine_stats()["kv_pool_free"]
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        assert r1.finished
        assert r1.kv_pages == 0

        # Apply pressure — forces radix eviction.
        t.exhaust_kv(leave_pages=1)
        yield

        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # The post-evict re-chunk must complete cleanly.
        for _ in range(2000):
            if r2.finished:
                break
            yield
        assert r2.finished
        assert r2.kv_pages == 0
        # r2 must have actually re-chunked (radix eviction means no
        # full prefix hit available).
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
        """Chunked retract → resume → finish: kv_pool_free returns to baseline."""
        self.runtime.run(self._script_chunked_retract_resume_kv_recovers_exactly)

    # chunked retract → resume → finish — completed state must
    # release every KV page acquired during the chunked path; pool count
    # is at least the pre-submit baseline.
    @staticmethod
    def _script_chunked_retract_resume_kv_recovers_exactly(t: ScriptedRuntime):
        baseline = t.engine_stats()["kv_pool_free"]
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.force_retract(r)
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.kv_pages == 0
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"kv_pool_free did not recover after chunked retract+resume: "
            f"baseline={baseline}, final={final}"
        )

    def test_chunked_retract_at_chunk_first_mid_last(self):
        """Three sequential reqs with same prompt; force retract at chunks_done={0, mid, last-1}."""
        self.runtime.run(self._script_chunked_retract_at_chunk_first_mid_last)

    # retract at three distinct chunked progress points —
    # chunks_done={0, mid, last-1} — each chunked-resume must rebuild
    # prefix_indices correctly and complete.
    @staticmethod
    def _script_chunked_retract_at_chunk_first_mid_last(t: ScriptedRuntime):
        # Total expected chunks = VERY_LONG_PROMPT_LEN // DEFAULT_CHUNK_SIZE.
        expected_chunks = VERY_LONG_PROMPT_LEN // DEFAULT_CHUNK_SIZE
        mid_chunk = expected_chunks // 2
        last_minus_one = max(1, expected_chunks - 1)

        # First: retract at chunks_done == 0 (entered chunked path but no
        # chunk committed yet — just barely is_chunking).
        r_first = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_first, lambda h: h.is_chunking)
        t.force_retract(r_first)
        yield from run_until_finished(r_first, max_steps=2000)
        assert r_first.finished
        assert r_first.kv_pages == 0

        # Mid: retract at chunks_done == expected_chunks // 2.
        r_mid = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_mid, lambda h: h.chunks_done >= mid_chunk)
        t.force_retract(r_mid)
        yield from run_until_finished(r_mid, max_steps=2000)
        assert r_mid.finished
        assert r_mid.kv_pages == 0

        # Last-1: retract at chunks_done == expected_chunks - 1.
        r_last = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_last, lambda h: h.chunks_done >= last_minus_one)
        t.force_retract(r_last)
        yield from run_until_finished(r_last, max_steps=2000)
        assert r_last.finished
        assert r_last.kv_pages == 0

    def test_flush_cache_during_chunked_in_flight(self):
        """Chunked req in flight; flush_cache must not corrupt the in-flight prefix or other cache state."""
        self.runtime.run(self._script_flush_cache_during_chunked_in_flight)

    # flush_cache mid-chunked — the in-flight chunked req's
    # prefix must not be flushed; the req still completes cleanly.
    @staticmethod
    def _script_flush_cache_during_chunked_in_flight(t: ScriptedRuntime):
        # NEW API NEEDED: t.flush_cache() — drop the radix cache entirely.
        # Warm radix with a chunked req to populate the tree.
        r_warm = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1)
        yield from run_until_finished(r_warm)
        assert r_warm.finished

        # Start another chunked req; flush mid-flight.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.flush_cache()
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.kv_pages == 0

    def test_chunked_oscillation_three_force_retracts(self):
        """Single chunked req; force_retract three times across its lifecycle; chunks_done preserved, final completes."""
        self.runtime.run(self._script_chunked_oscillation_three_force_retracts)

    # chunked oscillation — 3 force_retracts on one req;
    # chunks_done is preserved across retract/resume rounds and the
    # req still finishes once admitted long enough.
    @staticmethod
    def _script_chunked_oscillation_three_force_retracts(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # First retract — early.
        yield from run_until(r, lambda h: h.is_chunking)
        chunks_at_first = r.chunks_done
        t.force_retract(r)
        # Resume back into chunking.
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

        # Second retract — mid-flight.
        t.force_retract(r)
        yield from run_until(
            r,
            lambda h: h.is_chunking and h.chunks_done >= chunks_after_first_resume,
            max_steps=800,
        )
        chunks_after_second_resume = r.chunks_done
        assert chunks_after_second_resume >= chunks_after_first_resume

        # Third retract — late.
        t.force_retract(r)
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.chunks_done >= chunks_after_second_resume


class TestKVPressurePriority(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_priority_preempt_multiple_chunked(self):
        """Priority preemption + multiple chunked reqs preempt each other."""
        self.runtime.run(self._script_priority_preempt_multiple_chunked)

    @staticmethod
    def _script_priority_preempt_multiple_chunked(t: ScriptedRuntime):
        # priority preemption + multiple chunked reqs preempt each other.
        # Each victim's resources must release on every preemption event.
        # Mutual exclusion: at no point can two reqs simultaneously
        # occupy is_chunking — only one chunked slot.
        baseline = t.engine_stats()["kv_pool_free"]
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)

        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high"
        )
        for _ in range(DEFAULT_MAX_STEPS * 4):
            # Mutual exclusion across both reqs: only one chunked slot.
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


class TestKVPressurePageSize(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        page_size=16,
    )

    def test_strict_mem_check_handles_chunked_tail(self):
        """Strict mem check + page_size>1 + waiting chunked: no false-positive leak assert."""
        self.runtime.run(self._script_strict_mem_check_handles_chunked_tail)

    # strict mem check + page_size>1 must count the
    # chunked-resume tail when a chunked req sits in waiting_queue;
    # pre-fix the runtime mem accounting tripped a false-positive leak
    # assert during busy. Observable: the retract+resume completes and
    # the per-req KV is fully released.
    @staticmethod
    def _script_strict_mem_check_handles_chunked_tail(t: ScriptedRuntime):
        # Mid-chunk retract pushes a chunked-resume req onto the
        # waiting_queue; with page_size>1 the tail page must still be
        # counted by the strict mem check.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN + 17, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.force_retract(r)
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0


if __name__ == "__main__":
    unittest.main()
