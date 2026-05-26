"""Multi-req scripted tests for chunked prefill.

Verifies the single-in-flight chunked
invariant from main-upstream (``len(chunked_reqs()) <= 1``), and the
mixed chunked-in-flight + normal-decode batch composition.

Covers A.3 series from the expansion plan plus parametrised
stress / interleaving / rid-reuse scenarios. Verifies that
arbitrary submission timings (back-to-back, mid-yield, sustained
trickle) all reach a clean terminal state for every req.

Also covers B.2 series from the expansion plan plus fan-out across
batch sizes and mixed-shape concurrency.

Round-1 counter wiring: every multi-req test that runs at least one
chunked req asserts ``t.chunked_req_in_batch_violation_count() == 0``
(S2 invariant — ``self.chunked_req`` must be exclusive of
``running_batch.reqs``) and ``r.inflight_middle_chunks_premature_decrement_count
== 0`` on chunked reqs (R1 invariant — decrement only on final chunk).
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


class TestMultiReqBasic(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_at_most_one_chunked_in_flight(self):
        """Two long requests submitted back-to-back; main-upstream invariant says at most one is chunked-in-flight at any moment."""
        self.runtime.run(self._script_at_most_one_chunked_in_flight)

    # two long requests submitted back-to-back; main-upstream
    # invariant says at most one is chunked-in-flight at any moment.
    @staticmethod
    def _script_at_most_one_chunked_in_flight(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)

        # Drive the scheduler until both complete, asserting the
        # single-in-flight invariant AND the S2 violation counter at
        # every step.
        for _ in range(DEFAULT_MAX_STEPS):
            in_flight = t.chunked_in_flight_count()
            assert in_flight <= 1, (
                f"single-in-flight invariant violated: "
                f"chunked_in_flight_count()={in_flight}"
            )
            assert (
                t.chunked_req_in_batch_violation_count() == 0
            ), "S2 violated: chunked_req appeared inside running_batch.reqs"
            if r1.finished and r2.finished:
                # R1 invariant: no premature inflight_middle_chunks
                # decrement observed on either chunked req.
                assert r1.inflight_middle_chunks_premature_decrement_count == 0
                assert r2.inflight_middle_chunks_premature_decrement_count == 0
                return
            yield
        raise AssertionError("r1 and r2 did not both finish within step budget")

    def test_second_chunked_waits(self):
        """R1 chunked mid-stream + r2 submitted long."""
        self.runtime.run(self._script_second_chunked_waits)

    # r1 chunked mid-stream + r2 submitted long. r2 must wait for
    # r1's chunk loop to clear before starting its own chunking.
    @staticmethod
    def _script_second_chunked_waits(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        # r1 is mid-chunk-loop. Submit a second long req.
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield  # give scheduler one step to observe r2

        assert r1.is_chunking, "r1 should still be chunking"
        assert (
            not r2.is_chunking
        ), "r2 must wait for r1's chunk loop to clear before chunking"
        # S2: while r1 is chunked-in-flight, the scheduler must not
        # have admitted it into running_batch.reqs.
        assert t.chunked_req_in_batch_violation_count() == 0

        yield from run_until_all_finished([r1, r2])
        # R1 invariant after completion.
        assert r1.inflight_middle_chunks_premature_decrement_count == 0
        assert r2.inflight_middle_chunks_premature_decrement_count == 0
        assert t.chunked_req_in_batch_violation_count() == 0

    def test_chunked_plus_decode_in_batch(self):
        """R1 chunked mid-stream + r2 short decode-only."""
        self.runtime.run(self._script_chunked_plus_decode_in_batch)

    # r1 chunked mid-stream + r2 short decode-only. r2 should be
    # admittable into the running batch alongside r1's chunked extend.
    @staticmethod
    def _script_chunked_plus_decode_in_batch(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)

        r2 = t.start_req(prompt_len=8, max_new_tokens=4)
        yield  # admission step

        comp = t.batch_composition()
        assert r1.rid in comp.get(
            "chunked", []
        ), f"r1 should be in chunked subset of batch; got {comp}"
        assert r2.rid in comp.get("prefill", []) + comp.get(
            "decode", []
        ), f"r2 should be in prefill or decode subset; got {comp}"
        # S2: chunked subset must be exclusive of decode/prefill subsets
        # at the scheduler level — assert via counter rather than just
        # the batch_composition snapshot.
        assert t.chunked_req_in_batch_violation_count() == 0

        yield from run_until_all_finished([r1, r2])
        assert r1.inflight_middle_chunks_premature_decrement_count == 0
        assert t.chunked_req_in_batch_violation_count() == 0

    def test_hundred_short_reqs(self):
        """100 short reqs back-to-back: all complete, no leak, no S2 violation."""
        self.runtime.run(self._script_hundred_short_reqs)

    @staticmethod
    def _script_hundred_short_reqs(t: ScriptedRuntime):
        # 100 short reqs back-to-back. Beyond r.finished, assert the
        # scheduler-side S2 invariant and confirm KV pool returns to
        # baseline (no leak).
        baseline = t.engine_stats()
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(100)]
        yield from run_until_all_finished(reqs, max_steps=2000)
        for r in reqs:
            assert r.finished
        assert t.chunked_req_in_batch_violation_count() == 0
        final = t.engine_stats()
        assert (
            final["kv_pool_free"] >= baseline["kv_pool_free"]
        ), f"KV leak: {baseline['kv_pool_free']} -> {final['kv_pool_free']}"

    def test_five_hundred_short_reqs(self):
        """500 short reqs: sustained pressure, KV returns to baseline."""
        self.runtime.run(self._script_five_hundred_short_reqs)

    @staticmethod
    def _script_five_hundred_short_reqs(t: ScriptedRuntime):
        # 500 short reqs. Beyond r.finished, assert S2 stays clean and
        # KV pool returns to baseline.
        # max_steps bumped to 20000 for the same reason as the 200-req case.
        baseline = t.engine_stats()
        reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(500)]
        yield from run_until_all_finished(reqs, max_steps=20000)
        for r in reqs:
            assert r.finished
        assert t.chunked_req_in_batch_violation_count() == 0
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_mixed_ten_chunked_ten_short(self):
        """10 chunked + 10 short, all submitted back-to-back."""
        self.runtime.run(self._script_mixed_ten_chunked_ten_short)

    @staticmethod
    def _script_mixed_ten_chunked_ten_short(t: ScriptedRuntime):
        # 10 chunked + 10 short, all submitted back-to-back. Single
        # chunked-in-flight invariant preserved at every step, and
        # S2 scheduler-side counter must stay 0 across the whole run.
        chunked = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(10)
        ]
        shorts = [t.start_req(prompt_len=8, max_new_tokens=2) for _ in range(10)]
        all_reqs = chunked + shorts
        for _ in range(DEFAULT_MAX_STEPS * 20):
            assert t.chunked_in_flight_count() <= 1
            assert t.chunked_req_in_batch_violation_count() == 0
            if all(r.finished for r in all_reqs):
                for r in chunked:
                    assert r.inflight_middle_chunks_premature_decrement_count == 0
                return
            yield
        raise AssertionError("not all reqs finished")

    def test_submit_during_chunk_mid(self):
        """R1 in mid-chunk; r2 submitted after 1 yield; r3 after another."""
        self.runtime.run(self._script_submit_during_chunk_mid)

    @staticmethod
    def _script_submit_during_chunk_mid(t: ScriptedRuntime):
        # r1 in mid-chunk; r2 submitted after 1 yield; r3 after another.
        # All three complete; S2 + R1 stay clean across the run.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield
        r3 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_all_finished([r1, r2, r3])
        assert r1.finished and r2.finished and r3.finished
        assert t.chunked_req_in_batch_violation_count() == 0
        # R1: chunked reqs must never have premature decrement.
        assert r1.inflight_middle_chunks_premature_decrement_count == 0
        assert r2.inflight_middle_chunks_premature_decrement_count == 0

    def test_five_identical_prompts(self):
        """5 identical prompts: r1 chunks; r2..r5 hit radix."""
        self.runtime.run(self._script_five_identical_prompts)

    @staticmethod
    def _script_five_identical_prompts(t: ScriptedRuntime):
        # 5 identical prompts: r1 chunks; r2..r5 hit radix.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)

        others = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(4)
        ]
        yield from run_until_all_finished(others)
        for r in others:
            assert r.finished
            # Cached prefix — should not re-chunk.
            assert r.chunks_done == 0
            # C1: radix cached path must not double-count chunked hits.
            assert r.radix_chunked_hit_inflation_count == 0
        # S2 across the whole multi-req radix-hit sequence.
        assert t.chunked_req_in_batch_violation_count() == 0
        assert r1.inflight_middle_chunks_premature_decrement_count == 0

    def test_sibling_shared_prefix(self):
        """Two reqs share the first N tokens: each runs to completion."""
        self.runtime.run(self._script_sibling_shared_prefix)

    @staticmethod
    def _script_sibling_shared_prefix(t: ScriptedRuntime):
        # Two reqs share the first N tokens: each runs to completion.
        # The shared-prefix branch hits radix; verify C1 (no inflation
        # on chunked hits) and S2 stay clean.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN + 8, max_new_tokens=2)
        yield from run_until_finished(r2)
        assert r1.finished and r2.finished
        assert r1.radix_chunked_hit_inflation_count == 0
        assert r2.radix_chunked_hit_inflation_count == 0
        assert r1.inflight_middle_chunks_premature_decrement_count == 0
        assert r2.inflight_middle_chunks_premature_decrement_count == 0
        assert t.chunked_req_in_batch_violation_count() == 0

    def test_trickle_per_yield_50(self):
        """Submit one new req per yield for 50 yields."""
        self.runtime.run(self._script_trickle_per_yield_50)

    @staticmethod
    def _script_trickle_per_yield_50(t: ScriptedRuntime):
        # Submit one new req per yield for 50 yields. All complete and
        # KV pool returns to baseline. S2 stays 0 across the run.
        baseline = t.engine_stats()
        reqs = []
        for _ in range(50):
            reqs.append(t.start_req(prompt_len=8, max_new_tokens=2))
            yield
            assert t.chunked_req_in_batch_violation_count() == 0
        yield from run_until_all_finished(reqs, max_steps=2000)
        for r in reqs:
            assert r.finished
        assert t.chunked_req_in_batch_violation_count() == 0
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_submit_then_immediate_abort(self):
        """Start_req then abort in same yield step: clean state, no double-release."""
        self.runtime.run(self._script_submit_then_immediate_abort)

    @staticmethod
    def _script_submit_then_immediate_abort(t: ScriptedRuntime):
        # start_req then abort in same yield step: clean state.
        # Note: same-step abort vs admission ordering is impl-defined;
        # assertion holds either way (whether or not the req was admitted
        # before the abort took effect, no KV/row should remain after).
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        t.abort(r)
        # Drain a few iters to let abort propagate (the dual-queue path
        # can take a couple of steps to settle).
        for _ in range(5):
            yield
        assert r.kv_pages == 0
        assert r.row_idx is None
        # W3: abort path must not double-release the req's resources.
        assert r.abort_double_release_count == 0
        assert t.chunked_req_in_batch_violation_count() == 0

    def test_rid_reuse_after_finish(self):
        """Submit r1, wait for finish, then submit r2 with same rid."""
        self.runtime.run(self._script_rid_reuse_after_finish)

    @staticmethod
    def _script_rid_reuse_after_finish(t: ScriptedRuntime):
        # Submit r1, wait for finish, then submit r2 with same rid.
        # NEW API NEEDED: start_req(..., rid="...") — explicit rid control.
        baseline = t.engine_stats()
        r1 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-rid")
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-rid")
        yield from run_until_finished(r2)
        assert r1.finished and r2.finished
        assert t.chunked_req_in_batch_violation_count() == 0
        # Pool must have recovered after r1 finished + r2 finished.
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_three_long_back_to_back(self):
        """Three long chunked reqs submitted back-to-back."""
        self.runtime.run(self._script_three_long_back_to_back)

    @staticmethod
    def _script_three_long_back_to_back(t: ScriptedRuntime):
        # Three long chunked reqs submitted back-to-back.
        # Beyond r.finished, assert chunked_in_flight stays <= 1 and
        # S2 / R1 counters remain clean across the lifecycle.
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(3)
        ]
        for _ in range(DEFAULT_MAX_STEPS * 10):
            assert t.chunked_in_flight_count() <= 1
            assert t.chunked_req_in_batch_violation_count() == 0
            if all(r.finished for r in reqs):
                break
            yield
        for r in reqs:
            assert r.finished
            assert r.inflight_middle_chunks_premature_decrement_count == 0

    def test_submit_pause_n_resubmit_same_rid(self):
        """Submit and complete r1, then 200 yields, then resubmit with same rid."""
        self.runtime.run(self._script_submit_pause_n_resubmit_same_rid)

    @staticmethod
    def _script_submit_pause_n_resubmit_same_rid(t: ScriptedRuntime):
        # Submit and complete r1, then 200 yields, then resubmit with
        # same rid. The idle yield-gap must not leave any state behind.
        baseline = t.engine_stats()
        r1 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-200")
        yield from run_until_finished(r1)
        # After r1 finishes, the pool should be free + no chunked req
        # should be in flight across the idle window.
        for _ in range(200):
            assert t.chunked_in_flight_count() == 0
            assert t.chunked_req_in_batch_violation_count() == 0
            yield
        r2 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-200")
        yield from run_until_finished(r2)
        assert r2.finished
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_submit_during_decode_of_other(self):
        """R1 in decode phase; submit r2 (chunked)."""
        self.runtime.run(self._script_submit_during_decode_of_other)

    @staticmethod
    def _script_submit_during_decode_of_other(t: ScriptedRuntime):
        # r1 in decode phase; submit r2 (chunked). Both complete.
        # S2 + R1 must stay clean since r2 will be the active chunked
        # req while r1 decodes alongside.
        r1 = t.start_req(prompt_len=16, max_new_tokens=16)
        yield from run_until(r1, lambda h: h.status == "running")
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS * 5):
            assert t.chunked_req_in_batch_violation_count() == 0
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished and r2.finished
        assert r2.inflight_middle_chunks_premature_decrement_count == 0

    def test_unique_rids_distinct(self):
        """Many reqs with unique explicit rids."""
        self.runtime.run(self._script_unique_rids_distinct)

    @staticmethod
    def _script_unique_rids_distinct(t: ScriptedRuntime):
        # Many reqs with unique explicit rids. KV pool must return to
        # baseline and S2 must remain clean.
        baseline = t.engine_stats()
        reqs = [
            t.start_req(prompt_len=16, max_new_tokens=2, rid=f"unique-{i}")
            for i in range(20)
        ]
        yield from run_until_all_finished(reqs)
        rids = {r.rid for r in reqs}
        assert len(rids) == 20
        assert t.chunked_req_in_batch_violation_count() == 0
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_two_small_parallel(self):
        """Two short parallel reqs both finish; no chunked admission triggered."""
        self.runtime.run(self._script_two_small_parallel)

    @staticmethod
    def _script_two_small_parallel(t: ScriptedRuntime):
        # Two short reqs. Negative control for chunked admission —
        # the scheduler's chunked slot must never be populated.
        r1 = t.start_req(prompt_len=16, max_new_tokens=4)
        r2 = t.start_req(prompt_len=16, max_new_tokens=4)
        for _ in range(DEFAULT_MAX_STEPS):
            assert t.get_chunked_req_rid() is None
            assert t.chunked_req_in_batch_violation_count() == 0
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished and r2.finished

    def test_one_chunked_plus_many_short(self):
        """1 long chunked + 5 short, all parallel."""
        self.runtime.run(self._script_one_chunked_plus_many_short)

    @staticmethod
    def _script_one_chunked_plus_many_short(t: ScriptedRuntime):
        # 1 long chunked + 5 short, all parallel.
        chunked = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        shorts = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(5)]
        for _ in range(DEFAULT_MAX_STEPS * 5):
            assert t.chunked_in_flight_count() <= 1
            assert t.chunked_req_in_batch_violation_count() == 0
            if chunked.finished and all(s.finished for s in shorts):
                break
            yield
        assert chunked.chunks_done >= 2
        assert chunked.inflight_middle_chunks_premature_decrement_count == 0

    def test_multiple_chunked_staggered(self):
        """Submit chunked reqs every few yields, serial chunking."""
        self.runtime.run(self._script_multiple_chunked_staggered)

    @staticmethod
    def _script_multiple_chunked_staggered(t: ScriptedRuntime):
        # Submit chunked reqs every few yields, serial chunking.
        reqs = []
        for _ in range(4):
            reqs.append(t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2))
            yield
            assert t.chunked_in_flight_count() <= 1
            assert t.chunked_req_in_batch_violation_count() == 0
            yield
        for _ in range(DEFAULT_MAX_STEPS * 10):
            assert t.chunked_in_flight_count() <= 1
            assert t.chunked_req_in_batch_violation_count() == 0
            if all(r.finished for r in reqs):
                break
            yield
        for r in reqs:
            assert r.finished
            assert r.inflight_middle_chunks_premature_decrement_count == 0

    def test_eight_concurrent_chunked(self):
        """8 chunked reqs submitted together."""
        self.runtime.run(self._script_eight_concurrent_chunked)

    @staticmethod
    def _script_eight_concurrent_chunked(t: ScriptedRuntime):
        # 8 chunked reqs submitted together. Single-in-flight invariant
        # holds, S2 stays clean, R1 stays clean across the whole run.
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(8)
        ]
        for _ in range(DEFAULT_MAX_STEPS * 5):
            assert t.chunked_in_flight_count() <= 1
            assert t.chunked_req_in_batch_violation_count() == 0
            if all(r.finished for r in reqs):
                for r in reqs:
                    assert r.inflight_middle_chunks_premature_decrement_count == 0
                return
            yield
        raise AssertionError("not all reqs finished")

    def test_decode_only_batch(self):
        """10 short reqs — pure decode batch, chunked slot must remain None."""
        self.runtime.run(self._script_decode_only_batch)

    @staticmethod
    def _script_decode_only_batch(t: ScriptedRuntime):
        # 10 short reqs — pure decode batch. Negative control for the
        # chunked admission path: the scheduler's chunked_req slot must
        # remain None and S2 must stay clean across the whole lifecycle.
        reqs = [t.start_req(prompt_len=4, max_new_tokens=8) for _ in range(10)]
        for _ in range(DEFAULT_MAX_STEPS * 3):
            assert (
                t.get_chunked_req_rid() is None
            ), "pure decode workload must never populate chunked_req"
            assert t.chunked_in_flight_count() == 0
            assert t.chunked_req_in_batch_violation_count() == 0
            if all(r.finished for r in reqs):
                return
            yield
        raise AssertionError("not all reqs finished")

    def test_mixed_prefill_lengths(self):
        """Variable prompt lengths in same batch."""
        self.runtime.run(self._script_mixed_prefill_lengths)

    @staticmethod
    def _script_mixed_prefill_lengths(t: ScriptedRuntime):
        # Variable prompt lengths in same batch. Some will chunk
        # (the longer ones), others won't. S2 + R1 must stay clean
        # across the whole run.
        lens = [8, 16, 32, 64, 128, 256, 512, 1024]
        reqs = [t.start_req(prompt_len=L, max_new_tokens=2) for L in lens]
        for _ in range(DEFAULT_MAX_STEPS * 10):
            assert t.chunked_in_flight_count() <= 1
            assert t.chunked_req_in_batch_violation_count() == 0
            if all(r.finished for r in reqs):
                break
            yield
        for r in reqs:
            assert r.finished
            assert r.inflight_middle_chunks_premature_decrement_count == 0

    def test_two_chunked_one_decode(self):
        """2 chunked + 1 decode-only."""
        self.runtime.run(self._script_two_chunked_one_decode)

    @staticmethod
    def _script_two_chunked_one_decode(t: ScriptedRuntime):
        # 2 chunked + 1 decode-only. Verify in-flight invariant +
        # S2 + R1 across the whole run.
        chunked1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        chunked2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        short = t.start_req(prompt_len=8, max_new_tokens=4)
        for _ in range(DEFAULT_MAX_STEPS * 10):
            assert t.chunked_in_flight_count() <= 1
            assert t.chunked_req_in_batch_violation_count() == 0
            if chunked1.finished and chunked2.finished and short.finished:
                break
            yield
        assert chunked1.finished and chunked2.finished and short.finished
        assert chunked1.inflight_middle_chunks_premature_decrement_count == 0
        assert chunked2.inflight_middle_chunks_premature_decrement_count == 0

    def test_batch_with_finish_event_count(self):
        """Each req emits exactly 1 finish event."""
        self.runtime.run(self._script_batch_with_finish_event_count)

    @staticmethod
    def _script_batch_with_finish_event_count(t: ScriptedRuntime):
        # Each req emits exactly 1 finish event.
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(6)]
        yield from run_until_all_finished(reqs)
        for r in reqs:
            assert r.finish_event_count == 1
        assert t.chunked_req_in_batch_violation_count() == 0

    def test_batch_state_query_during_run(self):
        """Query batch_composition every step while batch is active."""
        self.runtime.run(self._script_batch_state_query_during_run)

    @staticmethod
    def _script_batch_state_query_during_run(t: ScriptedRuntime):
        # Query batch_composition every step while batch is active.
        # Beyond the type check, verify that prefill / decode / chunked
        # subsets remain disjoint at every step, and S2 stays clean.
        reqs = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(4)]
        for _ in range(DEFAULT_MAX_STEPS):
            comp = t.batch_composition()
            assert isinstance(comp, dict)
            prefill = set(comp.get("prefill", []))
            decode = set(comp.get("decode", []))
            chunked = set(comp.get("chunked", []))
            assert prefill & decode == set()
            assert prefill & chunked == set()
            assert decode & chunked == set()
            assert t.chunked_req_in_batch_violation_count() == 0
            if all(r.finished for r in reqs):
                return
            yield
        raise AssertionError("not all reqs finished")

    def test_mixed_lengths_then_more_arrivals(self):
        """First batch starts; midway, more reqs arrive."""
        self.runtime.run(self._script_mixed_lengths_then_more_arrivals)

    @staticmethod
    def _script_mixed_lengths_then_more_arrivals(t: ScriptedRuntime):
        # First batch starts; midway, more reqs arrive. S2 stays clean
        # across the admission of the second batch.
        initial = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(3)]
        yield
        yield
        assert t.chunked_req_in_batch_violation_count() == 0
        more = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(3)]
        yield from run_until_all_finished(initial + more)
        for r in initial + more:
            assert r.finished
        assert t.chunked_req_in_batch_violation_count() == 0


class TestMultiReqPriority(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_parallel_with_priority(self):
        """3 normal + 2 high-priority reqs all complete; S2 stays clean."""
        self.runtime.run(self._script_parallel_with_priority)

    @staticmethod
    def _script_parallel_with_priority(t: ScriptedRuntime):
        # 3 normal + 2 high-priority reqs. Priority scheduling must not
        # break the S2 exclusivity invariant.
        baseline = t.engine_stats()
        normal = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(3)]
        high = [
            t.start_req(prompt_len=16, max_new_tokens=2, priority="high")
            for _ in range(2)
        ]
        yield from run_until_all_finished(normal + high)
        for r in normal + high:
            assert r.finished
        assert t.chunked_req_in_batch_violation_count() == 0
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]


if __name__ == "__main__":
    unittest.main()
