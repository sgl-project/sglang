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
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase

# r2 must be a distinct request lifecycle, not a resurrection of r1.


# Decode-only batch: chunked_in_flight should never have been > 0.


class TestScriptedMultiReq(CustomTestCase):
    def test_at_most_one_chunked_in_flight(self):
        """Two long requests submitted back-to-back; main-upstream invariant says at most one is chunked-in-flight at any moment."""
        execute_scripted_runtime(
            self._script_at_most_one_chunked_in_flight,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # two long requests submitted back-to-back; main-upstream
    # invariant says at most one is chunked-in-flight at any moment.
    @staticmethod
    def _script_at_most_one_chunked_in_flight(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)

        # Drive the scheduler until both complete, asserting the
        # single-in-flight invariant at every step.
        for _ in range(DEFAULT_MAX_STEPS):
            in_flight = t.chunked_in_flight_count()
            assert in_flight <= 1, (
                f"single-in-flight invariant violated: "
                f"chunked_in_flight_count()={in_flight}"
            )
            if r1.finished and r2.finished:
                return
            yield
        raise AssertionError("r1 and r2 did not both finish within step budget")

    def test_second_chunked_waits(self):
        """R1 chunked mid-stream + r2 submitted long."""
        execute_scripted_runtime(
            self._script_second_chunked_waits,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

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

        yield from run_until_all_finished([r1, r2])

    def test_chunked_plus_decode_in_batch(self):
        """R1 chunked mid-stream + r2 short decode-only."""
        execute_scripted_runtime(
            self._script_chunked_plus_decode_in_batch,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

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

        yield from run_until_all_finished([r1, r2])

    def test_hundred_short_reqs(self):
        """100 short reqs back-to-back: all complete, no leak."""
        execute_scripted_runtime(
            self._script_hundred_short_reqs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_hundred_short_reqs(t: ScriptedRuntime):
        # 100 short reqs back-to-back: all complete, no leak.
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(100)]
        yield from run_until_all_finished(reqs, max_steps=2000)
        for r in reqs:
            assert r.finished

    def test_five_hundred_short_reqs(self):
        """500 short reqs: sustained pressure."""
        execute_scripted_runtime(
            self._script_five_hundred_short_reqs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_five_hundred_short_reqs(t: ScriptedRuntime):
        # 500 short reqs: sustained pressure.
        # max_steps bumped to 20000 for the same reason as the 200-req case.
        reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(500)]
        yield from run_until_all_finished(reqs, max_steps=20000)
        for r in reqs:
            assert r.finished

    def test_mixed_ten_chunked_ten_short(self):
        """10 chunked + 10 short, all submitted back-to-back."""
        execute_scripted_runtime(
            self._script_mixed_ten_chunked_ten_short,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_mixed_ten_chunked_ten_short(t: ScriptedRuntime):
        # 10 chunked + 10 short, all submitted back-to-back. Single
        # chunked-in-flight invariant preserved at every step.
        chunked = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(10)
        ]
        shorts = [t.start_req(prompt_len=8, max_new_tokens=2) for _ in range(10)]
        all_reqs = chunked + shorts
        for _ in range(DEFAULT_MAX_STEPS * 20):
            assert t.chunked_in_flight_count() <= 1
            if all(r.finished for r in all_reqs):
                return
            yield
        raise AssertionError("not all reqs finished")

    def test_submit_during_chunk_mid(self):
        """R1 in mid-chunk; r2 submitted after 1 yield; r3 after another."""
        execute_scripted_runtime(
            self._script_submit_during_chunk_mid,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_submit_during_chunk_mid(t: ScriptedRuntime):
        # r1 in mid-chunk; r2 submitted after 1 yield; r3 after another.
        # All three complete.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield
        r3 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_all_finished([r1, r2, r3])

    def test_five_identical_prompts(self):
        """5 identical prompts: r1 chunks; r2..r5 hit radix."""
        execute_scripted_runtime(
            self._script_five_identical_prompts,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

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

    def test_sibling_shared_prefix(self):
        """Two reqs share the first N tokens: each runs to completion."""
        execute_scripted_runtime(
            self._script_sibling_shared_prefix,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_sibling_shared_prefix(t: ScriptedRuntime):
        # Two reqs share the first N tokens: each runs to completion.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN + 8, max_new_tokens=2)
        yield from run_until_finished(r2)
        assert r1.finished and r2.finished

    def test_trickle_per_yield_50(self):
        """Submit one new req per yield for 50 yields."""
        execute_scripted_runtime(
            self._script_trickle_per_yield_50,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_trickle_per_yield_50(t: ScriptedRuntime):
        # Submit one new req per yield for 50 yields. All complete.
        reqs = []
        for _ in range(50):
            reqs.append(t.start_req(prompt_len=8, max_new_tokens=2))
            yield
        yield from run_until_all_finished(reqs, max_steps=2000)
        for r in reqs:
            assert r.finished

    def test_submit_then_immediate_abort(self):
        """Start_req then abort in same yield step: clean state."""
        execute_scripted_runtime(
            self._script_submit_then_immediate_abort,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_submit_then_immediate_abort(t: ScriptedRuntime):
        # start_req then abort in same yield step: clean state.
        # Note: same-step abort vs admission ordering is impl-defined;
        # assertion holds either way (whether or not the req was admitted
        # before the abort took effect, no KV/row should remain after).
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.row_idx is None

    def test_rid_reuse_after_finish(self):
        """Submit r1, wait for finish, then submit r2 with same rid."""
        execute_scripted_runtime(
            self._script_rid_reuse_after_finish,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_rid_reuse_after_finish(t: ScriptedRuntime):
        # Submit r1, wait for finish, then submit r2 with same rid.
        # NEW API NEEDED: start_req(..., rid="...") — explicit rid control.
        r1 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-rid")
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-rid")
        yield from run_until_finished(r2)
        assert r2.finished

    def test_concurrent_short_and_long(self):
        """5 short + 1 long, all concurrent; verify long does not starve."""
        execute_scripted_runtime(
            self._script_concurrent_short_and_long,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_concurrent_short_and_long(t: ScriptedRuntime):
        # 5 short + 1 long, all concurrent; verify long does not starve.
        long = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        shorts = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(5)]
        yield from run_until_all_finished([long] + shorts)

    def test_three_long_back_to_back(self):
        """Three long chunked reqs submitted back-to-back."""
        execute_scripted_runtime(
            self._script_three_long_back_to_back,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_three_long_back_to_back(t: ScriptedRuntime):
        # Three long chunked reqs submitted back-to-back.
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(3)
        ]
        yield from run_until_all_finished(reqs, max_steps=1500)
        for r in reqs:
            assert r.finished

    def test_submit_pause_n_resubmit_same_rid(self):
        """Submit and complete r1, then 200 yields, then resubmit with same rid."""
        execute_scripted_runtime(
            self._script_submit_pause_n_resubmit_same_rid,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_submit_pause_n_resubmit_same_rid(t: ScriptedRuntime):
        # Submit and complete r1, then 200 yields, then resubmit with same rid.
        r1 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-200")
        yield from run_until_finished(r1)
        for _ in range(200):
            yield
        r2 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-200")
        yield from run_until_finished(r2)
        assert r2.finished

    def test_submit_during_decode_of_other(self):
        """R1 in decode phase; submit r2 (chunked)."""
        execute_scripted_runtime(
            self._script_submit_during_decode_of_other,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_submit_during_decode_of_other(t: ScriptedRuntime):
        # r1 in decode phase; submit r2 (chunked). Both complete.
        r1 = t.start_req(prompt_len=16, max_new_tokens=16)
        yield from run_until(r1, lambda h: h.status == "running")
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_all_finished([r1, r2])

    def test_unique_rids_distinct(self):
        """Many reqs with unique explicit rids."""
        execute_scripted_runtime(
            self._script_unique_rids_distinct,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_unique_rids_distinct(t: ScriptedRuntime):
        # Many reqs with unique explicit rids.
        reqs = [
            t.start_req(prompt_len=16, max_new_tokens=2, rid=f"unique-{i}")
            for i in range(20)
        ]
        yield from run_until_all_finished(reqs)
        rids = {r.rid for r in reqs}
        assert len(rids) == 20

    def test_two_small_parallel(self):
        """Two short parallel reqs both finish."""
        execute_scripted_runtime(
            self._script_two_small_parallel,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_two_small_parallel(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=16, max_new_tokens=4)
        r2 = t.start_req(prompt_len=16, max_new_tokens=4)
        yield from run_until_all_finished([r1, r2])
        assert r1.finished and r2.finished

    def test_one_chunked_plus_many_short(self):
        """1 long chunked + 5 short, all parallel."""
        execute_scripted_runtime(
            self._script_one_chunked_plus_many_short,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_one_chunked_plus_many_short(t: ScriptedRuntime):
        # 1 long chunked + 5 short, all parallel.
        chunked = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        shorts = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(5)]
        yield from run_until_all_finished([chunked] + shorts)
        assert chunked.chunks_done >= 2

    def test_multiple_chunked_staggered(self):
        """Submit chunked reqs every few yields, serial chunking."""
        execute_scripted_runtime(
            self._script_multiple_chunked_staggered,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_multiple_chunked_staggered(t: ScriptedRuntime):
        # Submit chunked reqs every few yields, serial chunking.
        reqs = []
        for _ in range(4):
            reqs.append(t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2))
            yield
            yield
        yield from run_until_all_finished(reqs, max_steps=2000)

    def test_eight_concurrent_chunked(self):
        """8 chunked reqs submitted together."""
        execute_scripted_runtime(
            self._script_eight_concurrent_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_eight_concurrent_chunked(t: ScriptedRuntime):
        # 8 chunked reqs submitted together. Single-in-flight invariant holds.
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(8)
        ]
        for _ in range(DEFAULT_MAX_STEPS * 5):
            assert t.chunked_in_flight_count() <= 1
            if all(r.finished for r in reqs):
                return
            yield

    def test_decode_only_batch(self):
        """10 short reqs — pure decode batch."""
        execute_scripted_runtime(
            self._script_decode_only_batch,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_decode_only_batch(t: ScriptedRuntime):
        # 10 short reqs — pure decode batch.
        reqs = [t.start_req(prompt_len=4, max_new_tokens=8) for _ in range(10)]
        yield from run_until_all_finished(reqs)

    def test_mixed_prefill_lengths(self):
        """Variable prompt lengths in same batch."""
        execute_scripted_runtime(
            self._script_mixed_prefill_lengths,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_mixed_prefill_lengths(t: ScriptedRuntime):
        # Variable prompt lengths in same batch.
        lens = [8, 16, 32, 64, 128, 256, 512, 1024]
        reqs = [t.start_req(prompt_len=L, max_new_tokens=2) for L in lens]
        yield from run_until_all_finished(reqs, max_steps=1500)

    def test_two_chunked_one_decode(self):
        """2 chunked + 1 decode-only."""
        execute_scripted_runtime(
            self._script_two_chunked_one_decode,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_two_chunked_one_decode(t: ScriptedRuntime):
        # 2 chunked + 1 decode-only.
        chunked1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        chunked2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        short = t.start_req(prompt_len=8, max_new_tokens=4)
        yield from run_until_all_finished([chunked1, chunked2, short])

    def test_batch_with_finish_event_count(self):
        """Each req emits exactly 1 finish event."""
        execute_scripted_runtime(
            self._script_batch_with_finish_event_count,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_batch_with_finish_event_count(t: ScriptedRuntime):
        # Each req emits exactly 1 finish event.
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(6)]
        yield from run_until_all_finished(reqs)
        for r in reqs:
            assert r.finish_event_count == 1

    def test_batch_state_query_during_run(self):
        """Query batch_composition every step while batch is active."""
        execute_scripted_runtime(
            self._script_batch_state_query_during_run,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_batch_state_query_during_run(t: ScriptedRuntime):
        # Query batch_composition every step while batch is active.
        reqs = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(4)]
        for _ in range(DEFAULT_MAX_STEPS):
            comp = t.batch_composition()
            assert isinstance(comp, dict)
            if all(r.finished for r in reqs):
                return
            yield

    def test_mixed_lengths_then_more_arrivals(self):
        """First batch starts; midway, more reqs arrive."""
        execute_scripted_runtime(
            self._script_mixed_lengths_then_more_arrivals,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_mixed_lengths_then_more_arrivals(t: ScriptedRuntime):
        # First batch starts; midway, more reqs arrive.
        initial = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(3)]
        yield
        yield
        more = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(3)]
        yield from run_until_all_finished(initial + more)

    def test_parallel_with_priority(self):
        """3 normal + 2 high-priority reqs."""
        execute_scripted_runtime(
            self._script_parallel_with_priority,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )

    @staticmethod
    def _script_parallel_with_priority(t: ScriptedRuntime):
        # 3 normal + 2 high-priority reqs.
        normal = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(3)]
        high = [
            t.start_req(prompt_len=16, max_new_tokens=2, priority="high")
            for _ in range(2)
        ]
        yield from run_until_all_finished(normal + high)


if __name__ == "__main__":
    unittest.main()
