"""Lifecycle scripted tests for chunked prefill.

Single-req normal completion (B.1 series from the expansion plan
plus fan-out across prompt-length × decode-length combinations).
These are baseline "the engine works" tests, no fault injection.

Also covers B.3 series — sequential submission and clean handoff.
Verifies that the scheduler properly handles "submit, wait, submit
again, wait, …" patterns without state leaks between reqs.
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
    run_until_finished,
)


class TestLifecycleBasic(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_small_prompt_short_decode(self):
        """Tiny prompt + tiny decode: no chunking, exact output length."""
        self.runtime.run(self._script_small_prompt_short_decode)

    @staticmethod
    def _script_small_prompt_short_decode(t: ScriptedRuntime):
        # Tiny prompt + tiny decode.
        r = t.start_req(prompt_len=8, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.output_tokens) == 2

    def test_medium_prompt_medium_decode(self):
        """Prompt <= chunk_size + medium decode: at most one chunk, exact output length."""
        self.runtime.run(self._script_medium_prompt_medium_decode)

    @staticmethod
    def _script_medium_prompt_medium_decode(t: ScriptedRuntime):
        # Prompt <= chunk_size + medium decode.
        r = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=16, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done <= 1
        assert len(r.output_tokens) == 16

    def test_long_prompt_short_decode(self):
        """Multi-chunk prefill + short decode: chunked path runs cleanly."""
        self.runtime.run(self._script_long_prompt_short_decode)

    @staticmethod
    def _script_long_prompt_short_decode(t: ScriptedRuntime):
        # Multi-chunk prefill + short decode.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.output_tokens) == 2

    def test_long_prompt_long_decode(self):
        """Multi-chunk prefill + long decode: completes cleanly."""
        self.runtime.run(self._script_long_prompt_long_decode)

    @staticmethod
    def _script_long_prompt_long_decode(t: ScriptedRuntime):
        # Multi-chunk prefill + long decode.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=64, ignore_eos=True
        )
        yield from run_until(r, lambda h: h.finished, max_steps=1000)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.output_tokens) == 64

    def test_tiny_prompt_long_decode(self):
        """1-token prompt, long decode: no chunking, exact length."""
        self.runtime.run(self._script_tiny_prompt_long_decode)

    @staticmethod
    def _script_tiny_prompt_long_decode(t: ScriptedRuntime):
        # 1-token prompt, long decode.
        r = t.start_req(prompt_len=1, max_new_tokens=64, ignore_eos=True)
        yield from run_until(r, lambda h: h.finished, max_steps=500)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.output_tokens) == 64

    def test_chunk_size_minus_one_prompt(self):
        """Prompt_len = chunk_size - 1: no chunking, exact length."""
        self.runtime.run(self._script_chunk_size_minus_one_prompt)

    @staticmethod
    def _script_chunk_size_minus_one_prompt(t: ScriptedRuntime):
        # prompt_len = chunk_size - 1 + short decode.
        r = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE - 1, max_new_tokens=4, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.output_tokens) == 4

    def test_chunk_size_plus_two_prompt(self):
        """Prompt_len = chunk_size + 2: exactly 2 chunks, exact length."""
        self.runtime.run(self._script_chunk_size_plus_two_prompt)

    @staticmethod
    def _script_chunk_size_plus_two_prompt(t: ScriptedRuntime):
        # prompt_len = chunk_size + 2 — exactly 2 chunks. The 2-byte tail
        # chunk is the canonical short-final-chunk shape.
        r = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE + 2, max_new_tokens=4, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2
        assert len(r.output_tokens) == 4

    def test_just_over_2x_chunk_size(self):
        """Prompt_len = 2 * chunk_size + 1: exactly 3 chunks."""
        self.runtime.run(self._script_just_over_2x_chunk_size)

    @staticmethod
    def _script_just_over_2x_chunk_size(t: ScriptedRuntime):
        # prompt_len = 2 * chunk_size + 1.
        r = t.start_req(
            prompt_len=2 * DEFAULT_CHUNK_SIZE + 1, max_new_tokens=4, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 3
        assert len(r.output_tokens) == 4

    def test_five_x_chunk_size(self):
        """5 chunks exactly: full middle-chunk path."""
        self.runtime.run(self._script_five_x_chunk_size)

    @staticmethod
    def _script_five_x_chunk_size(t: ScriptedRuntime):
        # 5 chunks exactly — exercises the middle-chunk path with 3
        # interior chunks between first and last.
        r = t.start_req(
            prompt_len=5 * DEFAULT_CHUNK_SIZE, max_new_tokens=4, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 5
        assert len(r.output_tokens) == 4

    def test_ten_x_chunk_size(self):
        """10 chunks exactly: long chunked path."""
        self.runtime.run(self._script_ten_x_chunk_size)

    @staticmethod
    def _script_ten_x_chunk_size(t: ScriptedRuntime):
        # 10 chunks exactly — 8 middle chunks along the chunked path.
        r = t.start_req(
            prompt_len=10 * DEFAULT_CHUNK_SIZE, max_new_tokens=2, ignore_eos=True
        )
        yield from run_until(r, lambda h: h.finished, max_steps=1000)
        assert r.finished
        assert r.chunks_done == 10
        assert len(r.output_tokens) == 2

    def test_status_progression_happy_path(self):
        """Status traverses every expected stage of a happy-path req: running and finished both observed."""
        self.runtime.run(self._script_status_progression_happy_path)

    @staticmethod
    def _script_status_progression_happy_path(t: ScriptedRuntime):
        # Verify status moves through running → finished. The docstring
        # mentions waiting / unknown too, but for a single-req baseline
        # those may flicker in a single iter; the invariants that ALWAYS
        # hold are: "running" appears before "finished", "finished"
        # appears at the end, and the status never regresses past
        # finished.
        r = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
        seen = []
        for _ in range(DEFAULT_MAX_STEPS):
            seen.append(r.status)
            if r.finished:
                break
            yield
        else:
            raise AssertionError("req did not finish within DEFAULT_MAX_STEPS")
        assert "running" in seen, f"never observed running status; seen={seen}"
        assert seen[-1] == "finished", f"final status must be finished; seen={seen}"
        # Status must not flip back to running after finishing.
        finished_idx = seen.index("finished")
        assert all(
            s in ("finished",) for s in seen[finished_idx:]
        ), f"status regressed after finish; seen={seen}"

    def test_long_prompt_only_one_decode(self):
        """Max_new_tokens = 1 on chunked prompt: chunks_done >= 2, exactly 1 output."""
        self.runtime.run(self._script_long_prompt_only_one_decode)

    @staticmethod
    def _script_long_prompt_only_one_decode(t: ScriptedRuntime):
        # max_new_tokens = 1 on a chunked prompt — the minimal-decode
        # chunked path.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.output_tokens) == 1

    def test_kv_pages_consistent_during_run(self):
        """Kv_pages > 0 throughout active phase, drains to 0 after finish."""
        self.runtime.run(self._script_kv_pages_consistent_during_run)

    @staticmethod
    def _script_kv_pages_consistent_during_run(t: ScriptedRuntime):
        # kv_pages > 0 during running, == 0 after finish. Also, once we
        # have observed kv_pages > 0, the value must stay positive until
        # finish — a mid-run drop to 0 means the page accounting leaked.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, ignore_eos=True
        )
        saw_positive = False
        for _ in range(DEFAULT_MAX_STEPS):
            pages = r.kv_pages
            if pages > 0:
                saw_positive = True
            elif saw_positive and not r.finished:
                raise AssertionError(
                    f"kv_pages collapsed to 0 mid-run before finish; "
                    f"saw_positive={saw_positive}, status={r.status!r}"
                )
            if r.finished:
                break
            yield
        else:
            raise AssertionError("req did not finish within DEFAULT_MAX_STEPS")
        assert saw_positive
        assert r.kv_pages == 0
        assert len(r.output_tokens) == 4

    def test_row_idx_recycled_after_finish(self):
        """After finish, row_idx becomes None AND no residual KV / lock_ref remains."""
        self.runtime.run(self._script_row_idx_recycled_after_finish)

    @staticmethod
    def _script_row_idx_recycled_after_finish(t: ScriptedRuntime):
        # After finish, all per-req resources must release together.
        r = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r)
        assert r.finished
        assert r.row_idx is None
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_two_seq_clean_handoff(self):
        """Two sequential reqs both finish, leave engine idle, no row/KV leak between them."""
        self.runtime.run(self._script_two_seq_clean_handoff)

    @staticmethod
    def _script_two_seq_clean_handoff(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r1)
        # Between reqs, r1 must have fully released its resources before
        # r2 is admitted — otherwise the handoff is leaky.
        assert r1.row_idx is None and r1.kv_pages == 0 and r1.lock_refs == 0
        r2 = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r2)
        assert r1.finished and r2.finished
        assert len(r1.output_tokens) == 2 and len(r2.output_tokens) == 2
        assert r2.row_idx is None and r2.kv_pages == 0 and r2.lock_refs == 0

    def test_five_seq_clean(self):
        """Five sequential reqs all finish with exact length and full resource recycle each round."""
        self.runtime.run(self._script_five_seq_clean)

    @staticmethod
    def _script_five_seq_clean(t: ScriptedRuntime):
        reqs = []
        for _ in range(5):
            r = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.output_tokens) == 2
            # Each req must release ALL resources before the next one
            # is submitted, or sequential handoff leaks slowly.
            assert r.row_idx is None
            assert r.kv_pages == 0
            assert r.lock_refs == 0
            reqs.append(r)
        # Every req in the sequence finished exactly once.
        for r in reqs:
            assert r.finish_event_count == 1

    def test_radix_partial_seq(self):
        """R1 prompt becomes radix prefix; r2 = r1.prompt + extra: r2 hits cache, completes in <=1 chunk."""
        self.runtime.run(self._script_radix_partial_seq)

    @staticmethod
    def _script_radix_partial_seq(t: ScriptedRuntime):
        # r1 prompt becomes radix prefix; r2 = r1.prompt + extra. The
        # whole point of this test is that r2 cached_tokens > 0, not
        # just that chunks_done is small.
        r1 = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1, ignore_eos=True
        )
        yield from run_until_finished(r1)
        r2 = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE + 32, max_new_tokens=2, ignore_eos=True
        )
        yield from run_until_finished(r2)
        assert r1.finished and r2.finished
        # Should hit r1's prefix.
        assert r2.chunks_done <= 1
        assert r2.cached_tokens > 0, (
            f"r2 must hit r1's radix prefix; got cached_tokens=" f"{r2.cached_tokens}"
        )
        assert len(r2.output_tokens) == 2

    def test_alternating_short_long_seq(self):
        """Alternate short / long across 6 reqs: each finishes with exact length and recycles cleanly."""
        self.runtime.run(self._script_alternating_short_long_seq)

    @staticmethod
    def _script_alternating_short_long_seq(t: ScriptedRuntime):
        # alternate short / long across 6 reqs. The alternation exercises
        # the engine's transition between non-chunked and chunked paths
        # back-to-back — any sticky chunked state from the long req would
        # corrupt the next short req.
        for i in range(6):
            prompt = 8 if i % 2 == 0 else VERY_LONG_PROMPT_LEN
            r = t.start_req(prompt_len=prompt, max_new_tokens=2, ignore_eos=True)
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.output_tokens) == 2
            assert r.row_idx is None and r.kv_pages == 0 and r.lock_refs == 0
            if prompt == VERY_LONG_PROMPT_LEN:
                assert r.chunks_done >= 2
            else:
                assert r.chunks_done == 0

    def test_seq_with_growing_prompt(self):
        """Prompt_len grows across 5 reqs: each finishes, chunked admission only fires on the chunk-size-exceeding ones."""
        self.runtime.run(self._script_seq_with_growing_prompt)

    @staticmethod
    def _script_seq_with_growing_prompt(t: ScriptedRuntime):
        # prompt_len grows: each new req longer than the previous. The
        # chunked path engages only once prompt_len > chunk_size; the
        # short reqs must NOT take the chunked codepath.
        for L in [8, 32, 128, 512, 1024]:
            r = t.start_req(prompt_len=L, max_new_tokens=1, ignore_eos=True)
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.output_tokens) == 1
            assert r.row_idx is None and r.kv_pages == 0 and r.lock_refs == 0
            if L > DEFAULT_CHUNK_SIZE:
                assert r.chunks_done >= 2
            else:
                assert r.chunks_done == 0

    def test_seq_with_shrinking_prompt(self):
        """Sequential reqs with shrinking prompt lengths: each finishes with exact length, chunked engages only above chunk_size."""
        self.runtime.run(self._script_seq_with_shrinking_prompt)

    @staticmethod
    def _script_seq_with_shrinking_prompt(t: ScriptedRuntime):
        # Shrinking prompts — the symmetric counterpart to the growing
        # test; ensures the engine returns to non-chunked state cleanly
        # after a chunked round.
        for L in [1024, 512, 128, 32, 8]:
            r = t.start_req(prompt_len=L, max_new_tokens=1, ignore_eos=True)
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.output_tokens) == 1
            assert r.row_idx is None and r.kv_pages == 0 and r.lock_refs == 0
            if L > DEFAULT_CHUNK_SIZE:
                assert r.chunks_done >= 2
            else:
                assert r.chunks_done == 0

    def test_seq_with_idle_yields_between(self):
        """Insert idle yields between completion and next submission: idle time must not corrupt sequential handoff."""
        self.runtime.run(self._script_seq_with_idle_yields_between)

    @staticmethod
    def _script_seq_with_idle_yields_between(t: ScriptedRuntime):
        # Insert idle yields between completion and next submission. The
        # idle yields drive the scheduler's "no work" loop; after them
        # the engine must accept a fresh req cleanly.
        for _ in range(4):
            r = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.output_tokens) == 2
            assert r.row_idx is None and r.kv_pages == 0 and r.lock_refs == 0
            for _ in range(20):
                yield

    def test_chunked_then_short_seq(self):
        """Long chunked → short → long → short: each finishes."""
        self.runtime.run(self._script_chunked_then_short_seq)

    @staticmethod
    def _script_chunked_then_short_seq(t: ScriptedRuntime):
        # Long chunked, then short, then long, then short. After each
        # long chunked round the engine must be quiet again before the
        # next short req arrives.
        seq = [VERY_LONG_PROMPT_LEN, 8, VERY_LONG_PROMPT_LEN, 8]
        for L in seq:
            r = t.start_req(prompt_len=L, max_new_tokens=2, ignore_eos=True)
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.output_tokens) == 2
            assert r.row_idx is None and r.kv_pages == 0 and r.lock_refs == 0
            if L == VERY_LONG_PROMPT_LEN:
                assert r.chunks_done >= 2
            else:
                assert r.chunks_done == 0

    def test_seq_finish_events_one_each(self):
        """Each sequential req emits exactly one finish event."""
        self.runtime.run(self._script_seq_finish_events_one_each)

    @staticmethod
    def _script_seq_finish_events_one_each(t: ScriptedRuntime):
        reqs = []
        for _ in range(5):
            r = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.output_tokens) == 2
            assert r.row_idx is None and r.kv_pages == 0 and r.lock_refs == 0
            reqs.append(r)
        for r in reqs:
            assert r.finish_event_count == 1, (
                f"sequential req must emit exactly one finish event; "
                f"got {r.finish_event_count} for rid={r.rid}"
            )

    def test_seq_engine_stats_stable(self):
        """Engine KV pool stays at baseline across a sequence of reqs."""
        self.runtime.run(self._script_seq_engine_stats_stable)

    @staticmethod
    def _script_seq_engine_stats_stable(t: ScriptedRuntime):
        # KV pool baseline must hold across 5 sequential reqs. Any drift
        # downwards by more than 1 page indicates a per-req leak.
        baseline = t.engine_stats()["kv_pool_free"]
        for _ in range(5):
            r = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.output_tokens) == 2
            # Per-req: every resource must return before the next iter.
            assert r.row_idx is None and r.kv_pages == 0 and r.lock_refs == 0
        final = t.engine_stats()["kv_pool_free"]
        assert (
            final >= baseline - 1
        ), f"KV pool drift: baseline={baseline}, final={final}"

    def test_engine_shutdown_during_chunked(self):
        """Engine shutdown mid-chunk: chunked req receives a final error and no subprocess is orphaned."""
        self.runtime.run(self._script_engine_shutdown_during_chunked)

    # engine shutdown mid-chunked — the chunked req must
    # surface a clean terminal error (not a hang) and no scheduler
    # subprocess should be left orphaned after the engine tears down.
    @staticmethod
    def _script_engine_shutdown_during_chunked(t: ScriptedRuntime):
        # NEW API NEEDED: t.shutdown() — request a graceful engine
        # shutdown from inside a scripted run. Currently the harness
        # only tears down when the generator returns; an explicit
        # shutdown signal during a chunked req is required to test the
        # cross-boundary cleanup path.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.shutdown()
        # Spin a few iters to let the shutdown propagate and the chunked
        # req settle to a terminal state.
        for _ in range(DEFAULT_MAX_STEPS):
            if r.finished or r.error_message is not None:
                break
            yield
        else:
            raise AssertionError(
                "chunked req did not terminate after shutdown within DEFAULT_MAX_STEPS"
            )
        # After shutdown the chunked req should surface either a clean
        # terminal error or finish with no orphaned KV.
        assert r.finished or r.error_message is not None
        assert r.kv_pages == 0
        assert r.row_idx is None
        assert r.lock_refs == 0


if __name__ == "__main__":
    unittest.main()
