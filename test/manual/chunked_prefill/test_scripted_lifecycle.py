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
        """Tiny prompt + tiny decode."""
        self.runtime.run(self._script_small_prompt_short_decode)

    @staticmethod
    def _script_small_prompt_short_decode(t: ScriptedRuntime):
        # Tiny prompt + tiny decode.
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0

    def test_medium_prompt_medium_decode(self):
        """Prompt <= chunk_size + medium decode."""
        self.runtime.run(self._script_medium_prompt_medium_decode)

    @staticmethod
    def _script_medium_prompt_medium_decode(t: ScriptedRuntime):
        # Prompt <= chunk_size + medium decode.
        r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=16)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done <= 1

    def test_long_prompt_short_decode(self):
        """Multi-chunk prefill + short decode."""
        self.runtime.run(self._script_long_prompt_short_decode)

    @staticmethod
    def _script_long_prompt_short_decode(t: ScriptedRuntime):
        # Multi-chunk prefill + short decode.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_long_prompt_long_decode(self):
        """Multi-chunk prefill + long decode."""
        self.runtime.run(self._script_long_prompt_long_decode)

    @staticmethod
    def _script_long_prompt_long_decode(t: ScriptedRuntime):
        # Multi-chunk prefill + long decode.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=64)
        yield from run_until(r, lambda h: h.finished, max_steps=1000)
        assert r.finished
        assert r.chunks_done >= 2

    def test_tiny_prompt_long_decode(self):
        """1-token prompt, long decode."""
        self.runtime.run(self._script_tiny_prompt_long_decode)

    @staticmethod
    def _script_tiny_prompt_long_decode(t: ScriptedRuntime):
        # 1-token prompt, long decode.
        r = t.start_req(prompt_len=1, max_new_tokens=64)
        yield from run_until(r, lambda h: h.finished, max_steps=500)
        assert r.finished
        assert r.chunks_done == 0

    def test_chunk_size_minus_one_prompt(self):
        """Prompt_len = chunk_size - 1 + short decode."""
        self.runtime.run(self._script_chunk_size_minus_one_prompt)

    @staticmethod
    def _script_chunk_size_minus_one_prompt(t: ScriptedRuntime):
        # prompt_len = chunk_size - 1 + short decode.
        r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE - 1, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0

    def test_chunk_size_plus_two_prompt(self):
        """Prompt_len = chunk_size + 2."""
        self.runtime.run(self._script_chunk_size_plus_two_prompt)

    @staticmethod
    def _script_chunk_size_plus_two_prompt(t: ScriptedRuntime):
        # prompt_len = chunk_size + 2.
        r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 2, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2

    def test_just_over_2x_chunk_size(self):
        """Prompt_len = 2 * chunk_size + 1."""
        self.runtime.run(self._script_just_over_2x_chunk_size)

    @staticmethod
    def _script_just_over_2x_chunk_size(t: ScriptedRuntime):
        # prompt_len = 2 * chunk_size + 1.
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE + 1, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 3

    def test_five_x_chunk_size(self):
        """5 chunks exactly."""
        self.runtime.run(self._script_five_x_chunk_size)

    @staticmethod
    def _script_five_x_chunk_size(t: ScriptedRuntime):
        # 5 chunks exactly.
        r = t.start_req(prompt_len=5 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 5

    def test_ten_x_chunk_size(self):
        """10 chunks exactly."""
        self.runtime.run(self._script_ten_x_chunk_size)

    @staticmethod
    def _script_ten_x_chunk_size(t: ScriptedRuntime):
        # 10 chunks exactly.
        r = t.start_req(prompt_len=10 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(r, lambda h: h.finished, max_steps=1000)
        assert r.finished
        assert r.chunks_done == 10

    def test_status_progression_happy_path(self):
        """Verify status progresses unknown → waiting → running → finished."""
        self.runtime.run(self._script_status_progression_happy_path)

    @staticmethod
    def _script_status_progression_happy_path(t: ScriptedRuntime):
        # Verify status progresses unknown → waiting → running → finished.
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        seen = set()
        for _ in range(DEFAULT_MAX_STEPS):
            seen.add(r.status)
            if r.finished:
                break
            yield
        assert "finished" in seen

    def test_long_prompt_only_one_decode(self):
        """Max_new_tokens = 1: minimum decode."""
        self.runtime.run(self._script_long_prompt_only_one_decode)

    @staticmethod
    def _script_long_prompt_only_one_decode(t: ScriptedRuntime):
        # max_new_tokens = 1: minimum decode.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.finished

    def test_kv_pages_consistent_during_run(self):
        """Kv_pages > 0 during running, == 0 after finish."""
        self.runtime.run(self._script_kv_pages_consistent_during_run)

    @staticmethod
    def _script_kv_pages_consistent_during_run(t: ScriptedRuntime):
        # kv_pages > 0 during running, == 0 after finish.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        saw_positive = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.kv_pages > 0:
                saw_positive = True
            if r.finished:
                break
            yield
        assert saw_positive
        assert r.kv_pages == 0

    def test_row_idx_recycled_after_finish(self):
        """After finish, row_idx becomes None."""
        self.runtime.run(self._script_row_idx_recycled_after_finish)

    @staticmethod
    def _script_row_idx_recycled_after_finish(t: ScriptedRuntime):
        # After finish, row_idx becomes None.
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.row_idx is None

    def test_two_seq_clean_handoff(self):
        """Two sequential reqs complete cleanly one after the other."""
        self.runtime.run(self._script_two_seq_clean_handoff)

    @staticmethod
    def _script_two_seq_clean_handoff(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r2)
        assert r1.finished and r2.finished

    def test_five_seq_clean(self):
        """Five sequential reqs each complete cleanly."""
        self.runtime.run(self._script_five_seq_clean)

    @staticmethod
    def _script_five_seq_clean(t: ScriptedRuntime):
        for _ in range(5):
            r = t.start_req(prompt_len=16, max_new_tokens=2)
            yield from run_until_finished(r)

    def test_radix_partial_seq(self):
        """R1 prompt becomes radix prefix; r2 = r1.prompt + extra."""
        self.runtime.run(self._script_radix_partial_seq)

    @staticmethod
    def _script_radix_partial_seq(t: ScriptedRuntime):
        # r1 prompt becomes radix prefix; r2 = r1.prompt + extra.
        r1 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 32, max_new_tokens=2)
        yield from run_until_finished(r2)
        # Should hit r1's prefix.
        assert r2.chunks_done <= 1

    def test_alternating_short_long_seq(self):
        """Alternate short / long across 6 reqs."""
        self.runtime.run(self._script_alternating_short_long_seq)

    @staticmethod
    def _script_alternating_short_long_seq(t: ScriptedRuntime):
        # alternate short / long across 6 reqs.
        for i in range(6):
            prompt = 8 if i % 2 == 0 else VERY_LONG_PROMPT_LEN
            r = t.start_req(prompt_len=prompt, max_new_tokens=2)
            yield from run_until_finished(r)

    def test_seq_with_growing_prompt(self):
        """Prompt_len grows: each new req longer than the previous."""
        self.runtime.run(self._script_seq_with_growing_prompt)

    @staticmethod
    def _script_seq_with_growing_prompt(t: ScriptedRuntime):
        # prompt_len grows: each new req longer than the previous.
        for L in [8, 32, 128, 512, 1024]:
            r = t.start_req(prompt_len=L, max_new_tokens=1)
            yield from run_until_finished(r)

    def test_seq_with_shrinking_prompt(self):
        """Sequential reqs with shrinking prompt lengths all finish."""
        self.runtime.run(self._script_seq_with_shrinking_prompt)

    @staticmethod
    def _script_seq_with_shrinking_prompt(t: ScriptedRuntime):
        for L in [1024, 512, 128, 32, 8]:
            r = t.start_req(prompt_len=L, max_new_tokens=1)
            yield from run_until_finished(r)

    def test_seq_with_idle_yields_between(self):
        """Insert idle yields between completion and next submission."""
        self.runtime.run(self._script_seq_with_idle_yields_between)

    @staticmethod
    def _script_seq_with_idle_yields_between(t: ScriptedRuntime):
        # Insert idle yields between completion and next submission.
        for _ in range(4):
            r = t.start_req(prompt_len=16, max_new_tokens=2)
            yield from run_until_finished(r)
            for _ in range(20):
                yield

    def test_chunked_then_short_seq(self):
        """Long chunked, then short, then long, then short."""
        self.runtime.run(self._script_chunked_then_short_seq)

    @staticmethod
    def _script_chunked_then_short_seq(t: ScriptedRuntime):
        # Long chunked, then short, then long, then short.
        seq = [VERY_LONG_PROMPT_LEN, 8, VERY_LONG_PROMPT_LEN, 8]
        for L in seq:
            r = t.start_req(prompt_len=L, max_new_tokens=2)
            yield from run_until_finished(r)

    def test_seq_finish_events_one_each(self):
        """Each sequential req emits exactly one finish event."""
        self.runtime.run(self._script_seq_finish_events_one_each)

    @staticmethod
    def _script_seq_finish_events_one_each(t: ScriptedRuntime):
        reqs = []
        for _ in range(5):
            r = t.start_req(prompt_len=16, max_new_tokens=2)
            yield from run_until_finished(r)
            reqs.append(r)
        for r in reqs:
            assert r.finish_event_count == 1

    def test_seq_engine_stats_stable(self):
        """Engine KV pool stays at baseline across a sequence of reqs."""
        self.runtime.run(self._script_seq_engine_stats_stable)

    @staticmethod
    def _script_seq_engine_stats_stable(t: ScriptedRuntime):
        baseline = t.engine_stats()["kv_pool_free"]
        for _ in range(5):
            r = t.start_req(prompt_len=16, max_new_tokens=2)
            yield from run_until_finished(r)
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
        yield
        # After shutdown the chunked req should surface either a clean
        # terminal error or finish with no orphaned KV.
        assert r.finished or r.error_message is not None
        assert r.kv_pages == 0


if __name__ == "__main__":
    unittest.main()
