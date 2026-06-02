import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)

_SWA_MODEL = "openai/gpt-oss-20b"
_SWA_WINDOW = 4096


class TestSWABasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        mem_fraction_static=0.70,
        disable_piecewise_cuda_graph=True,
    )

    def test_naive_swa_chunked(self):
        self.server.execute_script(self._script_naive_swa_chunked)

    @staticmethod
    def _script_naive_swa_chunked(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN + 4096, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        # prompt_len = 2048 + 4096 = 6144, chunk size = 256, so the prompt needs
        # at least ceil(6144 / 256) = 24 partial prefill iterations. SWA may add
        # extra chunked iterations because the sliding window can shrink the
        # usable budget per chunk, so this is a tight lower bound, not an equality.
        assert r.chunks_done >= 24
        assert len(r.req.output_ids) == 4

    def test_swa_prompt_equals_window(self):
        self.server.execute_script(self._script_swa_prompt_equals_window)

    @staticmethod
    def _script_swa_prompt_equals_window(t: ScriptedContext):
        r = t.start_req(prompt_len=_SWA_WINDOW, max_new_tokens=4)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_swa_budget_for_chunked_req_math(self):
        self.server.execute_script(self._script_swa_budget_for_chunked_req_math)

    @staticmethod
    def _script_swa_budget_for_chunked_req_math(t: ScriptedContext):
        # No swa_budget_overflow counter exists. The real invariant a budget
        # miscalculation would break is that a prompt straddling the SWA window
        # still prefills chunk-by-chunk to completion and releases all its SWA KV
        # afterwards. Assert the request finishes, chunked (so the across-window
        # budget path ran), and leaves no held pages or radix lock refs.
        baseline_free = t.engine_stats()["kv_pool_free"]
        r = t.start_req(prompt_len=_SWA_WINDOW + 13, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        # The finished req commits its prompt prefix to the radix tree, so the SWA
        # pool stays below baseline until the overlap lag drains and the cache is
        # flushed. Drain, flush, then measure recovery.
        for _ in range(5):
            yield
        t.flush_cache()
        yield
        assert t.engine_stats()["kv_pool_free"] >= baseline_free, (
            "SWA pool failed to recover after a window-straddling chunked req: "
            f"baseline={baseline_free}, "
            f"final={t.engine_stats()['kv_pool_free']}"
        )

    def test_swa_chunked_park_on_budget_exhaustion(self):
        self.server.execute_script(self._script_swa_chunked_park_on_budget_exhaustion)

    @staticmethod
    def _script_swa_chunked_park_on_budget_exhaustion(t: ScriptedContext):
        # add_chunked_req hybrid-SWA early-return (schedule_policy.py:679-681):
        # when _rem_tokens (= rem_swa_tokens - page_size) drops to <= 0 while a
        # chunked req is in flight, the scheduler returns the req WITHOUT
        # scheduling its next chunk -- it is parked, not advanced and not dropped.
        # chunked_parks counts exactly those held-but-not-run iterations. Drive
        # the SWA pool to exhaustion mid-chunk to force at least one park, then
        # assert the req still completes with no leaked pages or lock refs (park,
        # not leak; the non-SWA branch would instead force-admit at line 682).
        #
        # The prompt must exceed the sliding window (_SWA_WINDOW=4096): once
        # prefill passes the window, SWA evicts the req's own out-of-window KV
        # back to the pool, which is what lets a parked chunk eventually resume
        # even though exhaust_kv holds every externally-free page. A prompt that
        # fits inside the window (e.g. VERY_LONG_PROMPT_LEN=2048) never self-evicts
        # and would park forever after exhaustion.
        r = t.start_req(prompt_len=_SWA_WINDOW + VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.exhaust_kv(leave_pages=0)
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.chunked_parks >= 1, (
            "SWA budget exhaustion mid-chunk must park the chunked req at least "
            f"once (schedule_policy.py:680-681); got chunked_parks={r.chunked_parks}"
        )
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_swa_chunked_resume_kv_committed_bound(self):
        self.server.execute_script(self._script_swa_chunked_resume_kv_committed_bound)

    @staticmethod
    def _script_swa_chunked_resume_kv_committed_bound(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(400):
            if r.is_chunking:
                assert len(r.req.prefix_indices) <= r.req.kv_committed_len, (
                    f"prefix_indices must be bounded by kv_committed_len, "
                    f"got prefix_indices_len={len(r.req.prefix_indices)}, "
                    f"kv_committed_len={r.req.kv_committed_len}"
                )
            if r.finished:
                break
            yield
        assert r.finished


class TestSWAHalfWindowChunk(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=_SWA_WINDOW // 2,
        mem_fraction_static=0.70,
        disable_piecewise_cuda_graph=True,
    )

    def test_swa_prompt_2x_window_half_chunks(self):
        self.server.execute_script(self._script_swa_prompt_2x_window_half_chunks)

    @staticmethod
    def _script_swa_prompt_2x_window_half_chunks(t: ScriptedContext):
        r = t.start_req(prompt_len=2 * _SWA_WINDOW, max_new_tokens=4)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert (
            r.chunks_done >= 4
        ), f"expected >=4 chunks for 2*window / (window/2), got {r.chunks_done}"
        assert len(r.req.output_ids) == 4


class TestSWAChunkSizeExceedsWindow(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=_SWA_WINDOW * 2,
        mem_fraction_static=0.70,
        disable_piecewise_cuda_graph=True,
    )

    def test_swa_chunk_size_exceeds_window(self):
        self.server.execute_script(self._script_swa_chunk_size_exceeds_window)

    @staticmethod
    def _script_swa_chunk_size_exceeds_window(t: ScriptedContext):
        r = t.start_req(prompt_len=3 * _SWA_WINDOW, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) == 2


# Removed TestSWAOverlap.test_swa_chunk_cache_evict_skips_first_two_extends: it
# asserted swa_chunk_cache_first_two_evict_skips >= 2, a counter the engine does
# not keep. The "first two extends skip evict" behavior is a pure internal
# optimization detail of _evict_swa (it only advances swa_evicted_seqlen once the
# prefill passes the sliding window), not an externally observable invariant. The
# durable SWA-no-leak invariant it gestured at is already covered by the kv_pages
# / lock_refs / kv_pool_free checks in test_swa_budget_for_chunked_req_math and
# test_swa_prompt_equals_window.


class TestSWARadix(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        mem_fraction_static=0.70,
        disable_radix_cache=False,
        disable_piecewise_cuda_graph=True,
    )

    def test_swa_radix_partial_hit_straddles_window(self):
        self.server.execute_script(self._script_swa_radix_partial_hit_straddles_window)

    @staticmethod
    def _script_swa_radix_partial_hit_straddles_window(t: ScriptedContext):
        r1 = t.start_req(prompt_len=_SWA_WINDOW + DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until_finished(r1, max_steps=800)
        assert r1.finished
        r2 = t.start_req(
            prompt_len=_SWA_WINDOW + DEFAULT_CHUNK_SIZE * 2, max_new_tokens=2
        )
        yield from run_until_finished(r2, max_steps=800)
        assert r2.finished
        assert (
            r2.req.cached_tokens > 0
        ), f"r2 must hit the radix prefix, got cached_tokens={r2.req.cached_tokens}"


if __name__ == "__main__":
    unittest.main()
