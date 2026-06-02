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
        # pool stays below baseline until the cache is flushed. flush_cache is a
        # no-op unless the scheduler is fully idle (it bails while the just-finished
        # req still lingers in the overlap pipeline), so drain until is_fully_idle
        # BEFORE flushing -- a fixed 5-yield drain is not always enough and leaves
        # the committed prefix un-flushed, reading as a ~prompt-sized pool deficit.
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        assert t.engine_stats()["kv_pool_free"] >= baseline_free, (
            "SWA pool failed to recover after a window-straddling chunked req: "
            f"baseline={baseline_free}, "
            f"final={t.engine_stats()['kv_pool_free']}"
        )

    def test_swa_chunked_resume_under_swa_pressure(self):
        self.server.execute_script(self._script_swa_chunked_resume_under_swa_pressure)

    @staticmethod
    def _script_swa_chunked_resume_under_swa_pressure(t: ScriptedContext):
        # Exercise the hybrid-SWA across-window budget path in add_chunked_req
        # (schedule_policy.py:671-682) under genuine SWA-pool pressure: grab most
        # of the SWA pool mid-chunk (leaving a small residual) so every subsequent
        # chunk admission recomputes the SWA budget (rem_swa_tokens - page_size)
        # against a nearly-empty pool. The prompt straddles the sliding window
        # (_SWA_WINDOW + VERY_LONG_PROMPT_LEN), so the across-window branch runs
        # repeatedly, and the req must still drive to completion and release all
        # SWA KV / radix lock refs.
        #
        # Why this and not "park to >=1 then resume": on v1 with the SWA *radix*
        # cache a single chunked req can never resume from a *full* SWA exhaustion.
        # Radix-mode self-eviction of a req's own out-of-window KV happens only in
        # the chunk-cache extend branch of maybe_evict_swa, and the in-flight
        # chunked req holds a swa_lock_ref on its committed prefix, so it cannot
        # reclaim its own SWA pages. With leave_pages=0 the req deadlocks at chunk 1
        # (swa_available_size()==0, never finishing) and the park is never even
        # recorded -- no batch runs, so chunked_parks stays 0. The real, observable
        # v1 invariant is the tight-budget across-window resume asserted below.
        r = t.start_req(prompt_len=_SWA_WINDOW + VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        chunks_at_pressure = r.chunks_done
        t.exhaust_kv(leave_pages=1000)
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        # Prefill kept advancing under SWA pressure past the chunk it was on when
        # the pool was squeezed -- the across-window budget path did not stall it.
        assert r.chunks_done > chunks_at_pressure, (
            f"chunked prefill stalled under SWA pressure: chunks_done="
            f"{r.chunks_done} did not advance past chunks_at_pressure="
            f"{chunks_at_pressure}"
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
