
import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
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
        assert r.chunks_done >= 2

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
        r = t.start_req(prompt_len=_SWA_WINDOW + 13, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert (
            r.swa_budget_overflow_count == 0
        ), f"SWA budget overflowed {r.swa_budget_overflow_count} times"

    def test_swa_chunked_resume_kv_committed_bound(self):
        self.server.execute_script(self._script_swa_chunked_resume_kv_committed_bound)

    @staticmethod
    def _script_swa_chunked_resume_kv_committed_bound(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(400):
            if r.is_chunking:
                assert r.prefix_indices_len <= r.kv_committed_len, (
                    f"prefix_indices must be bounded by kv_committed_len, "
                    f"got prefix_indices_len={r.prefix_indices_len}, "
                    f"kv_committed_len={r.kv_committed_len}"
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
        assert len(r.output_tokens) == 4


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
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert len(r.output_tokens) == 2


class TestSWAOverlap(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        mem_fraction_static=0.70,
        disable_overlap_schedule=False,
        disable_piecewise_cuda_graph=True,
    )

    def test_swa_chunk_cache_evict_skips_first_two_extends(self):
        self.server.execute_script(
            self._script_swa_chunk_cache_evict_skips_first_two_extends
        )

    @staticmethod
    def _script_swa_chunk_cache_evict_skips_first_two_extends(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.swa_chunk_cache_first_two_evict_skips >= 2, (
            f"expected first two extends to skip evict, got "
            f"{r.swa_chunk_cache_first_two_evict_skips}"
        )


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
            r2.cached_tokens > 0
        ), f"r2 must hit the radix prefix, got cached_tokens={r2.cached_tokens}"


if __name__ == "__main__":
    unittest.main()
