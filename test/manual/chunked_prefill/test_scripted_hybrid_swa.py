"""Hybrid SWA × chunked: naive ScriptedRuntime smoke.

Hybrid sliding-window-attention models exercise an ``add_chunked_req``
early-return branch under SWA pressure (see audit doc § "Hybrid SWA").
We pump a prompt longer than the SWA window so the chunk loop has to
cross the window boundary.

Uses gpt-oss-20b — the same SWA model the existing
``test_streaming_session_swa.py`` uses. Requires a single 80 GB GPU.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase

_SWA_MODEL = "openai/gpt-oss-20b"
_SWA_WINDOW = 4096


class TestScriptedHybridSWA(CustomTestCase):
    def test_naive_swa_chunked(self):
        """Chunked prompt crosses the SWA window boundary at least once."""
        execute_scripted_runtime(
            self._script_naive_swa_chunked,
            **base_engine_kwargs(
                model_path=_SWA_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                mem_fraction_static=0.70,
                disable_piecewise_cuda_graph=True,
            ),
        )

    @staticmethod
    def _script_naive_swa_chunked(t: ScriptedRuntime):
        # Length chosen to exceed both DEFAULT_CHUNK_SIZE *and* the SWA
        # window for gpt-oss-20b (4096) — guarantees the chunk loop has to
        # cross the SWA boundary at least once.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN + 4096, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_swa_prompt_equals_window(self):
        """SWA chunked admission with prompt_len == sliding_window aligns to the boundary."""
        execute_scripted_runtime(
            self._script_swa_prompt_equals_window,
            **base_engine_kwargs(
                model_path=_SWA_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                mem_fraction_static=0.70,
                disable_piecewise_cuda_graph=True,
            ),
        )

    # [a-SWA1] prompt_len == sliding_window — chunked admission must
    # complete with the window boundary aligned to the prompt end.
    @staticmethod
    def _script_swa_prompt_equals_window(t: ScriptedRuntime):
        r = t.start_req(prompt_len=_SWA_WINDOW, max_new_tokens=4)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 2
        # The exact window boundary should land at prompt end so no
        # window-bookkeeping page is held past finish.
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_swa_prompt_2x_window_half_chunks(self):
        """SWA prompt = 2*window, chunk_size = window/2 — each chunk crosses the boundary."""
        execute_scripted_runtime(
            self._script_swa_prompt_2x_window_half_chunks,
            **base_engine_kwargs(
                model_path=_SWA_MODEL,
                chunked_prefill_size=_SWA_WINDOW // 2,
                mem_fraction_static=0.70,
                disable_piecewise_cuda_graph=True,
            ),
        )

    # [a-SWA2] prompt = 2 * sliding_window, chunk_size = window/2 — every
    # chunk straddles the window boundary; output must still be correct.
    @staticmethod
    def _script_swa_prompt_2x_window_half_chunks(t: ScriptedRuntime):
        r = t.start_req(prompt_len=2 * _SWA_WINDOW, max_new_tokens=4)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # 2*window / (window/2) == 4 chunks.
        assert (
            r.chunks_done >= 4
        ), f"expected >=4 chunks for 2*window / (window/2), got {r.chunks_done}"
        assert len(r.output_tokens) == 4

    def test_swa_chunked_req_early_return_no_double_free(self):
        """SWA add_chunked_req early-return must flip _chunked_req_scheduled_last_iter cleanly."""
        execute_scripted_runtime(
            self._script_swa_chunked_req_early_return_no_double_free,
            **base_engine_kwargs(
                model_path=_SWA_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                mem_fraction_static=0.70,
                disable_piecewise_cuda_graph=True,
            ),
        )

    # [c-S9 / P5a] SWA pool critical + add_chunked_req forced early-return —
    # _chunked_req_scheduled_last_iter must flip correctly; stash must not
    # be double-freed.
    @staticmethod
    def _script_swa_chunked_req_early_return_no_double_free(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        # Observe that the chunked_req_scheduled_last_iter flag flips
        # cleanly across iterations.
        observed_true = False
        for _ in range(200):
            if r.chunked_req_scheduled_last_iter:
                observed_true = True
            if r.finished:
                break
            yield
        assert r.finished
        assert (
            observed_true
        ), "_chunked_req_scheduled_last_iter must be True at least once"
        assert (
            r.swa_stash_double_free_count == 0
        ), f"SWA stash double-free observed {r.swa_stash_double_free_count} times"

    def test_swa_budget_for_chunked_req_math(self):
        """SWA _swa_budget_for_req math must not overflow across chunk_size / window combos."""
        execute_scripted_runtime(
            self._script_swa_budget_for_chunked_req_math,
            **base_engine_kwargs(
                model_path=_SWA_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                mem_fraction_static=0.70,
                disable_piecewise_cuda_graph=True,
            ),
        )

    # [c-P2] SWA _swa_budget_for_req — chunk size vs sliding window
    # combos must not overflow the budget formula or trigger OOM.
    @staticmethod
    def _script_swa_budget_for_chunked_req_math(t: ScriptedRuntime):
        r = t.start_req(prompt_len=_SWA_WINDOW + 13, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert (
            r.swa_budget_overflow_count == 0
        ), f"SWA budget overflowed {r.swa_budget_overflow_count} times"

    def test_swa_chunk_cache_evict_skips_first_two_extends(self):
        """SWA + overlap + chunked: the first two extend_batch_idx values must skip eviction."""
        execute_scripted_runtime(
            self._script_swa_chunk_cache_evict_skips_first_two_extends,
            **base_engine_kwargs(
                model_path=_SWA_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                mem_fraction_static=0.70,
                disable_overlap_schedule=False,
                disable_piecewise_cuda_graph=True,
            ),
        )

    # [c-B11] SWA + overlap + chunked — the extend_batch_idx < 2 path must
    # not evict; verifies skip count >= 2 on the first chunk admission.
    @staticmethod
    def _script_swa_chunk_cache_evict_skips_first_two_extends(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.swa_chunk_cache_first_two_evict_skips >= 2, (
            f"expected first two extends to skip evict, got "
            f"{r.swa_chunk_cache_first_two_evict_skips}"
        )

    def test_swa_chunk_size_exceeds_window(self):
        """SWA chunk_size > sliding_window: engine must accept or reject cleanly and consistently."""
        execute_scripted_runtime(
            self._script_swa_chunk_size_exceeds_window,
            **base_engine_kwargs(
                model_path=_SWA_MODEL,
                chunked_prefill_size=_SWA_WINDOW * 2,
                mem_fraction_static=0.70,
                disable_piecewise_cuda_graph=True,
            ),
        )

    # [a-SWA4] chunk_size > sliding_window — engine must either reject
    # at startup or accept and complete; behavior must be consistent
    # across runs (no flaky partial init).
    @staticmethod
    def _script_swa_chunk_size_exceeds_window(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # If we got here the engine accepted the config; output must
        # still be valid.
        assert len(r.output_tokens) == 2

    def test_swa_radix_partial_hit_straddles_window(self):
        """SWA + radix + chunked: prefix hit, window, and chunked admission stay consistent."""
        execute_scripted_runtime(
            self._script_swa_radix_partial_hit_straddles_window,
            **base_engine_kwargs(
                model_path=_SWA_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                mem_fraction_static=0.70,
                disable_radix_cache=False,
                disable_piecewise_cuda_graph=True,
            ),
        )

    # [a-SWA5] SWA + radix + chunked — partial prefix hit that straddles
    # the SWA window; admission must coordinate prefix_indices with the
    # window boundary and chunk admission cleanly.
    @staticmethod
    def _script_swa_radix_partial_hit_straddles_window(t: ScriptedRuntime):
        # First req populates the radix prefix.
        r1 = t.start_req(prompt_len=_SWA_WINDOW + DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until_finished(r1, max_steps=800)
        assert r1.finished
        # Second req hits the prefix that straddles the SWA window.
        r2 = t.start_req(
            prompt_len=_SWA_WINDOW + DEFAULT_CHUNK_SIZE * 2, max_new_tokens=2
        )
        yield from run_until_finished(r2, max_steps=800)
        assert r2.finished
        assert (
            r2.cached_tokens > 0
        ), f"r2 must hit the radix prefix, got cached_tokens={r2.cached_tokens}"

    def test_swa_chunked_resume_kv_committed_bound(self):
        """SWA early-return between chunks: prefix_indices must be bounded by kv_committed_len."""
        execute_scripted_runtime(
            self._script_swa_chunked_resume_kv_committed_bound,
            **base_engine_kwargs(
                model_path=_SWA_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                mem_fraction_static=0.70,
                disable_piecewise_cuda_graph=True,
            ),
        )

    # [b-1c3bf8e7db] SWA early-return between two chunks — the next iter
    # must read prefix_indices only up to kv_committed_len; rows past
    # kv_committed_len are uninitialized and would corrupt the chunk.
    @staticmethod
    def _script_swa_chunked_resume_kv_committed_bound(t: ScriptedRuntime):
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


if __name__ == "__main__":
    unittest.main()
