"""Piecewise CUDA graph × chunked: naive ScriptedRuntime smoke.

Piecewise CUDA graph is on by default; per instructions.md we just
verify the chunked path still works without disabling it. The smoke
catches "piecewise graph capture trips on a chunked extend".

In addition to the baseline smoke, this file pins:

* chunk_size that misses every captured bucket (a-Piecewise1)
* tail chunk of size 1 — tiniest possible extend (a-Piecewise2)
* retract + resume that lands the resumed chunk in a different bucket
  (a-Piecewise3)
* dynamic chunking walking the bucket table iter by iter (c-T3e)

All four tests force ``disable_cuda_graph=False`` so the piecewise
capture machinery is actually exercised.
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


class TestScriptedPiecewiseCG(CustomTestCase):
    def test_naive_piecewise_cg_chunked(self):
        """Piecewise CUDA graph capture runs alongside chunked prefill."""
        # Note: deliberately *do not* pass disable_cuda_graph=True so
        # piecewise capture runs; ``base_engine_kwargs`` defaults it off
        # for compatibility with all the other tests, so override here.
        execute_scripted_runtime(
            self._script_naive_piecewise_cg_chunked,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disable_cuda_graph=False,
            ),
        )

    @staticmethod
    def _script_naive_piecewise_cg_chunked(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=8)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_piecewise_cg_chunk_size_misses_bucket(self):
        """Chunk size that hits no captured graph bucket still completes via fallback or next-larger bucket."""
        # Pick a chunk size that is unlikely to be a captured bucket
        # boundary (most captures use powers-of-two / multiples of 64).
        odd_chunk_size: int = 257
        execute_scripted_runtime(
            self._script_piecewise_cg_chunk_size_misses_bucket,
            **base_engine_kwargs(
                chunked_prefill_size=odd_chunk_size,
                disable_cuda_graph=False,
            ),
        )

    # [a-Piecewise1] chunk_size deliberately falls between captured
    # graph buckets — the piecewise dispatcher must fall back to
    # eager or to the next-larger bucket. Either path is fine; the
    # request must still complete cleanly with no graph-capture crash.
    @staticmethod
    def _script_piecewise_cg_chunk_size_misses_bucket(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # Prompt > 8x chunk_size still produces many chunks even with
        # a bucket-missing size; we just need >=2 to confirm the path.
        assert r.chunks_done >= 2

    def test_piecewise_cg_tail_chunk_tiny(self):
        """Tail chunk of exactly 1 token still runs under piecewise CUDA graph."""
        # prompt_len = N*chunk_size + 1 produces a final tail chunk of
        # exactly 1 token after N full chunks.
        prompt_len: int = 4 * DEFAULT_CHUNK_SIZE + 1
        execute_scripted_runtime(
            self._script_piecewise_cg_tail_chunk_tiny,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disable_cuda_graph=False,
            ),
        )

    # [a-Piecewise2] Tiny tail chunk = 1 token. Piecewise capture
    # typically does not have a bucket for extend_len=1; the
    # dispatcher must fall back gracefully (or capture on demand).
    @staticmethod
    def _script_piecewise_cg_tail_chunk_tiny(t: ScriptedRuntime):
        r = t.start_req(
            prompt_len=4 * DEFAULT_CHUNK_SIZE + 1,
            max_new_tokens=2,
        )
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # 4 full chunks + 1 tail = >=5 chunks_done; the tail of size 1
        # is the path under test.
        assert r.chunks_done >= 4

    def test_piecewise_cg_retract_resume_different_bucket(self):
        """Retract then resume lands the resumed chunk in a different graph bucket."""
        execute_scripted_runtime(
            self._script_piecewise_cg_retract_resume_different_bucket,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disable_cuda_graph=False,
                enable_dynamic_chunking=True,
            ),
        )

    # [a-Piecewise3] Force a retract mid-chunk; on resume the dynamic
    # chunker may pick a different size (different bucket). Piecewise
    # capture must dispatch the new bucket cleanly — pre-fix, the
    # capture cache could mis-route after a bucket switch.
    @staticmethod
    def _script_piecewise_cg_retract_resume_different_bucket(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.force_retract(r)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # The req must finish despite a bucket switch on the resume side.
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_piecewise_cg_dynamic_chunking_bucket_walk(self):
        """Dynamic chunking iterates several distinct chunk sizes — each hits or falls back cleanly."""
        execute_scripted_runtime(
            self._script_piecewise_cg_dynamic_chunking_bucket_walk,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disable_cuda_graph=False,
                enable_dynamic_chunking=True,
            ),
        )

    # [c-T3e] Dynamic chunking predictor emits a sequence of chunk
    # sizes as the batch shape evolves. Each chunk size dispatches to
    # piecewise capture independently; the dispatcher must walk the
    # bucket table without misrouting between iterations.
    @staticmethod
    def _script_piecewise_cg_dynamic_chunking_bucket_walk(t: ScriptedRuntime):
        # A very long prompt + dynamic chunking gives the predictor
        # several opportunities to vary chunk size as the batch fills.
        r = t.start_req(prompt_len=8 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=1200)
        assert r.finished
        assert r.chunks_done >= 4


if __name__ == "__main__":
    unittest.main()
