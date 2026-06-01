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

_HISPARSE_MODEL = "zai-org/GLM-5-FP8"


class TestHiSparseBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_HISPARSE_MODEL,
        tp_size=8,
        dp_size=8,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_dp_attention=True,
        enable_hisparse=True,
    )

    def test_naive_hisparse_chunked(self):
        self.server.execute_script(self._script_naive_hisparse_chunked)

    @staticmethod
    def _script_naive_hisparse_chunked(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN * 4, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        # No per-request staging-buffer count exists; the staging buffers live in
        # the hisparse allocator pool, gated per request by Req.hisparse_staging.
        # After finish that flag must be clear (no DMA staging left held).
        assert (
            not r.hisparse_dma_in_flight
        ), "hisparse + chunked left staging in flight after finish"
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert len(r.req.output_ids) == 4

    def test_hisparse_staging_dma_during_chunk_admit(self):
        self.server.execute_script(self._script_hisparse_staging_dma_during_chunk_admit)

    @staticmethod
    def _script_hisparse_staging_dma_during_chunk_admit(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN * 4, max_new_tokens=2)
        yield from run_until(r, lambda h: h.hisparse_dma_in_flight)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=2000)
        yield from run_until_finished(r2, max_steps=2000)
        assert r.finished and r2.finished
        assert not r.hisparse_dma_in_flight, "r still staging in flight after finish"
        assert not r2.hisparse_dma_in_flight, "r2 still staging in flight after finish"
        assert r.kv_pages == 0 and r2.kv_pages == 0
        assert r.lock_refs == 0 and r2.lock_refs == 0
        assert r.chunks_done >= 2

    def test_hisparse_abort_during_chunk_with_dma(self):
        self.server.execute_script(self._script_hisparse_abort_during_chunk_with_dma)

    @staticmethod
    def _script_hisparse_abort_during_chunk_with_dma(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN * 4, max_new_tokens=4)
        yield from run_until(
            r,
            lambda h: h.is_chunking and h.chunks_done >= 1 and h.hisparse_dma_in_flight,
        )
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert (
            not r.hisparse_dma_in_flight
        ), "abort during DMA must release staging (hisparse_staging cleared)"


if __name__ == "__main__":
    unittest.main()
