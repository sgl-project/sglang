"""HiSparse × chunked: naive ScriptedRuntime smoke.

HiSparse staging DMA can race with chunked admission (audit doc
§ "HiSparse"). A naive smoke just verifies "engine starts with both
flags enabled + chunked path completes". Deeper time-sensitive cases
live in ``test_scripted_special_case_coverage.py``.

Uses the same GLM-5-FP8 model and 8×H200 layout as
``test_dsa_models_hisparse.py``.
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

_HISPARSE_MODEL = "zai-org/GLM-5-FP8"


class TestScriptedHiSparse(CustomTestCase):
    def test_naive_hisparse_chunked(self):
        """Long enough to trigger both chunked prefill and a hisparse staging transfer mid-chunk."""
        execute_scripted_runtime(
            self._script_naive_hisparse_chunked,
            **base_engine_kwargs(
                model_path=_HISPARSE_MODEL,
                tp_size=8,
                dp_size=8,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dp_attention=True,
                enable_hisparse=True,
            ),
        )

    @staticmethod
    def _script_naive_hisparse_chunked(t: ScriptedRuntime):
        # Long enough to trigger both chunked prefill and a hisparse
        # staging transfer mid-chunk.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN * 4, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_hisparse_staging_dma_during_chunk_admit(self):
        """HiSparse staging DMA in flight + chunk admission must not deadlock."""
        execute_scripted_runtime(
            self._script_hisparse_staging_dma_during_chunk_admit,
            **base_engine_kwargs(
                model_path=_HISPARSE_MODEL,
                tp_size=8,
                dp_size=8,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dp_attention=True,
                enable_hisparse=True,
            ),
        )

    # HiSparse staging DMA + chunk admission race — when a
    # DMA transfer is in flight at admission time the scheduler must not
    # deadlock; the chunked req must still progress and finish.
    @staticmethod
    def _script_hisparse_staging_dma_during_chunk_admit(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN * 4, max_new_tokens=2)
        # Wait until a staging DMA is in flight before admitting more work.
        yield from run_until(r, lambda h: h.hisparse_dma_in_flight)
        # Admit a second req while DMA is still in flight — this is the
        # race window the bug targets.
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=2000)
        yield from run_until_finished(r2, max_steps=2000)
        assert r.finished and r2.finished

    def test_hisparse_abort_during_chunk_with_dma(self):
        """HiSparse DMA + chunk + abort three-way race must cancel cleanly."""
        execute_scripted_runtime(
            self._script_hisparse_abort_during_chunk_with_dma,
            **base_engine_kwargs(
                model_path=_HISPARSE_MODEL,
                tp_size=8,
                dp_size=8,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dp_attention=True,
                enable_hisparse=True,
            ),
        )

    # HiSparse DMA + mid-chunk + abort — cancellation must
    # complete cleanly with no orphaned staging buffers or stuck DMA.
    @staticmethod
    def _script_hisparse_abort_during_chunk_with_dma(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN * 4, max_new_tokens=4)
        yield from run_until(
            r,
            lambda h: h.is_chunking and h.chunks_done >= 1 and h.hisparse_dma_in_flight,
        )
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert r.hisparse_staging_buffers_held == 0, (
            f"abort during DMA must release staging buffers, got "
            f"{r.hisparse_staging_buffers_held} still held"
        )


if __name__ == "__main__":
    unittest.main()
