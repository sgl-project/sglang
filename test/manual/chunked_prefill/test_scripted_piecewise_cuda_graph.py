"""Piecewise CUDA graph × chunked: naive ScriptedRuntime smoke.

Piecewise CUDA graph is on by default; per instructions.md we just
verify the chunked path still works without disabling it. The smoke
catches "piecewise graph capture trips on a chunked extend".
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
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


if __name__ == "__main__":
    unittest.main()
