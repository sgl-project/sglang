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
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase

_SWA_MODEL = "openai/gpt-oss-20b"


def _script_naive_swa_chunked(t: ScriptedRuntime):
    # Length chosen to exceed both DEFAULT_CHUNK_SIZE *and* the SWA
    # window for gpt-oss-20b (4096) — guarantees the chunk loop has to
    # cross the SWA boundary at least once.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN + 4096, max_new_tokens=4)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done >= 2


class TestScriptedHybridSWA(CustomTestCase):
    def test_naive_swa_chunked(self):
        execute_scripted_runtime(
            _script_naive_swa_chunked,
            **base_engine_kwargs(
                model_path=_SWA_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                mem_fraction_static=0.70,
                disable_piecewise_cuda_graph=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
