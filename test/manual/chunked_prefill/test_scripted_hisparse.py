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
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase

_HISPARSE_MODEL = "zai-org/GLM-5-FP8"


def _script_naive_hisparse_chunked(t: ScriptedRuntime):
    # Long enough to trigger both chunked prefill and a hisparse
    # staging transfer mid-chunk.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN * 4, max_new_tokens=4)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done >= 2


class TestScriptedHiSparse(CustomTestCase):
    def test_naive_hisparse_chunked(self):
        execute_scripted_runtime(
            _script_naive_hisparse_chunked,
            **base_engine_kwargs(
                model_path=_HISPARSE_MODEL,
                tp_size=8,
                dp_size=8,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dp_attention=True,
                enable_hisparse=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
