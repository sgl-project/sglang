"""PP × chunked: naive ScriptedRuntime smoke.

Submit one long-prompt request that must be chunked across at least
two scheduler iterations, with ``pp_size=2`` and ``tp_size=2`` so the
chunked req crosses microbatch boundaries.

Asserts the request reaches ``finished`` and went through ≥ 2 chunks.
Does not attempt to reproduce the 309b6dc last-chunk-in-flight race —
that lives in ``test_scripted_regression_309b6dc.py``.

Requires 4 GPUs. ScriptedRuntime must support ``pp_size > 1`` / ``tp_size > 1``
(see wishlist §4 P2 (12)).
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
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, CustomTestCase


def _script_naive_pp_chunked(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
    yield from run_until_finished(r)
    assert r.finished
    # VERY_LONG_PROMPT_LEN / DEFAULT_CHUNK_SIZE chunks expected.
    assert r.chunks_done >= 2, f"expected ≥2 chunks, got {r.chunks_done}"


class TestScriptedPP(CustomTestCase):
    def test_naive_pp_chunked(self):
        execute_scripted_runtime(
            _script_naive_pp_chunked,
            **base_engine_kwargs(
                model_path=DEFAULT_MODEL_NAME_FOR_TEST,
                tp_size=2,
                pp_size=2,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dynamic_chunking=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
