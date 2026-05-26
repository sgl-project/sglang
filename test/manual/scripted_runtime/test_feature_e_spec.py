"""Feature e — Speculative decoding × chunked: naive ScriptedRuntime smoke.

EAGLE-style spec decoding adds a verify pass between prefill chunks
and decode. We want a long chunked prompt to be admitted, complete
its chunk loop, and then transition to spec-decoded output without
any state-machine surprises.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.test_utils import CustomTestCase

from test.manual.scripted_runtime.common import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until_finished,
)


_SPEC_MODEL = "Qwen/Qwen3-8B"
_SPEC_DRAFT = "Qwen/Qwen3-8B-EAGLE"


def _script_naive_spec_chunked(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done >= 2


class TestFeatureESpecChunked(CustomTestCase):
    def test_naive_spec_chunked(self):
        execute_scripted_runtime(
            _script_naive_spec_chunked,
            **base_engine_kwargs(
                model_path=_SPEC_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                speculative_algorithm="EAGLE",
                speculative_draft_model_path=_SPEC_DRAFT,
                speculative_num_steps=3,
                speculative_eagle_topk=4,
                speculative_num_draft_tokens=8,
            ),
        )


if __name__ == "__main__":
    unittest.main()
