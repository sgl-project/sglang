"""DP attention × chunked: naive ScriptedRuntime smoke.

DP attention shards attention across DP ranks; the chunked path runs
independently on each DP rank. The naive smoke just verifies a
chunked request completes when DP attention is on.

Requires 2 GPUs and ScriptedRuntime multi-rank support (wishlist §4
P2 (12)).
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


def _script_naive_dp_attention_chunked(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done >= 2


class TestScriptedDPAttention(CustomTestCase):
    def test_naive_dp_attention_chunked(self):
        execute_scripted_runtime(
            _script_naive_dp_attention_chunked,
            **base_engine_kwargs(
                tp_size=2,
                dp_size=2,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dp_attention=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
