"""Radix prefix cache × chunked: naive ScriptedRuntime smoke.

Two requests with the same long prefix. The second request hits the
radix cache after the first completes; what remains should still be
routed through the chunked path if it's longer than chunk_size, or
admit directly if it isn't. We pick prompt lengths such that:

* r1 prefix = long → ends up in radix cache
* r2 prefix = same long prefix + extra suffix > chunk_size → still
  chunks the *suffix* via the chunked-resume path.

Touches `init_next_round_input(tree_cache)` branching (audit doc
§ "Radix cache prefix match").
"""

import unittest
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until_finished,
)

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.test_utils import CustomTestCase


def _script_naive_radix_chunked(t: ScriptedRuntime):
    # First request populates the radix tree with a long shared prefix.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r1)
    assert r1.finished

    # Second request — same prefix tokens (start_req uses placeholder
    # token id 1, so r2's prefix is byte-identical to r1's). Add tail
    # > chunk_size so the residual still chunks.
    r2 = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN + DEFAULT_CHUNK_SIZE * 2,
        max_new_tokens=2,
    )
    yield from run_until_finished(r2)
    assert r2.finished


class TestFeatureRadixChunked(CustomTestCase):
    def test_naive_radix_chunked(self):
        execute_scripted_runtime(
            _script_naive_radix_chunked,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                schedule_policy="fcfs",
            ),
        )


if __name__ == "__main__":
    unittest.main()
