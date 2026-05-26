"""Feature g — Priority scheduling × chunked: naive ScriptedRuntime smoke.

Submit a low-priority long-prompt request that must be chunked, then
a high-priority short request. With priority preemption enabled the
high-priority req should not starve waiting on the low-priority one's
chunk loop.

Requires the wishlist API extension ``start_req(..., priority=...)``
(§4 P2 (10)). Until it lands the priority kwargs are passed as-is and
will surface as a clear AttributeError at script time.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.test_utils import CustomTestCase

from test.manual.scripted_runtime.common import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until_all_finished,
)


def _script_naive_priority_chunked(t: ScriptedRuntime):
    low = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, priority="low"
    )
    yield  # let scheduler pull `low` and begin its chunk loop

    high = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")

    yield from run_until_all_finished([low, high])
    assert low.finished and high.finished


class TestFeatureGPriorityChunked(CustomTestCase):
    def test_naive_priority_chunked(self):
        execute_scripted_runtime(
            _script_naive_priority_chunked,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
