"""HTTP-path smoke test for the ScriptedRuntime harness.

Exercises only the implemented ScriptedRuntime surface (``start_req``,
``ReqHandle.finished``, ``ReqHandle.is_chunking``, ``ReqHandle.req``) so it
can run end-to-end on the small CI model. The point is to verify the new
real-HTTP-server launch path and the ``start_req`` -> ``/generate`` ->
``wait_until_rid_arrived`` round-trip, not any wishlist observability.
"""

import unittest

from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.test_case import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    base_engine_kwargs,
    run_until_finished,
)


class TestScriptedHttpSmoke(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_single_short_req_finishes(self):
        """A short prompt submitted over HTTP runs to finish."""
        self.runtime.run(self._script_single_short_req_finishes)

    @staticmethod
    def _script_single_short_req_finishes(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished

    def test_chunked_req_is_chunking_then_finishes(self):
        """A prompt several chunks long is chunking mid-flight, then finishes."""
        self.runtime.run(self._script_chunked_req_is_chunking_then_finishes)

    @staticmethod
    def _script_chunked_req_is_chunking_then_finishes(t: ScriptedRuntime):
        r = t.start_req(prompt_len=4 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        saw_chunking = False
        for _ in range(800):
            if r.is_chunking:
                saw_chunking = True
            if r.finished:
                break
            yield
        assert r.finished
        assert (
            saw_chunking
        ), "expected the req to hold the chunked_req slot at least once"

    def test_two_reqs_finish(self):
        """Two requests submitted over HTTP both run to finish."""
        self.runtime.run(self._script_two_reqs_finish)

    @staticmethod
    def _script_two_reqs_finish(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=8, max_new_tokens=4)
        r2 = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        for _ in range(800):
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished
        assert r2.finished


if __name__ == "__main__":
    unittest.main()
