import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    base_engine_kwargs,
)


class TestScriptedHttpSmoke(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_chunked_req_is_chunking_then_finishes(self):
        self.server.execute_script(self._script_chunked_req_is_chunking_then_finishes)

    @staticmethod
    def _script_chunked_req_is_chunking_then_finishes(t: ScriptedContext):
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
        self.server.execute_script(self._script_two_reqs_finish)

    @staticmethod
    def _script_two_reqs_finish(t: ScriptedContext):
        r1 = t.start_req(prompt_len=8, max_new_tokens=4)
        r2 = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        # Probe BOTH handles every step (no short-circuit): a handle is only
        # registered for post-recycle finished-tracking by probing it, so the
        # faster req must not go unprobed while waiting on the slower one.
        done = {r1.rid: False, r2.rid: False}
        for _ in range(800):
            done[r1.rid] = done[r1.rid] or r1.finished
            done[r2.rid] = done[r2.rid] or r2.finished
            if all(done.values()):
                break
            yield
        assert done[r1.rid]
        assert done[r2.rid]
        assert r1.chunks_done == 0
        # r2's prompt is 2 * DEFAULT_CHUNK_SIZE (512), an exact multiple of the
        # 256 chunk size, so it chunks into ceil(512 / 256) = 2 partial prefill
        # iterations regardless of co-batching with r1.
        assert r2.chunks_done == 2


if __name__ == "__main__":
    unittest.main()
