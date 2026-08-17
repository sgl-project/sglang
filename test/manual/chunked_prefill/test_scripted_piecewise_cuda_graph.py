import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)


class TestPiecewiseBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disable_cuda_graph=False,
    )

    def test_naive_piecewise_cg_chunked(self):
        self.server.execute_script(self._script_naive_piecewise_cg_chunked)

    @staticmethod
    def _script_naive_piecewise_cg_chunked(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=8)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_piecewise_cg_tail_chunk_tiny(self):
        self.server.execute_script(self._script_piecewise_cg_tail_chunk_tiny)

    @staticmethod
    def _script_piecewise_cg_tail_chunk_tiny(t: ScriptedContext):
        r = t.start_req(
            prompt_len=4 * DEFAULT_CHUNK_SIZE + 1,
            max_new_tokens=2,
        )
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done == 5


class TestPiecewiseRetractResume(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disable_cuda_graph=False,
    )

    def test_piecewise_cg_retract_resume(self):
        self.server.execute_script(self._script_piecewise_cg_retract_resume)

    @staticmethod
    def _script_piecewise_cg_retract_resume(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0


if __name__ == "__main__":
    unittest.main()
