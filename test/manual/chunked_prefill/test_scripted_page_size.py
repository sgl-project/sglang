import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    base_engine_kwargs,
    run_until_finished,
)


class TestPageSize16ChunkSize16(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=16, page_size=16)

    def test_chunk_size_equals_page_size(self):
        self.server.execute_script(self._script_chunk_size_equals_page_size)

    @staticmethod
    def _script_chunk_size_equals_page_size(t: ScriptedContext):
        r = t.start_req(prompt_len=8 * 16, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert r.chunks_done == 8
        assert r.kv_pages == 0

    def test_chunk_size_equals_page_size_plus_one(self):
        self.server.execute_script(self._script_chunk_size_equals_page_size_plus_one)

    @staticmethod
    def _script_chunk_size_equals_page_size_plus_one(t: ScriptedContext):
        r = t.start_req(prompt_len=129, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert r.chunks_done == 9
        assert r.kv_pages == 0


class TestPageSize16ChunkSize16NonMultiplePrompt(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=16, page_size=16)

    def test_non_page_multiple_prompt_advances_in_page_steps(self):
        self.server.execute_script(
            self._script_non_page_multiple_prompt_advances_in_page_steps
        )

    @staticmethod
    def _script_non_page_multiple_prompt_advances_in_page_steps(t: ScriptedContext):
        r = t.start_req(prompt_len=60, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert r.chunks_done == 4
        assert r.kv_pages == 0


class TestPageSize16ChunkSize64(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=64, page_size=16)

    def test_chunk_spans_multiple_pages_exact(self):
        self.server.execute_script(self._script_chunk_spans_multiple_pages_exact)

    @staticmethod
    def _script_chunk_spans_multiple_pages_exact(t: ScriptedContext):
        r = t.start_req(prompt_len=4 * 64, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert r.chunks_done == 4
        assert r.kv_pages == 0

    def test_chunk_spans_multiple_pages_plus_one_page(self):
        self.server.execute_script(
            self._script_chunk_spans_multiple_pages_plus_one_page
        )

    @staticmethod
    def _script_chunk_spans_multiple_pages_plus_one_page(t: ScriptedContext):
        r = t.start_req(prompt_len=4 * 64 + 16, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert r.chunks_done == 5
        assert r.kv_pages == 0


class TestPageSize16RadixHit(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=16, page_size=16, disable_radix_cache=False
    )

    def test_radix_hit_on_page_boundary(self):
        self.server.execute_script(self._script_radix_hit_on_page_boundary)

    @staticmethod
    def _script_radix_hit_on_page_boundary(t: ScriptedContext):
        r_warm = t.start_req(prompt_len=4 * 16, max_new_tokens=1)
        yield from run_until_finished(r_warm, max_steps=400)
        assert r_warm.finished

        yield

        r = t.start_req(prompt_len=8 * 16, max_new_tokens=1)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert r.chunks_done == 4
        assert r.kv_pages == 0


if __name__ == "__main__":
    unittest.main()
