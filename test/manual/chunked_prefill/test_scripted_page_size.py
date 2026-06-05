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
        """Prompt that is an exact page multiple chunks one page per step."""
        self.server.execute_script(self._script_chunk_size_equals_page_size)

    @staticmethod
    def _script_chunk_size_equals_page_size(t: ScriptedContext):
        r = t.start_req(prompt_len=8 * 16, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert r.chunks_done == 8
        assert r.kv_pages == 0

    def test_chunk_size_equals_page_size_plus_one(self):
        """One token past an exact page multiple rounds up to one extra page-chunk."""
        self.server.execute_script(self._script_chunk_size_equals_page_size_plus_one)

    @staticmethod
    def _script_chunk_size_equals_page_size_plus_one(t: ScriptedContext):
        # One token past an exact 8-page (128-token) prompt: the single extra
        # token rounds up to its own 16-token page, adding exactly one more
        # page-padded chunk on top of the prompt_len=128 -> chunks_done=8 case.
        r = t.start_req(prompt_len=129, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert r.chunks_done == 9
        assert r.kv_pages == 0


class TestPageSize16ChunkSize16NonMultiplePrompt(ScriptedTestCase):
    # chunked_prefill_size must be a multiple of page_size (server_args), so a
    # sub-page chunk is impossible; chunk == page is the smallest valid "tiny
    # chunk".
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=16, page_size=16)

    def test_non_page_multiple_prompt_advances_in_page_steps(self):
        """A prompt that is not a page multiple still advances in page-aligned chunks."""
        self.server.execute_script(
            self._script_non_page_multiple_prompt_advances_in_page_steps
        )

    @staticmethod
    def _script_non_page_multiple_prompt_advances_in_page_steps(t: ScriptedContext):
        # prompt_len=60 with chunk==page==16 -> ceil(60/16) == 4 page-aligned
        # chunks (16, 16, 16, 12); the trailing partial page is its own chunk.
        r = t.start_req(prompt_len=60, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert r.chunks_done == 4
        assert r.kv_pages == 0


class TestPageSize16ChunkSize64(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=64, page_size=16)

    def test_chunk_spans_multiple_pages_exact(self):
        """A chunk that is a 4x page multiple consumes four pages per step."""
        self.server.execute_script(self._script_chunk_spans_multiple_pages_exact)

    @staticmethod
    def _script_chunk_spans_multiple_pages_exact(t: ScriptedContext):
        r = t.start_req(prompt_len=4 * 64, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert r.chunks_done == 4
        assert r.kv_pages == 0

    def test_chunk_spans_multiple_pages_plus_one_page(self):
        """A trailing partial page adds exactly one more page-aligned chunk."""
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
        """A page-aligned radix prefix hit removes its chunks from the next req."""
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
        # The first 4 pages hit the radix cache, leaving 4 fresh pages to chunk
        # one page at a time under chunk_size == page_size.
        assert r.chunks_done == 4
        assert r.kv_pages == 0


if __name__ == "__main__":
    unittest.main()
