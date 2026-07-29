import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)


def _expected_chunks(prompt_len: int, chunk_size: int) -> int:
    if prompt_len <= chunk_size:
        return 0
    return (prompt_len + chunk_size - 1) // chunk_size


class TestChunkSizeDefault(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_exact_chunk_size(self):
        self.server.execute_script(self._script_exact_chunk_size)

    @staticmethod
    def _script_exact_chunk_size(t: ScriptedContext):
        r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0, (
            f"prompt_len == chunk_size completes in one non-chunked shot, "
            f"got chunks_done={r.chunks_done}"
        )

    def test_one_token_over(self):
        self.server.execute_script(self._script_one_token_over)

    @staticmethod
    def _script_one_token_over(t: ScriptedContext):
        r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 1, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2, (
            f"prompt_len == chunk_size + 1 expected exactly 2 chunks, "
            f"got chunks_done={r.chunks_done}"
        )

    def test_two_chunks_exact(self):
        self.server.execute_script(self._script_two_chunks_exact)

    @staticmethod
    def _script_two_chunks_exact(t: ScriptedContext):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2

    def test_three_chunks_one_decode(self):
        self.server.execute_script(self._script_three_chunks_one_decode)

    @staticmethod
    def _script_three_chunks_one_decode(t: ScriptedContext):
        r = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 3

    def test_tiny_prompt(self):
        self.server.execute_script(self._script_tiny_prompt)

    @staticmethod
    def _script_tiny_prompt(t: ScriptedContext):
        r = t.start_req(prompt_len=1, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert (
            r.chunks_done == 0
        ), f"single-token prompt should not chunk, got chunks_done={r.chunks_done}"

    def test_chunk_size_256_prompt_100x(self):
        self.server.execute_script(self._script_chunk_size_256_prompt_100x)

    @staticmethod
    def _script_chunk_size_256_prompt_100x(t: ScriptedContext):
        r = t.start_req(prompt_len=100 * 256, max_new_tokens=2)
        yield from run_until(r, lambda h: h.finished, max_steps=4000)
        assert r.finished
        assert r.chunks_done == 100

    def test_prompt_n_chunks_plus_minus_1(self):
        self.server.execute_script(self._script_prompt_n_chunks_plus_minus_1)

    @staticmethod
    def _script_prompt_n_chunks_plus_minus_1(t: ScriptedContext):
        for n in range(1, 6):
            prompt_minus = n * DEFAULT_CHUNK_SIZE - 1
            r_minus = t.start_req(
                prompt_len=prompt_minus, max_new_tokens=1, prompt_token=2 * n
            )
            yield from run_until_finished(r_minus, max_steps=800)
            assert r_minus.finished
            expected_minus = _expected_chunks(prompt_minus, DEFAULT_CHUNK_SIZE)
            assert r_minus.chunks_done == expected_minus, (
                f"N={n} prompt_len={prompt_minus}: "
                f"expected chunks_done={expected_minus}, got {r_minus.chunks_done}"
            )

            prompt_plus = n * DEFAULT_CHUNK_SIZE + 1
            r_plus = t.start_req(
                prompt_len=prompt_plus, max_new_tokens=1, prompt_token=2 * n + 1
            )
            yield from run_until_finished(r_plus, max_steps=800)
            assert r_plus.finished
            expected_plus = _expected_chunks(prompt_plus, DEFAULT_CHUNK_SIZE)
            assert r_plus.chunks_done == expected_plus, (
                f"N={n} prompt_len={prompt_plus}: "
                f"expected chunks_done={expected_plus}, got {r_plus.chunks_done}"
            )


class TestChunkSizeDefaultRadixExplicit(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disable_radix_cache=False,
    )

    def test_radix_hit_minus_one(self):
        self.server.execute_script(self._script_radix_hit_minus_one)

    @staticmethod
    def _script_radix_hit_minus_one(t: ScriptedContext):
        r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE - 1, max_new_tokens=1)
        yield from run_until_finished(r_warm)
        assert r_warm.finished

        yield

        r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0, (
            f"radix hit should leave only 1 fresh token; chunked path should "
            f"not engage. Got chunks_done={r.chunks_done}"
        )


class TestChunkSize1(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=1)

    def test_chunk_size_one_token(self):
        self.server.execute_script(self._script_chunk_size_one_token)

    @staticmethod
    def _script_chunk_size_one_token(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 8, f"expected 8 chunks, got {r.chunks_done}"

    def test_chunks_done_monotone_under_chunk_size_1(self):
        self.server.execute_script(self._script_chunks_done_monotone_under_chunk_size_1)

    @staticmethod
    def _script_chunks_done_monotone_under_chunk_size_1(t: ScriptedContext):
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 16, (
            f"chunk_size=1 with prompt_len=16 chunks one token at a time "
            f"(ceil(16/1)=16), got chunks_done={r.chunks_done}"
        )

    def test_chunk_size_one_long_prompt(self):
        self.server.execute_script(self._script_chunk_size_one_long_prompt)

    @staticmethod
    def _script_chunk_size_one_long_prompt(t: ScriptedContext):
        r = t.start_req(prompt_len=64, max_new_tokens=2)
        yield from run_until(r, lambda h: h.finished, max_steps=500)
        assert r.finished
        assert r.chunks_done == 64


class TestChunkSize2(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=2)

    def test_chunk_size_two_prompt_two(self):
        self.server.execute_script(self._script_chunk_size_two_prompt_two)

    @staticmethod
    def _script_chunk_size_two_prompt_two(t: ScriptedContext):
        r = t.start_req(prompt_len=2, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0

    def test_chunk_size_two_prompt_five(self):
        self.server.execute_script(self._script_chunk_size_two_prompt_five)

    @staticmethod
    def _script_chunk_size_two_prompt_five(t: ScriptedContext):
        r = t.start_req(prompt_len=5, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 3, f"expected 3 chunks, got {r.chunks_done}"


class TestChunkSize4(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=4)

    def test_chunk_size_4_prompt_4(self):
        self.server.execute_script(self._script_chunk_size_4_prompt_4)

    @staticmethod
    def _script_chunk_size_4_prompt_4(t: ScriptedContext):
        r = t.start_req(prompt_len=4, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0

    def test_chunk_size_4_prompt_5(self):
        self.server.execute_script(self._script_chunk_size_4_prompt_5)

    @staticmethod
    def _script_chunk_size_4_prompt_5(t: ScriptedContext):
        r = t.start_req(prompt_len=5, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2


class TestChunkSize16(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=16)

    def test_chunk_size_16_prompt_1024(self):
        self.server.execute_script(self._script_chunk_size_16_prompt_1024)

    @staticmethod
    def _script_chunk_size_16_prompt_1024(t: ScriptedContext):
        r = t.start_req(prompt_len=1024, max_new_tokens=2)
        yield from run_until(r, lambda h: h.finished, max_steps=1000)
        assert r.finished
        assert r.chunks_done == 64


class TestChunkSize32(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=32)

    def test_chunk_size_32_prompt_33(self):
        self.server.execute_script(self._script_chunk_size_32_prompt_33)

    @staticmethod
    def _script_chunk_size_32_prompt_33(t: ScriptedContext):
        r = t.start_req(prompt_len=33, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2


class TestChunkSize64(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=64)

    def test_chunk_size_64_prompt_129(self):
        self.server.execute_script(self._script_chunk_size_64_prompt_129)

    @staticmethod
    def _script_chunk_size_64_prompt_129(t: ScriptedContext):
        r = t.start_req(prompt_len=129, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 3


class TestChunkSize128(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=128)

    def test_chunk_size_128_prompt_512(self):
        self.server.execute_script(self._script_chunk_size_128_prompt_512)

    @staticmethod
    def _script_chunk_size_128_prompt_512(t: ScriptedContext):
        r = t.start_req(prompt_len=512, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 4


class TestChunkSize1024(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=1024)

    def test_chunk_size_1024_prompt_4096(self):
        self.server.execute_script(self._script_chunk_size_1024_prompt_4096)

    @staticmethod
    def _script_chunk_size_1024_prompt_4096(t: ScriptedContext):
        r = t.start_req(prompt_len=4096, max_new_tokens=2)
        yield from run_until(r, lambda h: h.finished, max_steps=1000)
        assert r.finished
        assert r.chunks_done == 4


class TestChunkSize4096(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=4096)

    def test_chunk_size_4096_prompt_4097(self):
        self.server.execute_script(self._script_chunk_size_4096_prompt_4097)

    @staticmethod
    def _script_chunk_size_4096_prompt_4097(t: ScriptedContext):
        r = t.start_req(prompt_len=4097, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2


class TestChunkSize1024MaxPrefill1024(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=1024, max_prefill_tokens=1024
    )

    def test_chunk_size_equals_max_prefill_tokens(self):
        self.server.execute_script(self._script_chunk_size_equals_max_prefill_tokens)

    @staticmethod
    def _script_chunk_size_equals_max_prefill_tokens(t: ScriptedContext):
        r = t.start_req(prompt_len=1024, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0


class TestChunkSize2048MaxPrefill1024(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=2048, max_prefill_tokens=1024
    )

    def test_chunk_size_exceeds_max_prefill_tokens(self):
        self.server.execute_script(self._script_chunk_size_exceeds_max_prefill_tokens)

    @staticmethod
    def _script_chunk_size_exceeds_max_prefill_tokens(t: ScriptedContext):
        r = t.start_req(prompt_len=512, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0

    def test_prompt_between_max_prefill_and_chunk_size(self):
        self.server.execute_script(
            self._script_prompt_between_max_prefill_and_chunk_size
        )

    @staticmethod
    def _script_prompt_between_max_prefill_and_chunk_size(t: ScriptedContext):
        r = t.start_req(prompt_len=1536, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0


if __name__ == "__main__":
    unittest.main()
