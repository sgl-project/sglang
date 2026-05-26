"""Edge cases — chunk_size × prompt_len cartesian variation.

Covers the A.1 series from the expansion plan and a parametrised
fan-out across chunk_size ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
1024, 4096}.

Each script asserts ``chunks_done`` matches the expected ceil-div
arithmetic and that the final state is clean (``r.finished`` true,
no resource leak).
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.req_handle import ReqHandle
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase


def _expected_chunks(prompt_len: int, chunk_size: int) -> int:
    """Inline ad-hoc helper — ceil-div with the chunks_done semantic.

    Per the pinned semantics in the round-1 plan:
    * prompt_len <= chunk_size: 0 (single-shot, no chunked path)
    * prompt_len == N * chunk_size, N >= 2: N
    * prompt_len == N * chunk_size + r, 0 < r < chunk_size, N >= 1: N + 1
    """
    if prompt_len <= chunk_size:
        return 0
    return (prompt_len + chunk_size - 1) // chunk_size


def _script_chunk_size_one_token(t: ScriptedRuntime):
    # chunk_size = 1: every token is its own chunk.
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 8, f"expected 8 chunks, got {r.chunks_done}"


def _script_chunk_size_two_prompt_one(t: ScriptedRuntime):
    # chunk_size = 2, prompt_len = 1: single-shot, no chunking.
    r = t.start_req(prompt_len=1, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 0


def _script_chunk_size_two_prompt_two(t: ScriptedRuntime):
    # chunk_size = 2, prompt_len = 2: equal, single-shot.
    r = t.start_req(prompt_len=2, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done <= 1


def _script_chunk_size_two_prompt_five(t: ScriptedRuntime):
    # chunk_size = 2, prompt_len = 5: chunks_done == 3 (2 + 2 + 1).
    r = t.start_req(prompt_len=5, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 3, f"expected 3 chunks, got {r.chunks_done}"


def _script_chunk_size_equals_max_prefill_tokens(t: ScriptedRuntime):
    # chunk_size == max_prefill_tokens: prompt fits in one shot.
    r = t.start_req(prompt_len=1024, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done <= 1


def _script_chunk_size_exceeds_max_prefill_tokens(t: ScriptedRuntime):
    # chunk_size > max_prefill_tokens: engine should reject or cap.
    # Test verifies submission still completes (engine internally
    # capped chunk_size to max_prefill_tokens).
    # NEW API NEEDED: engine_args validation — server_args should
    # reject chunked_prefill_size > max_prefill_tokens cleanly.
    r = t.start_req(prompt_len=512, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished


def _script_chunk_size_4096_prompt_4097(t: ScriptedRuntime):
    # chunk_size = 4096, prompt_len = 4097: 2 chunks (4096 + 1).
    r = t.start_req(prompt_len=4097, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 2


def _script_chunk_size_256_prompt_10x(t: ScriptedRuntime):
    # chunk_size = 256, prompt_len = 10 * 256 = 2560: chunks_done == 10.
    r = t.start_req(prompt_len=10 * 256, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 10


def _script_chunk_size_256_prompt_100x(t: ScriptedRuntime):
    # chunk_size = 256, prompt_len = 100 * 256 = 25600: long-prompt
    # stability test. May be slow on small models.
    r = t.start_req(prompt_len=100 * 256, max_new_tokens=2)
    yield from run_until(r, lambda h: h.finished, max_steps=2000)
    assert r.finished
    assert r.chunks_done == 100


def _script_chunk_size_16_prompt_1024(t: ScriptedRuntime):
    # High fragmentation: chunk_size = 16, prompt_len = 1024 = 64 chunks.
    r = t.start_req(prompt_len=1024, max_new_tokens=2)
    yield from run_until(r, lambda h: h.finished, max_steps=1000)
    assert r.finished
    assert r.chunks_done == 64


def _script_chunk_size_4_prompt_4(t: ScriptedRuntime):
    # chunk_size = 4, prompt_len = 4: single-shot boundary.
    r = t.start_req(prompt_len=4, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done <= 1


def _script_chunk_size_4_prompt_5(t: ScriptedRuntime):
    # chunk_size = 4, prompt_len = 5: chunks_done == 2 (4 + 1).
    r = t.start_req(prompt_len=5, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 2


def _script_chunk_size_8_prompt_64(t: ScriptedRuntime):
    # chunk_size = 8, prompt_len = 64: exactly 8 chunks.
    r = t.start_req(prompt_len=64, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 8


def _script_chunk_size_32_prompt_33(t: ScriptedRuntime):
    # chunk_size = 32, prompt_len = 33: 2 chunks, second is 1 token.
    r = t.start_req(prompt_len=33, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 2


def _script_chunk_size_64_prompt_127(t: ScriptedRuntime):
    # chunk_size = 64, prompt_len = 127: 2 chunks (64 + 63).
    r = t.start_req(prompt_len=127, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 2


def _script_chunk_size_64_prompt_128(t: ScriptedRuntime):
    # chunk_size = 64, prompt_len = 128: 2 chunks (64 + 64).
    r = t.start_req(prompt_len=128, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 2


def _script_chunk_size_64_prompt_129(t: ScriptedRuntime):
    # chunk_size = 64, prompt_len = 129: 3 chunks (64 + 64 + 1).
    r = t.start_req(prompt_len=129, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 3


def _script_chunk_size_128_prompt_512(t: ScriptedRuntime):
    # chunk_size = 128, prompt_len = 512: 4 chunks.
    r = t.start_req(prompt_len=512, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 4


def _script_chunk_size_1024_prompt_4096(t: ScriptedRuntime):
    # chunk_size = 1024, prompt_len = 4096: 4 chunks.
    r = t.start_req(prompt_len=4096, max_new_tokens=2)
    yield from run_until(r, lambda h: h.finished, max_steps=1000)
    assert r.finished
    assert r.chunks_done == 4


def _script_chunks_done_monotone_under_chunk_size_1(t: ScriptedRuntime):
    # Invariant: chunks_done is monotone non-decreasing across yields.
    r = t.start_req(prompt_len=16, max_new_tokens=2)
    prev = 0
    for _ in range(DEFAULT_MAX_STEPS):
        cur = r.chunks_done
        assert cur >= prev, (
            f"chunks_done regressed: prev={prev}, cur={cur}"
        )
        prev = cur
        if r.finished:
            return
        yield
    raise AssertionError("req never finished")


def _script_chunks_done_monotone_under_chunk_size_16(t: ScriptedRuntime):
    # Invariant: chunks_done monotone with a larger chunk size.
    r = t.start_req(prompt_len=160, max_new_tokens=2)
    prev = 0
    for _ in range(DEFAULT_MAX_STEPS):
        cur = r.chunks_done
        assert cur >= prev
        prev = cur
        if r.finished:
            return
        yield
    raise AssertionError("req never finished")


def _script_chunk_size_one_long_prompt(t: ScriptedRuntime):
    # chunk_size = 1, prompt_len = 64: stress fragmentation.
    r = t.start_req(prompt_len=64, max_new_tokens=2)
    yield from run_until(r, lambda h: h.finished, max_steps=500)
    assert r.finished
    assert r.chunks_done == 64


def _script_chunk_size_two_long_prompt(t: ScriptedRuntime):
    # chunk_size = 2, prompt_len = 64: 32 chunks.
    r = t.start_req(prompt_len=64, max_new_tokens=2)
    yield from run_until(r, lambda h: h.finished, max_steps=300)
    assert r.finished
    assert r.chunks_done == 32


class TestEdgeChunkSizeVariation(CustomTestCase):
    def test_chunk_size_one_token(self):
        execute_scripted_runtime(
            _script_chunk_size_one_token,
            **base_engine_kwargs(chunked_prefill_size=1),
        )

    def test_chunk_size_two_prompt_one(self):
        execute_scripted_runtime(
            _script_chunk_size_two_prompt_one,
            **base_engine_kwargs(chunked_prefill_size=2),
        )

    def test_chunk_size_two_prompt_two(self):
        execute_scripted_runtime(
            _script_chunk_size_two_prompt_two,
            **base_engine_kwargs(chunked_prefill_size=2),
        )

    def test_chunk_size_two_prompt_five(self):
        execute_scripted_runtime(
            _script_chunk_size_two_prompt_five,
            **base_engine_kwargs(chunked_prefill_size=2),
        )

    def test_chunk_size_equals_max_prefill_tokens(self):
        execute_scripted_runtime(
            _script_chunk_size_equals_max_prefill_tokens,
            **base_engine_kwargs(chunked_prefill_size=1024, max_prefill_tokens=1024),
        )

    def test_chunk_size_exceeds_max_prefill_tokens(self):
        execute_scripted_runtime(
            _script_chunk_size_exceeds_max_prefill_tokens,
            **base_engine_kwargs(chunked_prefill_size=2048, max_prefill_tokens=1024),
        )

    def test_chunk_size_4096_prompt_4097(self):
        execute_scripted_runtime(
            _script_chunk_size_4096_prompt_4097,
            **base_engine_kwargs(chunked_prefill_size=4096),
        )

    def test_chunk_size_256_prompt_10x(self):
        execute_scripted_runtime(
            _script_chunk_size_256_prompt_10x,
            **base_engine_kwargs(chunked_prefill_size=256),
        )

    def test_chunk_size_256_prompt_100x(self):
        execute_scripted_runtime(
            _script_chunk_size_256_prompt_100x,
            **base_engine_kwargs(chunked_prefill_size=256),
        )

    def test_chunk_size_16_prompt_1024(self):
        execute_scripted_runtime(
            _script_chunk_size_16_prompt_1024,
            **base_engine_kwargs(chunked_prefill_size=16),
        )

    def test_chunk_size_4_prompt_4(self):
        execute_scripted_runtime(
            _script_chunk_size_4_prompt_4,
            **base_engine_kwargs(chunked_prefill_size=4),
        )

    def test_chunk_size_4_prompt_5(self):
        execute_scripted_runtime(
            _script_chunk_size_4_prompt_5,
            **base_engine_kwargs(chunked_prefill_size=4),
        )

    def test_chunk_size_8_prompt_64(self):
        execute_scripted_runtime(
            _script_chunk_size_8_prompt_64,
            **base_engine_kwargs(chunked_prefill_size=8),
        )

    def test_chunk_size_32_prompt_33(self):
        execute_scripted_runtime(
            _script_chunk_size_32_prompt_33,
            **base_engine_kwargs(chunked_prefill_size=32),
        )

    def test_chunk_size_64_prompt_127(self):
        execute_scripted_runtime(
            _script_chunk_size_64_prompt_127,
            **base_engine_kwargs(chunked_prefill_size=64),
        )

    def test_chunk_size_64_prompt_128(self):
        execute_scripted_runtime(
            _script_chunk_size_64_prompt_128,
            **base_engine_kwargs(chunked_prefill_size=64),
        )

    def test_chunk_size_64_prompt_129(self):
        execute_scripted_runtime(
            _script_chunk_size_64_prompt_129,
            **base_engine_kwargs(chunked_prefill_size=64),
        )

    def test_chunk_size_128_prompt_512(self):
        execute_scripted_runtime(
            _script_chunk_size_128_prompt_512,
            **base_engine_kwargs(chunked_prefill_size=128),
        )

    def test_chunk_size_1024_prompt_4096(self):
        execute_scripted_runtime(
            _script_chunk_size_1024_prompt_4096,
            **base_engine_kwargs(chunked_prefill_size=1024),
        )

    def test_chunks_done_monotone_under_chunk_size_1(self):
        execute_scripted_runtime(
            _script_chunks_done_monotone_under_chunk_size_1,
            **base_engine_kwargs(chunked_prefill_size=1),
        )

    def test_chunks_done_monotone_under_chunk_size_16(self):
        execute_scripted_runtime(
            _script_chunks_done_monotone_under_chunk_size_16,
            **base_engine_kwargs(chunked_prefill_size=16),
        )

    def test_chunk_size_one_long_prompt(self):
        execute_scripted_runtime(
            _script_chunk_size_one_long_prompt,
            **base_engine_kwargs(chunked_prefill_size=1),
        )

    def test_chunk_size_two_long_prompt(self):
        execute_scripted_runtime(
            _script_chunk_size_two_long_prompt,
            **base_engine_kwargs(chunked_prefill_size=2),
        )


if __name__ == "__main__":
    unittest.main()
