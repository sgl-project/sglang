"""Edge cases — chunk boundary arithmetic.

Verifies that prompt lengths exactly at /
just over / well over the chunk boundary all go through the right
path (chunked vs single-shot), and that ``chunks_done`` reports a
sensible count.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    base_engine_kwargs,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase


# prompt_len == chunk_size → no chunking, single-shot prefill.
def _script_exact_chunk_size(t: ScriptedRuntime):
    r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done <= 1, (
        f"prompt_len == chunk_size should fit in one chunk, "
        f"got chunks_done={r.chunks_done}"
    )


# prompt_len == chunk_size + 1 → exactly two chunks, second is 1 token.
def _script_one_token_over(t: ScriptedRuntime):
    r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 1, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 2, (
        f"prompt_len == chunk_size + 1 expected exactly 2 chunks, "
        f"got chunks_done={r.chunks_done}"
    )


# prompt_len == 2 * chunk_size → exactly 2 chunks.
def _script_two_chunks_exact(t: ScriptedRuntime):
    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 2


# prompt_len == 3 * chunk_size, max_new_tokens=1 → 3 chunks + 1 decode.
def _script_three_chunks_one_decode(t: ScriptedRuntime):
    r = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE, max_new_tokens=1)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 3


# prompt_len == 1 → no chunking; req runs straight through decode.
def _script_tiny_prompt(t: ScriptedRuntime):
    r = t.start_req(prompt_len=1, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert (
        r.chunks_done == 0
    ), f"single-token prompt should not chunk, got chunks_done={r.chunks_done}"


# long prefix already in radix; req that follows shares prefix
# such that only 1 token is uncached → no chunking required for the
# tail. Verifies the chunked-resume path is not taken when the radix
# match already covers everything but a sliver.
def _script_radix_hit_minus_one(t: ScriptedRuntime):
    # Warm radix with chunk_size-1 tokens.
    r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE - 1, max_new_tokens=1)
    yield from run_until_finished(r_warm)
    assert r_warm.finished

    # Same prefix + 1 extra token = total chunk_size. Even though the
    # *total* prompt equals chunk_size, only 1 token is fresh.
    r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 0, (
        f"radix hit should leave only 1 fresh token; chunked path should "
        f"not engage. Got chunks_done={r.chunks_done}"
    )


class TestEdgeChunkBoundary(CustomTestCase):
    def test_exact_chunk_size(self):
        execute_scripted_runtime(
            _script_exact_chunk_size,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_one_token_over(self):
        execute_scripted_runtime(
            _script_one_token_over,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_two_chunks_exact(self):
        execute_scripted_runtime(
            _script_two_chunks_exact,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_three_chunks_one_decode(self):
        execute_scripted_runtime(
            _script_three_chunks_one_decode,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_tiny_prompt(self):
        execute_scripted_runtime(
            _script_tiny_prompt,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_radix_hit_minus_one(self):
        execute_scripted_runtime(
            _script_radix_hit_minus_one,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
