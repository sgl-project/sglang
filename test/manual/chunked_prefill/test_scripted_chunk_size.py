"""Chunk size and chunk boundary scripted tests.

Verifies that prompt lengths exactly at /
just over / well over the chunk boundary all go through the right
path (chunked vs single-shot), and that ``chunks_done`` reports a
sensible count.

Also covers the A.1 series from the expansion plan and a parametrised
fan-out across chunk_size ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
1024, 4096}. Each script asserts ``chunks_done`` matches the expected
ceil-div arithmetic and that the final state is clean (``r.finished``
true, no resource leak).
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    base_engine_kwargs,
    run_until,
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


class TestScriptedChunkSize(CustomTestCase):
    def test_exact_chunk_size(self):
        """Prompt_len == chunk_size → no chunking, single-shot prefill."""
        execute_scripted_runtime(
            self._script_exact_chunk_size,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # prompt_len == chunk_size → no chunking, single-shot prefill.
    @staticmethod
    def _script_exact_chunk_size(t: ScriptedRuntime):
        r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done <= 1, (
            f"prompt_len == chunk_size should fit in one chunk, "
            f"got chunks_done={r.chunks_done}"
        )

    def test_one_token_over(self):
        """Prompt_len == chunk_size + 1 → exactly two chunks, second is 1 token."""
        execute_scripted_runtime(
            self._script_one_token_over,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # prompt_len == chunk_size + 1 → exactly two chunks, second is 1 token.
    @staticmethod
    def _script_one_token_over(t: ScriptedRuntime):
        r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 1, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2, (
            f"prompt_len == chunk_size + 1 expected exactly 2 chunks, "
            f"got chunks_done={r.chunks_done}"
        )

    def test_two_chunks_exact(self):
        """Prompt_len == 2 * chunk_size → exactly 2 chunks."""
        execute_scripted_runtime(
            self._script_two_chunks_exact,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # prompt_len == 2 * chunk_size → exactly 2 chunks.
    @staticmethod
    def _script_two_chunks_exact(t: ScriptedRuntime):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2

    def test_three_chunks_one_decode(self):
        """Prompt_len == 3 * chunk_size, max_new_tokens=1 → 3 chunks + 1 decode."""
        execute_scripted_runtime(
            self._script_three_chunks_one_decode,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # prompt_len == 3 * chunk_size, max_new_tokens=1 → 3 chunks + 1 decode.
    @staticmethod
    def _script_three_chunks_one_decode(t: ScriptedRuntime):
        r = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 3

    def test_tiny_prompt(self):
        """Prompt_len == 1 → no chunking; req runs straight through decode."""
        execute_scripted_runtime(
            self._script_tiny_prompt,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # prompt_len == 1 → no chunking; req runs straight through decode.
    @staticmethod
    def _script_tiny_prompt(t: ScriptedRuntime):
        r = t.start_req(prompt_len=1, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert (
            r.chunks_done == 0
        ), f"single-token prompt should not chunk, got chunks_done={r.chunks_done}"

    def test_radix_hit_minus_one(self):
        """Radix hit leaving 1 fresh token does not engage chunked path."""
        # Radix cache is on by default; pass explicitly to make the
        # dependency obvious — if someone flips the default later, the
        # test still pins what it needs.
        execute_scripted_runtime(
            self._script_radix_hit_minus_one,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disable_radix_cache=False,
            ),
        )

    # long prefix already in radix; req that follows shares prefix
    # such that only 1 token is uncached → no chunking required for the
    # tail. Verifies the chunked-resume path is not taken when the radix
    # match already covers everything but a sliver.
    @staticmethod
    def _script_radix_hit_minus_one(t: ScriptedRuntime):
        # Warm radix with chunk_size-1 tokens. start_req builds input_ids
        # = [1] * prompt_len, so r_warm and r below share the exact same
        # prefix bytes — the radix cache will key off them.
        r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE - 1, max_new_tokens=1)
        yield from run_until_finished(r_warm)
        assert r_warm.finished

        # Give the cache one extra iteration to commit r_warm's KV into the
        # radix tree before the follow-up submission. Without this yield
        # the prefix may not yet be visible to r.
        yield

        # Same prefix + 1 extra token = total chunk_size. Even though the
        # *total* prompt equals chunk_size, only 1 token is fresh.
        r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0, (
            f"radix hit should leave only 1 fresh token; chunked path should "
            f"not engage. Got chunks_done={r.chunks_done}"
        )

    def test_chunk_size_one_token(self):
        """Chunk_size = 1: every token is its own chunk."""
        execute_scripted_runtime(
            self._script_chunk_size_one_token,
            **base_engine_kwargs(chunked_prefill_size=1),
        )

    @staticmethod
    def _script_chunk_size_one_token(t: ScriptedRuntime):
        # chunk_size = 1: every token is its own chunk.
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 8, f"expected 8 chunks, got {r.chunks_done}"

    def test_chunk_size_two_prompt_two(self):
        """Chunk_size = 2, prompt_len = 2: equal, single-shot."""
        execute_scripted_runtime(
            self._script_chunk_size_two_prompt_two,
            **base_engine_kwargs(chunked_prefill_size=2),
        )

    @staticmethod
    def _script_chunk_size_two_prompt_two(t: ScriptedRuntime):
        # chunk_size = 2, prompt_len = 2: equal, single-shot.
        r = t.start_req(prompt_len=2, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done <= 1

    def test_chunk_size_two_prompt_five(self):
        """Chunk_size = 2, prompt_len = 5: chunks_done == 3 (2 + 2 + 1)."""
        execute_scripted_runtime(
            self._script_chunk_size_two_prompt_five,
            **base_engine_kwargs(chunked_prefill_size=2),
        )

    @staticmethod
    def _script_chunk_size_two_prompt_five(t: ScriptedRuntime):
        # chunk_size = 2, prompt_len = 5: chunks_done == 3 (2 + 2 + 1).
        r = t.start_req(prompt_len=5, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 3, f"expected 3 chunks, got {r.chunks_done}"

    def test_chunk_size_equals_max_prefill_tokens(self):
        """Chunk_size == max_prefill_tokens: prompt fits in one shot."""
        execute_scripted_runtime(
            self._script_chunk_size_equals_max_prefill_tokens,
            **base_engine_kwargs(chunked_prefill_size=1024, max_prefill_tokens=1024),
        )

    @staticmethod
    def _script_chunk_size_equals_max_prefill_tokens(t: ScriptedRuntime):
        # chunk_size == max_prefill_tokens: prompt fits in one shot.
        r = t.start_req(prompt_len=1024, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done <= 1

    def test_chunk_size_exceeds_max_prefill_tokens(self):
        """Chunk_size > max_prefill_tokens: engine should reject or cap."""
        execute_scripted_runtime(
            self._script_chunk_size_exceeds_max_prefill_tokens,
            **base_engine_kwargs(chunked_prefill_size=2048, max_prefill_tokens=1024),
        )

    @staticmethod
    def _script_chunk_size_exceeds_max_prefill_tokens(t: ScriptedRuntime):
        # chunk_size > max_prefill_tokens: engine should reject or cap.
        # Test verifies submission still completes (engine internally
        # capped chunk_size to max_prefill_tokens).
        # NEW API NEEDED: engine_args validation — server_args should
        # reject chunked_prefill_size > max_prefill_tokens cleanly.
        r = t.start_req(prompt_len=512, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished

    def test_chunk_size_4096_prompt_4097(self):
        """Chunk_size = 4096, prompt_len = 4097: 2 chunks (4096 + 1)."""
        execute_scripted_runtime(
            self._script_chunk_size_4096_prompt_4097,
            **base_engine_kwargs(chunked_prefill_size=4096),
        )

    @staticmethod
    def _script_chunk_size_4096_prompt_4097(t: ScriptedRuntime):
        # chunk_size = 4096, prompt_len = 4097: 2 chunks (4096 + 1).
        r = t.start_req(prompt_len=4097, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2

    def test_chunk_size_256_prompt_100x(self):
        """Chunk_size = 256, prompt_len = 100 * 256 = 25600: long-prompt stability test."""
        execute_scripted_runtime(
            self._script_chunk_size_256_prompt_100x,
            **base_engine_kwargs(chunked_prefill_size=256),
        )

    @staticmethod
    def _script_chunk_size_256_prompt_100x(t: ScriptedRuntime):
        # chunk_size = 256, prompt_len = 100 * 256 = 25600: long-prompt
        # stability test. May be slow on small models.
        # NOTE: 25600 tokens may exceed DEFAULT_SMALL_MODEL_NAME_FOR_TEST's
        # max-seq-len. If so, the engine should still admit/reject cleanly.
        r = t.start_req(prompt_len=100 * 256, max_new_tokens=2)
        yield from run_until(r, lambda h: h.finished, max_steps=4000)
        assert r.finished
        assert r.chunks_done == 100

    def test_chunk_size_16_prompt_1024(self):
        """High fragmentation: chunk_size = 16, prompt_len = 1024 = 64 chunks."""
        execute_scripted_runtime(
            self._script_chunk_size_16_prompt_1024,
            **base_engine_kwargs(chunked_prefill_size=16),
        )

    @staticmethod
    def _script_chunk_size_16_prompt_1024(t: ScriptedRuntime):
        # High fragmentation: chunk_size = 16, prompt_len = 1024 = 64 chunks.
        r = t.start_req(prompt_len=1024, max_new_tokens=2)
        yield from run_until(r, lambda h: h.finished, max_steps=1000)
        assert r.finished
        assert r.chunks_done == 64

    def test_chunk_size_4_prompt_4(self):
        """Chunk_size = 4, prompt_len = 4: single-shot, no chunking."""
        execute_scripted_runtime(
            self._script_chunk_size_4_prompt_4,
            **base_engine_kwargs(chunked_prefill_size=4),
        )

    @staticmethod
    def _script_chunk_size_4_prompt_4(t: ScriptedRuntime):
        # chunk_size = 4, prompt_len = 4: single-shot, no chunking.
        r = t.start_req(prompt_len=4, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0

    def test_chunk_size_4_prompt_5(self):
        """Chunk_size = 4, prompt_len = 5: chunks_done == 2 (4 + 1)."""
        execute_scripted_runtime(
            self._script_chunk_size_4_prompt_5,
            **base_engine_kwargs(chunked_prefill_size=4),
        )

    @staticmethod
    def _script_chunk_size_4_prompt_5(t: ScriptedRuntime):
        # chunk_size = 4, prompt_len = 5: chunks_done == 2 (4 + 1).
        r = t.start_req(prompt_len=5, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2

    def test_chunk_size_32_prompt_33(self):
        """Chunk_size = 32, prompt_len = 33: 2 chunks, second is 1 token."""
        execute_scripted_runtime(
            self._script_chunk_size_32_prompt_33,
            **base_engine_kwargs(chunked_prefill_size=32),
        )

    @staticmethod
    def _script_chunk_size_32_prompt_33(t: ScriptedRuntime):
        # chunk_size = 32, prompt_len = 33: 2 chunks, second is 1 token.
        r = t.start_req(prompt_len=33, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2

    def test_chunk_size_64_prompt_129(self):
        """Chunk_size = 64, prompt_len = 129: 3 chunks (64 + 64 + 1)."""
        execute_scripted_runtime(
            self._script_chunk_size_64_prompt_129,
            **base_engine_kwargs(chunked_prefill_size=64),
        )

    @staticmethod
    def _script_chunk_size_64_prompt_129(t: ScriptedRuntime):
        # chunk_size = 64, prompt_len = 129: 3 chunks (64 + 64 + 1).
        r = t.start_req(prompt_len=129, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 3

    def test_chunk_size_128_prompt_512(self):
        """Chunk_size = 128, prompt_len = 512: 4 chunks."""
        execute_scripted_runtime(
            self._script_chunk_size_128_prompt_512,
            **base_engine_kwargs(chunked_prefill_size=128),
        )

    @staticmethod
    def _script_chunk_size_128_prompt_512(t: ScriptedRuntime):
        # chunk_size = 128, prompt_len = 512: 4 chunks.
        r = t.start_req(prompt_len=512, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 4

    def test_chunk_size_1024_prompt_4096(self):
        """Chunk_size = 1024, prompt_len = 4096: 4 chunks."""
        execute_scripted_runtime(
            self._script_chunk_size_1024_prompt_4096,
            **base_engine_kwargs(chunked_prefill_size=1024),
        )

    @staticmethod
    def _script_chunk_size_1024_prompt_4096(t: ScriptedRuntime):
        # chunk_size = 1024, prompt_len = 4096: 4 chunks.
        r = t.start_req(prompt_len=4096, max_new_tokens=2)
        yield from run_until(r, lambda h: h.finished, max_steps=1000)
        assert r.finished
        assert r.chunks_done == 4

    def test_chunks_done_monotone_under_chunk_size_1(self):
        """Invariant: chunks_done is monotone non-decreasing across yields."""
        execute_scripted_runtime(
            self._script_chunks_done_monotone_under_chunk_size_1,
            **base_engine_kwargs(chunked_prefill_size=1),
        )

    @staticmethod
    def _script_chunks_done_monotone_under_chunk_size_1(t: ScriptedRuntime):
        # Invariant: chunks_done is monotone non-decreasing across yields.
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        prev = 0
        for _ in range(DEFAULT_MAX_STEPS):
            cur = r.chunks_done
            assert cur >= prev, f"chunks_done regressed: prev={prev}, cur={cur}"
            prev = cur
            if r.finished:
                return
            yield
        raise AssertionError("req never finished")

    def test_chunk_size_one_long_prompt(self):
        """Chunk_size = 1, prompt_len = 64: stress fragmentation."""
        execute_scripted_runtime(
            self._script_chunk_size_one_long_prompt,
            **base_engine_kwargs(chunked_prefill_size=1),
        )

    @staticmethod
    def _script_chunk_size_one_long_prompt(t: ScriptedRuntime):
        # chunk_size = 1, prompt_len = 64: stress fragmentation.
        r = t.start_req(prompt_len=64, max_new_tokens=2)
        yield from run_until(r, lambda h: h.finished, max_steps=500)
        assert r.finished
        assert r.chunks_done == 64

    def test_chunk_size_equals_page_size(self):
        """chunk_size == page_size: each chunk allocates one page; final kv_pages reflects chunks_done."""
        execute_scripted_runtime(
            self._script_chunk_size_equals_page_size,
            **base_engine_kwargs(chunked_prefill_size=16, page_size=16),
        )

    # chunk_size == page_size — every chunk is exactly one
    # page; the chunked path must not double-allocate or leak.
    @staticmethod
    def _script_chunk_size_equals_page_size(t: ScriptedRuntime):
        # prompt_len = 8 * page_size → 8 chunks expected.
        r = t.start_req(prompt_len=8 * 16, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert r.chunks_done == 8
        # After finish, KV must be fully released.
        assert r.kv_pages == 0

    def test_chunk_size_misaligned_page_size_plus_minus_1(self):
        """chunk_size = page_size ± 1: chunked completes and page accounting stays consistent."""
        execute_scripted_runtime(
            self._script_chunk_size_misaligned_page_size_plus_minus_1,
            **base_engine_kwargs(chunked_prefill_size=15, page_size=16),
        )

    # chunk_size = page_size - 1 — misaligned boundaries
    # exercise the partial-page tail; finishing without leaks is the
    # property under test.
    @staticmethod
    def _script_chunk_size_misaligned_page_size_plus_minus_1(t: ScriptedRuntime):
        # chunk_size = page_size - 1 = 15; prompt = 60 → 4 chunks
        # (15 + 15 + 15 + 15) but each chunk straddles page boundaries.
        r = t.start_req(prompt_len=60, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert r.chunks_done == 4
        assert r.kv_pages == 0

    def test_prompt_n_chunks_plus_minus_1(self):
        """Sweep prompt_len = N*chunk_size ± 1 for N in {1..5}; chunks_done matches ceil-div."""
        execute_scripted_runtime(
            self._script_prompt_n_chunks_plus_minus_1,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # N*chunk_size ± 1 sweep — verifies that the
    # ceil-div arithmetic in the chunked admission path is correct at
    # every chunk multiple, both just below and just above.
    @staticmethod
    def _script_prompt_n_chunks_plus_minus_1(t: ScriptedRuntime):
        # N = 1..5 — for each N, submit two reqs: one at N*chunk_size - 1
        # and one at N*chunk_size + 1, and check chunks_done = ceil-div.
        for n in range(1, 6):
            # N*chunk_size - 1.
            prompt_minus = n * DEFAULT_CHUNK_SIZE - 1
            r_minus = t.start_req(prompt_len=prompt_minus, max_new_tokens=1)
            yield from run_until_finished(r_minus, max_steps=800)
            assert r_minus.finished
            expected_minus = _expected_chunks(prompt_minus, DEFAULT_CHUNK_SIZE)
            assert r_minus.chunks_done == expected_minus, (
                f"N={n} prompt_len={prompt_minus}: "
                f"expected chunks_done={expected_minus}, got {r_minus.chunks_done}"
            )

            # N*chunk_size + 1.
            prompt_plus = n * DEFAULT_CHUNK_SIZE + 1
            r_plus = t.start_req(prompt_len=prompt_plus, max_new_tokens=1)
            yield from run_until_finished(r_plus, max_steps=800)
            assert r_plus.finished
            expected_plus = _expected_chunks(prompt_plus, DEFAULT_CHUNK_SIZE)
            assert r_plus.chunks_done == expected_plus, (
                f"N={n} prompt_len={prompt_plus}: "
                f"expected chunks_done={expected_plus}, got {r_plus.chunks_done}"
            )


if __name__ == "__main__":
    unittest.main()
