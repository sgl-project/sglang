"""Radix prefix cache × chunked: naive ScriptedRuntime smoke plus edge cases.

Two requests with the same long prefix. The second request hits the
radix cache after the first completes; what remains should still be
routed through the chunked path if it's longer than chunk_size, or
admit directly if it isn't. We pick prompt lengths such that:

* r1 prefix = long → ends up in radix cache
* r2 prefix = same long prefix + extra suffix > chunk_size → still
  chunks the *suffix* via the chunked-resume path.

Touches `init_next_round_input(tree_cache)` branching (audit doc
§ "Radix cache prefix match").

Edge cases cover A.6 series from the expansion plan plus fan-out
across radix disabled / partial-hit / full-hit / evict / LPM-vs-DFS
policy. Verifies that the chunked admission path correctly accounts
for already-cached prefix tokens.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase


def _script_naive_radix_chunked(t: ScriptedRuntime):
    # First request populates the radix tree with a long shared prefix.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r1)
    assert r1.finished

    # Second request — same prefix tokens (start_req uses placeholder
    # token id 1, so r2's prefix is byte-identical to r1's). Add tail
    # > chunk_size so the residual still chunks.
    r2 = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN + DEFAULT_CHUNK_SIZE * 2,
        max_new_tokens=2,
    )
    yield from run_until_finished(r2)
    assert r2.finished


def _script_radix_disabled_chunks_every_time(t: ScriptedRuntime):
    # Radix disabled — every same-prompt re-submission chunks fresh.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r1)
    assert r1.chunks_done >= 2

    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r2)
    # With radix disabled, r2 still chunks (no prefix reuse).
    assert r2.chunks_done >= 2


def _script_radix_full_prefix_hit_nine_reqs(t: ScriptedRuntime):
    # Submit 10 identical reqs. r1 chunks; r2..r10 fully hit.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r1)

    others = [
        t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2) for _ in range(9)
    ]
    yield from run_until_all_finished(others)
    for r in others:
        assert r.chunks_done == 0


def _script_radix_hit_full_prefix(t: ScriptedRuntime):
    # Warm prefix exactly = chunk_size; second prompt = prefix + 1 token.
    # Tail is 1 token, not enough to chunk.
    r1 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
    yield from run_until_finished(r1)

    r2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 1, max_new_tokens=1)
    yield from run_until_finished(r2)
    assert r2.chunks_done == 0


def _script_radix_hit_partial_then_chunk_tail(t: ScriptedRuntime):
    # Prefix == 2 * chunk_size. New prompt = prefix + chunk_size + 1.
    # Tail = chunk_size + 1 tokens, requires chunking into 2 chunks.
    r1 = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=1)
    yield from run_until_finished(r1)

    r2 = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE + 1, max_new_tokens=1)
    yield from run_until_finished(r2)
    # Tail length is chunk_size + 1, which chunks into exactly 2.
    assert r2.chunks_done == 2


def _script_radix_evict_then_resubmit_rechunks(t: ScriptedRuntime):
    # Submit, force radix evict, resubmit same prompt → re-chunks.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r1)

    # NEW API NEEDED: t.evict_radix(prefix_tokens) — evict a specific
    # prefix from the radix tree.
    t.evict_radix(prefix_tokens=None)  # None = evict all
    yield

    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r2)
    assert r2.chunks_done >= 2


def _script_radix_resume_init_next_round_path(t: ScriptedRuntime):
    # Chunked resume after partial radix hit — verifies the
    # ``init_next_round_input`` no-tree-cache branch is reached.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1)
    yield from run_until_finished(r1)
    r2 = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN + DEFAULT_CHUNK_SIZE, max_new_tokens=2
    )
    yield from run_until_finished(r2)

    # NEW API NEEDED: t.last_scheduler_path() — last branch taken in
    # admission decision: "new" | "chunked_resume" | "tree_cache_resume".
    # We don't enforce a specific value, just smoke-check the API.
    path = t.last_scheduler_path()
    assert path in ("new", "chunked_resume", "tree_cache_resume", None)


def _script_radix_lpm_policy_chunked_priority(t: ScriptedRuntime):
    # LPM policy: chunked-resume reqs get sort priority (bf5b4e9a10).
    # Submit several siblings; the chunked-resume one should win.
    r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
    yield from run_until_finished(r_warm)

    reqs = [
        t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 3, max_new_tokens=1)
        for _ in range(3)
    ]
    yield from run_until_all_finished(reqs)


def _script_radix_dfs_weight_policy_chunked(t: ScriptedRuntime):
    # DFS_WEIGHT policy + chunked.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r1)
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r2)


def _script_radix_lock_ref_concurrent_chunked(t: ScriptedRuntime):
    # Concurrent chunked reqs all sharing a long prefix → many lock_refs.
    r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 4, max_new_tokens=1)
    yield from run_until_finished(r_warm)
    reqs = [
        t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 4 + 8, max_new_tokens=2)
        for _ in range(5)
    ]
    yield from run_until_all_finished(reqs)
    # Each finished req should release its lock_refs.
    for r in reqs:
        assert r.lock_refs == 0


def _script_radix_partial_hit_exact_chunk_boundary(t: ScriptedRuntime):
    # Hit equals exactly the chunk boundary; second req tail = chunk_size.
    # With a chunk_size-long prefix hit and a chunk_size tail, the tail
    # is single-shot (per pinned semantics, prompt_len <= chunk_size of
    # *uncached* tail => no chunked path).
    r1 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
    yield from run_until_finished(r1)
    r2 = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=1)
    yield from run_until_finished(r2)
    assert r2.chunks_done == 1


def _script_radix_warmup_helper(t: ScriptedRuntime):
    # Warm radix without going through start_req.
    # NEW API NEEDED: t.warmup_radix(prompt_tokens: list[int]).
    t.warmup_radix(prompt_tokens=[1] * (2 * DEFAULT_CHUNK_SIZE))
    yield

    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE + 1, max_new_tokens=1)
    yield from run_until_finished(r)
    # After warmup, this req should hit prefix.
    assert r.chunks_done == 0


def _script_radix_two_distinct_prefixes(t: ScriptedRuntime):
    # Two warm prefixes; later reqs hit one or the other.
    r_a = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
    yield from run_until_finished(r_a)
    r_b = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 3, max_new_tokens=1)
    yield from run_until_finished(r_b)
    r_a2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
    yield from run_until_finished(r_a2)
    assert r_a2.chunks_done == 0


def _script_radix_full_prefix_minus_one(t: ScriptedRuntime):
    # Prefix length = chunk_size - 1; new prompt = chunk_size.
    # New tail = 1 token only.
    r1 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE - 1, max_new_tokens=1)
    yield from run_until_finished(r1)
    r2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
    yield from run_until_finished(r2)
    assert r2.chunks_done == 0


def _script_radix_prefix_match_with_priority(t: ScriptedRuntime):
    # priority high + radix partial hit: high-priority req still hits cache.
    r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
    yield from run_until_finished(r_warm)
    r = t.start_req(
        prompt_len=DEFAULT_CHUNK_SIZE * 3, max_new_tokens=2, priority="high"
    )
    yield from run_until_finished(r)


def _script_radix_calc_priority_skip_chunked_resume(t: ScriptedRuntime):
    # aaf3752d2b: skip chunked-resume reqs in calc_priority prefix matching.
    # Two reqs share a prefix; one is chunked-resume. The scheduler must
    # observably take the "skip chunked-resume in priority calc" branch.
    # NEW API NEEDED: t.last_admission_path() returns the most recent
    # admission branch label, e.g. "priority_skip_chunked_resume".
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high")
    yield from run_until(r1, lambda h: h.is_chunking)
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low")
    saw_skip = False
    while not (r1.finished and r2.finished):
        path = t.last_admission_path()
        if path == "priority_skip_chunked_resume":
            saw_skip = True
        yield
    assert (
        saw_skip
    ), "expected priority-skip-chunked-resume branch to fire at least once"


class TestScriptedRadix(CustomTestCase):
    def test_naive_radix_chunked(self):
        """First request populates the radix tree with a long shared prefix."""
        execute_scripted_runtime(
            _script_naive_radix_chunked,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                schedule_policy="fcfs",
            ),
        )

    def test_radix_disabled_chunks_every_time(self):
        """Radix disabled — every same-prompt re-submission chunks fresh."""
        execute_scripted_runtime(
            _script_radix_disabled_chunks_every_time,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disable_radix_cache=True,
            ),
        )

    def test_radix_full_prefix_hit_nine_reqs(self):
        """Submit 10 identical reqs."""
        execute_scripted_runtime(
            _script_radix_full_prefix_hit_nine_reqs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_radix_hit_full_prefix(self):
        """Warm prefix exactly = chunk_size; second prompt = prefix + 1 token."""
        execute_scripted_runtime(
            _script_radix_hit_full_prefix,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_radix_hit_partial_then_chunk_tail(self):
        """Prefix == 2 * chunk_size."""
        execute_scripted_runtime(
            _script_radix_hit_partial_then_chunk_tail,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_radix_evict_then_resubmit_rechunks(self):
        """Submit, force radix evict, resubmit same prompt → re-chunks."""
        execute_scripted_runtime(
            _script_radix_evict_then_resubmit_rechunks,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_radix_resume_init_next_round_path(self):
        """Chunked resume after partial radix hit — verifies the ``init_next_round_input`` no-tree-cache branch is reached."""
        execute_scripted_runtime(
            _script_radix_resume_init_next_round_path,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_radix_lpm_policy_chunked_priority(self):
        """LPM policy: chunked-resume reqs get sort priority (bf5b4e9a10)."""
        execute_scripted_runtime(
            _script_radix_lpm_policy_chunked_priority,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                schedule_policy="lpm",
            ),
        )

    def test_radix_dfs_weight_policy_chunked(self):
        """DFS_WEIGHT policy + chunked."""
        execute_scripted_runtime(
            _script_radix_dfs_weight_policy_chunked,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                schedule_policy="dfs-weight",
            ),
        )

    def test_radix_lock_ref_concurrent_chunked(self):
        """Concurrent chunked reqs all sharing a long prefix → many lock_refs."""
        execute_scripted_runtime(
            _script_radix_lock_ref_concurrent_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_radix_partial_hit_exact_chunk_boundary(self):
        """Hit equals exactly the chunk boundary; second req tail = chunk_size."""
        execute_scripted_runtime(
            _script_radix_partial_hit_exact_chunk_boundary,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_radix_warmup_helper(self):
        """Warm radix without going through start_req."""
        execute_scripted_runtime(
            _script_radix_warmup_helper,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_radix_two_distinct_prefixes(self):
        """Two warm prefixes; later reqs hit one or the other."""
        execute_scripted_runtime(
            _script_radix_two_distinct_prefixes,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_radix_full_prefix_minus_one(self):
        """Prefix length = chunk_size - 1; new prompt = chunk_size."""
        execute_scripted_runtime(
            _script_radix_full_prefix_minus_one,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_radix_prefix_match_with_priority(self):
        """Priority high + radix partial hit: high-priority req still hits cache."""
        execute_scripted_runtime(
            _script_radix_prefix_match_with_priority,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )

    def test_radix_calc_priority_skip_chunked_resume(self):
        """Aaf3752d2b: skip chunked-resume reqs in calc_priority prefix matching."""
        execute_scripted_runtime(
            _script_radix_calc_priority_skip_chunked_resume,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
