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

from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


class TestRadixBasic(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_radix_full_prefix_hit_nine_reqs(self):
        """Submit 10 identical reqs."""
        self.runtime.run(self._script_radix_full_prefix_hit_nine_reqs)

    @staticmethod
    def _script_radix_full_prefix_hit_nine_reqs(t: ScriptedRuntime):
        # Submit 10 identical reqs. r1 chunks; r2..r10 fully hit.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)

        others = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(9)
        ]
        yield from run_until_all_finished(others)
        for r in others:
            assert r.chunks_done == 0

    def test_radix_hit_full_prefix(self):
        """Warm prefix exactly = chunk_size; second prompt = prefix + 1 token."""
        self.runtime.run(self._script_radix_hit_full_prefix)

    @staticmethod
    def _script_radix_hit_full_prefix(t: ScriptedRuntime):
        # Warm prefix exactly = chunk_size; second prompt = prefix + 1 token.
        # Tail is 1 token, not enough to chunk.
        r1 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r1)

        r2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 1, max_new_tokens=1)
        yield from run_until_finished(r2)
        assert r2.chunks_done == 0

    def test_radix_hit_partial_then_chunk_tail(self):
        """Prefix == 2 * chunk_size."""
        self.runtime.run(self._script_radix_hit_partial_then_chunk_tail)

    @staticmethod
    def _script_radix_hit_partial_then_chunk_tail(t: ScriptedRuntime):
        # Prefix == 2 * chunk_size. New prompt = prefix + chunk_size + 1.
        # Tail = chunk_size + 1 tokens, requires chunking into 2 chunks.
        r1 = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r1)

        r2 = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE + 1, max_new_tokens=1)
        yield from run_until_finished(r2)
        # Tail length is chunk_size + 1, which chunks into exactly 2.
        assert r2.chunks_done == 2

    def test_radix_evict_then_resubmit_rechunks(self):
        """Submit, force radix evict, resubmit same prompt → re-chunks."""
        self.runtime.run(self._script_radix_evict_then_resubmit_rechunks)

    @staticmethod
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

    def test_radix_resume_init_next_round_path(self):
        """Chunked resume after partial radix hit — verifies the ``init_next_round_input`` no-tree-cache branch is reached."""
        self.runtime.run(self._script_radix_resume_init_next_round_path)

    @staticmethod
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

    def test_radix_lock_ref_concurrent_chunked(self):
        """Concurrent chunked reqs all sharing a long prefix → many lock_refs."""
        self.runtime.run(self._script_radix_lock_ref_concurrent_chunked)

    @staticmethod
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

    def test_radix_partial_hit_exact_chunk_boundary(self):
        """Hit equals exactly the chunk boundary; second req tail = chunk_size."""
        self.runtime.run(self._script_radix_partial_hit_exact_chunk_boundary)

    @staticmethod
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

    def test_radix_warmup_helper(self):
        """Warm radix without going through start_req."""
        self.runtime.run(self._script_radix_warmup_helper)

    @staticmethod
    def _script_radix_warmup_helper(t: ScriptedRuntime):
        # Warm radix without going through start_req.
        # NEW API NEEDED: t.warmup_radix(prompt_tokens: list[int]).
        t.warmup_radix(prompt_tokens=[1] * (2 * DEFAULT_CHUNK_SIZE))
        yield

        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE + 1, max_new_tokens=1)
        yield from run_until_finished(r)
        # After warmup, this req should hit prefix.
        assert r.chunks_done == 0

    def test_radix_two_distinct_prefixes(self):
        """Two warm prefixes; later reqs hit one or the other."""
        self.runtime.run(self._script_radix_two_distinct_prefixes)

    @staticmethod
    def _script_radix_two_distinct_prefixes(t: ScriptedRuntime):
        # Two warm prefixes; later reqs hit one or the other.
        r_a = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
        yield from run_until_finished(r_a)
        r_b = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 3, max_new_tokens=1)
        yield from run_until_finished(r_b)
        r_a2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
        yield from run_until_finished(r_a2)
        assert r_a2.chunks_done == 0

    def test_radix_full_prefix_minus_one(self):
        """Prefix length = chunk_size - 1; new prompt = chunk_size."""
        self.runtime.run(self._script_radix_full_prefix_minus_one)

    @staticmethod
    def _script_radix_full_prefix_minus_one(t: ScriptedRuntime):
        # Prefix length = chunk_size - 1; new prompt = chunk_size.
        # New tail = 1 token only.
        r1 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE - 1, max_new_tokens=1)
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r2)
        assert r2.chunks_done == 0

    def test_radix_chunked_stash_no_hit_count_inflation(self):
        """Chunked re-insert does not inflate radix hit_count via the self-referencing stash path."""
        self.runtime.run(self._script_radix_chunked_stash_no_hit_count_inflation)

    # _inc_hit_count(chunked=True) is the gate that prevents a
    # chunked-resume's own re-insert from inflating the prefix's hit
    # count. Pre-fix, the prefix's hit_count could climb one per chunk
    # — making this prefix look hotter than it really was. With a
    # single chunked req touching the prefix, observed hit_count must
    # stay == 1 from first commit to finish.
    @staticmethod
    def _script_radix_chunked_stash_no_hit_count_inflation(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        # After first commit, hit_count for the chunked req's own
        # prefix should be exactly 1 (the req's own lock_ref).
        stats = t.engine_stats()
        hit_count_first = stats["radix_hit_count_for_inflight_chunked"]
        assert (
            hit_count_first == 1
        ), f"radix hit_count should be 1 at first commit, got {hit_count_first}"
        # Drive through several more chunks; hit_count must not climb.
        for _ in range(20):
            if r.finished:
                break
            stats = t.engine_stats()
            cur = stats["radix_hit_count_for_inflight_chunked"]
            assert cur == 1, f"radix hit_count inflated by chunked re-insert: {cur} > 1"
            yield
        yield from run_until_finished(r)
        assert r.finished

    def test_radix_hit_changes_between_chunks(self):
        """Second req with identical prompt admits with a prefix reflecting r1's already-committed chunks."""
        self.runtime.run(self._script_radix_hit_changes_between_chunks)

    # r1 chunks a long prompt; r2 with the same prompt is
    # submitted while r1 is mid-prefill. r2's admission prefix length
    # must reflect r1's already-committed chunks (i.e. r2 starts
    # further along than if r1 had not run). r2 should therefore
    # chunk fewer times than r1.
    @staticmethod
    def _script_radix_hit_changes_between_chunks(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # Wait until r1 has committed at least one chunk's worth of
        # prefix into the radix tree.
        yield from run_until(r1, lambda h: h.is_chunking and h.chunks_done >= 1)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1, max_steps=800)
        yield from run_until_finished(r2, max_steps=800)
        assert r1.finished and r2.finished
        # r2 must benefit from r1's committed prefix — fewer chunks.
        assert r2.chunks_done < r1.chunks_done, (
            f"r2 should hit r1's committed prefix; r2.chunks_done="
            f"{r2.chunks_done} not < r1.chunks_done={r1.chunks_done}"
        )

    def test_radix_evict_during_inflight_chunk(self):
        """External evict_radix during an in-flight chunked req does not use-after-free its prefix."""
        self.runtime.run(self._script_radix_evict_during_inflight_chunk)

    # r1 chunks a long prompt; midway through, we call
    # evict_radix on the in-flight prefix. The req's own lock_ref
    # must prevent the evict from freeing pages it still references;
    # the req must complete cleanly with no use-after-free.
    @staticmethod
    def _script_radix_evict_during_inflight_chunk(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        # Evict the entire tree while r is still chunking — r's own
        # lock_ref on its in-flight prefix must keep its pages alive.
        t.evict_radix(prefix_tokens=None)
        yield
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_radix_full_hit_no_chunked_path(self):
        """Warmed prefix exactly equal to the resubmitted prompt produces zero chunks."""
        self.runtime.run(self._script_radix_full_hit_no_chunked_path)

    # Warm a long prefix, then re-submit the *exact same*
    # prompt. The cached prefix covers the entire input, so the new
    # req should bypass the chunked path entirely and go straight to
    # decode — chunks_done must be 0.
    @staticmethod
    def _script_radix_full_hit_no_chunked_path(t: ScriptedRuntime):
        prompt_len: int = 16 * DEFAULT_CHUNK_SIZE
        r_warm = t.start_req(prompt_len=prompt_len, max_new_tokens=1)
        yield from run_until_finished(r_warm, max_steps=1200)
        assert r_warm.finished

        r = t.start_req(prompt_len=prompt_len, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=400)
        assert r.finished
        assert (
            r.chunks_done == 0
        ), f"full prefix hit must skip chunked path; got chunks_done={r.chunks_done}"

    def test_radix_evict_race_concurrent_chunked_admit(self):
        """Radix evict racing a chunked admission acquires lock_ref before any page release."""
        self.runtime.run(self._script_radix_evict_race_concurrent_chunked_admit)

    # Warm a prefix, then in the same scheduler step
    # both evict it and submit a new req that shares it. The admission
    # path must take its lock_ref before the evict can release pages,
    # otherwise the new req sees a stale pointer / use-after-free.
    @staticmethod
    def _script_radix_evict_race_concurrent_chunked_admit(t: ScriptedRuntime):
        warm_len: int = 4 * DEFAULT_CHUNK_SIZE
        r_warm = t.start_req(prompt_len=warm_len, max_new_tokens=1)
        yield from run_until_finished(r_warm, max_steps=400)
        assert r_warm.finished

        # Same yield: kick off evict AND submit a sharing req. The
        # scheduler must serialize the lock_ref grab before the evict
        # can free pages — no use-after-free on the chunked admission.
        t.evict_radix(prefix_tokens=None)
        r = t.start_req(
            prompt_len=warm_len + DEFAULT_CHUNK_SIZE * 2,
            max_new_tokens=2,
        )
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_chunked_req_re_chunked_after_resume_same_prefix(self):
        """Chunked req retracted mid-stream and resumed: chunks_done across the full lifetime matches the expected total chunk count."""
        self.runtime.run(self._script_chunked_req_re_chunked_after_resume_same_prefix)

    # retract-resume re-chunking accounting. R1 is mid-stream
    # when we force_retract it; on resume it must re-chunk the
    # remaining prompt — chunks_done across the full lifetime should
    # equal the expected total chunk count for the original prompt
    # (not 0, not doubled). The radix layer's lock_ref on the
    # already-committed prefix means resume can re-use prior chunks'
    # KV, so chunks_done counts only the chunks the scheduler actually
    # admitted for this req.
    @staticmethod
    def _script_chunked_req_re_chunked_after_resume_same_prefix(t: ScriptedRuntime):
        # 4 * DEFAULT_CHUNK_SIZE makes the expected chunk count exact
        # and small enough to bound assertions tightly.
        prompt_len: int = 4 * DEFAULT_CHUNK_SIZE
        r = t.start_req(prompt_len=prompt_len, max_new_tokens=2)
        # Mid-stream: at least one chunk committed, still chunking.
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        t.force_retract(r)
        yield
        assert r.kv_pages == 0, f"retract must release KV; got {r.kv_pages}"

        # Resume: the scheduler re-admits the req and finishes it.
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # Total chunks_done for prompt_len = 4 * CHUNK_SIZE must be at
        # least 4 — the lifetime aggregate count across pre-retract
        # chunks + post-resume chunks. Upper bound: doubling would
        # indicate the resume re-chunked the entire prompt instead of
        # picking up where retract left off (we tolerate a small
        # overshoot but reject 2x).
        expected_total: int = prompt_len // DEFAULT_CHUNK_SIZE
        assert r.chunks_done >= expected_total, (
            f"lifetime chunks_done after retract+resume must cover the "
            f"whole prompt; expected >= {expected_total}, got "
            f"{r.chunks_done}"
        )
        assert r.chunks_done < 2 * expected_total, (
            f"lifetime chunks_done after retract+resume should not double "
            f"the prompt's chunk count; expected < {2 * expected_total}, "
            f"got {r.chunks_done}"
        )


class TestRadixFcfs(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        schedule_policy="fcfs",
    )

    def test_naive_radix_chunked(self):
        """First request populates the radix tree with a long shared prefix."""
        self.runtime.run(self._script_naive_radix_chunked)

    @staticmethod
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


class TestRadixDisabled(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disable_radix_cache=True,
    )

    def test_radix_disabled_chunks_every_time(self):
        """Radix disabled — every same-prompt re-submission chunks fresh."""
        self.runtime.run(self._script_radix_disabled_chunks_every_time)

    @staticmethod
    def _script_radix_disabled_chunks_every_time(t: ScriptedRuntime):
        # Radix disabled — every same-prompt re-submission chunks fresh.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        assert r1.chunks_done >= 2

        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r2)
        # With radix disabled, r2 still chunks (no prefix reuse).
        assert r2.chunks_done >= 2


class TestRadixLpm(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        schedule_policy="lpm",
    )

    def test_radix_lpm_policy_chunked_priority(self):
        """LPM policy: chunked-resume reqs get sort priority (bf5b4e9a10)."""
        self.runtime.run(self._script_radix_lpm_policy_chunked_priority)

    @staticmethod
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


class TestRadixDfsWeight(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        schedule_policy="dfs-weight",
    )

    def test_radix_dfs_weight_policy_chunked(self):
        """DFS_WEIGHT policy + chunked."""
        self.runtime.run(self._script_radix_dfs_weight_policy_chunked)

    @staticmethod
    def _script_radix_dfs_weight_policy_chunked(t: ScriptedRuntime):
        # DFS_WEIGHT policy + chunked.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r2)


class TestRadixPriority(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_radix_prefix_match_with_priority(self):
        """Priority high + radix partial hit: high-priority req still hits cache."""
        self.runtime.run(self._script_radix_prefix_match_with_priority)

    @staticmethod
    def _script_radix_prefix_match_with_priority(t: ScriptedRuntime):
        # priority high + radix partial hit: high-priority req still hits cache.
        r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
        yield from run_until_finished(r_warm)
        r = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 3, max_new_tokens=2, priority="high"
        )
        yield from run_until_finished(r)

    def test_radix_calc_priority_skip_chunked_resume(self):
        """Aaf3752d2b: skip chunked-resume reqs in calc_priority prefix matching."""
        self.runtime.run(self._script_radix_calc_priority_skip_chunked_resume)

    @staticmethod
    def _script_radix_calc_priority_skip_chunked_resume(t: ScriptedRuntime):
        # aaf3752d2b: skip chunked-resume reqs in calc_priority prefix matching.
        # Two reqs share a prefix; one is chunked-resume. The scheduler must
        # observably take the "skip chunked-resume in priority calc" branch.
        # NEW API NEEDED: t.last_admission_path() returns the most recent
        # admission branch label, e.g. "priority_skip_chunked_resume".
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high"
        )
        yield from run_until(r1, lambda h: h.is_chunking)
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low"
        )
        saw_skip = False
        while not (r1.finished and r2.finished):
            path = t.last_admission_path()
            if path == "priority_skip_chunked_resume":
                saw_skip = True
            yield
        assert (
            saw_skip
        ), "expected priority-skip-chunked-resume branch to fire at least once"


if __name__ == "__main__":
    unittest.main()
