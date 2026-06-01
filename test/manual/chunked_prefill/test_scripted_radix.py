import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
    warmup_radix,
)


class TestRadixBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_radix_full_prefix_hit_nine_reqs(self):
        self.server.execute_script(self._script_radix_full_prefix_hit_nine_reqs)

    @staticmethod
    def _script_radix_full_prefix_hit_nine_reqs(t: ScriptedContext):
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
        self.server.execute_script(self._script_radix_hit_full_prefix)

    @staticmethod
    def _script_radix_hit_full_prefix(t: ScriptedContext):
        r1 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r1)

        r2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 1, max_new_tokens=1)
        yield from run_until_finished(r2)
        assert r2.chunks_done == 0

    def test_radix_hit_partial_then_chunk_tail(self):
        self.server.execute_script(self._script_radix_hit_partial_then_chunk_tail)

    @staticmethod
    def _script_radix_hit_partial_then_chunk_tail(t: ScriptedContext):
        r1 = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r1)

        r2 = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE + 1, max_new_tokens=1)
        yield from run_until_finished(r2)
        assert r2.chunks_done == 2

    def test_radix_evict_then_resubmit_rechunks(self):
        self.server.execute_script(self._script_radix_evict_then_resubmit_rechunks)

    @staticmethod
    def _script_radix_evict_then_resubmit_rechunks(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        assert r1.finished
        t.evict_radix(prefix_tokens=None)
        yield

        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r2)
        assert r2.finished
        assert r2.chunks_done >= 2
        assert r2.req.cached_tokens == 0
        assert r2.kv_pages == 0
        assert r2.lock_refs == 0

    def test_radix_resume_init_next_round_path(self):
        self.server.execute_script(self._script_radix_resume_init_next_round_path)

    @staticmethod
    def _script_radix_resume_init_next_round_path(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1)
        yield from run_until_finished(r1)
        assert r1.finished

        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN + DEFAULT_CHUNK_SIZE, max_new_tokens=2
        )
        yield from run_until_finished(r2)
        assert r2.finished
        assert r2.req.cached_tokens > 0, (
            f"r2 must hit r1's prefix to exercise the partial-hit chunked-"
            f"resume branch; got cached_tokens={r2.req.cached_tokens}"
        )
        assert r2.chunks_done >= 1, (
            f"residual tail beyond cached prefix should still chunk; got "
            f"chunks_done={r2.chunks_done}"
        )

    def test_radix_lock_ref_concurrent_chunked(self):
        self.server.execute_script(self._script_radix_lock_ref_concurrent_chunked)

    @staticmethod
    def _script_radix_lock_ref_concurrent_chunked(t: ScriptedContext):
        r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 4, max_new_tokens=1)
        yield from run_until_finished(r_warm)
        reqs = [
            t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 4 + 8, max_new_tokens=2)
            for _ in range(5)
        ]
        yield from run_until_all_finished(reqs)
        for r in reqs:
            assert r.lock_refs == 0

    def test_radix_partial_hit_exact_chunk_boundary(self):
        self.server.execute_script(self._script_radix_partial_hit_exact_chunk_boundary)

    @staticmethod
    def _script_radix_partial_hit_exact_chunk_boundary(t: ScriptedContext):
        r1 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r2)
        assert r2.chunks_done == 1

    def test_radix_warmup_helper(self):
        self.server.execute_script(self._script_radix_warmup_helper)

    @staticmethod
    def _script_radix_warmup_helper(t: ScriptedContext):
        yield from warmup_radix(t, [1] * (2 * DEFAULT_CHUNK_SIZE))

        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE + 1, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.chunks_done == 0

    def test_radix_two_distinct_prefixes(self):
        self.server.execute_script(self._script_radix_two_distinct_prefixes)

    @staticmethod
    def _script_radix_two_distinct_prefixes(t: ScriptedContext):
        r_a = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
        yield from run_until_finished(r_a)
        r_b = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 3, max_new_tokens=1)
        yield from run_until_finished(r_b)
        r_a2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
        yield from run_until_finished(r_a2)
        assert r_a2.chunks_done == 0

    def test_radix_full_prefix_minus_one(self):
        self.server.execute_script(self._script_radix_full_prefix_minus_one)

    @staticmethod
    def _script_radix_full_prefix_minus_one(t: ScriptedContext):
        r1 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE - 1, max_new_tokens=1)
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r2)
        assert r2.chunks_done == 0

    def test_radix_hit_changes_between_chunks(self):
        self.server.execute_script(self._script_radix_hit_changes_between_chunks)

    @staticmethod
    def _script_radix_hit_changes_between_chunks(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking and h.chunks_done >= 1)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1, max_steps=800)
        yield from run_until_finished(r2, max_steps=800)
        assert r1.finished and r2.finished
        assert r2.chunks_done < r1.chunks_done, (
            f"r2 should hit r1's committed prefix; r2.chunks_done="
            f"{r2.chunks_done} not < r1.chunks_done={r1.chunks_done}"
        )
        assert r2.req.cached_tokens > 0

    def test_radix_evict_during_inflight_chunk(self):
        self.server.execute_script(self._script_radix_evict_during_inflight_chunk)

    @staticmethod
    def _script_radix_evict_during_inflight_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.evict_radix(prefix_tokens=None)
        yield
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_radix_full_hit_no_chunked_path(self):
        self.server.execute_script(self._script_radix_full_hit_no_chunked_path)

    @staticmethod
    def _script_radix_full_hit_no_chunked_path(t: ScriptedContext):
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
        self.server.execute_script(
            self._script_radix_evict_race_concurrent_chunked_admit
        )

    @staticmethod
    def _script_radix_evict_race_concurrent_chunked_admit(t: ScriptedContext):
        warm_len: int = 4 * DEFAULT_CHUNK_SIZE
        r_warm = t.start_req(prompt_len=warm_len, max_new_tokens=1)
        yield from run_until_finished(r_warm, max_steps=400)
        assert r_warm.finished

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
        self.server.execute_script(
            self._script_chunked_req_re_chunked_after_resume_same_prefix
        )

    @staticmethod
    def _script_chunked_req_re_chunked_after_resume_same_prefix(t: ScriptedContext):
        prompt_len: int = 4 * DEFAULT_CHUNK_SIZE
        r = t.start_req(prompt_len=prompt_len, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        t.pause_generation(mode="retract")
        yield
        assert r.kv_pages == 0, f"retract must release KV; got {r.kv_pages}"

        t.continue_generation()
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
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


class TestRadixHitCountInvariant(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_chunked_stash_no_hit_count_inflation_invariant(self):
        self.server.execute_script(
            self._script_chunked_stash_no_hit_count_inflation_invariant
        )

    @staticmethod
    def _script_chunked_stash_no_hit_count_inflation_invariant(t: ScriptedContext):
        def _snapshot_hit_counts(root) -> dict:
            snapshot: dict = {}
            stack = [root]
            while stack:
                node = stack.pop()
                snapshot[node.id] = node.hit_count
                stack.extend(node.children.values())
            return snapshot

        s = t._scheduler
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        baseline = _snapshot_hit_counts(s.tree_cache.root_node)

        prev_chunks: int = r.chunks_done
        observed_chunk_admissions: int = 0
        for _ in range(800):
            cur_chunks: int = r.chunks_done
            if cur_chunks > prev_chunks:
                observed_chunk_admissions += 1
                cur = _snapshot_hit_counts(s.tree_cache.root_node)
                for node_id, base_count in baseline.items():
                    if node_id in cur:
                        assert cur[node_id] == base_count, (
                            f"_inc_hit_count(chunked=True) inflated existing "
                            f"node id={node_id} hit_count: baseline={base_count}, "
                            f"now={cur[node_id]}"
                        )
                prev_chunks = cur_chunks
            if r.finished:
                break
            yield
        assert r.finished
        assert observed_chunk_admissions >= 2, (
            f"test must exercise at least 2 chunk admissions after baseline; "
            f"observed {observed_chunk_admissions}"
        )


class TestRadixPartialPage(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        page_size=4,
    )

    def test_partial_page_tail_no_double_free_invariant(self):
        self.server.execute_script(
            self._script_partial_page_tail_no_double_free_invariant
        )

    @staticmethod
    def _script_partial_page_tail_no_double_free_invariant(t: ScriptedContext):
        s = t._scheduler
        allocator = s.token_to_kv_pool_allocator
        free_before: int = allocator.available_size()
        prompt_len: int = 4 * DEFAULT_CHUNK_SIZE + 7
        r = t.start_req(prompt_len=prompt_len, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        observed_partial_tail: bool = False
        for _ in range(800):
            req = s.chunked_req
            if req is not None and req.rid == r.rid:
                prefix_len: int = len(req.prefix_indices)
                protected_len: int = req.cache_protected_len
                assert prefix_len >= protected_len, (
                    f"len(prefix_indices)={prefix_len} dropped below "
                    f"cache_protected_len={protected_len}: tail was freed "
                    f"prematurely"
                )
                if prefix_len > protected_len:
                    observed_partial_tail = True
            if r.finished:
                break
            yield
        assert r.finished
        free_after: int = allocator.available_size()
        assert free_after == free_before, (
            f"KV pool free count delta on chunked req lifecycle must be 0; "
            f"got free_before={free_before}, free_after={free_after} "
            f"(double-free or leak of partial-page tail)"
        )
        assert observed_partial_tail, (
            "test must observe len(prefix_indices) > cache_protected_len at "
            "least once (partial-page tail window); the page_size > 1 "
            "branch was never exercised"
        )


class TestRadixFcfs(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        schedule_policy="fcfs",
    )

    def test_naive_radix_chunked(self):
        self.server.execute_script(self._script_naive_radix_chunked)

    @staticmethod
    def _script_naive_radix_chunked(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        assert r1.finished

        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN + DEFAULT_CHUNK_SIZE * 2,
            max_new_tokens=2,
        )
        yield from run_until_finished(r2)
        assert r2.finished
        assert r2.req.cached_tokens > 0
        assert r2.chunks_done >= 1


class TestRadixDisabled(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disable_radix_cache=True,
    )

    def test_radix_disabled_chunks_every_time(self):
        self.server.execute_script(self._script_radix_disabled_chunks_every_time)

    @staticmethod
    def _script_radix_disabled_chunks_every_time(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        assert r1.chunks_done >= 2

        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r2)
        assert r2.chunks_done >= 2


class TestRadixLpm(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        schedule_policy="lpm",
    )

    def test_radix_lpm_policy_chunked_priority(self):
        self.server.execute_script(self._script_radix_lpm_policy_chunked_priority)

    @staticmethod
    def _script_radix_lpm_policy_chunked_priority(t: ScriptedContext):
        r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
        yield from run_until_finished(r_warm)
        assert r_warm.finished

        reqs = [
            t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 3, max_new_tokens=1)
            for _ in range(3)
        ]
        yield from run_until_all_finished(reqs)
        for r in reqs:
            assert r.finished
            assert r.kv_pages == 0
            assert r.lock_refs == 0
            assert r.req.cached_tokens > 0


class TestRadixDfsWeight(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        schedule_policy="dfs-weight",
    )

    def test_radix_dfs_weight_policy_chunked(self):
        self.server.execute_script(self._script_radix_dfs_weight_policy_chunked)

    @staticmethod
    def _script_radix_dfs_weight_policy_chunked(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        assert r1.finished
        assert r1.chunks_done >= 2

        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r2)
        assert r2.finished
        assert r2.req.cached_tokens > 0
        assert r2.kv_pages == 0
        assert r2.lock_refs == 0


class TestRadixPriority(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_radix_prefix_match_with_priority(self):
        self.server.execute_script(self._script_radix_prefix_match_with_priority)

    @staticmethod
    def _script_radix_prefix_match_with_priority(t: ScriptedContext):
        r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
        yield from run_until_finished(r_warm)
        assert r_warm.finished

        r = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 3, max_new_tokens=2, priority="high"
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.req.cached_tokens > 0
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_radix_calc_priority_skip_chunked_resume(self):
        self.server.execute_script(self._script_radix_calc_priority_skip_chunked_resume)

    @staticmethod
    def _script_radix_calc_priority_skip_chunked_resume(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high"
        )
        yield from run_until(r1, lambda h: h.is_chunking)
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low"
        )

        # The scheduler does not record which admission branch it took. The
        # observable consequence of skipping r1's chunked-resume in the priority
        # calc is that the lower-priority r2 never preempts r1's in-flight chunked
        # prefill: r1's chunk progress only advances (a preemption/retract would
        # drop it back to waiting and reset chunks_done), and r1 stays ahead of r2.
        prev_chunks_done = r1.chunks_done
        r1_finished_first = False
        while not (r1.finished and r2.finished):
            assert r1.chunks_done >= prev_chunks_done, (
                f"r1 chunked prefill was preempted by lower-priority r2: "
                f"chunks_done regressed {prev_chunks_done} -> {r1.chunks_done}"
            )
            prev_chunks_done = r1.chunks_done
            if r1.finished and not r2.finished:
                r1_finished_first = True
            yield

        assert r1.finished and r2.finished
        assert r1_finished_first, (
            "high-priority r1 must finish its chunked prefill before low-priority "
            "r2 completes; lower-priority r2 must not jump ahead of r1's resume"
        )


if __name__ == "__main__":
    unittest.main()
