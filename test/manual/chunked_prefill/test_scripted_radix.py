import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
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
        # evict_radix(None) flushes the tree, but flush only frees nodes once they
        # are unreferenced. r1 lingers in the running batch for one overlap step
        # after finishing, so drain to idle before flushing or the flush is a no-op
        # and r2 re-hits r1's stale prefix.
        for _ in range(5):
            yield
        t.evict_radix(prefix_tokens=None)
        yield

        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r2)
        assert r2.finished
        assert r2.chunks_done >= 2, (
            f"after eviction r2 must re-chunk from scratch; "
            f"chunks_done={r2.chunks_done} cached_tokens={r2.req.cached_tokens}"
        )
        assert (
            r2.req.cached_tokens == 0
        ), f"eviction must clear r1's prefix; cached_tokens={r2.req.cached_tokens}"
        assert r2.kv_pages == 0
        assert r2.lock_refs == 0

    def test_radix_resume_init_next_round_path(self):
        self.server.execute_script(self._script_radix_resume_init_next_round_path)

    @staticmethod
    def _script_radix_resume_init_next_round_path(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1)
        yield from run_until_finished(r1)
        assert r1.finished

        # Residual past the cached prefix must EXCEED chunk_size, or it completes in
        # one non-chunked prefill (a residual of exactly chunk_size does not chunk).
        # 2 * chunk_size of fresh tail guarantees the chunked-resume branch runs.
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN + 2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2
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
            assert r.finished
            # The shared 4-chunk prefix must actually be hit (this pins that
            # prefix-sharing happened); the 8-token tail is too short to chunk,
            # so cached_tokens -- not chunks_done -- is the right witness.
            assert r.req.cached_tokens > 0
            assert r.lock_refs == 0

    def test_radix_partial_hit_exact_chunk_boundary(self):
        self.server.execute_script(self._script_radix_partial_hit_exact_chunk_boundary)

    @staticmethod
    def _script_radix_partial_hit_exact_chunk_boundary(t: ScriptedContext):
        r1 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r2)
        # r2 hits r1's full chunk_size prefix, leaving a residual of exactly
        # chunk_size. A residual equal to chunk_size completes in one non-chunked
        # prefill (the chunked path engages only when the residual exceeds the chunk
        # budget), so chunks_done is 0 -- the defining behavior at this boundary.
        assert r2.chunks_done == 0, (
            f"residual of exactly chunk_size must not chunk; "
            f"chunks_done={r2.chunks_done} cached_tokens={r2.req.cached_tokens}"
        )

    def test_radix_two_distinct_prefixes(self):
        self.server.execute_script(self._script_radix_two_distinct_prefixes)

    @staticmethod
    def _script_radix_two_distinct_prefixes(t: ScriptedContext):
        # Distinct prompt_token values fork the radix tree into two sibling
        # branches off the root instead of a single linear chain, so each
        # re-submission must hit only its own branch's cached prefix.
        r_a = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1, prompt_token=11
        )
        yield from run_until_finished(r_a)
        r_b = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1, prompt_token=22
        )
        yield from run_until_finished(r_b)

        r_a2 = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1, prompt_token=11
        )
        yield from run_until_finished(r_a2)
        assert r_a2.chunks_done == 0
        assert r_a2.req.cached_tokens > 0

        r_b2 = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1, prompt_token=22
        )
        yield from run_until_finished(r_b2)
        assert r_b2.chunks_done == 0
        assert r_b2.req.cached_tokens > 0

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

        # Drain the overlap lag so r_warm is fully unreferenced before flushing;
        # otherwise evict_radix's flush is a no-op and the warm prefix survives.
        for _ in range(5):
            yield
        t.evict_radix(prefix_tokens=None)
        r = t.start_req(
            prompt_len=warm_len + DEFAULT_CHUNK_SIZE * 2,
            max_new_tokens=2,
        )
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # The warm prefix was evicted before r was submitted, so r must miss
        # the cache and re-chunk its whole 6-chunk prompt from scratch.
        assert r.req.cached_tokens == 0, (
            f"eviction must clear the warm prefix; "
            f"cached_tokens={r.req.cached_tokens} chunks_done={r.chunks_done}"
        )
        assert r.chunks_done >= 2
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


class TestRadixNoTailChunked(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_page_size_one_chunked_has_no_partial_page_tail(self):
        """At page_size=1 chunked caching takes the no-tail else branch every step."""
        self.server.execute_script(
            self._script_page_size_one_chunked_has_no_partial_page_tail
        )

    @staticmethod
    def _script_page_size_one_chunked_has_no_partial_page_tail(t: ScriptedContext):
        # GPU validation pending (manual scripted-runtime suite).
        # Contrast to TestRadixPartialPage (page_size=4 takes the `if` partial-
        # tail branch at radix_cache.py:517-520, where len(prefix_indices) >
        # cache_protected_len). At the default page_size=1 every kv index is page
        # aligned, so cache_unfinished_req always lands on the `else` no-tail
        # branch at radix_cache.py:521-522 (req.prefix_indices = new_indices),
        # making len(prefix_indices) == cache_protected_len at every mid-chunk
        # observation.
        s = t.scheduler
        prompt_len: int = 4 * DEFAULT_CHUNK_SIZE
        r = t.start_req(prompt_len=prompt_len, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        observed_mid_chunk: bool = False
        for _ in range(800):
            req = s.chunked_req
            if req is not None and req.rid == r.rid:
                observed_mid_chunk = True
                prefix_len: int = len(req.prefix_indices)
                protected_len: int = req.cache_protected_len
                assert prefix_len == protected_len, (
                    f"page_size=1 must take the no-tail else branch: "
                    f"len(prefix_indices)={prefix_len} != "
                    f"cache_protected_len={protected_len} (a partial-page tail "
                    f"was appended, which only happens for page_size > 1)"
                )
            if r.finished:
                break
            yield
        assert r.finished
        assert observed_mid_chunk, (
            "test must observe r as the in-flight chunked_req at least once; the "
            "no-tail else branch was never exercised"
        )
        assert (
            r.kv_pages == 0
        ), f"finished chunked req must release KV; got {r.kv_pages}"


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

        s = t.scheduler
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


class TestRadixHitCountNonChunked(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_non_chunked_prefix_hit_increments_hit_count_by_one(self):
        """Non-chunked finished req hitting a warm prefix bumps node hit_count by 1."""
        self.server.execute_script(
            self._script_non_chunked_prefix_hit_increments_hit_count_by_one
        )

    @staticmethod
    def _script_non_chunked_prefix_hit_increments_hit_count_by_one(t: ScriptedContext):
        # GPU validation pending (manual scripted-runtime suite).
        # Contrast to TestRadixHitCountInvariant (chunked=True leaves hit_count
        # untouched): when a NON-chunked finished req re-traverses an existing
        # warm prefix, _inc_hit_count(chunked=False) at radix_cache.py:672 runs
        # and bumps the matched ancestor node's hit_count by exactly 1.
        def _snapshot_hit_counts(root) -> dict:
            snapshot: dict = {}
            stack = [root]
            while stack:
                node = stack.pop()
                snapshot[node.id] = node.hit_count
                stack.extend(node.children.values())
            return snapshot

        s = t.scheduler

        # Warm a 2-chunk prefix on its own radix branch (prompt_token=11). This
        # creates the node(s) whose hit_count we then watch.
        r_warm = t.start_req(
            prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=1, prompt_token=11
        )
        yield from run_until_finished(r_warm)
        assert r_warm.finished

        baseline = _snapshot_hit_counts(s.tree_cache.root_node)

        # A full-fit re-submission on the same prefix: residual beyond the cached
        # 2-chunk prefix is a single token (<= chunk_size), so it never chunks
        # (chunks_done == 0) and finishes via the non-chunked cache_finished_req
        # -> insert(chunked=False) path.
        r2 = t.start_req(
            prompt_len=2 * DEFAULT_CHUNK_SIZE + 1, max_new_tokens=1, prompt_token=11
        )
        yield from run_until_finished(r2)
        assert r2.finished
        assert r2.chunks_done == 0, (
            f"residual past the warm prefix is one token and must not chunk; "
            f"got chunks_done={r2.chunks_done}"
        )
        assert r2.req.cached_tokens > 0, (
            f"r2 must hit the warm 2-chunk prefix to drive the non-chunked "
            f"hit_count increment; got cached_tokens={r2.req.cached_tokens}"
        )

        cur = _snapshot_hit_counts(s.tree_cache.root_node)
        # At least one pre-existing node that r2 traversed must show exactly a +1
        # increment, and no pre-existing node may move by more than 1 (a single
        # non-chunked insert visits each matched node once).
        incremented_by_one = 0
        for node_id, base_count in baseline.items():
            if node_id not in cur:
                continue
            delta: int = cur[node_id] - base_count
            assert delta in (0, 1), (
                f"non-chunked insert moved existing node id={node_id} hit_count "
                f"by {delta} (expected 0 or 1): baseline={base_count}, "
                f"now={cur[node_id]}"
            )
            if delta == 1:
                incremented_by_one += 1
        assert incremented_by_one >= 1, (
            f"_inc_hit_count(chunked=False) at radix_cache.py:672 must bump at "
            f"least one matched warm-prefix node by exactly 1; saw "
            f"{incremented_by_one} such nodes"
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
        s = t.scheduler
        allocator = s.token_to_kv_pool_allocator
        free_before: int = allocator.available_size()
        prompt_len: int = 4 * DEFAULT_CHUNK_SIZE + 7
        r = t.start_req(prompt_len=prompt_len, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        # While the req is the in-flight chunked_req, cache_unfinished_req only ever
        # commits page-aligned slices: chunked_prefill_size must be divisible by
        # page_size (server_args validation), so every mid-chunk fill_ids length is
        # page aligned and len(new_indices) == len(kv_indices). The partial-page
        # tail branch at radix_cache.py:517-520 (len(prefix_indices) >
        # cache_protected_len) is therefore NOT reachable mid-chunk -- it fires only
        # at the completing extend, when the non-page-aligned prompt remainder (+7)
        # is finally committed, and that batch no longer carries the rid as
        # chunked_req. The observable, durable safety property mid-chunk is the
        # one-directional invariant that prefix_indices never drops below
        # cache_protected_len (the committed tail is never freed prematurely).
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
            if r.finished:
                break
            yield
        assert r.finished
        # The real partial-page-tail invariant: a non-page-aligned prompt (+7 over a
        # page boundary) is committed through the chunked lifecycle and the trailing
        # partial page is freed exactly once. flush_cache is a no-op unless the
        # scheduler is fully idle (it bails while the just-finished req still lingers
        # in the overlap pipeline), so drain until is_fully_idle BEFORE flushing -- a
        # fixed 5-yield drain is not always enough and leaves the committed prefix
        # un-flushed, reading as a ~prompt-sized leak. A net-zero free-count delta
        # proves the partial-page tail was neither leaked nor double-freed.
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        free_after: int = allocator.available_size()
        assert free_after == free_before, (
            f"KV pool free count delta on chunked req lifecycle must be 0; "
            f"got free_before={free_before}, free_after={free_after} "
            f"(double-free or leak of partial-page tail)"
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
    # disable_radix_cache=True swaps the radix tree for a ChunkCache, which the
    # scripted harness's KV-canary walker (walk_radix_cache_for_canary) does not
    # support -- it raises NotImplementedError on ChunkCache and crashes the
    # server at startup. The canary is a harness-side integrity layer, not part of
    # the v1 engine under test, so turn it off for this radix-disabled class
    # (sweep_interval must also drop to 0 or server_args validation rejects the
    # combination).
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disable_radix_cache=True,
        kv_canary="none",
        kv_canary_real_data="none",
        kv_canary_sweep_interval=0,
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
        # Warm a 2-chunk prefix on the prompt_token=1 branch.
        r_warm = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1, prompt_token=1
        )
        yield from run_until_finished(r_warm)
        assert r_warm.finished
        # Drain the overlap lag so r_warm fully leaves the running batch before the
        # competitors start; otherwise it lingers as the "first admitted" rid.
        for _ in range(5):
            yield

        # Submit two reqs at once so calc_priority orders the waiting queue:
        # r_long shares the full 2-chunk warm prefix (longest prefix match),
        # r_short is on a disjoint branch (zero prefix match). LPM must admit
        # r_long first, so it is the first rid to enter chunked prefill.
        r_long = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 4, max_new_tokens=1, prompt_token=1
        )
        r_short = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 4, max_new_tokens=1, prompt_token=7
        )

        # cached_tokens is set at admission but the req object is cleared on
        # finish (handle.req becomes None), so capture it live each step.
        first_admitted = None
        cached_tokens_by_rid: dict = {}
        for _ in range(DEFAULT_MAX_STEPS):
            comp = t.batch_composition()
            active = (
                comp.get("chunked", [])
                + comp.get("prefill", [])
                + comp.get("running", [])
            )
            if first_admitted is None and active:
                first_admitted = active[0]
            for r in (r_long, r_short):
                if r.req is not None:
                    cached_tokens_by_rid[r.rid] = r.req.cached_tokens
            if r_long.finished and r_short.finished:
                break
            yield
        assert r_long.finished and r_short.finished
        assert first_admitted == r_long.rid, (
            f"LPM must admit the longest-prefix-match req first; first admitted "
            f"rid was {first_admitted!r}, expected r_long={r_long.rid!r}"
        )
        assert cached_tokens_by_rid.get(r_long.rid, 0) > 0
        assert cached_tokens_by_rid.get(r_short.rid, 0) == 0
        for r in (r_long, r_short):
            assert r.kv_pages == 0
            assert r.lock_refs == 0


class TestRadixDfsWeight(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        schedule_policy="dfs-weight",
    )

    def test_radix_dfs_weight_policy_chunked(self):
        self.server.execute_script(self._script_radix_dfs_weight_policy_chunked)

    @staticmethod
    def _script_radix_dfs_weight_policy_chunked(t: ScriptedContext):
        # Warm two distinct branches (different prompt_token) so waiting reqs
        # hang off two different radix nodes.
        warm_a = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1, prompt_token=3
        )
        yield from run_until_finished(warm_a)
        warm_b = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1, prompt_token=4
        )
        yield from run_until_finished(warm_b)

        # Branch A carries a heavier subtree weight (3 waiting reqs) than branch
        # B (1 waiting req). dfs-weight traverses the heavier child subtree
        # first, so all branch-A reqs must finish before the lone branch-B req.
        heavy = [
            t.start_req(
                prompt_len=DEFAULT_CHUNK_SIZE * 4, max_new_tokens=1, prompt_token=3
            )
            for _ in range(3)
        ]
        light = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 4, max_new_tokens=1, prompt_token=4
        )
        all_reqs = heavy + [light]

        # cached_tokens is set at admission and is stable, but req objects are
        # cleared on finish (handle.req becomes None), so capture it live each step.
        finish_order: list = []
        cached_tokens_by_rid: dict = {}
        for _ in range(DEFAULT_MAX_STEPS * 4):
            for r in all_reqs:
                if r.req is not None:
                    cached_tokens_by_rid[r.rid] = r.req.cached_tokens
                if r.finished and r.rid not in finish_order:
                    finish_order.append(r.rid)
            if all(r.finished for r in all_reqs):
                break
            yield
        assert all(r.finished for r in all_reqs)
        assert finish_order[-1] == light.rid, (
            f"dfs-weight must drain the heavier branch-A subtree before the "
            f"lighter branch-B req; finish order was {finish_order!r}, "
            f"light={light.rid!r}"
        )
        for r in heavy:
            assert cached_tokens_by_rid.get(r.rid, 0) > 0
        assert cached_tokens_by_rid.get(light.rid, 0) > 0
        for r in all_reqs:
            assert r.kv_pages == 0
            assert r.lock_refs == 0


class TestRadixPriority(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_radix_prefix_match_with_priority(self):
        self.server.execute_script(self._script_radix_prefix_match_with_priority)

    @staticmethod
    def _script_radix_prefix_match_with_priority(t: ScriptedContext):
        # enable_priority_scheduling forces fcfs+priority sorting; with the
        # default schedule_low_priority_values_first=False a larger priority
        # value wins. Both reqs hit the same warm prefix, so priority -- not
        # prefix length -- decides admission order.
        r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
        yield from run_until_finished(r_warm)
        assert r_warm.finished
        # Drain the overlap lag so r_warm fully leaves the running batch before the
        # competitors start; otherwise it lingers as the "first admitted" rid.
        for _ in range(5):
            yield

        r_low = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 4, max_new_tokens=1, priority=0
        )
        r_high = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE * 4, max_new_tokens=1, priority=10
        )

        # Evaluate both finished flags into locals every step. A short-circuited
        # `r_low.finished and r_high.finished` would skip the r_high.finished probe
        # on every step where r_low is not yet finished -- but r_high is admitted
        # first (priority) and finishes first, so it would never be observed live
        # and is_finished would report it "unknown" forever, hanging the loop. We
        # must call find_req_by_rid on r_high while it is still live.
        # cached_tokens is set at admission but the req object is cleared on finish
        # (handle.req becomes None), so capture it live each step.
        first_admitted = None
        low_done = False
        high_done = False
        cached_tokens_by_rid: dict = {}
        for _ in range(DEFAULT_MAX_STEPS):
            comp = t.batch_composition()
            active = (
                comp.get("chunked", [])
                + comp.get("prefill", [])
                + comp.get("running", [])
            )
            if first_admitted is None and active:
                first_admitted = active[0]
            for r in (r_low, r_high):
                if r.req is not None:
                    cached_tokens_by_rid[r.rid] = r.req.cached_tokens
            low_done = low_done or r_low.finished
            high_done = high_done or r_high.finished
            if low_done and high_done:
                break
            yield
        assert low_done and high_done
        assert first_admitted == r_high.rid, (
            f"higher-priority req must be admitted first; first admitted rid "
            f"was {first_admitted!r}, expected r_high={r_high.rid!r}"
        )
        for r in (r_low, r_high):
            assert cached_tokens_by_rid.get(r.rid, 0) > 0
            assert r.kv_pages == 0
            assert r.lock_refs == 0

    def test_radix_calc_priority_skip_chunked_resume(self):
        self.server.execute_script(self._script_radix_calc_priority_skip_chunked_resume)

    @staticmethod
    def _script_radix_calc_priority_skip_chunked_resume(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority=10)
        yield from run_until(r1, lambda h: h.is_chunking)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority=0)

        # The scheduler does not record which admission branch it took. The
        # observable consequence of skipping r1's chunked-resume in the priority
        # calc is that the lower-priority r2 never preempts r1's in-flight chunked
        # prefill: r1's chunk progress only advances (a preemption/retract would
        # drop it back to waiting and reset chunks_done), and r1 -- which started
        # chunking first and outranks r2 -- finishes no later than r2.
        #
        # Note: do NOT assert r1 finishes *strictly* before r2. Both reqs have the
        # same prompt and max_new_tokens, and under the overlap scheduler the coarse
        # per-step observation lands their completions on the same step (r1 and r2
        # both observed finished at the same yield). The genuine v1 invariant is
        # "r1 finishes no later than r2", captured as r1_fin_step <= r2_fin_step.
        prev_chunks_done = r1.chunks_done
        r1_fin_step = None
        r2_fin_step = None
        step = 0
        while not (r1.finished and r2.finished):
            assert r1.chunks_done >= prev_chunks_done, (
                f"r1 chunked prefill was preempted by lower-priority r2: "
                f"chunks_done regressed {prev_chunks_done} -> {r1.chunks_done}"
            )
            prev_chunks_done = r1.chunks_done
            if r1.finished and r1_fin_step is None:
                r1_fin_step = step
            if r2.finished and r2_fin_step is None:
                r2_fin_step = step
            step += 1
            yield
        if r1.finished and r1_fin_step is None:
            r1_fin_step = step
        if r2.finished and r2_fin_step is None:
            r2_fin_step = step

        assert r1.finished and r2.finished
        assert r1_fin_step is not None and r2_fin_step is not None
        assert r1_fin_step <= r2_fin_step, (
            f"high-priority r1 must finish its chunked prefill no later than "
            f"low-priority r2; r1 finished at step {r1_fin_step}, r2 at step "
            f"{r2_fin_step} (lower-priority r2 jumped ahead of r1's resume)"
        )


if __name__ == "__main__":
    unittest.main()
