import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    BALLAST_MAX_NEW_TOKENS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    SMALL_KV_POOL_BALLAST_MAX_NEW_TOKENS,
    SMALL_KV_POOL_BALLAST_PROMPT_LEN,
    SMALL_KV_POOL_MAX_TOTAL_TOKENS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


class TestKVPressureBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_lock_refs_tight_concurrent_prefix(self):
        self.server.execute_script(self._script_lock_refs_tight_concurrent_prefix)

    @staticmethod
    def _script_lock_refs_tight_concurrent_prefix(t: ScriptedContext):
        """A long chunked req sharing a pinned warm prefix finishes and leaks no lock_refs."""
        # GPU validation pending.
        #
        # Warm a shared prefix: a short req that finishes leaves cached,
        # evictable prefix nodes in the radix tree. Use a distinct prompt_token
        # so the long req below provably hits THIS prefix (not stray nodes).
        warm_token = 7
        warm_len = DEFAULT_CHUNK_SIZE
        r_warm = t.start_req(
            prompt_len=warm_len, max_new_tokens=1, prompt_token=warm_token
        )
        yield from run_until_finished(r_warm)
        # A second, distinct warm prefix: with a single cached node in the tree,
        # exhaust_lock_refs(leave_refs=1) would leave exactly that node unpinned
        # and pin nothing.
        r_warm2 = t.start_req(prompt_len=warm_len, max_new_tokens=1, prompt_token=8)
        yield from run_until_finished(r_warm2)
        assert r_warm.finished
        # the overlap scheduler drops the finished req's protective lock a few
        # steps after the finish is observed; drain before asserting.
        for _ in range(12):
            if r_warm.lock_refs == 0:
                break
            yield
        assert r_warm.lock_refs == 0

        # Baseline AFTER warming: this is the global lock_ref state the reset
        # path must restore to. The warm prefix nodes are present and evictable
        # (lock_ref == 0) at this point.
        baseline_lock_refs = t.get_all_node_lock_refs()

        # Pin most evictable nodes via the real inc_lock_ref path, leaving only a
        # few unlocked. The KV is still present, just protected/non-evictable.
        t.exhaust_lock_refs(leave_refs=1)
        yield
        pinned_lock_refs = t.get_all_node_lock_refs()
        assert any(
            pinned_lock_refs.get(node_id, 0) > baseline_lock_refs.get(node_id, 0)
            for node_id in pinned_lock_refs
        ), "exhaust_lock_refs(leave_refs=1) must pin at least one warm-prefix node"

        # Run a long chunked req that shares the warm prefix (same prompt_token,
        # longer prompt). It must really chunk while the cache is pinned.
        r_long = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            prompt_token=warm_token,
        )
        yield from run_until(r_long, lambda h: h.is_chunking)
        yield from run_until_finished(r_long, max_steps=2000)
        assert r_long.finished
        # Even with the warm prefix hit, the remaining new tokens span multiple
        # chunks: (VERY_LONG_PROMPT_LEN - warm_len) / DEFAULT_CHUNK_SIZE == 7.
        assert r_long.chunks_done >= 2, (
            f"long req must really chunk under pinned cache; got chunks_done="
            f"{r_long.chunks_done}"
        )
        # The finished req must hold no locks of its own.
        assert (
            r_long.lock_refs == 0
        ), f"req {r_long.rid} leaked {r_long.lock_refs} lock_refs after finish"

        # Release the exhauster's locks (mirrors the reset path) and confirm the
        # global lock_ref state returns EXACTLY to the post-warm baseline: no
        # leaked locks from either the exhauster or the chunked req.
        t._release_exhausted_pools()
        final_lock_refs = t.get_all_node_lock_refs()
        for node_id, baseline in baseline_lock_refs.items():
            assert final_lock_refs.get(node_id, 0) == baseline, (
                f"node {node_id} lock_ref leaked: baseline={baseline}, "
                f"final={final_lock_refs.get(node_id, 0)}"
            )

    def test_kv_pressure_with_retract_resume(self):
        self.server.execute_script(self._script_kv_pressure_with_retract_resume)

    @staticmethod
    def _script_kv_pressure_with_retract_resume(t: ScriptedContext):
        baseline = t.engine_stats()["kv_pool_free"]
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=60
        )
        yield from run_until(r, lambda h: h.is_chunking)
        chunks_before_retract = r.chunks_done
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        # Force real forward progress past the retract point: the resumed req must
        # re-chunk and run a strictly later chunk than it had before the retract,
        # not merely fail to regress (which a finished-or-aborted req satisfies
        # vacuously). Only then do we drive it to completion.
        yield from run_until(
            r,
            lambda h: h.chunks_done > chunks_before_retract,
            max_steps=2000,
        )
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.kv_pages == 0
        # The finished prompt stays cached in the radix tree; drain and flush
        # before the leak comparison so cached (not leaked) pages do not count
        # against the baseline.
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"KV pool failed to recover after retract+resume: "
            f"baseline={baseline}, final={final}"
        )

    def test_chunked_batch_recovers_pools_to_steady_state(self):
        self.server.execute_script(
            self._script_chunked_batch_recovers_pools_to_steady_state
        )

    @staticmethod
    def _script_chunked_batch_recovers_pools_to_steady_state(t: ScriptedContext):
        before = t.engine_stats()
        # prompt_len must exceed chunk_size so each req actually walks the chunked
        # prefill path; prompt_len=16 (< chunk size) never chunks. A distinct
        # prompt_token per req keeps every prompt cold: with identical prompts only
        # the first would chunk and the rest would cache-hit its committed prefix.
        reqs = [
            t.start_req(
                prompt_len=DEFAULT_CHUNK_SIZE + 1,
                max_new_tokens=2,
                prompt_token=500 + i,
            )
            for i in range(50)
        ]
        yield from run_until_all_finished(reqs, max_steps=2000)
        for r in reqs:
            assert r.finished
            assert r.kv_pages == 0, f"req {r.rid} kept {r.kv_pages} pages after finish"
        # The finished prompts legitimately stay committed in the radix tree
        # (cached != leaked), so kv_pool_free stays below baseline until eviction;
        # drain to idle and flush before the leak comparison.
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        after = t.engine_stats()
        assert after["kv_pool_free"] >= before["kv_pool_free"], (
            f"50 chunked reqs leaked KV: baseline={before['kv_pool_free']}, "
            f"final={after['kv_pool_free']}"
        )
        assert after["req_pool_free"] >= before["req_pool_free"], (
            f"50 chunked reqs leaked req-pool rows: "
            f"baseline={before['req_pool_free']}, final={after['req_pool_free']}"
        )

    def test_chunked_retract_at_chunk_first_mid_last(self):
        self.server.execute_script(self._script_chunked_retract_at_chunk_first_mid_last)

    @staticmethod
    def _script_chunked_retract_at_chunk_first_mid_last(t: ScriptedContext):
        expected_chunks = VERY_LONG_PROMPT_LEN // DEFAULT_CHUNK_SIZE
        mid_chunk = expected_chunks // 2
        last_minus_one = max(1, expected_chunks - 1)

        # Distinct prompt_token per req: with identical prompts the second/third
        # req would cache-hit the prior finished req's full prefix and never
        # chunk, so chunks_done would stay 0 and the chunk-progress predicate
        # would never be satisfied.
        r_first = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=31
        )
        yield from run_until(r_first, lambda h: h.is_chunking)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r_first, max_steps=2000)
        assert r_first.finished
        assert r_first.kv_pages == 0

        r_mid = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=32
        )
        yield from run_until(r_mid, lambda h: h.chunks_done >= mid_chunk)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r_mid, max_steps=2000)
        assert r_mid.finished
        assert r_mid.kv_pages == 0

        r_last = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=33
        )
        yield from run_until(r_last, lambda h: h.chunks_done >= last_minus_one)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r_last, max_steps=2000)
        assert r_last.finished
        assert r_last.kv_pages == 0

    def test_flush_cache_during_chunked_in_flight(self):
        self.server.execute_script(self._script_flush_cache_during_chunked_in_flight)

    @staticmethod
    def _script_flush_cache_during_chunked_in_flight(t: ScriptedContext):
        # Distinct prompt_token: r must genuinely chunk while in flight, so it
        # must NOT cache-hit the warm req's full prefix (identical prompts would
        # let it skip chunking entirely and chunks_done would stay 0). The warm
        # req still populates the radix tree so flush_cache has cached nodes to
        # clear mid-flight.
        r_warm = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1, prompt_token=41
        )
        yield from run_until_finished(r_warm)
        assert r_warm.finished

        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=42
        )
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.flush_cache()
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.kv_pages == 0

    def test_chunked_oscillation_three_force_retracts(self):
        self.server.execute_script(
            self._script_chunked_oscillation_three_force_retracts
        )

    @staticmethod
    def _script_chunked_oscillation_three_force_retracts(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=70
        )
        yield from run_until(r, lambda h: h.is_chunking)
        chunks_at_first = r.chunks_done
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until(
            r,
            lambda h: h.is_chunking and h.chunks_done >= chunks_at_first,
            max_steps=800,
        )
        chunks_after_first_resume = r.chunks_done
        assert chunks_after_first_resume >= chunks_at_first, (
            f"chunks_done regressed across retract: "
            f"before={chunks_at_first}, after={chunks_after_first_resume}"
        )

        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until(
            r,
            lambda h: h.is_chunking and h.chunks_done >= chunks_after_first_resume,
            max_steps=800,
        )
        chunks_after_second_resume = r.chunks_done
        assert chunks_after_second_resume >= chunks_after_first_resume

        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.chunks_done >= chunks_after_second_resume


class TestKVPressureSmallRowPool(ScriptedTestCase):
    # req_to_token_pool.size == max_running_requests. The default pool is
    # thousands of rows that ballast reqs can never fully occupy, because the KV
    # pool exhausts long before the row pool does. Cap max_running_requests small
    # so a handful of never-finishing ballast reqs really hold every row and the
    # admit-after-release path can be exercised.
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        max_running_requests=8,
    )

    def test_row_pool_tight_admits_after_release(self):
        self.server.execute_script(self._script_row_pool_tight_admits_after_release)

    @staticmethod
    def _script_row_pool_tight_admits_after_release(t: ScriptedContext):
        # Honest pressure pattern: hold every row with never-finishing ballast
        # reqs (each occupies exactly one row for its whole life), confirm fresh
        # reqs are stuck waiting because no row is allocatable, then release the
        # ballast and confirm the waiting reqs are admitted, finish, and leak no
        # rows. "Admits after release" with nothing ever freeing a row is
        # structurally impossible, so the release is part of the path under test.
        baseline_rows_used = (
            t.scheduler.req_to_token_pool.size
            - t.scheduler.req_to_token_pool.available_size()
        )
        row_pool_size = t.scheduler.req_to_token_pool.size
        ballast = [
            t.start_req(
                prompt_len=1, max_new_tokens=BALLAST_MAX_NEW_TOKENS, ignore_eos=True
            )
            for _ in range(row_pool_size)
        ]
        for _ in range(DEFAULT_MAX_STEPS):
            if t.scheduler.req_to_token_pool.available_size() == 0:
                break
            yield
        assert t.scheduler.req_to_token_pool.available_size() == 0, (
            f"ballast must hold every row; "
            f"available={t.scheduler.req_to_token_pool.available_size()}"
        )

        reqs = [
            t.start_req(prompt_len=8, max_new_tokens=1, prompt_token=50 + i)
            for i in range(5)
        ]
        for _ in range(6):
            yield
        for r in reqs:
            assert r.status == "waiting", (
                f"fresh req must be unschedulable under a full row pool; "
                f"rid={r.rid}, status={r.status}"
            )

        for b in ballast:
            t.abort(b)
        yield from run_until_all_finished(reqs, max_steps=2000)
        for r in reqs:
            assert r.finished, f"req {r.rid} did not finish after release"
            assert r.kv_pages == 0, (
                f"row-pool pressure must not leave KV held: rid={r.rid}, "
                f"kv_pages={r.kv_pages}"
            )

        # Drain the aborted ballast to idle before the row-leak comparison.
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        final_rows_used = (
            t.scheduler.req_to_token_pool.size
            - t.scheduler.req_to_token_pool.available_size()
        )
        assert final_rows_used <= baseline_rows_used, (
            f"row pool leak after admit-after-release: baseline used="
            f"{baseline_rows_used}, final used={final_rows_used}"
        )


class TestKVPressureSmallPool(ScriptedTestCase):
    # True, engine-resolvable KV pressure with a SMALL KV pool and REAL requests,
    # not raw exhaust_kv pages. exhaust_kv grabs raw allocator pages that have no
    # backing Req, so they are neither evictable cache nor retractable-req-locked;
    # a chunked req that force-re-adds (add_chunked_req: _rem_tokens <= 0 ->
    # rem_chunk_tokens, schedule_policy.py) against that raw exhaustion hard-OOMs
    # and crashes the shared scheduler. Here the pressure comes from a long-lived
    # ballast decode req plus an in-flight chunked req sharing a capped pool: when
    # their combined reservation drives rem_total_tokens <= 0 the force-re-add
    # still fires, but the pages it needs are physically free, so the alloc
    # succeeds and the engine resolves the over-commitment by retracting the
    # ballast on the decode-OOM path (retract_decode) -- never a crash.
    #
    # max_total_tokens caps the KV pool (model_runner_kv_cache_mixin.py:
    # _apply_token_constraints -> min(profiled, max_total_tokens)); page_size == 1
    # on the small test model, so the cap is exact in tokens.
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        max_total_tokens=SMALL_KV_POOL_MAX_TOTAL_TOKENS,
    )

    @staticmethod
    def _start_ballast(t: ScriptedContext, *, prompt_token: int):
        # A long-lived decode req that holds a big share of the capped pool and
        # reserves decode headroom. ignore_eos keeps it from finishing early; its
        # clipped max_new stays small enough to be ADMITTED into the 4096-token
        # pool (see SMALL_KV_POOL_BALLAST_MAX_NEW_TOKENS) yet reserves enough that,
        # once a chunked req is also in flight, rem_total_tokens goes <= 0.
        return t.start_req(
            prompt_len=SMALL_KV_POOL_BALLAST_PROMPT_LEN,
            max_new_tokens=SMALL_KV_POOL_BALLAST_MAX_NEW_TOKENS,
            ignore_eos=True,
            prompt_token=prompt_token,
        )

    def test_chunked_completes_when_ballast_retracted(self):
        self.server.execute_script(
            self._script_chunked_completes_when_ballast_retracted
        )

    @staticmethod
    def _script_chunked_completes_when_ballast_retracted(t: ScriptedContext):
        # Admit the chunked req FIRST, while the pool is empty, so it is guaranteed
        # into the chunked slot (rem_total_tokens = full pool > its admission cost).
        # Then add the ballast: with both reqs live the pool runs out, the chunked
        # resume hits the force-re-add branch, and the engine retracts the ballast
        # (decode-OOM) so the chunked req completes.
        baseline = t.engine_stats()["kv_pool_free"]
        r_chunk = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=600
        )
        yield from run_until(r_chunk, lambda h: h.is_chunking and h.chunks_done >= 1)

        ballast = TestKVPressureSmallPool._start_ballast(t, prompt_token=601)

        # Drive the run to completion of the chunked req. Latch whether the ballast
        # was ever retracted (returned to waiting_queue) -- the observable witness
        # that real retractable pressure was hit and engine-resolved.
        ballast_retracted = False
        for _ in range(2000):
            if ballast.status == "waiting":
                ballast_retracted = True
            if r_chunk.finished:
                break
            yield
        assert r_chunk.finished, (
            f"chunked req must complete once the engine retracts the ballast under "
            f"real KV pressure; status={r_chunk.status}, kv_pages={r_chunk.kv_pages}"
        )
        assert r_chunk.kv_pages == 0, f"kv_pages={r_chunk.kv_pages}"
        assert r_chunk.lock_refs == 0, f"lock_refs={r_chunk.lock_refs}"

        # The ballast was genuinely retracted at some point, or (if it could not be
        # re-admitted) is finished/aborted by the end -- either way it must not still
        # be running and holding the pool.
        ballast_resolved = (
            ballast_retracted
            or ballast.finished
            or ballast.status in ("waiting", "finished", "unknown")
        )
        assert ballast_resolved, (
            f"ballast must be retracted/aborted under pressure; "
            f"status={ballast.status}, retracted={ballast_retracted}"
        )

        # Abort the ballast (it is ignore_eos and would otherwise decode for a long
        # time) and drain to idle so the pool returns to baseline.
        t.abort(ballast)
        for _ in range(200):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"KV pool not recovered after ballast-retract pressure: "
            f"baseline={baseline}, final={final}"
        )

    def test_chunked_completes_under_ballast_then_aborts_chunked(self):
        self.server.execute_script(
            self._script_chunked_completes_under_ballast_then_aborts_chunked
        )

    @staticmethod
    def _script_chunked_completes_under_ballast_then_aborts_chunked(
        t: ScriptedContext,
    ):
        # Abort-under-pressure variant (replaces the raw-exhaust abort test): a
        # chunked req held in flight under real ballast pressure is aborted mid-run
        # and must release its own KV/row/lock even while the ballast still holds
        # the rest of the pool -- no crash, no leak.
        baseline = t.engine_stats()["kv_pool_free"]
        r_chunk = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=610
        )
        yield from run_until(r_chunk, lambda h: h.is_chunking and h.chunks_done >= 1)

        ballast = TestKVPressureSmallPool._start_ballast(t, prompt_token=611)
        # Let the pressure build for a few steps so the chunked req is genuinely
        # contending with the ballast for the capped pool.
        for _ in range(6):
            yield

        t.abort(r_chunk)
        for _ in range(12):
            if (
                r_chunk.kv_pages == 0
                and r_chunk.lock_refs == 0
                and (r_chunk.req is None or r_chunk.req.req_pool_idx is None)
            ):
                break
            yield
        assert r_chunk.kv_pages == 0, f"kv_pages={r_chunk.kv_pages}"
        assert r_chunk.lock_refs == 0, f"lock_refs={r_chunk.lock_refs}"
        assert r_chunk.req is None or r_chunk.req.req_pool_idx is None

        # Clean up the ballast and confirm the pool returns to baseline.
        t.abort(ballast)
        for _ in range(200):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"KV pool not recovered after abort-under-pressure: "
            f"baseline={baseline}, final={final}"
        )

    def test_kv_recovery_after_full(self):
        self.server.execute_script(self._script_kv_recovery_after_full)

    @staticmethod
    def _script_kv_recovery_after_full(t: ScriptedContext):
        baseline = t.engine_stats()["kv_pool_free"]
        # Proven pressure regime (same mechanics as the passing SmallPool
        # tests): one ballast decode req plus an in-flight 2048-token chunked
        # req drive rem_total (free minus clipped decode reservations) to <= 0
        # while the chunked req is mid-chunking. A fresh 16-token req started
        # inside that window cannot admit and parks in waiting_queue.
        b1 = t.start_req(
            prompt_len=SMALL_KV_POOL_BALLAST_PROMPT_LEN,
            max_new_tokens=SMALL_KV_POOL_BALLAST_MAX_NEW_TOKENS,
            ignore_eos=True,
            prompt_token=620,
        )
        yield from run_until(b1, lambda h: h.status == "running")

        big = t.start_req(prompt_len=2048, max_new_tokens=2, prompt_token=621)
        yield from run_until(big, lambda h: h.is_chunking and h.chunks_done >= 2)

        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield
        assert r.status == "waiting", (
            f"16-token req must be unschedulable while the ballast and the "
            f"mid-chunk req own the pool; status={r.status}"
        )

        # Real release: abort the ballast; the chunked req completes (the
        # engine may retract the ballast first on its own -- either way the
        # release path is engine-native) and the newcomer then admits.
        t.abort(b1)
        yield from run_until(big, lambda h: h.finished, max_steps=3000)
        yield from run_until(r, lambda h: h.finished, max_steps=3000)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

        # The finished prompts stay cached in the radix tree; drain to idle and
        # flush before the leak check so the pool returns to baseline.
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"pool must recover to baseline after release: "
            f"baseline={baseline}, final={final}"
        )

class TestKVPressurePriority(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_priority_preempt_multiple_chunked(self):
        self.server.execute_script(self._script_priority_preempt_multiple_chunked)

    @staticmethod
    def _script_priority_preempt_multiple_chunked(t: ScriptedContext):
        baseline = t.engine_stats()["kv_pool_free"]
        # Distinct prompt_token per req: with identical prompts r2 would hit
        # r1's cached prefix and never genuinely chunk, so the preemption under
        # test would not occur.
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=11
        )
        yield from run_until(r1, lambda h: h.is_chunking)

        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            priority=10,
            prompt_token=12,
        )
        # Probe BOTH handles every step (no short-circuit) so the faster req is
        # registered before recycle.
        done = {r1.rid: False, r2.rid: False}
        for _ in range(DEFAULT_MAX_STEPS * 4):
            assert not (r1.is_chunking and r2.is_chunking), (
                f"two reqs cannot share the chunked slot; "
                f"r1.is_chunking={r1.is_chunking}, r2.is_chunking={r2.is_chunking}"
            )
            done[r1.rid] = done[r1.rid] or r1.finished
            done[r2.rid] = done[r2.rid] or r2.finished
            if all(done.values()):
                break
            yield
        assert done[r1.rid] and done[r2.rid]
        assert r1.kv_pages == 0
        assert r2.kv_pages == 0
        # The finished prompts legitimately stay committed in the radix tree
        # (cached != leaked); drain to idle and flush before the leak comparison.
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        final = t.engine_stats()["kv_pool_free"]
        assert final >= baseline, (
            f"KV pool not fully released after preemption: "
            f"baseline={baseline}, final={final}"
        )


class TestKVPressurePageSize(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        page_size=16,
    )

    def test_strict_mem_check_handles_chunked_tail(self):
        self.server.execute_script(self._script_strict_mem_check_handles_chunked_tail)

    @staticmethod
    def _script_strict_mem_check_handles_chunked_tail(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN + 17, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0


if __name__ == "__main__":
    unittest.main()
