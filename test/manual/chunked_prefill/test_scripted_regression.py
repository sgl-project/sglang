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
    warmup_radix,
)


def _drain_until_released(t, *handles):
    for _ in range(12):
        if all(
            h.kv_pages == 0
            and h.lock_refs == 0
            and (h.req is None or h.req.req_pool_idx is None)
            for h in handles
        ):
            return
        yield


class TestRegressionBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_abort_waiting_releases_all(self):
        self.server.execute_script(self._script_abort_waiting_releases_all)

    @staticmethod
    def _script_abort_waiting_releases_all(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        t.abort(r)
        yield from _drain_until_released(t, r)

        assert r.kv_pages == 0
        assert r.req.req_pool_idx is None
        assert r.lock_refs == 0
        assert not r.is_chunking
        assert r.req.inflight_middle_chunks == 0

    def test_pause_covers_waiting_chunked(self):
        self.server.execute_script(self._script_pause_covers_waiting_chunked)

    @staticmethod
    def _script_pause_covers_waiting_chunked(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        t.pause_generation(mode="retract")
        yield

        assert r.kv_pages == 0
        assert r.req.req_pool_idx is None
        assert r.lock_refs == 0
        assert not r.is_chunking

        t.continue_generation()

    def test_inflight_middle_chunks_invariant(self):
        self.server.execute_script(self._script_inflight_middle_chunks_invariant)

    @staticmethod
    def _script_inflight_middle_chunks_invariant(t: ScriptedContext):
        r = t.start_req(
            prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4, ignore_eos=True
        )

        observed_max = 0
        saw_chunking_bump = False
        cleared_inflight = False
        for _ in range(DEFAULT_MAX_STEPS):
            req = r.req
            if req is not None:
                observed_max = max(observed_max, req.inflight_middle_chunks)
                if r.chunks_done >= 1 and r.is_chunking:
                    saw_chunking_bump = saw_chunking_bump or (
                        req.inflight_middle_chunks > 0
                    )
                if not r.is_chunking and r.chunks_done >= 2:
                    cleared_inflight = req.inflight_middle_chunks == 0
                    break
            if r.finished:
                cleared_inflight = True
                break
            yield
        else:
            raise AssertionError("chunk loop did not clear within DEFAULT_MAX_STEPS")

        assert saw_chunking_bump, "last-chunk admit must bump inflight_middle_chunks"
        assert observed_max == 1, (
            f"e875cd36e4: inflight_middle_chunks must be a 0/1 latch; "
            f"observed max={observed_max} (pre-fix bug would bump to 2 "
            f"at the last-chunk admit boundary)"
        )
        assert (
            cleared_inflight
        ), "inflight_middle_chunks should be 0 once the chunk loop clears"

        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    @unittest.skip(
        "dbdcdde245 mamba_pool_idx cleanup-skip is mamba-architecture-specific. "
        "The shared test fixture does not configure a mamba model; running this "
        "regression against the default transformer model would not exercise the "
        "mamba NO_TOKEN cleanup path, so the body would be a pure smoke test "
        "with no real protection. Re-enable when a mamba fixture is wired in."
    )
    def test_mamba_chunked_resume_no_token(self):
        self.server.execute_script(self._script_mamba_chunked_resume_no_token)

    @staticmethod
    def _script_mamba_chunked_resume_no_token(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished

    def test_merge_batch_assert_widened(self):
        self.server.execute_script(self._script_merge_batch_assert_widened)

    @staticmethod
    def _script_merge_batch_assert_widened(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, ignore_eos=True
        )
        r2 = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
        yield from run_until_all_finished([r1, r2])
        assert r1.finished and r2.finished
        assert r1.chunks_done >= 2 and r2.chunks_done == 0
        assert len(r1.req.output_ids) == 2 and len(r2.req.output_ids) == 2

    def test_chunked_pending_tokens_subtract_prefix(self):
        self.server.execute_script(self._script_chunked_pending_tokens_subtract_prefix)

    @staticmethod
    def _script_chunked_pending_tokens_subtract_prefix(t: ScriptedContext):
        r1 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 4, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking and h.chunks_done >= 1)

        r2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(r2, lambda h: h.status == "waiting")

        r1_prefix = len(r1.req.prefix_indices)
        assert r1.is_chunking, "r1 must still be the in-flight chunked req"
        assert r1_prefix > 0, "r1 must hold a committed prefix as the chunked req"

        observed_pending = t.scheduler.load_inquirer._get_num_pending_tokens()

        r1_seqlen = r1.req.seqlen
        r2_seqlen = r2.req.seqlen
        expected_post_fix = (r1_seqlen - r1_prefix) + r2_seqlen
        pre_fix_bad = r1_seqlen + r2_seqlen

        assert observed_pending == expected_post_fix, (
            f"c79a73bec4: load_inquirer must subtract the chunked req's "
            f"prefix_indices_len from its pending-token contribution; "
            f"observed={observed_pending}, expected_post_fix="
            f"{expected_post_fix}, pre_fix_bad={pre_fix_bad}"
        )
        assert observed_pending < pre_fix_bad, (
            f"c79a73bec4: observed pending tokens did not subtract the chunked "
            f"prefix (matches the no-subtraction sum {pre_fix_bad}) — fix is regressed"
        )

        yield from run_until_all_finished([r1, r2])
        assert r1.finished and r2.finished

    def test_chunked_admission_reuse_branch_balanced(self):
        self.server.execute_script(self._script_chunked_admission_reuse_branch_balanced)

    @staticmethod
    def _script_chunked_admission_reuse_branch_balanced(t: ScriptedContext):
        baseline_refs = sum(t.get_all_node_lock_refs().values())
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)

        for target_chunk in (3, 4, 5):
            yield from run_until(
                r, lambda h: h.chunks_done >= target_chunk and h.is_chunking
            )
            assert r.lock_refs == 1, (
                f"reuse branch re-acquired lock_ref on the chunked req's node at "
                f"chunk {target_chunk}: r.lock_refs={r.lock_refs} (expected 1)"
            )
            per_node = t.get_all_node_lock_refs()
            assert max(per_node.values(), default=0) <= 1, (
                f"reuse branch double-locked a single node at chunk "
                f"{target_chunk}: per-node lock_refs={per_node}"
            )

        yield from run_until_finished(r)
        final_refs = sum(t.get_all_node_lock_refs().values())
        assert final_refs == baseline_refs, (
            f"chunked lifecycle must net to zero lock_ref delta; "
            f"baseline={baseline_refs}, final={final_refs}"
        )
        assert r.lock_refs == 0

    def test_multiturn_full_hit_no_reuse_branch(self):
        self.server.execute_script(self._script_multiturn_full_hit_no_reuse_branch)

    @staticmethod
    def _script_multiturn_full_hit_no_reuse_branch(t: ScriptedContext):
        baseline_refs = sum(t.get_all_node_lock_refs().values())
        r1 = t.start_req(prompt_len=64, max_new_tokens=2)
        yield from run_until_finished(r1)

        r2 = t.start_req(prompt_len=64, max_new_tokens=2)
        yield from run_until_finished(r2)

        assert not r2.is_chunking
        assert r2.req.cached_tokens > 0, (
            f"follow-up req must fully hit the warm prefix to exercise the "
            f"no-reuse path; got cached_tokens={r2.req.cached_tokens}"
        )
        for _ in range(5):
            yield
        assert sum(t.get_all_node_lock_refs().values()) == baseline_refs, (
            f"full-prefix-hit follow-up must not take the chunked reuse "
            f"branch; lock_refs drifted from {baseline_refs} to "
            f"{sum(t.get_all_node_lock_refs().values())}"
        )

    def test_abort_chunked_resume_releases_all_resources(self):
        self.server.execute_script(
            self._script_abort_chunked_resume_releases_all_resources
        )

    @staticmethod
    def _script_abort_chunked_resume_releases_all_resources(t: ScriptedContext):
        baseline_refs = sum(t.get_all_node_lock_refs().values())
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        assert r.req.req_pool_idx is not None, "row must be held mid-chunk"
        assert r.kv_pages > 0, "committed KV must be held mid-chunk"
        assert r.lock_refs >= 1, "radix lock_ref must be held mid-chunk"

        t.abort(r)
        yield from _drain_until_released(t, r)

        assert (
            r.req.req_pool_idx is None
        ), f"96d4749094: abort must release row; got row_idx={r.req.req_pool_idx!r}"
        assert (
            r.kv_pages == 0
        ), f"96d4749094: abort must release KV; got kv_pages={r.kv_pages}"
        assert (
            r.lock_refs == 0
        ), f"96d4749094: abort must release lock_ref; got lock_refs={r.lock_refs}"
        assert not r.is_chunking
        assert r.req.inflight_middle_chunks == 0
        assert sum(t.get_all_node_lock_refs().values()) == baseline_refs

    def test_pause_retract_releases_waiting_chunked_resume(self):
        self.server.execute_script(
            self._script_pause_retract_releases_waiting_chunked_resume
        )

    @staticmethod
    def _script_pause_retract_releases_waiting_chunked_resume(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        assert r.req.req_pool_idx is not None and r.kv_pages > 0 and r.lock_refs >= 1

        t.pause_generation(mode="retract")
        yield

        assert r.req.req_pool_idx is None, (
            f"f38e69f87d: pause(retract) must release waiting "
            f"chunked-resume row; got row_idx={r.req.req_pool_idx!r}"
        )
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert not r.is_chunking
        assert r.status == "waiting", (
            f"f38e69f87d: pause(retract) must re-queue the retracted "
            f"chunked-resume req; got status={r.status!r}"
        )
        assert t.scheduler.chunked_req is None
        assert t.scheduler.running_batch.is_empty()

        t.continue_generation()
        yield from run_until_finished(r)
        assert r.finished
        assert len(r.req.output_ids) == 2

    def test_retract_all_clears_batch_with_chunked(self):
        self.server.execute_script(self._script_retract_all_clears_batch_with_chunked)

    @staticmethod
    def _script_retract_all_clears_batch_with_chunked(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking and h.chunks_done >= 1)
        yield from run_until(r2, lambda h: h.status == "waiting")

        t.pause_generation(mode="retract")
        yield

        assert len(t.scheduler.running_batch.reqs) == 0, (
            f"f0388931bf: retract_all must clear batch; got batch_size="
            f"{len(t.scheduler.running_batch.reqs)}"
        )
        assert (
            t.scheduler.chunked_req.rid if t.scheduler.chunked_req is not None else None
        ) is None
        for r in (r1, r2):
            assert r.status == "waiting"
            assert r.kv_pages == 0
            assert not r.is_chunking

        t.continue_generation()
        yield from run_until_all_finished([r1, r2])


class TestRegressionPp(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        tp_size=2,
        pp_size=2,
    )

    def test_pp_abort_dedup(self):
        self.server.execute_script(self._script_pp_abort_dedup)

    @staticmethod
    def _script_pp_abort_dedup(t: ScriptedContext):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)

        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)

        t.abort(r)
        yield

        rids_after_abort = [req.rid for req in t.scheduler.running_batch.reqs]
        occurrences = sum(1 for rid in rids_after_abort if rid == r.rid)
        assert occurrences <= 1, (
            f"b823c16e60: batch_rids must dedup across mbs + "
            f"waiting_queue; got {occurrences} occurrences of rid="
            f"{r.rid} (pre-fix bug would yield 3)"
        )
        yield from _drain_until_released(t, r)
        assert r.finished


class TestRegressionPriority(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_priority_skips_chunked_in_prefix_match(self):
        self.server.execute_script(self._script_priority_skips_chunked_in_prefix_match)

    @staticmethod
    def _script_priority_skips_chunked_in_prefix_match(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            priority=0,
            ignore_eos=True,
        )
        yield from run_until(r1, lambda h: h.is_chunking and h.chunks_done >= 1)
        r1_host_hit_before = r1.req.host_hit_length

        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            priority=10,
            ignore_eos=True,
        )
        prev_chunks = r1.chunks_done
        while not (r1.finished and r2.finished):
            assert r1.req is None or r1.req.host_hit_length == r1_host_hit_before, (
                f"aaf3752d2b: priority calc re-matched the chunked-resume req; "
                f"host_hit_length changed {r1_host_hit_before} -> "
                f"{r1.req.host_hit_length}"
            )
            assert r1.chunks_done >= prev_chunks, (
                f"r1 chunked prefill was preempted by higher-priority r2: "
                f"chunks_done regressed {prev_chunks} -> {r1.chunks_done}"
            )
            prev_chunks = r1.chunks_done
            yield

        assert r1.finished and r2.finished
        assert r1.chunks_done >= 2
        assert len(r1.req.output_ids) == 2 and len(r2.req.output_ids) == 2


class TestRegressionLpm(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        schedule_policy="lpm",
    )

    def test_chunked_resume_priority_in_sort(self):
        self.server.execute_script(self._script_chunked_resume_priority_in_sort)

    @staticmethod
    def _script_chunked_resume_priority_in_sort(t: ScriptedContext):
        r_long = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_long, lambda h: h.is_chunking)

        shorts = [t.start_req(prompt_len=4, max_new_tokens=1) for _ in range(8)]

        initial_chunks = r_long.chunks_done
        for _ in range(200):
            if r_long.chunks_done > initial_chunks:
                break
            yield
        else:
            raise AssertionError(
                f"chunked req starved by short req flood; "
                f"chunks_done stuck at {initial_chunks}"
            )

        all_reqs = [r_long, *shorts]
        done = {r.rid: False for r in all_reqs}
        for _ in range(DEFAULT_MAX_STEPS):
            for r in all_reqs:
                done[r.rid] = done[r.rid] or r.finished
            if all(done.values()):
                break
            yield
        else:
            raise AssertionError(
                f"reqs did not all finish; done={done}, "
                f"long_chunks={r_long.chunks_done}"
            )
        assert r_long.finished
        assert r_long.chunks_done >= 2

    def test_lpm_skips_chunked_resume_prefix_match(self):
        self.server.execute_script(self._script_lpm_skips_chunked_resume_prefix_match)

    @staticmethod
    def _script_lpm_skips_chunked_resume_prefix_match(t: ScriptedContext):
        yield from warmup_radix(t, [1] * (2 * DEFAULT_CHUNK_SIZE))

        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.chunks_done >= 1 and h.is_chunking)

        host_hit_before = r1.req.host_hit_length

        r2 = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        while not (r1.finished and r2.finished):
            assert r1.req is None or r1.req.host_hit_length == host_hit_before, (
                f"calc_priority re-matched the chunked-resume req; host_hit_length "
                f"changed {host_hit_before} -> {r1.req.host_hit_length}"
            )
            yield

    def test_chunked_resume_priority_under_lpm(self):
        self.server.execute_script(self._script_chunked_resume_priority_under_lpm)

    @staticmethod
    def _script_chunked_resume_priority_under_lpm(t: ScriptedContext):
        long_prefix_tokens = [1] * (3 * DEFAULT_CHUNK_SIZE)
        yield from warmup_radix(t, long_prefix_tokens)

        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking and h.chunks_done >= 1)

        competitors = [
            t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE + 32, max_new_tokens=2)
            for _ in range(6)
        ]

        baseline_chunks = r1.chunks_done
        for _ in range(50):
            if r1.chunks_done > baseline_chunks:
                break
            yield
        else:
            raise AssertionError(
                f"bf5b4e9a10: chunked-resume starved under LPM by "
                f"long-prefix competitors; chunks_done stuck at "
                f"{baseline_chunks}"
            )

        yield from run_until_all_finished([r1, *competitors])


class TestRegressionGptOss(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        model_path="openai/gpt-oss-20b",
        mem_fraction_static=0.70,
        disable_piecewise_cuda_graph=True,
    )

    def test_chunked_stash_bounded_by_kv_committed_len(self):
        self.server.execute_script(
            self._script_chunked_stash_bounded_by_kv_committed_len
        )

    @staticmethod
    def _script_chunked_stash_bounded_by_kv_committed_len(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        committed = r.req.kv_committed_len
        assert committed > 0

        assert len(r.req.prefix_indices) <= committed, (
            f"cache_unfinished_req over-read past kv_committed_len: "
            f"prefix_indices_len={len(r.req.prefix_indices)}, "
            f"kv_committed_len={committed}"
        )
        yield from run_until_finished(r)


if __name__ == "__main__":
    unittest.main()
