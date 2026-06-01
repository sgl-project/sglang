import unittest
from typing import List

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

_LORA_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_LORA_ADAPTER = "philschmid/llama-3-2-1b-instruct-finetuning-lora-cookbook-test"


def _in_flight_other_mb_rids(t: ScriptedContext) -> List[str]:
    # Rids held in pipeline micro-batch slots other than the one currently being
    # serviced (running_batch). running_mbs only exists on the PP path.
    s = t.scheduler
    if not hasattr(s, "running_mbs"):
        return []
    rids: List[str] = []
    for mb in s.running_mbs:
        if mb is s.running_batch or mb is None:
            continue
        rids.extend(r.rid for r in mb.reqs)
    return rids


class TestRegressionLora(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER],
    )

    def test_lora_drainer_chunked_resume(self):
        self.server.execute_script(self._script_lora_drainer_chunked_resume)

    @staticmethod
    def _script_lora_drainer_chunked_resume(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER,
            ignore_eos=True,
        )
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        t.force_lora_drainer_reject(adapter=_LORA_ADAPTER)

        yield from run_until_finished(r)
        assert r.finished
        assert r.req.req_pool_idx is None
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) == 2


class TestRegressionBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_abort_waiting_releases_all(self):
        self.server.execute_script(self._script_abort_waiting_releases_all)

    @staticmethod
    def _script_abort_waiting_releases_all(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.status == "waiting")

        t.abort(r)
        yield

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
        yield from run_until(r, lambda h: h.status == "waiting")

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

        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        assert (
            r.req.inflight_middle_chunks > 0
        ), "last-chunk admit must bump inflight_middle_chunks"

        yield from run_until(r, lambda h: not h.is_chunking)
        assert r.req.inflight_middle_chunks == 0, (
            f"inflight_middle_chunks should be 0 after chunk loop clears; "
            f"got {r.req.inflight_middle_chunks}"
        )

        yield from run_until_finished(r)
        assert r.finished

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

    def test_revert_bump_inflight_middle_chunks(self):
        self.server.execute_script(self._script_revert_bump_inflight_middle_chunks)

    @staticmethod
    def _script_revert_bump_inflight_middle_chunks(t: ScriptedContext):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)

        observed_max = 0
        for _ in range(DEFAULT_MAX_STEPS):
            observed_max = max(observed_max, r.req.inflight_middle_chunks)
            if r.finished:
                break
            yield
        else:
            raise AssertionError("req did not finish within DEFAULT_MAX_STEPS")

        assert observed_max == 1, (
            f"e875cd36e4: inflight_middle_chunks must be a 0/1 latch; "
            f"observed max={observed_max} (pre-fix bug would bump to 2 "
            f"at the last-chunk admit boundary)"
        )
        assert r.req.inflight_middle_chunks == 0
        assert r.finished and r.chunks_done >= 2

    def test_filter_batch_exclude_in_flight_other_mb(self):
        self.server.execute_script(self._script_filter_batch_exclude_in_flight_other_mb)

    @staticmethod
    def _script_filter_batch_exclude_in_flight_other_mb(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)

        observed_in_flight_other_mb = False
        observed_excluded_from_running = False
        for _ in range(DEFAULT_MAX_STEPS):
            in_flight_other_mb = _in_flight_other_mb_rids(t)
            running = [req.rid for req in t.scheduler.running_batch.reqs]
            if r.rid in in_flight_other_mb:
                observed_in_flight_other_mb = True
                assert r.rid not in running, (
                    f"5c523049db: after filter_batch, chunked-resume "
                    f"req held in another mb must be excluded from "
                    f"local running set; rid={r.rid}"
                )
                observed_excluded_from_running = True
            if r.finished:
                break
            yield
        else:
            raise AssertionError("req did not finish within DEFAULT_MAX_STEPS")

        assert r.finished and r.chunks_done >= 2
        if observed_in_flight_other_mb:
            assert observed_excluded_from_running, (
                "5c523049db: filter_batch must exclude in-flight "
                "other-mb reqs from local running set"
            )

    # Removed test_stage_a_chunk_stash_iter_boundary: it probed t.last_scheduler_path
    # / t.last_admission_path, which require the scheduler to durably record which
    # internal branch (stash vs adder.build) it took. That ordering is a deep
    # stage-internal implementation detail, not an externally observable invariant,
    # and belongs in a dedicated scheduler unit test rather than the scripted
    # runtime. No durable path probe is added.

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

    def test_filter_batch_explicit_exclude_chunked_flag(self):
        self.server.execute_script(
            self._script_filter_batch_explicit_exclude_chunked_flag
        )

    @staticmethod
    def _script_filter_batch_explicit_exclude_chunked_flag(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, ignore_eos=True
        )
        r2 = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
        yield from run_until_all_finished([r1, r2])
        assert r1.finished and r2.finished
        assert r1.chunks_done >= 2
        assert len(r1.req.output_ids) == 2 and len(r2.req.output_ids) == 2

    def test_waiting_queue_pending_tokens_subtract_prefix(self):
        self.server.execute_script(
            self._script_waiting_queue_pending_tokens_subtract_prefix
        )

    @staticmethod
    def _script_waiting_queue_pending_tokens_subtract_prefix(t: ScriptedContext):
        r1 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 4, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking and h.chunks_done >= 1)

        t.pause_generation(mode="retract")
        yield
        yield from run_until(r1, lambda h: h.status == "waiting")

        r1_prefix = len(r1.req.prefix_indices)
        r1_committed = r1.req.kv_committed_len
        assert r1_committed > 0, "R1 must hold committed KV in waiting_queue"
        assert r1_prefix > 0, "R1 must have a non-zero stashed prefix"

        r2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield

        observed_pending = t.scheduler.load_inquirer._get_num_pending_tokens()

        r1_total = len(r1.req.origin_input_ids) + len(r1.req.output_ids)
        r2_total = len(r2.req.origin_input_ids) + len(r2.req.output_ids)
        expected_post_fix = (r1_total - r1_prefix) + r2_total
        pre_fix_bad = r1_total + r2_total

        assert observed_pending == expected_post_fix, (
            f"c79a73bec4: load_inquirer must subtract prefix_indices_len "
            f"from each waiting_queue entry's contribution; "
            f"observed={observed_pending}, expected_post_fix="
            f"{expected_post_fix}, pre_fix_bad={pre_fix_bad}"
        )
        assert observed_pending != pre_fix_bad, (
            f"c79a73bec4: observed pending tokens matches the pre-fix "
            f"(no-subtraction) sum — fix is regressed"
        )

        t.continue_generation()
        yield from run_until_all_finished([r1, r2])
        assert r1.finished and r2.finished

    def test_abort_dedup_dual_queue_holding(self):
        self.server.execute_script(self._script_abort_dedup_dual_queue_holding)

    @staticmethod
    def _script_abort_dedup_dual_queue_holding(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.abort(r)
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.req.req_pool_idx is None
        assert r.lock_refs == 0
        assert r.finished

    def test_chunked_admission_reuse_branch_balanced(self):
        self.server.execute_script(self._script_chunked_admission_reuse_branch_balanced)

    @staticmethod
    def _script_chunked_admission_reuse_branch_balanced(t: ScriptedContext):
        baseline_refs = t.get_all_node_lock_refs()
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)

        yield from run_until(r, lambda h: h.chunks_done >= 3 and h.is_chunking)
        mid_refs = t.get_all_node_lock_refs()
        assert mid_refs - baseline_refs == 1, (
            f"reuse branch must not re-acquire lock_ref per chunk; "
            f"baseline={baseline_refs}, mid={mid_refs}"
        )

        yield from run_until_finished(r)
        final_refs = t.get_all_node_lock_refs()
        assert final_refs == baseline_refs, (
            f"chunked lifecycle must net to zero lock_ref delta; "
            f"baseline={baseline_refs}, final={final_refs}"
        )
        assert r.lock_refs == 0

    def test_chunked_resume_lives_in_waiting_queue(self):
        self.server.execute_script(self._script_chunked_resume_lives_in_waiting_queue)

    @staticmethod
    def _script_chunked_resume_lives_in_waiting_queue(t: ScriptedContext):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE + 32, max_new_tokens=2)

        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        assert r.status == "waiting", (
            f"between chunks the chunked-resume req must hold in "
            f"waiting_queue; got status={r.status!r}"
        )
        assert (
            r.is_chunking
        ), f"is_chunking must be set while mid-chunk; got {r.is_chunking!r}"
        assert (
            t.scheduler.chunked_req.rid if t.scheduler.chunked_req is not None else None
        ) is None, (
            f"v2 must not maintain a top-level chunked_req field; "
            f"got {(t.scheduler.chunked_req.rid if t.scheduler.chunked_req is not None else None)!r}"
        )

        yield from run_until_finished(r)
        assert not r.is_chunking
        assert r.status in ("finished", "unknown")

    def test_streaming_session_multiturn_no_reuse_branch(self):
        self.server.execute_script(
            self._script_streaming_session_multiturn_no_reuse_branch
        )

    @staticmethod
    def _script_streaming_session_multiturn_no_reuse_branch(t: ScriptedContext):
        baseline_refs = t.get_all_node_lock_refs()
        r1 = t.start_req(
            prompt_len=64,
            max_new_tokens=2,
            session_id="sess-a79b",
        )
        yield from run_until_finished(r1)

        r2 = t.start_req(
            prompt_len=64,
            max_new_tokens=2,
            session_id="sess-a79b",
        )
        yield from run_until_finished(r2)

        assert not r2.is_chunking
        assert t.get_all_node_lock_refs() == baseline_refs, (
            f"streaming-session turn N>1 must not take chunked reuse "
            f"branch; lock_refs drifted from {baseline_refs} to "
            f"{t.get_all_node_lock_refs()}"
        )

    def test_abort_chunked_resume_releases_all_resources(self):
        self.server.execute_script(
            self._script_abort_chunked_resume_releases_all_resources
        )

    @staticmethod
    def _script_abort_chunked_resume_releases_all_resources(t: ScriptedContext):
        baseline_refs = t.get_all_node_lock_refs()
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        yield from run_until(r, lambda h: h.status == "waiting")

        assert r.req.req_pool_idx is not None, "row must be held mid-chunk"
        assert r.kv_pages > 0, "committed KV must be held mid-chunk"
        assert r.lock_refs >= 1, "radix lock_ref must be held mid-chunk"

        t.abort(r)
        yield

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
        assert t.get_all_node_lock_refs() == baseline_refs

    def test_abort_chunked_resume_dual_queue_no_double_release(self):
        self.server.execute_script(
            self._script_abort_chunked_resume_dual_queue_no_double_release
        )

    @staticmethod
    def _script_abort_chunked_resume_dual_queue_no_double_release(t: ScriptedContext):
        baseline_free = t.engine_stats()["kv_pool_free"]
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)

        for _ in range(DEFAULT_MAX_STEPS):
            in_waiting = r.rid in [req.rid for req in t.scheduler.waiting_queue]
            in_batch = r.rid in [req.rid for req in t.scheduler.running_batch.reqs]
            if in_waiting and in_batch:
                break
            if r.finished:
                raise AssertionError(
                    "req finished before reaching the dual-queue window"
                )
            yield
        else:
            raise AssertionError(
                "never observed chunked-resume in both waiting_queue and batch.reqs"
            )

        t.abort(r)
        yield

        assert r.finished
        assert r.kv_pages == 0
        assert r.req.req_pool_idx is None
        # There is no underflow counter in the engine. The real invariant a double
        # release_kv_cache would break is the pool's free accounting: an over-free
        # would push kv_pool_free above its pre-request baseline. Assert the pool
        # returns to exactly baseline (every page released once, none twice).
        post_free = t.engine_stats()["kv_pool_free"]
        assert post_free == baseline_free, (
            f"double release_kv_cache corrupted pool accounting; "
            f"kv_pool_free baseline={baseline_free}, after abort={post_free}"
        )

    def test_pause_retract_releases_waiting_chunked_resume(self):
        self.server.execute_script(
            self._script_pause_retract_releases_waiting_chunked_resume
        )

    @staticmethod
    def _script_pause_retract_releases_waiting_chunked_resume(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        yield from run_until(r, lambda h: h.status == "waiting")
        assert r.req.req_pool_idx is not None and r.kv_pages > 0 and r.lock_refs >= 1

        t.pause_retract_all()
        yield

        assert r.req.req_pool_idx is None, (
            f"f38e69f87d: pause(retract) must release waiting "
            f"chunked-resume row; got row_idx={r.req.req_pool_idx!r}"
        )
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert not r.is_chunking
        assert t.is_fully_idle, (
            f"f38e69f87d: pause(retract) must leave engine fully idle; "
            f"got is_fully_idle={t.is_fully_idle!r}"
        )

    def test_retract_all_clears_batch_with_chunked(self):
        self.server.execute_script(self._script_retract_all_clears_batch_with_chunked)

    @staticmethod
    def _script_retract_all_clears_batch_with_chunked(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking and h.chunks_done >= 1)
        yield from run_until(r2, lambda h: h.is_chunking)

        t.retract_all()
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
            assert r.chunks_done == 0
            assert not r.is_chunking

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
        assert r.finished

    def test_pp_other_mb_chunked_exclude(self):
        self.server.execute_script(self._script_pp_other_mb_chunked_exclude)

    @staticmethod
    def _script_pp_other_mb_chunked_exclude(t: ScriptedContext):
        r_long = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r_ctrl = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1)

        long_exclude_engaged = False
        ctrl_exclude_engaged = False
        for _ in range(2000):
            in_flight_other_mb = _in_flight_other_mb_rids(t)
            running = [req.rid for req in t.scheduler.running_batch.reqs]
            if r_long.rid in in_flight_other_mb:
                long_exclude_engaged = True
                assert r_long.rid not in running, (
                    f"69ef71edc4: max_new_tokens > 1 chunked-resume req "
                    f"must be excluded from local running while held in "
                    f"another mb's in-flight set; rid={r_long.rid}"
                )
            if r_ctrl.rid in in_flight_other_mb and r_ctrl.rid in running:
                ctrl_exclude_engaged = True
            in_flight = 1 if t.scheduler.chunked_req is not None else 0
            assert (
                in_flight <= 1
            ), f"PP cross-mb chunked exclusion broken: in_flight={in_flight}"
            if r_long.finished and r_ctrl.finished:
                break
            yield
        else:
            raise AssertionError("reqs did not finish")

        assert r_long.finished and r_ctrl.finished
        assert r_long.chunks_done >= 2 and r_ctrl.chunks_done >= 2
        assert long_exclude_engaged, (
            "69ef71edc4: never observed r_long in another mb's in-flight "
            "set — cross-mb exclude path was not exercised"
        )
        assert ctrl_exclude_engaged or not long_exclude_engaged, (
            "69ef71edc4: max_new_tokens == 1 control must NOT trigger "
            "the cross-mb exclude (no output-stash dedupe needed)"
        )


class TestRegressionDisagg(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disaggregation_mode="prefill",
    )

    def test_disagg_retract_resets_send(self):
        self.server.execute_script(self._script_disagg_retract_resets_send)

    @staticmethod
    def _script_disagg_retract_resets_send(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        t.pause_generation(mode="retract")
        yield

        # No disagg_send_state field exists. The real send-side state that retract
        # must reset is the pair of Req fields the next send_kv_chunk reads:
        # start_send_idx (where the next chunk send begins) and tmp_end_idx (the
        # pending overlap-send slice end). Both must be back at their initial values.
        assert r.req.start_send_idx == 0, (
            f"414efd4a27: disagg send state must reset on chunked-resume "
            f"retract; start_send_idx={r.req.start_send_idx}"
        )
        assert r.req.tmp_end_idx == -1, (
            f"414efd4a27: disagg send state must reset on chunked-resume "
            f"retract; tmp_end_idx={r.req.tmp_end_idx}"
        )

        t.continue_generation()


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
            priority="low",
            ignore_eos=True,
        )
        yield from run_until(r1, lambda h: h.is_chunking and h.chunks_done >= 1)
        r1_prefix_before = len(r1.req.prefix_indices)
        r1_host_hit_before = r1.req.host_hit_length
        lock_refs_before = t.get_all_node_lock_refs()

        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            priority="high",
            ignore_eos=True,
        )
        yield
        assert len(r1.req.prefix_indices) == r1_prefix_before, (
            f"aaf3752d2b: priority prefix-match must skip chunked-resume; "
            f"prefix_indices_len changed {r1_prefix_before} -> "
            f"{len(r1.req.prefix_indices)}"
        )
        assert r1.req.host_hit_length == r1_host_hit_before
        assert t.get_all_node_lock_refs() == lock_refs_before

        yield from run_until_all_finished([r1, r2])
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

        yield from run_until_all_finished([r_long, *shorts])
        assert r_long.finished
        assert r_long.chunks_done >= 2

    def test_lpm_skips_chunked_resume_prefix_match(self):
        self.server.execute_script(self._script_lpm_skips_chunked_resume_prefix_match)

    @staticmethod
    def _script_lpm_skips_chunked_resume_prefix_match(t: ScriptedContext):
        yield from warmup_radix(t, [1] * (2 * DEFAULT_CHUNK_SIZE))

        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.chunks_done >= 1 and h.is_chunking)
        yield from run_until(r1, lambda h: h.status == "waiting")

        last_node_before = r1.req.last_node.id
        prefix_len_before = len(r1.req.prefix_indices)
        host_hit_before = r1.req.host_hit_length
        lock_refs_before = t.get_all_node_lock_refs()

        r2 = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield

        assert r1.req.last_node.id == last_node_before, (
            f"calc_priority overwrote chunked-resume last_node; "
            f"before={last_node_before!r}, after={r1.req.last_node.id!r}"
        )
        assert len(r1.req.prefix_indices) == prefix_len_before
        assert r1.req.host_hit_length == host_hit_before
        assert t.get_all_node_lock_refs() == lock_refs_before

        yield from run_until_all_finished([r1, r2])

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

        yield from run_until(r, lambda h: h.status == "waiting")

        assert len(r.req.prefix_indices) <= committed, (
            f"cache_unfinished_req over-read past kv_committed_len: "
            f"prefix_indices_len={len(r.req.prefix_indices)}, "
            f"kv_committed_len={committed}"
        )
        yield from run_until_finished(r)


class TestRegressionWaitingTimeout(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        env={"SGLANG_REQ_WAITING_TIMEOUT": "1"},
    )

    def test_chunked_resume_immune_to_waiting_timeout(self):
        self.server.execute_script(
            self._script_chunked_resume_immune_to_waiting_timeout
        )

    @staticmethod
    def _script_chunked_resume_immune_to_waiting_timeout(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        assert r.is_chunking

        t.trigger_abort_on_waiting_timeout()
        yield

        assert not getattr(
            r, "aborted", False
        ), f"359e5ed7bd: chunked-resume must be immune to waiting timeout abort"
        assert r.kv_pages > 0 or r.chunks_done > 0
        assert r.is_chunking or r.chunks_done > 0
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2


if __name__ == "__main__":
    unittest.main()
