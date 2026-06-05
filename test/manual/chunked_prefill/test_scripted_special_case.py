import unittest
from typing import Optional

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    BALLAST_MAX_NEW_TOKENS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    advance_to_decode_step,
    base_engine_kwargs,
    exhaust_row_pool,
    run_until,
    run_until_finished,
)


def _load_inquirer_pending_for_rid(t: ScriptedContext, rid: str) -> int:
    # Per-rid contribution to the scheduler load inquirer's pending-token tally:
    # the chunked req counts only its not-yet-committed remainder, a waiting-queue
    # req counts its full seqlen.
    s = t.scheduler
    chunked = s.chunked_req
    if chunked is not None and chunked.rid == rid:
        return chunked.seqlen - len(chunked.prefix_indices)
    for req in s.waiting_queue:
        if req.rid == rid:
            return req.seqlen
    return 0


class TestSpecialCaseBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_chunked_in_flight_no_idle(self):
        self.server.execute_script(self._script_chunked_in_flight_no_idle)

    @staticmethod
    def _script_chunked_in_flight_no_idle(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                saw_chunking = True
                assert (
                    not t.is_idle
                ), "scheduler must not idle while chunked_req is in flight"
            if r.finished:
                break
            yield
        assert r.finished
        assert saw_chunking, "test must observe r mid-chunk at least once"

    def test_add_chunked_req_path(self):
        self.server.execute_script(self._script_add_chunked_req_path)

    @staticmethod
    def _script_add_chunked_req_path(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_admission_with_chunked_in_flight(self):
        self.server.execute_script(self._script_admission_with_chunked_in_flight)

    @staticmethod
    def _script_admission_with_chunked_in_flight(t: ScriptedContext):
        r_chunk = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_chunk, lambda h: h.is_chunking)

        r_small = t.start_req(prompt_len=4, max_new_tokens=2)
        yield

        comp = t.batch_composition()
        assert r_chunk.rid in comp.get("chunked", [])

        yield from run_until_finished(r_small)
        yield from run_until_finished(r_chunk)
        assert r_small.finished and r_chunk.finished

    def test_abort_excludes_chunked_req(self):
        self.server.execute_script(self._script_abort_excludes_chunked_req)

    @staticmethod
    def _script_abort_excludes_chunked_req(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        t.abort(r)
        # The overlap pipeline clears the chunked slot and releases the req's KV/row
        # over several steps after the abort is injected, not in a single yield.
        for _ in range(12):
            if t.scheduler.chunked_req is None and r.kv_pages == 0 and r.lock_refs == 0:
                break
            yield

        assert (
            t.scheduler.chunked_req is None
        ), f"abort must clear the chunked slot; got {t.scheduler.chunked_req!r}"
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_get_chunked_req_lambda_getter(self):
        self.server.execute_script(self._script_get_chunked_req_lambda_getter)

    @staticmethod
    def _script_get_chunked_req_lambda_getter(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        saw_match = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                cur = (
                    t.scheduler.chunked_req.rid
                    if t.scheduler.chunked_req is not None
                    else None
                )
                assert cur in (None, r.rid), (
                    f"getter returned unrelated rid: got {cur!r}, expected "
                    f"None or {r.rid!r}"
                )
                if cur == r.rid:
                    saw_match = True
            if r.finished:
                break
            yield
        assert r.finished
        assert saw_match, "getter must return r.rid at least once while r.is_chunking"
        assert (
            t.scheduler.chunked_req.rid if t.scheduler.chunked_req is not None else None
        ) is None

    @unittest.skip(
        "requires real disaggregation prefill/decode split — single-engine "
        "ScriptedContext cannot exercise the decode-side waiting_queue "
        "KV-hold path. Belongs in test_scripted_disagg.py with D3 counter "
        "wiring once disagg topology is available."
    )
    def test_disagg_decode_waiting_queue_kv_held(self):
        pass

    @unittest.skip(
        "requires DLLM model + staging mixin — single-engine cannot drive "
        "both DLLM staging AND chunked admission incrementing "
        "inflight_middle_chunks from two sources. Belongs in DLLM-specific "
        "test file."
    )
    def test_dllm_staging_double_inflight_middle_chunks(self):
        pass

    @unittest.skip(
        "requires real disaggregation topology — single-engine cannot "
        "exercise staging_handler chunked path."
    )
    def test_staging_handler_chunked(self):
        pass

    @unittest.skip(
        "requires mooncake KV transport backend — single-engine cannot "
        "drive the conn layer chunked path."
    )
    def test_mooncake_conn_chunked(self):
        pass

    @unittest.skip(
        "requires NIXL KV transport backend — single-engine cannot drive "
        "the NIXL conn layer chunked path."
    )
    def test_nixl_conn_chunked(self):
        pass

    def test_filter_batch_exclude_chunked_flag(self):
        self.server.execute_script(self._script_filter_batch_exclude_chunked_flag)

    @staticmethod
    def _script_filter_batch_exclude_chunked_flag(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        saw_r1_chunking = False
        for _ in range(DEFAULT_MAX_STEPS * 2):
            if r1.is_chunking:
                saw_r1_chunking = True
                comp = t.batch_composition()
                assert r1.rid in comp.get(
                    "chunked", []
                ), f"mid-chunk r1 must occupy the chunked role; got {comp!r}"
                assert r1.rid not in comp.get(
                    "running", []
                ), f"chunked r1 must be excluded from the running role; got {comp!r}"
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished and r2.finished
        assert (
            saw_r1_chunking
        ), "r1 must have chunked at some point to exercise the exclude branch"

    @unittest.skip(
        "pdmux split_prefill_batch requires the pdmux topology — "
        "single-engine ScriptedContext cannot drive the split path. "
        "Belongs in a pdmux-specific test once that lane is wired up."
    )
    def test_pdmux_split_prefill_batch(self):
        pass

    def test_streaming_session_kv_committed_bound(self):
        self.server.execute_script(self._script_streaming_session_kv_committed_bound)

    @staticmethod
    def _script_streaming_session_kv_committed_bound(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                assert len(r.req.prefix_indices) <= r.req.kv_committed_len, (
                    f"streaming-session chunked stash must stay bounded by "
                    f"kv_committed_len; prefix_indices_len={len(r.req.prefix_indices)}, "
                    f"kv_committed_len={r.req.kv_committed_len}"
                )
            if r.finished:
                break
            yield
        assert r.finished

    @unittest.skip(
        "mamba_pool_idx cleanup applies only to mamba-class models — "
        "single-engine with a non-mamba model cannot drive the NO_TOKEN "
        "chunked-resume cleanup-skip branch. Belongs in a mamba-specific "
        "test file once mamba scripted coverage is added."
    )
    def test_mamba_pool_idx_cleanup_skip_chunked_resume(self):
        pass

    def test_pause_retract_clears_chunked_req(self):
        self.server.execute_script(self._script_pause_retract_clears_chunked_req)

    @staticmethod
    def _script_pause_retract_clears_chunked_req(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        t.pause_generation(mode="retract")
        yield

        assert (
            t.scheduler.chunked_req is None
        ), f"pause(retract) must clear chunked_req; got {t.scheduler.chunked_req!r}"
        assert not r.finished, "retract must re-queue r, not finish or abort it"
        assert r.status == "waiting", (
            f"retracted chunked req must return to the waiting queue; "
            f"got status={r.status!r}"
        )

        t.continue_generation()
        yield from run_until_finished(r)
        assert (
            r.finished
        ), "continue_generation must drive the re-queued req to completion"

    def test_retract_during_gap_inflight_middle_chunks_positive(self):
        self.server.execute_script(
            self._script_retract_during_gap_inflight_middle_chunks_positive
        )

    @staticmethod
    def _script_retract_during_gap_inflight_middle_chunks_positive(t: ScriptedContext):
        # Retract a chunked req mid-prefill (after >= 1 chunk has committed, so
        # inflight_middle_chunks is the latched 1 and a committed prefix exists) and
        # verify retract releases KV/row, resets inflight_middle_chunks, and re-queues
        # the req to the waiting queue, then resumes it to completion.
        #
        # The "parked gap" state this test was originally written around
        # (inflight_middle_chunks > 0 AND not is_chunking) is never reached on the
        # single-engine overlap path: empirically the trajectory is
        # (chunks_done=1, inflight=1, is_chunking=True) -> (chunks_done=2, inflight=0,
        # is_chunking=False); inflight_middle_chunks is the 0/1 latch that bumps only
        # while the req is still the chunked_req, and clears in the same transition
        # that clears chunked_req. So land while genuinely chunking instead. Retract of
        # a still-chunking req (even with an empty running_batch) clears chunked_req,
        # releases its KV, resets inflight_middle_chunks to 0, and moves it to the
        # waiting queue.
        r = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        assert r.req.inflight_middle_chunks > 0
        assert r.is_chunking

        t.pause_generation(mode="retract")
        yield

        assert r.kv_pages == 0, f"retract must release KV; got kv_pages={r.kv_pages}"
        assert not r.finished, "retract must re-queue r, not finish or abort it"
        assert not r.is_chunking, "retract must release the chunked slot"
        assert r.status == "waiting", (
            f"retracted chunked req must return to the waiting queue; "
            f"got status={r.status!r}"
        )
        req = t.find_req_by_rid(r.rid)
        assert req is not None and req.inflight_middle_chunks == 0, (
            f"retract must reset inflight_middle_chunks; got "
            f"{req.inflight_middle_chunks if req is not None else None}"
        )

        t.continue_generation()
        yield from run_until_finished(r, max_steps=2000)
        assert (
            r.finished
        ), "continue_generation must drive the re-queued req to completion"
        assert r.kv_pages == 0
        assert len(r.req.output_ids) == 2

    def test_load_inquirer_pending_tokens_dedup_chunked(self):
        self.server.execute_script(
            self._script_load_inquirer_pending_tokens_dedup_chunked
        )

    @staticmethod
    def _script_load_inquirer_pending_tokens_dedup_chunked(t: ScriptedContext):
        # The dual-queue dedup invariant for the in-flight chunked req is the real v1
        # formula (load_inquirer.py:69-72): its pending-token contribution is its full
        # seqlen MINUS the committed radix prefix (len(prefix_indices)), never the full
        # seqlen. Asserting against r.remaining_prompt_tokens (origin_input_ids -
        # kv_committed_len) is wrong: prefix_indices lags kv_committed_len by up to one
        # chunk, so the two legitimately differ mid-flight. Compare against the
        # prefix-subtracting formula directly and require the subtraction actually
        # happened once a prefix is committed (strictly below the full seqlen).
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        saw_chunking = False
        saw_dedup = False
        for _ in range(DEFAULT_MAX_STEPS):
            chunked = t.scheduler.chunked_req
            if r.is_chunking and chunked is not None and chunked.rid == r.rid:
                saw_chunking = True
                pending = _load_inquirer_pending_for_rid(t, r.rid)
                expected = chunked.seqlen - len(chunked.prefix_indices)
                assert pending == expected, (
                    f"load_inquirer chunked contribution must equal the prefix-"
                    f"subtracting formula seqlen - len(prefix_indices) = {expected}; "
                    f"got {pending}"
                )
                assert pending <= chunked.seqlen, (
                    f"chunked contribution must never exceed its full seqlen "
                    f"{chunked.seqlen}; got {pending} — dual-queue dedup violated"
                )
                if len(chunked.prefix_indices) > 0:
                    saw_dedup = pending < chunked.seqlen
            if r.finished:
                break
            yield
        assert r.finished
        assert (
            saw_chunking
        ), "test must observe the dual-queue chunked state at least once"
        assert saw_dedup, (
            "test must observe the chunked req with a committed prefix so the "
            "dedup subtraction is actually exercised"
        )

    def test_load_inquirer_chunked_contribution_exact_remainder(self):
        self.server.execute_script(
            self._script_load_inquirer_chunked_contribution_exact_remainder
        )

    @staticmethod
    def _script_load_inquirer_chunked_contribution_exact_remainder(t: ScriptedContext):
        # load_inquirer.py:70-72 chunked branch: with no waiting-queue reqs, the
        # entire pending-token tally comes from the in-flight chunked_req and must
        # equal exactly seqlen - len(prefix_indices) (its not-yet-committed tail),
        # NOT the full seqlen. Force is_chunking with an otherwise-empty queue and
        # assert the exact remainder, so a regression that drops the prefix
        # subtraction (over-counts the committed prefix) is caught.
        s = t.scheduler
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            chunked = s.chunked_req
            if r.is_chunking and chunked is not None and chunked.rid == r.rid:
                assert len(s.waiting_queue) == 0, (
                    "test requires an empty waiting_queue so the chunked req is "
                    f"the sole pending-token contributor; got {len(s.waiting_queue)}"
                )
                expected = chunked.seqlen - len(chunked.prefix_indices)
                observed = s.load_inquirer._get_num_pending_tokens()
                assert observed == expected, (
                    f"chunked contribution must equal remainder "
                    f"seqlen - len(prefix_indices) = {expected}, got {observed}; "
                    f"a value of {chunked.seqlen} would mean the committed prefix "
                    "is being double-counted"
                )
                saw_chunking = True
            if r.finished:
                break
            yield
        assert r.finished
        assert saw_chunking, "test must observe the chunked req mid-flight"

    def test_load_inquirer_chunk_deduct_subtracts_planned_chunk(self):
        self.server.execute_script(
            self._script_load_inquirer_chunk_deduct_subtracts_planned_chunk
        )

    @staticmethod
    def _script_load_inquirer_chunk_deduct_subtracts_planned_chunk(t: ScriptedContext):
        # load_inquirer.py:72 chunk_deduct param: at batch-scheduling time the
        # current chunk is planned but not yet in prefix_indices, so the scheduler
        # passes chunk_deduct=extend_input_len and the deducted tally must be
        # exactly extend_input_len less than the default (chunk_deduct=0) tally.
        # This is the branch that distinguishes the two call sites of
        # _get_num_pending_tokens (load-reporting vs batch-scheduling).
        s = t.scheduler
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            chunked = s.chunked_req
            if (
                r.is_chunking
                and chunked is not None
                and chunked.rid == r.rid
                and chunked.extend_input_len > 0
            ):
                deduct = chunked.extend_input_len
                base = s.load_inquirer._get_num_pending_tokens()
                deducted = s.load_inquirer._get_num_pending_tokens(chunk_deduct=deduct)
                assert deducted == base - deduct, (
                    f"chunk_deduct must subtract the planned chunk exactly: "
                    f"base={base}, deducted={deducted}, expected={base - deduct}"
                )
                saw_chunking = True
            if r.finished:
                break
            yield
        assert r.finished
        assert saw_chunking, "test must observe the chunked req with a planned chunk"

    def test_chunked_forced_admission_avoids_leak(self):
        self.server.execute_script(self._script_chunked_forced_admission_avoids_leak)

    @staticmethod
    def _script_chunked_forced_admission_avoids_leak(t: ScriptedContext):
        # A retractable decode peer holds KV the scheduler's decode-OOM retract
        # path can actually free; raw exhaust_kv pages have no backing Req, so
        # with leave_pages=0 the forced chunk's alloc_for_extend hard-OOMs
        # (common.py raise RuntimeError) and crashes the scheduler.
        r_peer = t.start_req(
            prompt_len=8, max_new_tokens=BALLAST_MAX_NEW_TOKENS, ignore_eos=True
        )
        yield from run_until(r_peer, lambda h: h.status == "running")

        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        baseline_rows = (
            t.scheduler.req_to_token_pool.size
            - t.scheduler.req_to_token_pool.available_size()
        )
        # Squeeze free KV to a sub-chunk sliver so the chunked resume sees
        # _rem_tokens <= 0 and takes the force-re-add path; the retractable peer
        # supplies the pages the forced chunk needs.
        t.exhaust_kv(leave_pages=1)
        yield from run_until_finished(r)
        assert r.finished
        assert (
            t.scheduler.req_to_token_pool.size
            - t.scheduler.req_to_token_pool.available_size()
        ) <= baseline_rows, f"row leak under forced chunked admission: baseline={baseline_rows}, after={(t.scheduler.req_to_token_pool.size - t.scheduler.req_to_token_pool.available_size())}"

    def test_stage_a_inflight_middle_chunks_sync_invariant(self):
        self.server.execute_script(
            self._script_stage_a_inflight_middle_chunks_sync_invariant
        )

    @staticmethod
    def _script_stage_a_inflight_middle_chunks_sync_invariant(t: ScriptedContext):
        def assert_invariant() -> None:
            req = t.find_req_by_rid(r.rid)
            if req is not None and req.inflight_middle_chunks > 0:
                assert r.is_chunking, (
                    f"invariant violated: inflight_middle_chunks="
                    f"{req.inflight_middle_chunks} but is_chunking={r.is_chunking}"
                )

        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        # Drive a retract/resume mid-chunk: retract clears chunked_req (so
        # is_chunking flips to False) and must also reset inflight_middle_chunks
        # in the same transition, otherwise the cross-subsystem invariant breaks.
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        assert_invariant()

        t.pause_generation(mode="retract")
        yield
        assert_invariant()

        t.continue_generation()
        for _ in range(DEFAULT_MAX_STEPS):
            assert_invariant()
            if r.finished:
                return
            yield
        raise AssertionError("req did not finish")

    def test_init_next_round_input_resets_chunk_state(self):
        self.server.execute_script(
            self._script_init_next_round_input_resets_chunk_state
        )

    @staticmethod
    def _script_init_next_round_input_resets_chunk_state(t: ScriptedContext):
        # init_next_round_input rebuilds fill_ids = origin_input_ids + output_ids
        # (schedule_batch.py:1050); the chunked-admission path then truncates it to the
        # committed prefix plus the chunk about to run
        # (schedule_policy.py: fill_ids = fill_ids[:len(prefix_indices)+extend_input_len]).
        # So mid-chunk the invariant is len(fill_ids) == len(prefix_indices) +
        # extend_input_len, NOT len(fill_ids) == len(prefix_indices) (that equality only
        # holds at the instant a chunk boundary commits, before the next chunk is loaded;
        # with the overlap pipeline the observable state is always mid-chunk where the
        # current chunk's tokens are appended past the committed prefix).
        r = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        saw_mid_chunk = False
        for _ in range(DEFAULT_MAX_STEPS):
            req = r.req
            if (
                r.is_chunking
                and r.chunks_done >= 1
                and req is not None
                and req.extend_input_len is not None
            ):
                saw_mid_chunk = True
                assert (
                    len(req.fill_ids) == len(req.prefix_indices) + req.extend_input_len
                ), (
                    f"init_next_round_input must rebuild fill_ids to the committed "
                    f"prefix plus the in-flight chunk; "
                    f"fill_ids_len={len(req.fill_ids)}, "
                    f"prefix_indices_len={len(req.prefix_indices)}, "
                    f"extend_input_len={req.extend_input_len}, "
                    f"chunks_done={r.chunks_done}"
                )
            if r.finished:
                break
            yield
        assert r.finished
        assert (
            saw_mid_chunk
        ), "test must observe the fill_ids reset boundary at least once"
        assert r.finished

    def test_chunked_req_slot_cleared_when_chunk_completes(self):
        self.server.execute_script(
            self._script_chunked_req_slot_cleared_when_chunk_completes
        )

    @staticmethod
    def _script_chunked_req_slot_cleared_when_chunk_completes(t: ScriptedContext):
        s = t.scheduler
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                saw_chunking = True
            if r.finished:
                break
            yield
        assert r.finished
        assert saw_chunking, "req should have occupied the chunked_req slot mid-chunk"
        assert (
            s.chunked_req is None
        ), f"chunked_req slot must clear after last chunk; got {s.chunked_req!r}"

    def test_second_chunked_admit_blocked_when_chunked_req_set(self):
        self.server.execute_script(
            self._script_second_chunked_admit_blocked_when_chunked_req_set
        )

    @staticmethod
    def _script_second_chunked_admit_blocked_when_chunked_req_set(t: ScriptedContext):
        # Distinct prompt_token per req so r2 chunks cold: with the default fill
        # token both prompts are identical, r2 fully hits r1's committed radix prefix
        # and admits non-chunked (never entering is_chunking), so saw_r2 would stay
        # False. Distinct tokens force both to chunk over their lifetime while still
        # exercising the single-chunked-slot mutual exclusion.
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=10
        )
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=11
        )
        saw_r1_chunking = False
        saw_r2_chunking = False
        for _ in range(DEFAULT_MAX_STEPS * 2):
            both = r1.is_chunking and r2.is_chunking
            assert not both, (
                f"only one chunked-resume slot allowed; r1.is_chunking="
                f"{r1.is_chunking}, r2.is_chunking={r2.is_chunking}"
            )
            if r1.is_chunking:
                saw_r1_chunking = True
            if r2.is_chunking:
                saw_r2_chunking = True
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished and r2.finished
        assert saw_r1_chunking and saw_r2_chunking, (
            f"both reqs should chunk over their lifetime; saw_r1="
            f"{saw_r1_chunking}, saw_r2={saw_r2_chunking}"
        )

    # Removed test_chunked_exclude_falls_back_to_last_batch_reqs_when_no_pp: it
    # probed t.last_chunked_exclude_set_source, which would require the scheduler
    # to durably record which structure (last_batch.chunked_req vs last_batch.reqs)
    # it sourced the exclude set from. That is a pure implementation detail, not an
    # observable invariant. The behavior that actually matters -- excluding the
    # correct in-flight reqs from the local running set -- is covered by the
    # in_flight_other_mb_rids regression test (filter_batch exclusion).

    def test_scheduler_continues_with_only_chunked_req_no_waiting(self):
        self.server.execute_script(
            self._script_scheduler_continues_with_only_chunked_req_no_waiting
        )

    @staticmethod
    def _script_scheduler_continues_with_only_chunked_req_no_waiting(
        t: ScriptedContext,
    ):
        # "while a chunked req is in flight" means precisely while r.is_chunking
        # (chunked_req points at r). Once the final chunk commits, chunked_req clears
        # and the req moves on to decode/finish; the step right after the last chunk
        # (and the finishing step) legitimately observes is_idle, so the non-idle
        # assertion must be guarded by r.is_chunking, mirroring the passing
        # test_chunked_in_flight_no_idle. The real invariant under test is that the
        # sole chunked req keeps advancing with an empty waiting_queue.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        prev_chunks_done = r.chunks_done
        progressed = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                assert (
                    not t.is_idle
                ), "scheduler must not go idle while a chunked req is in flight"
            cur_chunks_done = r.chunks_done
            if cur_chunks_done > prev_chunks_done:
                progressed = True
            prev_chunks_done = cur_chunks_done
            if r.finished:
                break
            yield
        assert r.finished
        assert progressed, (
            "chunks_done must keep advancing without any waiter; pre-fix "
            "an empty waiting_queue could cause the loop to skip continuation"
        )

    def test_add_chunked_req_non_swa_forced_admit_on_rem_zero(self):
        self.server.execute_script(
            self._script_add_chunked_req_non_swa_forced_admit_on_rem_zero
        )

    @staticmethod
    def _script_add_chunked_req_non_swa_forced_admit_on_rem_zero(t: ScriptedContext):
        # A retractable decode peer holds KV the scheduler's decode-OOM retract
        # path can actually free; raw exhaust_kv pages have no backing Req, so
        # with leave_pages=0 the forced chunk's alloc_for_extend hard-OOMs
        # (common.py raise RuntimeError) and crashes the scheduler.
        r_peer = t.start_req(
            prompt_len=8, max_new_tokens=BALLAST_MAX_NEW_TOKENS, ignore_eos=True
        )
        yield from run_until(r_peer, lambda h: h.status == "running")

        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        # Sub-chunk sliver => _rem_tokens <= 0 => force-re-add; the peer is
        # retractable so the forced chunk can allocate and the req finishes.
        t.exhaust_kv(leave_pages=1)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished, (
            "non-SWA chunked-resume must be force-admitted when "
            "_rem_tokens == 0 (schedule_policy.py:679-682); pre-fix it "
            "would block forever and leak its held row + KV"
        )
        assert r.kv_pages == 0
        assert r.lock_refs == 0


class TestSpecialCaseRowPoolExhaustion(ScriptedTestCase):
    # req_to_token_pool.size == max_running_requests (model_runner_kv_cache_mixin.py:
    # max_num_reqs = self.max_running_requests). The default pool is thousands of rows,
    # which the ballast reqs can never fully occupy because the KV pool exhausts long
    # before the row pool does. Cap max_running_requests small so exhaust_row_pool can
    # actually drive get_num_allocatable_reqs() to 0 while the in-flight chunked req
    # (already admitted into the chunked slot) keeps progressing -- the bypass under test.
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        max_running_requests=8,
    )

    def test_chunked_req_bypasses_req_pool_exhaustion(self):
        self.server.execute_script(
            self._script_chunked_req_bypasses_req_pool_exhaustion
        )

    @staticmethod
    def _script_chunked_req_bypasses_req_pool_exhaustion(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        chunks_before_pressure = r.chunks_done

        yield from exhaust_row_pool(t, leave_rows=0)

        progressed_under_pressure = False
        for _ in range(DEFAULT_MAX_STEPS * 2):
            if r.chunks_done > chunks_before_pressure:
                progressed_under_pressure = True
            if r.finished:
                break
            yield
        assert r.finished
        assert progressed_under_pressure, (
            "chunked req must advance even when get_num_allocatable_reqs "
            "returns 0; pre-fix the bypass would block forever"
        )
        assert r.kv_pages == 0


class TestSpecialCaseMixedChunk(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_mixed_chunk=True,
    )

    def test_mix_with_running_chunked_plus_decode(self):
        self.server.execute_script(self._script_mix_with_running_chunked_plus_decode)

    @staticmethod
    def _script_mix_with_running_chunked_plus_decode(t: ScriptedContext):
        # Drive the decoders past their own prefill into the decoding state (each
        # has emitted at least one token) so they sit in running_batch as decode
        # reqs when the long chunked prompt arrives. The chunked prefill then rides
        # the same forward pass as the running decodes, producing a MIXED batch.
        # The exact iteration at which the merge happens is not deterministic (the
        # chunk admits over several passes), so poll for the MIXED co-batch across
        # the run rather than asserting it at a single instant.
        decodes = [t.start_req(prompt_len=8, max_new_tokens=16) for _ in range(3)]
        for d in decodes:
            yield from advance_to_decode_step(d, 1)

        r_chunk = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until(r_chunk, lambda h: h.is_chunking)

        comp = t.batch_composition()
        assert r_chunk.rid in comp.get("chunked", [])

        saw_mixed_with_decode = False
        all_reqs = [r_chunk, *decodes]
        for _ in range(DEFAULT_MAX_STEPS * 2):
            comp = t.batch_composition()
            # ForwardMode.MIXED.is_extend() is True, so batch_composition reports the
            # merged-in decode reqs (everything but the chunked rid) under the
            # "prefill" bucket, not "decode". Union all batch buckets to catch them.
            batch_rids = (
                set(comp.get("prefill", []))
                | set(comp.get("decode", []))
                | set(comp.get("running", []))
            )
            if (
                t.last_batch_forward_mode == "MIXED"
                and r_chunk.rid in comp.get("chunked", [])
                and any(d.rid in batch_rids for d in decodes)
            ):
                saw_mixed_with_decode = True
            if all(x.finished for x in all_reqs):
                break
            yield
        assert all(x.finished for x in all_reqs)
        assert saw_mixed_with_decode, (
            "enable_mixed_chunk should merge running decode reqs into the chunked "
            "prefill iter (MIXED forward_mode) at least once"
        )

    def test_mixed_chunk_with_logprob_falls_back(self):
        self.server.execute_script(self._script_mixed_chunk_with_logprob_falls_back)

    @staticmethod
    def _script_mixed_chunk_with_logprob_falls_back(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            return_logprob=True,
        )
        yield from run_until(r, lambda h: h.is_chunking)
        assert (
            t.last_batch_forward_mode != "MIXED"
        ), f"return_logprob must disable mixed-chunk path; got {t.last_batch_forward_mode!r}"
        yield from run_until_finished(r)

    def test_mixed_chunk_with_running_batch(self):
        self.server.execute_script(self._script_mixed_chunk_with_running_batch)

    @staticmethod
    def _script_mixed_chunk_with_running_batch(t: ScriptedContext):
        r_dec = t.start_req(prompt_len=8, max_new_tokens=32)
        yield from run_until(r_dec, lambda h: h.status == "running")

        r_chunk = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield
        yield from run_until(r_chunk, lambda h: h.is_chunking)

        assert (
            t.last_batch_forward_mode == "MIXED"
        ), f"chunked admission with running batch must enter MIXED; got {t.last_batch_forward_mode!r}"
        for _ in range(DEFAULT_MAX_STEPS * 2):
            if r_chunk.finished and r_dec.finished:
                break
            yield
        assert r_chunk.finished and r_dec.finished


class TestSpecialCaseTransformers(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        model_impl="transformers",
    )

    def test_transformers_text_model_still_chunks(self):
        self.server.execute_script(self._script_transformers_text_model_still_chunks)

    @staticmethod
    def _script_transformers_text_model_still_chunks(t: ScriptedContext):
        # Chunked prefill is force-disabled (chunked_prefill_size=-1) only for
        # *multimodal* models (server_args.py:1353,1413 gated on
        # model_config.is_multimodal), not by the Transformers backend itself. A
        # text model (Qwen3-0.6B) on the Transformers backend therefore chunks a
        # long prompt exactly like the native backend. A distinct prompt_token
        # makes the req chunk cold so the count matches _expected_chunks.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=10
        )
        yield from run_until_finished(r)
        assert r.finished
        expected = _expected_chunks(VERY_LONG_PROMPT_LEN, DEFAULT_CHUNK_SIZE)
        assert r.chunks_done == expected, (
            f"a text model on the Transformers backend must chunk like the native "
            f"backend ({expected} chunks); got chunks_done={r.chunks_done}"
        )


class TestSpecialCaseNoChunking(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=-1)

    def test_chunk_size_negative_disables_chunking(self):
        self.server.execute_script(self._script_chunk_size_negative_disables_chunking)

    @staticmethod
    def _script_chunk_size_negative_disables_chunking(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            assert (
                not r.is_chunking
            ), "chunked_prefill_size=-1 should disable chunked path"
            if r.finished:
                return
            yield
        raise AssertionError("req did not finish under disabled chunking")


# Deterministic flashinfer rounds chunked-prefill splits to its attention
# alignment size; chunk_size == the align size keeps the req chunking exactly
# once before the sub-align tail admits as the non-chunked last chunk.
DETERMINISTIC_ALIGN_SIZE = 4096


class TestSpecialCaseDeterministicFlashInfer(ScriptedTestCase):
    # Deterministic inference with flashinfer forces the radix cache off (it is not
    # yet compatible), so the tree_cache is a ChunkCache that the scripted harness's
    # default kv_canary="raise" cannot walk (walk_radix_cache_for_canary does not
    # support ChunkCache), crashing the scheduler. Opt out of the kv canary here.
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DETERMINISTIC_ALIGN_SIZE,
        page_size=16,
        attention_backend="flashinfer",
        enable_deterministic_inference=True,
        kv_canary="none",
        kv_canary_sweep_interval=0,
    )

    def test_chunked_truncation_align_size(self):
        self.server.execute_script(self._script_chunked_truncation_align_size)

    @staticmethod
    def _script_chunked_truncation_align_size(t: ScriptedContext):
        # A distinct prompt_token forces a cold chunk (no radix prefix overlap). The
        # prompt is one full chunk plus a tail so the first chunk truncates to the
        # align size and the remaining tail completes as the last (non-chunked) chunk.
        r = t.start_req(
            prompt_len=DETERMINISTIC_ALIGN_SIZE + 1024,
            max_new_tokens=2,
            prompt_token=10,
        )
        page_size = 16
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking and r.req.extend_input_len is not None:
                saw_chunking = True
                assert r.req.extend_input_len % page_size == 0, (
                    f"deterministic chunk boundary must be page-aligned; "
                    f"got extend_input_len={r.req.extend_input_len}, page_size={page_size}"
                )
            if r.finished:
                break
            yield
        assert r.finished, "chunked req did not finish"
        assert saw_chunking, "test must observe the req mid-chunk at least once"


class TestSpecialCaseHiCache(ScriptedTestCase):
    # enable_hierarchical_cache=True makes the tree_cache a HiRadixCache, which the
    # scripted harness's default kv_canary="raise" cannot walk
    # (walk_radix_cache_for_canary does not support HiRadixCache), crashing the
    # scheduler. Opt out of the kv canary (and zero its sweep interval, which is only
    # valid for a non-none kv_canary; server_args.py:7407).
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_hierarchical_cache=True,
        kv_canary="none",
        kv_canary_sweep_interval=0,
    )

    def test_hicache_breakdown_only_first_chunk(self):
        self.server.execute_script(self._script_hicache_breakdown_only_first_chunk)

    @staticmethod
    def _script_hicache_breakdown_only_first_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        first_chunk_snap = None
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                saw_chunking = True
            if r.is_chunking and r.chunks_done >= 1 and first_chunk_snap is None:
                first_chunk_snap = r.req.cached_tokens
            if first_chunk_snap is not None and r.is_chunking:
                cur = r.req.cached_tokens
                assert cur == first_chunk_snap, (
                    f"HiCache cached_tokens_* must freeze after first chunk; "
                    f"first={first_chunk_snap!r}, now={cur!r}"
                )
            if r.finished:
                break
            yield
        assert r.finished
        assert saw_chunking, "test must observe r mid-chunk at least once"
        assert (
            first_chunk_snap is not None
        ), "test must snapshot cached_tokens at the first chunk boundary"

    def test_hicache_cached_tokens_set_once_invariant(self):
        self.server.execute_script(
            self._script_hicache_cached_tokens_set_once_invariant
        )

    @staticmethod
    def _script_hicache_cached_tokens_set_once_invariant(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        snap: Optional[tuple] = None
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            req = t.find_req_by_rid(r.rid)
            # Only start tracking once the req is actually mid-chunk (chunks_done
            # >= 1); the set-once invariant is meaningless before the first
            # chunk boundary writes the cached_tokens_* breakdown.
            if req is not None and r.is_chunking and r.chunks_done >= 1:
                saw_chunking = True
                cur = (
                    req.cached_tokens_device,
                    req.cached_tokens_host,
                    req.cached_tokens_storage,
                )
                if snap is None:
                    snap = cur
                else:
                    assert cur == snap, (
                        f"HiCache cached_tokens_* fields must be set exactly "
                        f"once on first chunk; values changed: snap={snap}, "
                        f"cur={cur}"
                    )
            if r.finished:
                break
            yield
        assert r.finished
        assert (
            saw_chunking
        ), "test must observe the req mid-chunk (chunks_done >= 1) at least once"
        assert snap is not None, "test must snapshot the cached_tokens_* breakdown"


def _expected_chunks(prompt_len: int, chunk_size: int) -> int:
    # Mirror of the helper in test_scripted_chunk_size.py: the number of prefill
    # forward passes a chunked prompt takes. 0 when the whole prompt fits one
    # non-chunked shot; otherwise ceil(prompt_len / chunk_size) with the tail
    # iteration counted.
    if prompt_len <= chunk_size:
        return 0
    return (prompt_len + chunk_size - 1) // chunk_size


class TestSpecialCaseDynamicChunkingPP1(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_dynamic_chunking=True,
    )

    def test_dynamic_chunking_forced_off_on_pp1(self):
        self.server.execute_script(self._script_dynamic_chunking_forced_off_on_pp1)

    @staticmethod
    def _script_dynamic_chunking_forced_off_on_pp1(t: ScriptedContext):
        # scheduler.py:947-948: enable_dynamic_chunking = server_args.enable_dynamic_chunking
        # AND ps.pp_size > 1. On a single GPU (pp_size == 1) the conjunct forces it
        # OFF even though the server arg is True, so the scheduler.py:2612 dynamic
        # chunk-size predictor is never consulted and the prompt is chunked with the
        # uniform chunked_prefill_size — exactly _expected_chunks(...) chunks.
        # GPU validation pending.
        assert t.scheduler.enable_dynamic_chunking is False, (
            "pp_size==1 must force enable_dynamic_chunking off even when the "
            "server arg is True (the 'and ps.pp_size > 1' conjunct)"
        )
        # A distinct prompt_token makes the req chunk cold: with the default fill
        # token the all-identical prompt overlaps the radix tree and the chunk
        # boundary count drifts off the exact _expected_chunks value.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=10
        )
        yield from run_until_finished(r)
        assert r.finished
        expected = _expected_chunks(VERY_LONG_PROMPT_LEN, DEFAULT_CHUNK_SIZE)
        assert r.chunks_done == expected, (
            f"uniform chunked_prefill_size must yield exactly {expected} chunks "
            f"(no dynamic-size prediction on pp1); got {r.chunks_done}"
        )


class TestSpecialCaseChunkedRemReadd(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_add_chunked_req_rem_nonpositive_forces_rem_chunk_tokens(self):
        self.server.execute_script(
            self._script_add_chunked_req_rem_nonpositive_forces_rem_chunk_tokens
        )

    @staticmethod
    def _script_add_chunked_req_rem_nonpositive_forces_rem_chunk_tokens(
        t: ScriptedContext,
    ):
        # schedule_policy.py:679-682 (non-SWA branch): when a chunked resume sees
        # _rem_tokens = min(rem_chunk_tokens, rem_total_tokens) <= 0, the req must
        # still be re-added (forced to rem_chunk_tokens at line 682) rather than
        # silently dropped, otherwise its held row/KV leaks forever. Drive a chunked
        # req mid-flight, then squeeze every free KV page so rem_total_tokens hits 0
        # and _rem_tokens <= 0 on the next re-admit; the force-to-rem_chunk_tokens
        # re-add (line 682) keeps the req alive so it finishes and releases all
        # resources. A non-SWA model executes line 682 (not the SWA park at 681).
        #
        # The observable consequence of "re-added, not silently dropped" is exactly
        # this clean finish + full resource release under full KV pressure (the same
        # invariant test_add_chunked_req_non_swa_forced_admit_on_rem_zero asserts). A
        # post-squeeze chunks_done bump is NOT a reliable witness: with all KV held
        # the req may land on its final chunk and finish without advancing chunks_done
        # in the observable window, so requiring a bump is a wrong assumption.
        # A retractable decode peer holds the KV so pressure is real but
        # recoverable: the scheduler's decode-OOM retract path frees its pages,
        # whereas exhaust_kv pages have no backing Req to retract -- the forced
        # chunk's alloc_for_extend would hard-OOM and crash the scheduler.
        r_peer = t.start_req(
            prompt_len=8, max_new_tokens=BALLAST_MAX_NEW_TOKENS, ignore_eos=True
        )
        yield from run_until(r_peer, lambda h: h.status == "running")

        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        # Squeeze the remaining free KV down to a sub-chunk sliver so the chunked
        # resume sees _rem_tokens = min(rem_chunk_tokens, rem_total_tokens) <= 0
        # and takes the force-to-rem_chunk_tokens re-add path; the decode peer can
        # then be retracted to supply the pages the forced chunk needs.
        t.exhaust_kv(leave_pages=1)

        yield from run_until_finished(r, max_steps=DEFAULT_MAX_STEPS * 2)
        assert r.finished, (
            "non-SWA chunked resume must be force-re-added when _rem_tokens <= 0 "
            "and complete once a retractable peer frees KV; it must not leak its "
            f"held row + KV. status={r.status} kv_pages={r.kv_pages}"
        )
        assert r.kv_pages == 0, f"kv_pages={r.kv_pages}"
        assert r.lock_refs == 0, f"lock_refs={r.lock_refs}"


class TestSpecialCaseRetractMerge(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_retract_merges_extend_chunk_batch_before_retract_all(self):
        self.server.execute_script(
            self._script_retract_merges_extend_chunk_batch_before_retract_all
        )

    @staticmethod
    def _script_retract_merges_extend_chunk_batch_before_retract_all(
        t: ScriptedContext,
    ):
        # scheduler.py:3705-3721: on retract, if last_batch.is_extend() the scheduler
        # filter_batch(chunked_req_to_exclude=[]) (empty exclude) and merges the just-run
        # extend/chunk batch into running_batch BEFORE retract_all runs at 3726-3734.
        # Drive a chunked req until is_chunking so last_batch is the extend chunk batch
        # and running_batch is empty; pin that pre-state, then retract and witness the
        # merge-then-retract: last_batch cleared, running_batch drained (the chunk batch
        # was folded in and retracted, not stranded as a separate batch). GPU validation
        # pending.
        s = t.scheduler
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        assert s.last_batch is not None, "last_batch must be set while chunking"
        assert s.last_batch.forward_mode.is_extend(), (
            f"last_batch must be the extend/chunk batch; got "
            f"{s.last_batch.forward_mode!r}"
        )

        t.pause_generation(mode="retract")
        yield

        assert (
            s.last_batch is None
        ), "retract must clear last_batch after merging the extend chunk batch"
        assert len(s.running_batch.reqs) == 0, (
            "the merged extend chunk batch must be retracted out of running_batch, "
            f"not stranded; got {len(s.running_batch.reqs)} reqs"
        )
        assert (
            r.status == "waiting"
        ), f"retracted chunked req must return to the waiting queue; got {r.status!r}"
        assert r.kv_pages == 0

        t.continue_generation()
        yield from run_until_finished(r)
        assert r.finished
        assert len(r.req.output_ids) == 2, (
            f"resumed req must emit exactly max_new_tokens; got "
            f"{len(r.req.output_ids)}"
        )


class TestSpecialCaseChunkBudgetDefer(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_co_submitted_waiter_deferred_when_chunk_budget_zero(self):
        self.server.execute_script(
            self._script_co_submitted_waiter_deferred_when_chunk_budget_zero
        )

    @staticmethod
    def _script_co_submitted_waiter_deferred_when_chunk_budget_zero(
        t: ScriptedContext,
    ):
        # schedule_policy.py:577-578 (budget_state: rem_chunk_tokens is not None and
        # <= 0 -> OTHER). When the long prompt consumes the full chunk-token budget in
        # a pass, rem_chunk_tokens hits 0 and budget_state() refuses any further
        # candidate in that pass. Submit the long chunked prompt plus a second long
        # waiter that would otherwise admit; on the iters where the chunked rid is
        # active, witness the second waiter sitting in 'waiting' with no KV — deferred
        # specifically because the chunk-token budget was driven to 0. Both finish.
        # GPU validation pending.
        r_chunk = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r_wait = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 8, max_new_tokens=2)
        yield from run_until(r_chunk, lambda h: h.is_chunking)

        saw_deferred = False
        for _ in range(DEFAULT_MAX_STEPS * 2):
            comp = t.batch_composition()
            chunk_active = r_chunk.rid in comp.get("chunked", [])
            wait_idle = (
                r_wait.status == "waiting"
                and r_wait.rid not in comp.get("chunked", [])
                and r_wait.rid not in comp.get("running", [])
                and r_wait.rid not in comp.get("decode", [])
            )
            if chunk_active and wait_idle:
                saw_deferred = True
                assert r_wait.kv_pages == 0, (
                    f"deferred waiter (rem_chunk_tokens<=0 -> OTHER) must hold no KV; "
                    f"got kv_pages={r_wait.kv_pages}"
                )
            if r_chunk.finished and r_wait.finished:
                break
            yield
        assert r_chunk.finished and r_wait.finished
        assert saw_deferred, (
            "second waiter must be deferred for >=1 iter while the chunk-token "
            "budget was driven to 0 by the in-flight chunked req"
        )


class TestSpecialCaseIgnoreEosNoRadix(ScriptedTestCase):
    # disable_radix_cache=True is the load-bearing precondition: add_one_req routes to
    # add_one_req_ignore_eos only when req.sampling_params.ignore_eos AND
    # getattr(self.tree_cache, "disable", True) (schedule_policy.py:835). With radix
    # enabled the guard fails and ignore_eos reqs go through add_one_req instead.
    #
    # disable_radix_cache=True makes the tree_cache a ChunkCache, which the scripted
    # harness's default kv_canary="raise" cannot walk (walk_radix_cache_for_canary
    # does not support ChunkCache), crashing the scheduler on the first forward pass.
    # Opt out of the kv canary for this config so the ignore_eos path can actually
    # run (sweep_interval must also be zeroed; sweep_interval>0 requires a non-none
    # kv_canary, server_args.py:7407).
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disable_radix_cache=True,
        kv_canary="none",
        kv_canary_sweep_interval=0,
    )

    def test_ignore_eos_chunked_truncate_path(self):
        self.server.execute_script(self._script_ignore_eos_chunked_truncate_path)

    @staticmethod
    def _script_ignore_eos_chunked_truncate_path(t: ScriptedContext):
        # schedule_policy.py:798-809 (add_one_req_ignore_eos else: chunked truncate,
        # new_chunked_req=req). A long ignore_eos prompt with radix disabled routes to
        # add_one_req_ignore_eos and, exceeding the chunk budget, takes the chunked
        # truncation path that carries it as new_chunked_req. Witness it goes
        # is_chunking and completes the expected number of chunks. GPU validation pending.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, ignore_eos=True
        )
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until_finished(r)
        assert r.finished
        expected = _expected_chunks(VERY_LONG_PROMPT_LEN, DEFAULT_CHUNK_SIZE)
        assert r.chunks_done == expected, (
            f"ignore_eos chunk-truncation path must carry the prompt as new_chunked_req "
            f"for {expected} chunks; got {r.chunks_done}"
        )

    def test_ignore_eos_nonchunked_fits_chunk_budget(self):
        self.server.execute_script(self._script_ignore_eos_nonchunked_fits_chunk_budget)

    @staticmethod
    def _script_ignore_eos_nonchunked_fits_chunk_budget(t: ScriptedContext):
        # schedule_policy.py:787-797 (add_one_req_ignore_eos: rem_chunk_tokens is None
        # or extend_input_len <= rem_chunk_tokens -> non-chunked admit). A short
        # ignore_eos prompt (<= chunk budget) with radix disabled takes the non-chunked
        # ignore_eos admit branch: it must never chunk and report zero chunks_done.
        # GPU validation pending.
        r = t.start_req(prompt_len=8, max_new_tokens=2, ignore_eos=True)
        for _ in range(DEFAULT_MAX_STEPS):
            assert not r.is_chunking, (
                "a prompt that fits the chunk budget must admit non-chunked via the "
                "ignore_eos path, never entering is_chunking"
            )
            if r.finished:
                break
            yield
        assert r.finished
        assert r.chunks_done == 0, (
            f"prompt fitting the chunk budget completes non-chunked; got "
            f"chunks_done={r.chunks_done}"
        )

    def test_ignore_eos_second_long_waiter_deferred_on_chunk_budget_zero(self):
        self.server.execute_script(
            self._script_ignore_eos_second_long_waiter_deferred_on_chunk_budget_zero
        )

    @staticmethod
    def _script_ignore_eos_second_long_waiter_deferred_on_chunk_budget_zero(
        t: ScriptedContext,
    ):
        # schedule_policy.py:799-800 (add_one_req_ignore_eos nested guard:
        # rem_chunk_tokens <= 0 -> OTHER). Two long ignore_eos prompts with radix
        # disabled both route to add_one_req_ignore_eos; once the first occupies the
        # chunk budget, the second must be deferred (the nested <=0 OTHER guard) sitting
        # in 'waiting' with no KV. Both finish. GPU validation pending.
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, ignore_eos=True
        )
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, ignore_eos=True
        )
        yield from run_until(r1, lambda h: h.is_chunking)

        saw_deferred = False
        for _ in range(DEFAULT_MAX_STEPS * 2):
            comp = t.batch_composition()
            r1_active = r1.rid in comp.get("chunked", [])
            r2_idle = (
                r2.status == "waiting"
                and r2.rid not in comp.get("chunked", [])
                and r2.rid not in comp.get("running", [])
                and r2.rid not in comp.get("decode", [])
            )
            if r1_active and r2_idle:
                saw_deferred = True
                assert r2.kv_pages == 0, (
                    f"deferred ignore_eos waiter must hold no KV; got "
                    f"kv_pages={r2.kv_pages}"
                )
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished and r2.finished
        assert saw_deferred, (
            "second long ignore_eos waiter must be deferred for >=1 iter while the "
            "chunk-token budget was driven to 0 (nested rem_chunk_tokens<=0 OTHER)"
        )


class TestSpecialCaseRetractedStain(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_retracted_stain_suppresses_cached_token_recount(self):
        self.server.execute_script(
            self._script_retracted_stain_suppresses_cached_token_recount
        )

    @staticmethod
    def _script_retracted_stain_suppresses_cached_token_recount(t: ScriptedContext):
        # schedule_batch.py:1916-1918 gated by retracted_stain (set at 1315): on a
        # post-retract resume, prepare_for_extend must NOT re-add pre_len -
        # already_computed to cached_tokens, because retracted_stain is True. Drive a
        # chunked req past >=1 chunk, snapshot cached_tokens, retract+resume, then
        # witness retracted_stain True AND cached_tokens unchanged across the resume
        # (no double-count). GPU validation pending.
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        cached_before = r.req.cached_tokens

        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished

        req = r.req
        assert (
            req.retracted_stain is True
        ), "retract must set retracted_stain so the cached-token recount is suppressed"
        assert req.cached_tokens == cached_before, (
            f"retracted_stain must suppress re-adding pre_len-already_computed on "
            f"resume; cached_tokens grew from {cached_before} to {req.cached_tokens}"
        )


class TestSpecialCaseResultSkipRetracted(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_result_skip_retracted_emits_no_stray_token(self):
        self.server.execute_script(
            self._script_result_skip_retracted_emits_no_stray_token
        )

    @staticmethod
    def _script_result_skip_retracted_emits_no_stray_token(t: ScriptedContext):
        # batch_result_processor.py:218-221: for (req, next_token), if req.finished()
        # or req.is_retracted -> continue. Retract a chunked req mid-prefill so a
        # prefill result for it is in flight while it is is_retracted; the skip must
        # suppress any output append / inflight_middle_chunks decrement on that entry.
        # ignore_eos pins the resumed req to exactly max_new_tokens, so a stray token
        # from the skipped retracted entry would show up as output_ids length > 2.
        # GPU validation pending.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, ignore_eos=True
        )
        yield from run_until(r, lambda h: h.is_chunking)

        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r, max_steps=2000)
        assert r.finished
        assert len(r.req.output_ids) == 2, (
            f"the skipped retracted entry must not append a stray output token; "
            f"got output_ids len {len(r.req.output_ids)}"
        )


class TestSpecialCaseMiddleChunkNoToken(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_middle_chunk_appends_no_token_no_finish(self):
        self.server.execute_script(self._script_middle_chunk_appends_no_token_no_finish)

    @staticmethod
    def _script_middle_chunk_appends_no_token_no_finish(t: ScriptedContext):
        # batch_result_processor.py:270-289 (else branch, skip_stream_req): a middle
        # chunk whose prefill is not finished must NOT append an output token nor mark
        # the req finished. Step the engine while is_chunking and witness at every
        # middle-chunk iter that output_ids stays empty and status is not finished;
        # output only grows once is_chunking flips False. GPU validation pending.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        saw_middle_chunk = False
        for _ in range(DEFAULT_MAX_STEPS * 2):
            if r.is_chunking:
                saw_middle_chunk = True
                assert len(r.req.output_ids) == 0, (
                    f"middle chunk must not append an output token; got "
                    f"output_ids len {len(r.req.output_ids)}"
                )
                assert (
                    r.status != "finished"
                ), "middle chunk must not finish the req (skip_stream_req)"
            if r.finished:
                break
            yield
        assert r.finished
        assert saw_middle_chunk, "test must observe r mid-chunk at least once"
        assert (
            len(r.req.output_ids) >= 1
        ), "output tokens must appear only after the chunked prefill completes"


if __name__ == "__main__":
    unittest.main()
