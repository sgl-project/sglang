import unittest
from typing import Optional

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    SMALL_KV_POOL_BALLAST_MAX_NEW_TOKENS,
    SMALL_KV_POOL_BALLAST_PROMPT_LEN,
    SMALL_KV_POOL_MAX_TOTAL_TOKENS,
    VERY_LONG_PROMPT_LEN,
    advance_to_decode_step,
    base_engine_kwargs,
    chunked_req_of,
    exhaust_row_pool,
    extend_input_len_of,
    inflight_middle_chunks_of,
    run_until,
    run_until_finished,
)


def _load_inquirer_pending_for_rid(t: ScriptedContext, rid: str) -> int:
    s = t.scheduler
    chunked = chunked_req_of(s)
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
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=100
        )
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
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=110
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_admission_with_chunked_in_flight(self):
        self.server.execute_script(self._script_admission_with_chunked_in_flight)

    @staticmethod
    def _script_admission_with_chunked_in_flight(t: ScriptedContext):
        r_chunk = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=120
        )
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
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=130
        )
        yield from run_until(r, lambda h: h.is_chunking)

        t.abort(r)
        for _ in range(12):
            if (
                chunked_req_of(t.scheduler) is None
                and r.kv_pages == 0
                and r.lock_refs == 0
            ):
                break
            yield

        assert (
            chunked_req_of(t.scheduler) is None
        ), f"abort must clear the chunked slot; got {chunked_req_of(t.scheduler)!r}"
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_get_chunked_req_lambda_getter(self):
        self.server.execute_script(self._script_get_chunked_req_lambda_getter)

    @staticmethod
    def _script_get_chunked_req_lambda_getter(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=140
        )
        yield from run_until(r, lambda h: h.is_chunking)
        saw_match = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                cur = (
                    chunked_req_of(t.scheduler).rid
                    if chunked_req_of(t.scheduler) is not None
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
            chunked_req_of(t.scheduler).rid
            if chunked_req_of(t.scheduler) is not None
            else None
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
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=150
        )
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
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=160
        )
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
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=170
        )
        yield from run_until(r, lambda h: h.is_chunking)

        t.pause_generation(mode="retract")
        yield

        assert (
            chunked_req_of(t.scheduler) is None
        ), f"pause(retract) must clear chunked_req; got {chunked_req_of(t.scheduler)!r}"
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
        r = t.start_req(
            prompt_len=3 * DEFAULT_CHUNK_SIZE, max_new_tokens=2, prompt_token=180
        )
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        assert inflight_middle_chunks_of(r.req) > 0
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
        assert req is not None and inflight_middle_chunks_of(req) == 0, (
            f"retract must reset inflight_middle_chunks; got "
            f"{inflight_middle_chunks_of(req) if req is not None else None}"
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
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=190
        )
        yield from run_until(r, lambda h: h.is_chunking)
        saw_chunking = False
        saw_dedup = False
        for _ in range(DEFAULT_MAX_STEPS):
            chunked = chunked_req_of(t.scheduler)
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
        s = t.scheduler
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=200
        )
        yield from run_until(r, lambda h: h.is_chunking)
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            chunked = chunked_req_of(s)
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
        s = t.scheduler
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=210
        )
        yield from run_until(r, lambda h: h.is_chunking)
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            chunked = chunked_req_of(s)
            if (
                r.is_chunking
                and chunked is not None
                and chunked.rid == r.rid
                and extend_input_len_of(chunked) > 0
            ):
                deduct = extend_input_len_of(chunked)
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

    def test_stage_a_inflight_middle_chunks_sync_invariant(self):
        self.server.execute_script(
            self._script_stage_a_inflight_middle_chunks_sync_invariant
        )

    @staticmethod
    def _script_stage_a_inflight_middle_chunks_sync_invariant(t: ScriptedContext):
        def assert_invariant() -> None:
            req = t.find_req_by_rid(r.rid)
            if req is not None and inflight_middle_chunks_of(req) > 0:
                assert r.is_chunking, (
                    f"invariant violated: inflight_middle_chunks="
                    f"{inflight_middle_chunks_of(req)} but is_chunking={r.is_chunking}"
                )

        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, prompt_token=220
        )
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
        r = t.start_req(
            prompt_len=3 * DEFAULT_CHUNK_SIZE, max_new_tokens=2, prompt_token=230
        )
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        saw_mid_chunk = False
        for _ in range(DEFAULT_MAX_STEPS):
            req = r.req
            if (
                r.is_chunking
                and r.chunks_done >= 1
                and req is not None
                and extend_input_len_of(req) is not None
            ):
                saw_mid_chunk = True
                assert len(req.get_fill_ids()) == len(
                    req.prefix_indices
                ) + extend_input_len_of(req), (
                    f"init_next_round_input must rebuild fill_ids to the committed "
                    f"prefix plus the in-flight chunk; "
                    f"fill_ids_len={len(req.get_fill_ids())}, "
                    f"prefix_indices_len={len(req.prefix_indices)}, "
                    f"extend_input_len={extend_input_len_of(req)}, "
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
        r = t.start_req(
            prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2, prompt_token=240
        )
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
            chunked_req_of(s) is None
        ), f"chunked_req slot must clear after last chunk; got {chunked_req_of(s)!r}"

    def test_second_chunked_admit_blocked_when_chunked_req_set(self):
        self.server.execute_script(
            self._script_second_chunked_admit_blocked_when_chunked_req_set
        )

    @staticmethod
    def _script_second_chunked_admit_blocked_when_chunked_req_set(t: ScriptedContext):
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

    def test_scheduler_continues_with_only_chunked_req_no_waiting(self):
        self.server.execute_script(
            self._script_scheduler_continues_with_only_chunked_req_no_waiting
        )

    @staticmethod
    def _script_scheduler_continues_with_only_chunked_req_no_waiting(
        t: ScriptedContext,
    ):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=250
        )
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


class TestSpecialCaseRowPoolExhaustion(ScriptedTestCase):
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
        decodes = [t.start_req(prompt_len=8, max_new_tokens=16) for _ in range(3)]
        for d in decodes:
            yield from advance_to_decode_step(d, 1)

        r_chunk = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, prompt_token=300
        )
        yield from run_until(r_chunk, lambda h: h.is_chunking)

        comp = t.batch_composition()
        assert r_chunk.rid in comp.get("chunked", [])

        saw_mixed_with_decode = False
        all_reqs = [r_chunk, *decodes]
        for _ in range(DEFAULT_MAX_STEPS * 2):
            comp = t.batch_composition()
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
            prompt_token=310,
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

        r_chunk = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=320
        )
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


DETERMINISTIC_ALIGN_SIZE = 4096


class TestSpecialCaseDeterministicFlashInfer(ScriptedTestCase):
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
        r = t.start_req(
            prompt_len=DETERMINISTIC_ALIGN_SIZE + 1024,
            max_new_tokens=2,
            prompt_token=10,
        )
        page_size = 16
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking and extend_input_len_of(r.req) is not None:
                saw_chunking = True
                assert extend_input_len_of(r.req) % page_size == 0, (
                    f"deterministic chunk boundary must be page-aligned; "
                    f"got extend_input_len={extend_input_len_of(r.req)}, page_size={page_size}"
                )
            if r.finished:
                break
            yield
        assert r.finished, "chunked req did not finish"
        assert saw_chunking, "test must observe the req mid-chunk at least once"


class TestSpecialCaseHiCache(ScriptedTestCase):
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
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=400
        )
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
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=410
        )
        snap: Optional[tuple] = None
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            req = t.find_req_by_rid(r.rid)
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
        assert t.scheduler.enable_dynamic_chunking is False, (
            "pp_size==1 must force enable_dynamic_chunking off even when the "
            "server arg is True (the 'and ps.pp_size > 1' conjunct)"
        )
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


class TestSpecialCaseSmallPool(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        max_total_tokens=SMALL_KV_POOL_MAX_TOTAL_TOKENS,
    )

    @staticmethod
    def _start_ballast(t: ScriptedContext, *, prompt_token: int):
        return t.start_req(
            prompt_len=SMALL_KV_POOL_BALLAST_PROMPT_LEN,
            max_new_tokens=SMALL_KV_POOL_BALLAST_MAX_NEW_TOKENS,
            ignore_eos=True,
            prompt_token=prompt_token,
        )

    @staticmethod
    def _run_force_readd_then_complete(
        t: ScriptedContext, *, chunk_token: int, ballast_token: int
    ):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=chunk_token
        )
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        ballast = TestSpecialCaseSmallPool._start_ballast(t, prompt_token=ballast_token)

        ballast_retracted = False
        for _ in range(2000):
            if ballast.status == "waiting":
                ballast_retracted = True
            if r.finished:
                break
            yield
        assert r.finished, (
            "force-re-added chunked resume must complete once the engine retracts "
            f"the ballast; status={r.status} kv_pages={r.kv_pages}"
        )
        assert r.kv_pages == 0, f"kv_pages={r.kv_pages}"
        assert r.lock_refs == 0, f"lock_refs={r.lock_refs}"
        assert (
            ballast_retracted
            or ballast.finished
            or ballast.status in ("waiting", "finished", "unknown")
        ), f"ballast must be retracted/aborted under pressure; status={ballast.status}"

        t.abort(ballast)
        for _ in range(200):
            if t.is_fully_idle:
                break
            yield

    def test_add_chunked_req_rem_nonpositive_forces_rem_chunk_tokens(self):
        self.server.execute_script(
            self._script_add_chunked_req_rem_nonpositive_forces_rem_chunk_tokens
        )

    @staticmethod
    def _script_add_chunked_req_rem_nonpositive_forces_rem_chunk_tokens(
        t: ScriptedContext,
    ):
        yield from TestSpecialCaseSmallPool._run_force_readd_then_complete(
            t, chunk_token=700, ballast_token=701
        )

    def test_chunked_forced_admission_avoids_leak(self):
        self.server.execute_script(self._script_chunked_forced_admission_avoids_leak)

    @staticmethod
    def _script_chunked_forced_admission_avoids_leak(t: ScriptedContext):
        yield from TestSpecialCaseSmallPool._run_force_readd_then_complete(
            t, chunk_token=710, ballast_token=711
        )

    def test_add_chunked_req_non_swa_forced_admit_on_rem_zero(self):
        self.server.execute_script(
            self._script_add_chunked_req_non_swa_forced_admit_on_rem_zero
        )

    @staticmethod
    def _script_add_chunked_req_non_swa_forced_admit_on_rem_zero(t: ScriptedContext):
        yield from TestSpecialCaseSmallPool._run_force_readd_then_complete(
            t, chunk_token=720, ballast_token=721
        )


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
