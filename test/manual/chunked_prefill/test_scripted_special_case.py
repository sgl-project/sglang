import unittest
from typing import Optional

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)


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

    def test_new_chunked_req_first_chunk(self):
        self.server.execute_script(self._script_new_chunked_req_first_chunk)

    @staticmethod
    def _script_new_chunked_req_first_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield
        yield from run_until(r, lambda h: h.chunks_done >= 1)
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                saw_chunking = True
            if r.finished:
                break
            yield
        assert r.finished
        assert r.chunks_done >= 1
        assert saw_chunking, "first-chunk assignment branch must be exercised"

    def test_inflight_middle_chunks_counter(self):
        self.server.execute_script(self._script_inflight_middle_chunks_counter)

    @staticmethod
    def _script_inflight_middle_chunks_counter(t: ScriptedContext):
        r = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.finished
        assert r.req.inflight_middle_chunks >= 1

    def test_chunked_req_passes_through_batch(self):
        self.server.execute_script(self._script_chunked_req_passes_through_batch)

    @staticmethod
    def _script_chunked_req_passes_through_batch(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        comp = t.batch_composition()
        assert r.rid in comp.get("chunked", [])

        yield from run_until_finished(r)
        assert r.finished

    def test_no_idle_during_chunked(self):
        self.server.execute_script(self._script_no_idle_during_chunked)

    @staticmethod
    def _script_no_idle_during_chunked(t: ScriptedContext):
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
        assert saw_chunking, "test must observe r.is_chunking at least once"

    def test_abort_excludes_chunked_req(self):
        self.server.execute_script(self._script_abort_excludes_chunked_req)

    @staticmethod
    def _script_abort_excludes_chunked_req(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        t.abort(r)
        yield

        assert (
            1 if t._scheduler.chunked_req is not None else 0
        ) == 0, f"abort must clear in-flight count; got {(1 if t._scheduler.chunked_req is not None else 0)}"
        assert (
            t._scheduler.chunked_req.rid
            if t._scheduler.chunked_req is not None
            else None
        ) is None
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
                    t._scheduler.chunked_req.rid
                    if t._scheduler.chunked_req is not None
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
            t._scheduler.chunked_req.rid
            if t._scheduler.chunked_req is not None
            else None
        ) is None

    def test_chunked_req_reset_to_none(self):
        self.server.execute_script(self._script_chunked_req_reset_to_none)

    @staticmethod
    def _script_chunked_req_reset_to_none(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        saw_chunking_match = False
        for _ in range(DEFAULT_MAX_STEPS):
            if (
                r.is_chunking
                and (
                    t._scheduler.chunked_req.rid
                    if t._scheduler.chunked_req is not None
                    else None
                )
                == r.rid
            ):
                saw_chunking_match = True
            if r.finished:
                break
            yield
        assert r.finished
        assert (
            saw_chunking_match
        ), "must observe scheduler.chunked_req == r at least once during chunking"
        assert (
            t._scheduler.chunked_req.rid
            if t._scheduler.chunked_req is not None
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

    def test_idle_path_chunked_req_none(self):
        self.server.execute_script(self._script_idle_path_chunked_req_none)

    @staticmethod
    def _script_idle_path_chunked_req_none(t: ScriptedContext):
        for _ in range(5):
            yield
        assert t.is_idle
        assert (
            t._scheduler.chunked_req.rid
            if t._scheduler.chunked_req is not None
            else None
        ) is None, (
            f"with no in-flight reqs, chunked slot must be None; "
            f"got {(t._scheduler.chunked_req.rid if t._scheduler.chunked_req is not None else None)!r}"
        )
        for _ in range(5):
            assert t.is_idle, "scheduler must remain idle with no work"
            assert (
                t._scheduler.chunked_req.rid
                if t._scheduler.chunked_req is not None
                else None
            ) is None
            yield

    def test_admission_path_with_chunked_inflight_flag(self):
        self.server.execute_script(
            self._script_admission_path_with_chunked_inflight_flag
        )

    @staticmethod
    def _script_admission_path_with_chunked_inflight_flag(t: ScriptedContext):
        r_chunked = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_chunked, lambda h: h.is_chunking)
        comp = t.batch_composition()
        assert r_chunked.rid in comp.get(
            "chunked", []
        ), f"r_chunked must occupy chunked slot before admission; got {comp!r}"
        r_new = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r_chunked)
        yield from run_until_finished(r_new)
        assert r_chunked.finished and r_new.finished

    def test_inflight_counter_increments_each_chunk(self):
        self.server.execute_script(self._script_inflight_counter_increments_each_chunk)

    @staticmethod
    def _script_inflight_counter_increments_each_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=4 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        saw_increment = False
        last = 0
        for _ in range(DEFAULT_MAX_STEPS):
            cur = r.req.inflight_middle_chunks
            if cur > last:
                saw_increment = True
            last = max(last, cur)
            if r.finished:
                break
            yield
        assert saw_increment, "expected inflight_middle_chunks to increment"
        assert r.finished

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

    def test_chunked_req_bypasses_req_pool_exhaustion(self):
        self.server.execute_script(
            self._script_chunked_req_bypasses_req_pool_exhaustion
        )

    @staticmethod
    def _script_chunked_req_bypasses_req_pool_exhaustion(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        chunks_before_pressure = r.chunks_done

        t.exhaust_row_pool(leave_rows=0)

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

    def test_pause_retract_clears_chunked_req(self):
        self.server.execute_script(self._script_pause_retract_clears_chunked_req)

    @staticmethod
    def _script_pause_retract_clears_chunked_req(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        t.force_retract(r)
        yield

        assert (
            t._scheduler.chunked_req.rid
            if t._scheduler.chunked_req is not None
            else None
        ) is None, (
            f"pause(retract) must clear chunked_req; "
            f"got {(t._scheduler.chunked_req.rid if t._scheduler.chunked_req is not None else None)!r}"
        )
        assert (1 if t._scheduler.chunked_req is not None else 0) == 0

    def test_load_inquirer_pending_tokens_dedup_chunked(self):
        self.server.execute_script(
            self._script_load_inquirer_pending_tokens_dedup_chunked
        )

    @staticmethod
    def _script_load_inquirer_pending_tokens_dedup_chunked(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                saw_chunking = True
                snap = t.load_inquirer_snapshot()
                pending = snap["pending_tokens_count_for_rid"](r.rid)
                assert pending <= r.remaining_prompt_tokens, (
                    f"load_inquirer tallied {pending} tokens for r but only "
                    f"{r.remaining_prompt_tokens} are still pending — "
                    "dual-queue dedup violated"
                )
            if r.finished:
                break
            yield
        assert r.finished
        assert (
            saw_chunking
        ), "test must observe the dual-queue chunked state at least once"

    def test_chunked_forced_admission_avoids_leak(self):
        self.server.execute_script(self._script_chunked_forced_admission_avoids_leak)

    @staticmethod
    def _script_chunked_forced_admission_avoids_leak(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        baseline_rows = (
            t._scheduler.req_to_token_pool.size
            - t._scheduler.req_to_token_pool.available_size()
        )
        t.exhaust_kv()
        yield from run_until_finished(r)
        assert r.finished
        assert (
            t._scheduler.req_to_token_pool.size
            - t._scheduler.req_to_token_pool.available_size()
        ) <= baseline_rows, f"row leak under forced chunked admission: baseline={baseline_rows}, after={(t._scheduler.req_to_token_pool.size - t._scheduler.req_to_token_pool.available_size())}"

    def test_stage_a_inflight_middle_chunks_sync_invariant(self):
        self.server.execute_script(
            self._script_stage_a_inflight_middle_chunks_sync_invariant
        )

    @staticmethod
    def _script_stage_a_inflight_middle_chunks_sync_invariant(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        for _ in range(DEFAULT_MAX_STEPS):
            if r.finished:
                return
            if r.req.inflight_middle_chunks > 0:
                assert r.is_chunking, (
                    f"invariant violated: inflight_middle_chunks="
                    f"{r.req.inflight_middle_chunks} but is_chunking={r.is_chunking}"
                )
            yield
        raise AssertionError("req did not finish")

    def test_init_next_round_input_resets_chunk_state(self):
        self.server.execute_script(
            self._script_init_next_round_input_resets_chunk_state
        )

    @staticmethod
    def _script_init_next_round_input_resets_chunk_state(t: ScriptedContext):
        r = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        saw_mid_chunk = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking and r.chunks_done >= 1:
                saw_mid_chunk = True
                assert len(r.req.fill_ids) == len(r.req.prefix_indices), (
                    f"init_next_round_input must reset fill_ids to "
                    f"prefix_indices at every chunk boundary; "
                    f"fill_ids_len={len(r.req.fill_ids)}, "
                    f"prefix_indices_len={len(r.req.prefix_indices)}, "
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
        s = t._scheduler
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
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
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

    def test_chunked_exclude_falls_back_to_last_batch_reqs_when_no_pp(self):
        self.server.execute_script(
            self._script_chunked_exclude_falls_back_to_last_batch_reqs_when_no_pp
        )

    @staticmethod
    def _script_chunked_exclude_falls_back_to_last_batch_reqs_when_no_pp(
        t: ScriptedContext,
    ):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        saw_reqs_branch = False
        for _ in range(DEFAULT_MAX_STEPS):
            source = t.last_chunked_exclude_set_source()
            if source == "last_batch_reqs":
                saw_reqs_branch = True
                assert source != "last_batch_chunked_req"
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished and r2.finished
        assert saw_reqs_branch, (
            "non-PP scheduler must source exclude set from last_batch.reqs "
            "(else branch) at least once during the multi-req lifetime"
        )

    def test_scheduler_continues_with_only_chunked_req_no_waiting(self):
        self.server.execute_script(
            self._script_scheduler_continues_with_only_chunked_req_no_waiting
        )

    @staticmethod
    def _script_scheduler_continues_with_only_chunked_req_no_waiting(
        t: ScriptedContext,
    ):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        prev_chunks_done = r.chunks_done
        progressed = False
        for _ in range(DEFAULT_MAX_STEPS):
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
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.exhaust_kv()
        yield from run_until_finished(r, max_steps=800)
        assert r.finished, (
            "non-SWA chunked-resume must be force-admitted when "
            "_rem_tokens == 0 (schedule_policy.py:679-682); pre-fix it "
            "would block forever and leak its held row + KV"
        )
        assert r.kv_pages == 0
        assert r.lock_refs == 0


class TestSpecialCaseDynamicChunking(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_dynamic_chunking=True,
    )

    def test_dynamic_chunking_history_len(self):
        self.server.execute_script(self._script_dynamic_chunking_history_len)

    @staticmethod
    def _script_dynamic_chunking_history_len(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2


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
        yield from run_until(decodes[0], lambda h: h.status == "running")

        r_chunk = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until(r_chunk, lambda h: h.is_chunking)

        comp = t.batch_composition()
        assert r_chunk.rid in comp.get("chunked", [])
        running_in_batch = set(comp.get("decode", [])) | set(comp.get("running", []))
        assert any(
            d.rid in running_in_batch for d in decodes
        ), f"enable_mixed_chunk should merge decode reqs into chunked iter; got {comp!r}"
        assert (
            t.last_batch_forward_mode == "MIXED"
        ), f"expected forward_mode == MIXED with enable_mixed_chunk, got {t.last_batch_forward_mode!r}"

        all_reqs = [r_chunk, *decodes]
        for _ in range(DEFAULT_MAX_STEPS * 2):
            if all(x.finished for x in all_reqs):
                break
            yield
        assert all(x.finished for x in all_reqs)

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
        impl="transformers",
    )

    def test_multimodal_transformers_disables_chunking(self):
        self.server.execute_script(
            self._script_multimodal_transformers_disables_chunking
        )

    @staticmethod
    def _script_multimodal_transformers_disables_chunking(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert (
            r.chunks_done == 0
        ), f"transformers backend should disable chunking; got chunks_done={r.chunks_done}"


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


class TestSpecialCaseTinyChunk(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=8,
        page_size=16,
    )

    def test_chunked_admission_trunc_lt_zero_returns_other(self):
        self.server.execute_script(
            self._script_chunked_admission_trunc_lt_zero_returns_other
        )

    @staticmethod
    def _script_chunked_admission_trunc_lt_zero_returns_other(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        saw_deferred_iter = False
        prev_chunks_done = r.chunks_done
        for _ in range(DEFAULT_MAX_STEPS):
            if r.finished:
                break
            cur_chunks_done = r.chunks_done
            if r.status == "waiting" and cur_chunks_done == prev_chunks_done:
                saw_deferred_iter = True
            prev_chunks_done = cur_chunks_done
            yield
        assert r.finished
        assert saw_deferred_iter, (
            "expected at least one iter where add_one_req returned OTHER "
            "(req status == waiting and chunks_done did not advance); "
            "chunked_prefill_size < page_size must defer admission"
        )


class TestSpecialCaseDeterministicFlashInfer(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        page_size=16,
        attention_backend="flashinfer",
        enable_deterministic_inference=True,
    )

    def test_chunked_truncation_align_size(self):
        self.server.execute_script(self._script_chunked_truncation_align_size)

    @staticmethod
    def _script_chunked_truncation_align_size(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        page_size = 16
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking and r.req.extend_input_len is not None:
                assert r.req.extend_input_len % page_size == 0, (
                    f"deterministic chunk boundary must be page-aligned; "
                    f"got extend_input_len={r.req.extend_input_len}, page_size={page_size}"
                )
            if r.finished:
                return
            yield
        raise AssertionError("chunked req did not finish")


class TestSpecialCaseHiCache(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_hierarchical_cache=True,
    )

    def test_hicache_breakdown_only_first_chunk(self):
        self.server.execute_script(self._script_hicache_breakdown_only_first_chunk)

    @staticmethod
    def _script_hicache_breakdown_only_first_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        first_chunk_snap = None
        for _ in range(DEFAULT_MAX_STEPS):
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

    def test_hicache_cached_tokens_set_once_invariant(self):
        self.server.execute_script(
            self._script_hicache_cached_tokens_set_once_invariant
        )

    @staticmethod
    def _script_hicache_cached_tokens_set_once_invariant(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        snap: Optional[tuple] = None
        for _ in range(DEFAULT_MAX_STEPS):
            req = t.find_req_by_rid(r.rid)
            if req is not None:
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
        assert snap is not None, "test must observe the req in scheduler at least once"

    def test_init_load_back_called_once_per_request_with_hicache(self):
        self.server.execute_script(
            self._script_init_load_back_called_once_per_request_with_hicache
        )

    @staticmethod
    def _script_init_load_back_called_once_per_request_with_hicache(
        t: ScriptedContext,
    ):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2, (
            f"req must actually chunk for this branch to fire; got "
            f"chunks_done={r.chunks_done}"
        )
        assert r.init_load_back_count == 1, (
            f"HiCache init_load_back must run exactly once per request, "
            f"not once per chunk; got init_load_back_count="
            f"{r.init_load_back_count}"
        )


if __name__ == "__main__":
    unittest.main()
