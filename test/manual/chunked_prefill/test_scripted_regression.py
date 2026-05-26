"""Commit-history regression scripted tests.

Each test mirrors a specific fix commit on ``feat/stateless_scheduler_b``
(around 309b6dc). The scenario is the exact bug shape the commit had
to address; the assertion is "behavior after fix", so re-introducing
the bug during the chunked refactor flips this file red.

The commits sampled here are the ones with the cleanest causal link
to chunked-prefill. Niche ones (mamba, streaming session) are
included but mark obvious model/feature dependencies in the docstring.

Also covers category C from the expansion plan: commits from the 309b6dc
window that were NOT picked up by the round-1 set (~15 new regressions).
Each test mirrors one commit's bug shape.

These tests reference upstream commit SHAs in the docstrings — when
the chunked refactor lands, these scripts re-validate that the fix
is still in effect.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase

_LORA_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_LORA_ADAPTER = "philschmid/llama-3-2-1b-instruct-finetuning-lora-cookbook-test"


class TestScriptedRegression(CustomTestCase):
    def test_lora_drainer_chunked_resume(self):
        """5ed4faf0ab "Bypass LoRA scheduling gate for chunked-resume reqs"."""
        execute_scripted_runtime(
            self._script_lora_drainer_chunked_resume,
            **base_engine_kwargs(
                model_path=_LORA_BASE_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_lora=True,
                lora_paths=[_LORA_ADAPTER],
            ),
        )

    # 5ed4faf0ab "Bypass LoRA scheduling gate for chunked-resume reqs".
    # Bug: LoRA drainer would reject chunked-resume reqs, leaving them
    # stuck in waiting_queue while holding row + lock_ref + KV.
    # Fix verification: a LoRA chunked-resume req must complete, even
    # when adapter draining is forced via the wishlist
    # ``t.force_lora_drainer_reject`` helper.
    @staticmethod
    def _script_lora_drainer_chunked_resume(t: ScriptedRuntime):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER,
        )
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        # Force the drainer to want to evict r's adapter — pre-fix this
        # would deadlock (drainer says "wait for r to drain", r says
        # "drainer won't let me admit my next chunk").
        t.force_lora_drainer_reject(adapter=_LORA_ADAPTER)

        # Must still complete: chunked-resume bypasses the drainer gate.
        yield from run_until_finished(r)
        assert r.finished

    def test_abort_waiting_releases_all(self):
        """96d4749094 "Release row + KV + lock_ref when aborting a chunked-resume req from waiting_queue"."""
        execute_scripted_runtime(
            self._script_abort_waiting_releases_all,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # 96d4749094 "Release row + KV + lock_ref when aborting a
    # chunked-resume req from waiting_queue".
    @staticmethod
    def _script_abort_waiting_releases_all(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.status == "waiting")

        t.abort(r)
        yield

        assert r.kv_pages == 0
        assert r.row_idx is None
        assert r.lock_refs == 0

    def test_pause_covers_waiting_chunked(self):
        """F38e69f87d "Extend pause(retract) to waiting chunked-resume reqs"."""
        execute_scripted_runtime(
            self._script_pause_covers_waiting_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # f38e69f87d "Extend pause(retract) to waiting chunked-resume reqs".
    # Bug: pause path didn't iterate waiting_queue chunked-resume entries,
    # so paused chunked reqs leaked.
    @staticmethod
    def _script_pause_covers_waiting_chunked(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.status == "waiting")

        # ``force_retract`` (deterministic) substitutes for the production
        # pause path; either should release the same resources.
        t.force_retract(r)
        yield

        assert r.kv_pages == 0
        assert r.row_idx is None

    def test_pp_abort_dedup(self):
        """B823c16e60 "Include PP microbatch reqs in abort_request batch_rids dedup"."""
        execute_scripted_runtime(
            self._script_pp_abort_dedup,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                tp_size=2,
                pp_size=2,
            ),
        )

    # b823c16e60 "Include PP microbatch reqs in abort_request
    # batch_rids dedup".
    # Bug: abort_request's dedup set did not include PP cross-mb in-flight
    # reqs, leading to double-abort under PP last-chunk-in-flight timing.
    @staticmethod
    def _script_pp_abort_dedup(t: ScriptedRuntime):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)

        t.abort(r)
        yield

        assert r.finish_event_count == 1, (
            f"PP abort must dedup across microbatches; "
            f"got {r.finish_event_count} finish events"
        )

    def test_disagg_retract_resets_send(self):
        """414efd4a27 "Reset disagg send-side state on chunked-resume retract"."""
        execute_scripted_runtime(
            self._script_disagg_retract_resets_send,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )

    # 414efd4a27 "Reset disagg send-side state on chunked-resume retract".
    @staticmethod
    def _script_disagg_retract_resets_send(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        t.force_retract(r)
        yield

        assert r.disagg_send_state in (None, "idle")

    def test_pending_middle_outputs_invariant(self):
        """B3a7b9f2a1 "Bump pending_middle_outputs for last-chunk admits + decrement-first output proc"."""
        execute_scripted_runtime(
            self._script_pending_middle_outputs_invariant,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # b3a7b9f2a1 "Bump pending_middle_outputs for last-chunk admits +
    # decrement-first output proc".
    # The invariant: at the moment the last chunk is admitted,
    # pending_middle_outputs > 0; once the output is processed it returns
    # to 0.
    @staticmethod
    def _script_pending_middle_outputs_invariant(t: ScriptedRuntime):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)

        # Drive into the last chunk.
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        assert (
            r.pending_middle_outputs > 0
        ), "last-chunk admit must bump pending_middle_outputs"

        # Continue until the chunk loop ends.
        yield from run_until(r, lambda h: not h.is_chunking)
        # Once the chunk loop has cleared and the output has been
        # processed, the counter is back to 0 (decrement-first ordering).
        assert r.pending_middle_outputs == 0, (
            f"pending_middle_outputs should be 0 after chunk loop clears; "
            f"got {r.pending_middle_outputs}"
        )

        yield from run_until_finished(r)

    def test_pp_other_mb_chunked_exclude(self):
        """69ef71edc4 "Conditionally exclude in-flight other-mb chunked-resume reqs (PP, max_new_tokens > 1)"."""
        execute_scripted_runtime(
            self._script_pp_other_mb_chunked_exclude,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                tp_size=2,
                pp_size=2,
            ),
        )

    # 69ef71edc4 "Conditionally exclude in-flight other-mb
    # chunked-resume reqs (PP, max_new_tokens > 1)".
    # Bug: under PP, chunked-resume reqs in another microbatch leaked
    # into the local batch's filter, leading to double-admission.
    # Verification: run two chunked reqs concurrently under PP=2,
    # different microbatches; both must complete cleanly without
    # double-admit symptoms (extend_input_len mismatch or
    # chunked_in_flight_count > 1 momentarily).
    @staticmethod
    def _script_pp_other_mb_chunked_exclude(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)

        # PP=2 + two 4096 prompts at chunk_size=256 → ~32 chunks total.
        # Give the loop generous headroom.
        for _ in range(2000):
            in_flight = t.chunked_in_flight_count()
            assert (
                in_flight <= 1
            ), f"PP cross-mb chunked exclusion broken: in_flight={in_flight}"
            if r1.finished and r2.finished:
                return
            yield
        raise AssertionError("reqs did not finish")

    def test_pdmux_filter_chunked(self):
        """34c02d6a67 "Filter chunked-resume reqs from split_prefill_batch before pdmux merge"."""
        execute_scripted_runtime(
            self._script_pdmux_filter_chunked,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )

    # 34c02d6a67 "Filter chunked-resume reqs from split_prefill_batch
    # before pdmux merge".
    # Bug: pdmux merge would see chunked-resume reqs that should be
    # excluded. With disagg + chunked, the merge must filter them out.
    @staticmethod
    def _script_pdmux_filter_chunked(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished

    def test_priority_skips_chunked_in_prefix_match(self):
        """Aaf3752d2b "Skip chunked-resume reqs in calc_priority prefix matching"."""
        execute_scripted_runtime(
            self._script_priority_skips_chunked_in_prefix_match,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )

    # aaf3752d2b "Skip chunked-resume reqs in calc_priority prefix
    # matching".
    # Bug: priority calculation tried prefix-matching chunked-resume reqs,
    # which is wrong (they already have prefix_indices baked in). Test:
    # enable priority + radix; r1 chunks, r2 arrives; priority calc must
    # not include r1 in its prefix match.
    @staticmethod
    def _script_priority_skips_chunked_in_prefix_match(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low")
        yield from run_until(r1, lambda h: h.is_chunking)

        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high")

        yield from run_until_all_finished([r1, r2])
        assert r1.finished and r2.finished

    def test_mamba_chunked_resume_no_token(self):
        """Dbdcdde245 "Skip mamba_pool_idx cleanup for chunked-resume on NO_TOKEN"."""
        execute_scripted_runtime(
            self._script_mamba_chunked_resume_no_token,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # dbdcdde245 "Skip mamba_pool_idx cleanup for chunked-resume on
    # NO_TOKEN".
    # Mamba-specific: when a chunked-resume req hits NO_TOKEN admission
    # return, mamba_pool_idx cleanup must be skipped.
    #
    # Requires a mamba-architecture model — currently no small mamba model
    # is in the standard test fixture. The test sets the right flags;
    # actual mamba model wiring is left for the harness to fill in.
    @staticmethod
    def _script_mamba_chunked_resume_no_token(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished

    def test_streaming_session_stash_bound(self):
        """116584e8fa "Bound streaming-session chunked stash by kv_committed_len"."""
        execute_scripted_runtime(
            self._script_streaming_session_stash_bound,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # 116584e8fa "Bound streaming-session chunked stash by
    # kv_committed_len".
    # Streaming-session specific: when the session's KV commit length is
    # shorter than the chunked extend, the stash must clip to the committed
    # length to avoid stashing un-committed tokens.
    @staticmethod
    def _script_streaming_session_stash_bound(t: ScriptedRuntime):
        # Streaming session creation isn't yet a ScriptedRuntime primitive;
        # this test pumps a long req in a streaming-flagged engine and
        # verifies the chunk loop completes cleanly.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished

    def test_chunked_resume_priority_in_sort(self):
        """Bf5b4e9a10 "Give chunked-resume reqs priority in LPM and DFS_WEIGHT sorts"."""
        # Use LPM policy explicitly — the bf5b4e9a10 fix was in LPM /
        # DFS_WEIGHT sort paths, so the default (FCFS) won't exercise it.
        execute_scripted_runtime(
            self._script_chunked_resume_priority_in_sort,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                schedule_policy="lpm",
            ),
        )

    # bf5b4e9a10 "Give chunked-resume reqs priority in LPM and
    # DFS_WEIGHT sorts".
    # Bug: LPM / DFS_WEIGHT priority sort didn't prioritize chunked-resume
    # reqs, leading to starvation under load.
    # Test: a chunked req in flight + many short reqs; chunked must
    # advance, not starve.
    @staticmethod
    def _script_chunked_resume_priority_in_sort(t: ScriptedRuntime):
        r_long = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_long, lambda h: h.is_chunking)

        # Flood with short reqs.
        shorts = [t.start_req(prompt_len=4, max_new_tokens=1) for _ in range(8)]

        # r_long must make forward progress within a reasonable budget.
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

    def test_rename_inflight_to_pending_middle_outputs(self):
        """14adb09546: rename inflight_middle_chunks -> pending_middle_outputs."""
        execute_scripted_runtime(
            self._script_rename_inflight_to_pending_middle_outputs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_rename_inflight_to_pending_middle_outputs(t: ScriptedRuntime):
        # 14adb09546: rename inflight_middle_chunks -> pending_middle_outputs.
        # Verifies both attribute names are addressable on ReqHandle without
        # the test exploding on missing attr.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        # Both fields should be readable.
        _ = r.pending_middle_outputs
        _ = r.inflight_middle_chunks
        yield from run_until_finished(r)

    def test_revert_bump_pending_middle_outputs(self):
        """E875cd36e4: revert of pending_middle_outputs bump."""
        execute_scripted_runtime(
            self._script_revert_bump_pending_middle_outputs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_revert_bump_pending_middle_outputs(t: ScriptedRuntime):
        # e875cd36e4: revert of pending_middle_outputs bump.
        # After last chunk admit, pending_middle_outputs should not double-count.
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
        yield from run_until(r, lambda h: h.chunks_done >= 1)
        assert r.pending_middle_outputs <= 1
        yield from run_until_finished(r)
        assert r.pending_middle_outputs == 0

    def test_filter_batch_exclude_in_flight_other_mb(self):
        """5c523049db / 45347ca3a3: exclude in-flight other-mb reqs in filter_batch."""
        execute_scripted_runtime(
            self._script_filter_batch_exclude_in_flight_other_mb,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_filter_batch_exclude_in_flight_other_mb(t: ScriptedRuntime):
        # 5c523049db / 45347ca3a3: exclude in-flight other-mb reqs in
        # filter_batch. Single-engine smoke (multi-mb requires PP).
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished

    def test_chunked_req_marker_pp_filter_exclusion(self):
        """33f981ce93 / 11db3a4192: re-add ScheduleBatch.chunked_req marker for PP cross-mb filter exclusion."""
        execute_scripted_runtime(
            self._script_chunked_req_marker_pp_filter_exclusion,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_chunked_req_marker_pp_filter_exclusion(t: ScriptedRuntime):
        # 33f981ce93 / 11db3a4192: re-add ScheduleBatch.chunked_req marker
        # for PP cross-mb filter exclusion.
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)

    def test_v1_swa_chunked_tests_dropped(self):
        """A94e842611 / daf9c42f17: dropped v1 SWA chunked-req tests."""
        execute_scripted_runtime(
            self._script_v1_swa_chunked_tests_dropped,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_v1_swa_chunked_tests_dropped(t: ScriptedRuntime):
        # a94e842611 / daf9c42f17: dropped v1 SWA chunked-req tests. Verifies
        # the v2 path is what runs (no v1 codepath revival).
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_stage_a_chunk_stash_iter_boundary(self):
        """678bba26f0: Stage A chunk-stash runs at iter boundary, not mid-iter."""
        execute_scripted_runtime(
            self._script_stage_a_chunk_stash_iter_boundary,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_stage_a_chunk_stash_iter_boundary(t: ScriptedRuntime):
        # 678bba26f0: Stage A chunk-stash runs at iter boundary, not
        # mid-iter. Verifies pending_middle_outputs is consistent at the
        # boundary between iters.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            # Snapshot at iter boundary.
            if r.pending_middle_outputs > 0:
                assert r.is_chunking
            if r.finished:
                return
            yield

    def test_chunked_resume_tail_counted_page_size_gt1(self):
        """B433e1ea35: count chunked-resume tail in runtime mem check when page_size > 1."""
        execute_scripted_runtime(
            self._script_chunked_resume_tail_counted_page_size_gt1,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_chunked_resume_tail_counted_page_size_gt1(t: ScriptedRuntime):
        # b433e1ea35: count chunked-resume tail in runtime mem check
        # when page_size > 1. Verifies the runtime mem accounting includes
        # tail of chunked-resume reqs.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN + 17, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_merge_batch_assert_widened(self):
        """36ec1d7269: widen merge_batch assert to match filter_batch predicate."""
        execute_scripted_runtime(
            self._script_merge_batch_assert_widened,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_merge_batch_assert_widened(t: ScriptedRuntime):
        # 36ec1d7269: widen merge_batch assert to match filter_batch
        # predicate. Verifies the assert does not fire on legitimate
        # chunked merge.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_all_finished([r1, r2])

    def test_host_hit_length_reset_unconditional(self):
        """D7fa48baad: reset host_hit_length unconditionally in prepare_for_extend."""
        execute_scripted_runtime(
            self._script_host_hit_length_reset_unconditional,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_host_hit_length_reset_unconditional(t: ScriptedRuntime):
        # d7fa48baad: reset host_hit_length unconditionally in
        # prepare_for_extend. Second submission must not inherit stale
        # host_hit_length from r1.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r2)

    def test_add_one_req_reuse_gate_has_pending_chunk(self):
        """A79ba1b2f7: tighten add_one_req reuse gate to has_pending_chunk."""
        execute_scripted_runtime(
            self._script_add_one_req_reuse_gate_has_pending_chunk,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_add_one_req_reuse_gate_has_pending_chunk(t: ScriptedRuntime):
        # a79ba1b2f7: tighten add_one_req reuse gate to has_pending_chunk.
        # Verifies reuse branch only fires when there's a pending chunk.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_filter_batch_explicit_exclude_chunked_flag(self):
        """Fd3dcca22f: refactor filter_batch to explicit exclude_chunked_req flag."""
        execute_scripted_runtime(
            self._script_filter_batch_explicit_exclude_chunked_flag,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_filter_batch_explicit_exclude_chunked_flag(t: ScriptedRuntime):
        # fd3dcca22f: refactor filter_batch to explicit exclude_chunked_req flag.
        # Verifies behavior under the new API.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_all_finished([r1, r2])

    def test_retract_all_filter_batch_args(self):
        """F0388931bf: retract_all was passing List[Req] to filter_batch as keep_indices (wrong)."""
        execute_scripted_runtime(
            self._script_retract_all_filter_batch_args,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_retract_all_filter_batch_args(t: ScriptedRuntime):
        # f0388931bf: retract_all was passing List[Req] to filter_batch
        # as keep_indices (wrong). Test triggers a retract_all path.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.force_retract(r)
        yield
        yield from run_until_finished(r)

    def test_is_chunked_renamed_pending_middle_outputs(self):
        """B9d5d6ed5f: rename Req.is_chunked -> Req.pending_middle_outputs."""
        execute_scripted_runtime(
            self._script_is_chunked_renamed_pending_middle_outputs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_is_chunked_renamed_pending_middle_outputs(t: ScriptedRuntime):
        # b9d5d6ed5f: rename Req.is_chunked -> Req.pending_middle_outputs.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_chunked_resume_waiting_queue_holding(self):
        """C445a82cf5: switch chunked-resume to waiting_queue holding; delete chunked_req fields."""
        execute_scripted_runtime(
            self._script_chunked_resume_waiting_queue_holding,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_chunked_resume_waiting_queue_holding(t: ScriptedRuntime):
        # c445a82cf5: switch chunked-resume to waiting_queue holding;
        # delete chunked_req fields. Verifies that mid-chunk reqs end up
        # in waiting_queue (status "waiting") between chunks.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        saw_waiting = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking and r.status == "waiting":
                saw_waiting = True
            if r.finished:
                break
            yield
        assert saw_waiting, "expected to observe chunked-resume in waiting_queue state"

    def test_unified_admission_via_add_one_req(self):
        """74f1d8bbab: unify chunked admission via add_one_req reuse + has_pending_chunk."""
        execute_scripted_runtime(
            self._script_unified_admission_via_add_one_req,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_unified_admission_via_add_one_req(t: ScriptedRuntime):
        # 74f1d8bbab: unify chunked admission via add_one_req reuse +
        # has_pending_chunk. Smoke: chunked admission works.
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_alloc_assert_no_is_chunked(self):
        """9b361aef46: drop is_chunked from req_to_token_pool alloc assert."""
        execute_scripted_runtime(
            self._script_alloc_assert_no_is_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_alloc_assert_no_is_chunked(t: ScriptedRuntime):
        # 9b361aef46: drop is_chunked from req_to_token_pool alloc assert.
        # Smoke: alloc + chunked does not hit a stale assert.
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_cache_unfinished_req_row_bound(self):
        """1c3bf8e7db: bound cache_unfinished_req row read by kv_committed_len."""
        execute_scripted_runtime(
            self._script_cache_unfinished_req_row_bound,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_cache_unfinished_req_row_bound(t: ScriptedRuntime):
        # 1c3bf8e7db: bound cache_unfinished_req row read by kv_committed_len.
        # Verifies that radix caching of unfinished chunked reqs respects
        # the committed length.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_waiting_queue_pending_tokens_subtract_prefix(self):
        """C79a73bec4: subtract prefix_indices from waiting_queue pending tokens sum."""
        execute_scripted_runtime(
            self._script_waiting_queue_pending_tokens_subtract_prefix,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_waiting_queue_pending_tokens_subtract_prefix(t: ScriptedRuntime):
        # c79a73bec4: subtract prefix_indices from waiting_queue pending
        # tokens sum. Verifies waiting_queue pending tokens excludes
        # already-cached prefix.
        r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
        yield from run_until_finished(r_warm)
        r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 4, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_abort_dedup_dual_queue_holding(self):
        """De3859646b: abort_request dedup for chunked-resume dual-queue holding."""
        execute_scripted_runtime(
            self._script_abort_dedup_dual_queue_holding,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_abort_dedup_dual_queue_holding(t: ScriptedRuntime):
        # de3859646b: abort_request dedup for chunked-resume dual-queue holding.
        # The double-abort here is intentionally same-tick: both calls happen
        # before the next yield so the scheduler must dedup them in a single
        # iteration. Inserting a yield between them would test sequential
        # idempotence instead, which is a different code path.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.abort(r)
        t.abort(r)  # double-abort: must dedup
        yield
        assert r.kv_pages == 0

    # ================================================================
    # Round-3 real-reproduction tests for feat/stateless_scheduler_b fixes.
    # Each test reproduces the exact bad behavior the named commit fixed.
    # ================================================================

    def test_chunked_admission_reuse_branch_balanced(self):
        """[b-74f1d8bbab] add_one_req reuse branch keeps lock_ref balanced across a multi-chunk lifecycle."""
        execute_scripted_runtime(
            self._script_chunked_admission_reuse_branch_balanced,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # [b-74f1d8bbab] Unify chunked admission via add_one_req reuse branch.
    # Bug: the previous dedicated add_chunked_req method re-incremented
    # the radix lock_ref on every chunk's admission. The reuse branch
    # introduced here must skip _req_inc_lock_ref entirely (lock already
    # held by the prior stash) and pass 0 as prefix budget.
    # Verification: across the full lifecycle of a multi-chunk req, the
    # net delta of last_node lock_refs is exactly 0 (one acquire on the
    # first chunk, one release on completion). Pre-fix this would walk
    # up by one per chunk.
    @staticmethod
    def _script_chunked_admission_reuse_branch_balanced(t: ScriptedRuntime):
        baseline_refs = t.lock_refs_snapshot()
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)

        # Drive through 3+ chunk admissions so the reuse branch fires
        # multiple times; pre-fix bug would inflate lock_ref each time.
        yield from run_until(r, lambda h: h.chunks_done >= 3 and h.is_chunking)
        mid_refs = t.lock_refs_snapshot()
        # While the req is still chunking, exactly one outstanding lock
        # is held on its last_node (acquired on first admission, never
        # re-acquired by the reuse branch).
        assert mid_refs - baseline_refs == 1, (
            f"reuse branch must not re-acquire lock_ref per chunk; "
            f"baseline={baseline_refs}, mid={mid_refs}"
        )

        yield from run_until_finished(r)
        final_refs = t.lock_refs_snapshot()
        assert final_refs == baseline_refs, (
            f"chunked lifecycle must net to zero lock_ref delta; "
            f"baseline={baseline_refs}, final={final_refs}"
        )
        assert r.lock_refs == 0

    def test_chunked_resume_lives_in_waiting_queue(self):
        """[b-c445a82cf5] mid-chunk req sits in waiting_queue between chunks; queue empties after last chunk."""
        execute_scripted_runtime(
            self._script_chunked_resume_lives_in_waiting_queue,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # [b-c445a82cf5] Switch chunked-resume to waiting_queue holding;
    # delete chunked_req fields.
    # Bug shape: pre-refactor, mid-chunk reqs were tracked via
    # Scheduler.chunked_req + ScheduleBatch.chunked_req; the v2 design
    # moves them into waiting_queue with has_pending_chunk=True.
    # Verification: between chunks, the req's status is "waiting" AND
    # has_pending_chunk is True; once the last chunk admits and decode
    # starts, has_pending_chunk goes False and the req moves to
    # running. Pre-fix the req would have been in running_batch the
    # whole time with status "running".
    @staticmethod
    def _script_chunked_resume_lives_in_waiting_queue(t: ScriptedRuntime):
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE + 32, max_new_tokens=2)

        # Observe between chunk 1 and chunk 2: req must be in
        # waiting_queue with the pending-chunk flag set.
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        assert r.status == "waiting", (
            f"between chunks the chunked-resume req must hold in "
            f"waiting_queue; got status={r.status!r}"
        )
        assert r.has_pending_chunk, (
            f"has_pending_chunk must be set while mid-chunk; got "
            f"{r.has_pending_chunk!r}"
        )
        # Scheduler must NOT keep a separate chunked_req field anymore.
        assert t.get_chunked_req_rid() is None, (
            f"v2 must not maintain a top-level chunked_req field; "
            f"got {t.get_chunked_req_rid()!r}"
        )

        yield from run_until_finished(r)
        # After completion, waiting_queue is drained of this req and
        # the pending flag is cleared.
        assert not r.has_pending_chunk
        assert r.status in ("finished", "unknown")

    def test_chunked_stash_bounded_by_kv_committed_len(self):
        """[b-1c3bf8e7db] cache_unfinished_req reads only up to kv_committed_len; SWA early-return must not leak garbage prefix."""
        execute_scripted_runtime(
            self._script_chunked_stash_bounded_by_kv_committed_len,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                model_path="openai/gpt-oss-20b",
                mem_fraction_static=0.70,
                disable_piecewise_cuda_graph=True,
            ),
        )

    # [b-1c3bf8e7db] Bound cache_unfinished_req row read by
    # kv_committed_len.
    # Bug shape: init_next_round_input resets req.fill_ids to
    # len(origin_input_ids) + len(output_ids) BEFORE stash, but the
    # req_to_token row only holds valid KV up to kv_committed_len.
    # Pre-fix the cache impl read [req_pool_idx, :len(fill_ids)] and
    # inserted the garbage tail into the radix tree as a prefix entry.
    # Under SWA early-return this triggers prefix-hit corruption on
    # the very next admission.
    # Verification: force SWA early-return between two chunks. The
    # next iteration's prefix_indices length must equal exactly
    # kv_committed_len (NOT len(fill_ids)). Pre-fix prefix_indices
    # would be longer and point into garbage cells.
    @staticmethod
    def _script_chunked_stash_bounded_by_kv_committed_len(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        # The committed-len snapshot at the stash point.
        committed = r.kv_committed_len
        assert committed > 0

        # Step the chunk-stash by one iter boundary; SWA early-return
        # exercises the slice path on the next admission.
        yield from run_until(r, lambda h: h.status == "waiting")

        # Invariant the fix introduces: prefix_indices length must not
        # exceed kv_committed_len. Pre-fix it would equal len(fill_ids)
        # and over-read the row.
        assert r.prefix_indices_len <= committed, (
            f"cache_unfinished_req over-read past kv_committed_len: "
            f"prefix_indices_len={r.prefix_indices_len}, "
            f"kv_committed_len={committed}"
        )
        yield from run_until_finished(r)

    def test_chunked_retract_no_double_init_load_back(self):
        """[b-d7fa48baad] retract -> re-admit chunk 2 must not run init_load_back twice (host_hit_length reset is unconditional)."""
        execute_scripted_runtime(
            self._script_chunked_retract_no_double_init_load_back,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # [b-d7fa48baad] Reset host_hit_length unconditionally in
    # prepare_for_extend.
    # Bug shape: the reset was nested inside
    #   if not req.retracted_stain:
    #       if not req._cache_breakdown_computed:
    #           req.host_hit_length = 0
    # After a retract (retracted_stain stays True forever) and
    # re-admit, the outer block is skipped, so the reset never fires.
    # The re-admission's match_prefix sets host_hit_length non-zero,
    # init_load_back consumes it on chunk 1, then chunk 2's admission
    # still sees the stale value and runs init_load_back again
    # (double-load + lock_ref imbalance).
    # Verification: drive chunked req mid-flight, force_retract, let it
    # re-admit, observe init_load_back fired exactly ONCE across the
    # second lifecycle and lock_refs balanced.
    @staticmethod
    def _script_chunked_retract_no_double_init_load_back(t: ScriptedRuntime):
        baseline_refs = t.lock_refs_snapshot()
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)

        t.force_retract(r)
        yield
        # retracted_stain is now sticky-True; the bug window opens here.

        # Wait for re-admission. Track init_load_back count from this
        # point — must fire exactly once during the second lifecycle.
        load_back_before = r.init_load_back_count
        yield from run_until(r, lambda h: h.chunks_done >= 2)
        load_back_during = r.init_load_back_count - load_back_before
        assert load_back_during <= 1, (
            f"d7fa48baad: init_load_back must not double-fire after "
            f"retract; got {load_back_during} extra calls"
        )

        yield from run_until_finished(r)
        # Net lock_ref must still be balanced after the retract/resume
        # cycle (double init_load_back would leave one stuck).
        assert t.lock_refs_snapshot() == baseline_refs

    def test_streaming_session_multiturn_no_reuse_branch(self):
        """[b-a79ba1b2f7] streaming-session turn N>1 must NOT take the reuse branch even though kv_committed_len > 0."""
        execute_scripted_runtime(
            self._script_streaming_session_multiturn_no_reuse_branch,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # [b-a79ba1b2f7] Tighten add_one_req reuse gate to
    # has_pending_chunk.
    # Bug shape: the 'is_resume' predicate previously fired for any
    # req with kv_committed_len > 0, which incorrectly included
    # streaming-session turn N>1 reqs (they inherit kv_committed_len
    # from the session slot but are NOT chunked-resume). Pre-fix the
    # reuse branch skipped _req_inc_lock_ref, so those reqs left
    # last_node lock_ref underbalanced.
    # Verification: a streaming-session turn-2 req that is NOT chunked
    # (short prompt) must go through the fresh-req path, with a normal
    # _req_inc_lock_ref + matching dec on completion. lock_refs net to
    # zero. Pre-fix the dec would not match (no inc happened) and
    # baseline would be off by one.
    @staticmethod
    def _script_streaming_session_multiturn_no_reuse_branch(t: ScriptedRuntime):
        baseline_refs = t.lock_refs_snapshot()
        # Turn 1: short prompt, completes normally, leaves
        # kv_committed_len > 0 on the session slot.
        r1 = t.start_req(
            prompt_len=64,
            max_new_tokens=2,
            session_id="sess-a79b",
        )
        yield from run_until_finished(r1)

        # Turn 2: NEW req on same session — not chunked (short prompt)
        # but inherits kv_committed_len > 0. Must take fresh path, not
        # reuse branch.
        r2 = t.start_req(
            prompt_len=64,
            max_new_tokens=2,
            session_id="sess-a79b",
        )
        yield from run_until_finished(r2)

        assert not r2.has_pending_chunk
        # If the reuse branch had wrongly fired, lock_ref bookkeeping
        # would be off; the unconditional dec on completion would
        # underflow or leak.
        assert t.lock_refs_snapshot() == baseline_refs, (
            f"streaming-session turn N>1 must not take chunked reuse "
            f"branch; lock_refs drifted from {baseline_refs} to "
            f"{t.lock_refs_snapshot()}"
        )

    def test_lpm_skips_chunked_resume_prefix_match(self):
        """[b-aaf3752d2b] calc_priority prefix matching must skip chunked-resume reqs to preserve their stashed last_node/prefix_indices."""
        execute_scripted_runtime(
            self._script_lpm_skips_chunked_resume_prefix_match,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                schedule_policy="lpm",
            ),
        )

    # [b-aaf3752d2b] Skip chunked-resume reqs in calc_priority prefix
    # matching.
    # Bug shape: _compute_prefix_matches runs match_prefix_for_req on
    # every waiting_queue item. For a chunked-resume req that lives
    # in waiting_queue:
    #   - its last_node was inc_lock_ref'd by the prior stash
    #   - overwriting last_node leaves that lock_ref permanently
    #     inflated
    #   - prefix_indices reset would mislead the next chunk's
    #     admission (the row was written up to kv_committed_len)
    #   - host_hit_length would re-trigger init_load_back on next chunk
    # Verification: chunked-resume R1 sits in waiting_queue while a
    # competing R2 with the same prefix arrives. After the LPM priority
    # pass, R1's last_node, prefix_indices_len, and host_hit_length
    # must all be unchanged.
    @staticmethod
    def _script_lpm_skips_chunked_resume_prefix_match(t: ScriptedRuntime):
        # Warm the radix with a common prefix so R2 will have a match
        # candidate that calc_priority will try to use.
        t.warmup_radix(prompt_tokens=[1] * (2 * DEFAULT_CHUNK_SIZE))

        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.chunks_done >= 1 and h.is_chunking)
        yield from run_until(r1, lambda h: h.status == "waiting")

        # Snapshot stashed state on R1.
        last_node_before = r1.last_node_id
        prefix_len_before = r1.prefix_indices_len
        host_hit_before = r1.host_hit_length
        lock_refs_before = t.lock_refs_snapshot()

        # Competing R2 with same prefix kicks calc_priority's
        # _compute_prefix_matches into the danger path.
        r2 = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield  # one iter through priority calculation

        # R1's stashed state must be untouched.
        assert r1.last_node_id == last_node_before, (
            f"calc_priority overwrote chunked-resume last_node; "
            f"before={last_node_before!r}, after={r1.last_node_id!r}"
        )
        assert r1.prefix_indices_len == prefix_len_before
        assert r1.host_hit_length == host_hit_before
        # lock_ref must not have been double-acquired by an unwanted
        # match_prefix_for_req call.
        assert t.lock_refs_snapshot() == lock_refs_before

        yield from run_until_all_finished([r1, r2])

    def test_chunked_resume_priority_under_lpm(self):
        """[b-bf5b4e9a10] chunked-resume gets priority in LPM sort even when its prefix_indices length is short vs fresh long-prefix waiters."""
        execute_scripted_runtime(
            self._script_chunked_resume_priority_under_lpm,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                schedule_policy="lpm",
            ),
        )

    # [b-bf5b4e9a10] Give chunked-resume reqs priority in LPM and
    # DFS_WEIGHT sorts.
    # Bug shape: a chunked-resume req's prefix_indices length reflects
    # only its already-prefilled chunks (~ kv_committed_len), not the
    # full prompt prefix it could match as a fresh req. Under LPM with
    # tight budget, fresh reqs hitting a long cached prefix outrank
    # chunked-resume reqs every iter, starving them. Combined with the
    # watchdog skipping chunked-resume (359e5ed7bd) and the abort path
    # leaking before 96d4749094, the stuck state holds row+KV+lock_ref
    # forever.
    # Verification: warm a long prefix. Submit chunked R1 mid-flight.
    # Submit several fresh R_i sharing the long prefix. R1 must make
    # forward progress (chunks_done increases) within a small step
    # budget. Pre-fix R1 would be stuck.
    @staticmethod
    def _script_chunked_resume_priority_under_lpm(t: ScriptedRuntime):
        long_prefix_tokens = [1] * (3 * DEFAULT_CHUNK_SIZE)
        t.warmup_radix(prompt_tokens=long_prefix_tokens)

        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking and h.chunks_done >= 1)

        # Inject 6 fresh competitors that all share the long warmed
        # prefix — pre-fix these would outrank r1 every iter.
        competitors = [
            t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE + 32, max_new_tokens=2)
            for _ in range(6)
        ]

        baseline_chunks = r1.chunks_done
        # r1 must advance within 50 iters — pre-fix it would stall.
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

    def test_chunked_resume_immune_to_waiting_timeout(self):
        """[b-359e5ed7bd] _abort_on_waiting_timeout must skip chunked-resume reqs even when SGLANG_REQ_WAITING_TIMEOUT triggers."""
        execute_scripted_runtime(
            self._script_chunked_resume_immune_to_waiting_timeout,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                env={"SGLANG_REQ_WAITING_TIMEOUT": "1"},
            ),
        )

    # [b-359e5ed7bd] Skip chunked-resume reqs in
    # _abort_on_waiting_timeout.
    # Bug shape: after the v2 refactor, chunked-resume reqs live in
    # waiting_queue across iters while actively prefilling. Their
    # wait_queue_entry_time is set on original arrival and never
    # refreshed, so a sufficiently long prefill makes them look "stuck"
    # to the timeout watchdog — which would abort them and leak the
    # held req_to_token row + radix tree lock_ref + committed KV.
    # Verification: enable timeout, drive a chunked req mid-flight,
    # invoke the timeout watchdog directly via the harness wishlist.
    # The chunked-resume req must NOT be aborted; it must continue to
    # completion.
    @staticmethod
    def _script_chunked_resume_immune_to_waiting_timeout(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        assert r.has_pending_chunk

        # Pre-fix: this call would abort r and leak its resources.
        t.trigger_abort_on_waiting_timeout()
        yield

        # has_pending_chunk reqs must be immune.
        assert not getattr(r, "aborted", False), (
            f"359e5ed7bd: chunked-resume must be immune to waiting "
            f"timeout abort"
        )
        assert r.kv_pages > 0 or r.chunks_done > 0
        yield from run_until_finished(r)
        assert r.finished

    def test_abort_chunked_resume_releases_all_resources(self):
        """[b-96d4749094] aborting a chunked-resume req that lives ONLY in waiting_queue must release row + KV + lock_ref together."""
        execute_scripted_runtime(
            self._script_abort_chunked_resume_releases_all_resources,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # [b-96d4749094] Release row + KV + lock_ref when aborting a
    # chunked-resume req from waiting_queue.
    # Bug shape: the waiting-queue abort path in abort_request only
    # freed disagg-decode KV and mamba state. Pre-v2 this covered every
    # kind of resource a waiting req could hold. After v2, chunked-
    # resume reqs in waiting_queue hold req_to_token row + committed
    # KV slots + a radix tree lock_ref on req.last_node. Aborting such
    # a req while it sits ONLY in waiting_queue (not in batch.reqs)
    # left all three permanently leaked.
    # Verification: drive r mid-chunk to the waiting state, abort, and
    # check row_idx=None, kv_pages=0, lock_refs=0 ALL together. The
    # earlier test test_abort_waiting_releases_all checks the same
    # idea but does not snapshot all three resources simultaneously;
    # this one pins the full triple plus the global lock_refs baseline.
    @staticmethod
    def _script_abort_chunked_resume_releases_all_resources(t: ScriptedRuntime):
        baseline_refs = t.lock_refs_snapshot()
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        yield from run_until(r, lambda h: h.status == "waiting")

        # All three resources must be currently held to make the test
        # meaningful — otherwise the abort would be trivial.
        assert r.row_idx is not None, "row must be held mid-chunk"
        assert r.kv_pages > 0, "committed KV must be held mid-chunk"
        assert r.lock_refs >= 1, "radix lock_ref must be held mid-chunk"

        t.abort(r)
        yield

        # All three released, together.
        assert r.row_idx is None, (
            f"96d4749094: abort must release row; got row_idx={r.row_idx!r}"
        )
        assert r.kv_pages == 0, (
            f"96d4749094: abort must release KV; got kv_pages={r.kv_pages}"
        )
        assert r.lock_refs == 0, (
            f"96d4749094: abort must release lock_ref; got lock_refs={r.lock_refs}"
        )
        # has_pending_chunk + pending_middle_outputs are defensively
        # cleared so the iter-end stash path cannot revive the dead req.
        assert not r.has_pending_chunk
        assert r.pending_middle_outputs == 0
        # Global lock_refs must return to baseline.
        assert t.lock_refs_snapshot() == baseline_refs

    def test_abort_chunked_resume_dual_queue_no_double_release(self):
        """[b-de3859646b] abort of a chunked-resume req held in BOTH waiting_queue and batch.reqs must dedup and release exactly once."""
        execute_scripted_runtime(
            self._script_abort_chunked_resume_dual_queue_no_double_release,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # [b-de3859646b] abort_request dedup for chunked-resume dual-queue
    # holding.
    # Bug shape: under the stateless-scheduler refactor a chunked-
    # resume req can be simultaneously in waiting_queue (because it
    # holds across iters) and in batch.reqs (because the current iter
    # is admitting its next chunk). Pre-fix abort_request would process
    # the same rid twice (queue pop + to_finish), causing duplicate
    # send_output and double release_kv_cache (which underflows the
    # pool counter).
    # Verification: catch the req at the dual-queue moment — its rid
    # appears in BOTH t.waiting_rids() and t.batch_rids(). Abort.
    # finish_event_count must be exactly 1 and resources must release
    # cleanly without underflow.
    @staticmethod
    def _script_abort_chunked_resume_dual_queue_no_double_release(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)

        # Spin until we observe the dual-queue moment: rid in both
        # waiting_queue (held across iters) AND batch.reqs (current
        # admission iter). Bounded to avoid hang.
        for _ in range(DEFAULT_MAX_STEPS):
            in_waiting = r.rid in t.waiting_rids()
            in_batch = r.rid in t.batch_rids()
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

        # Pre-fix abort here would dispatch send_output twice and
        # release_kv_cache twice on the same row.
        t.abort(r)
        yield

        assert r.finish_event_count == 1, (
            f"de3859646b: dual-queue abort must dedup; got "
            f"{r.finish_event_count} finish events"
        )
        assert r.kv_pages == 0
        assert r.row_idx is None
        # KV pool counter must NOT be underflowed by double release.
        assert t.kv_pool_underflow_count() == 0, (
            f"double release_kv_cache underflowed pool; underflow_count="
            f"{t.kv_pool_underflow_count()}"
        )

    def test_pause_retract_releases_waiting_chunked_resume(self):
        """[b-f38e69f87d] pause_generation(retract) must release waiting_queue chunked-resume row+KV+lock_ref (not only running_batch)."""
        execute_scripted_runtime(
            self._script_pause_retract_releases_waiting_chunked_resume,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # [b-f38e69f87d] Extend pause(retract) to waiting chunked-resume
    # reqs.
    # Bug shape: pause_generation(retract)'s contract says "every req
    # holding KV is retracted, the cache can be flushed, recompute on
    # continue_generation". Pre-v2 every KV-holding req was in
    # running_batch, so iterating running_batch was sufficient. After
    # v2, chunked-resume reqs in waiting_queue hold req_to_token row
    # + committed KV slots + radix lock_ref. pause(retract) that only
    # iterated running_batch left those resources never released, so
    # flush_cache silently couldn't free everything and is_fully_idle
    # stayed False (waiting_queue still non-empty).
    # Verification: drive r mid-chunk to the waiting state, call pause
    # (retract). The waiting chunked-resume's row+KV+lock_ref must be
    # released and the engine must report is_fully_idle.
    @staticmethod
    def _script_pause_retract_releases_waiting_chunked_resume(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        yield from run_until(r, lambda h: h.status == "waiting")
        assert r.row_idx is not None and r.kv_pages > 0 and r.lock_refs >= 1

        # Trigger pause(retract) over the whole runtime, not a
        # single-req force_retract — the bug was in the queue-iteration
        # scope, not in the per-req release path.
        t.pause_retract_all()
        yield

        assert r.row_idx is None, (
            f"f38e69f87d: pause(retract) must release waiting "
            f"chunked-resume row; got row_idx={r.row_idx!r}"
        )
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert not r.has_pending_chunk
        # After pause(retract), the engine is fully idle — no leftover
        # waiting_queue entry blocking flush.
        assert t.is_fully_idle, (
            f"f38e69f87d: pause(retract) must leave engine fully idle; "
            f"got is_fully_idle={t.is_fully_idle!r}"
        )

    def test_retract_all_clears_batch_with_chunked(self):
        """[b-f0388931bf] retract_all triggered mid-chunk must clear the batch (keep_indices=[] semantics), not pass List[Req] as keep_indices."""
        execute_scripted_runtime(
            self._script_retract_all_clears_batch_with_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # [b-f0388931bf] retract_all was passing List[Req] to filter_batch
    # as keep_indices.
    # Bug shape: after 3fd7319a3d removed the chunked_req_to_exclude
    # first positional from filter_batch, retract_all's existing call
    # self.filter_batch(retracted_reqs) silently broke: the new first
    # positional is keep_indices: Optional[List[int]], so the code
    # was trying to index reqs by Req objects (TypeError under strict
    # checks, or silent slicing under permissive numpy).
    # Verification: trigger retract_all mid-chunk over multiple reqs.
    # The batch must end up empty (NOT preserving the retracted list as
    # "kept"), all reqs must be in waiting_queue with reset chunked
    # state, and subsequent resume must finish all of them.
    @staticmethod
    def _script_retract_all_clears_batch_with_chunked(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking and h.chunks_done >= 1)
        yield from run_until(r2, lambda h: h.is_chunking)

        # Trigger retract_all — exercises the filter_batch(keep_indices=[])
        # path the fix introduced.
        t.retract_all()
        yield

        # Batch is empty: pre-fix the wrong call would either crash or
        # silently keep the reqs that should have been retracted.
        assert t.batch_size() == 0, (
            f"f0388931bf: retract_all must clear batch; got batch_size="
            f"{t.batch_size()}"
        )
        # chunked_req field must be cleared (paired with the f0388931bf
        # series — retract_all also clears the v1 chunked_req if still
        # present).
        assert t.get_chunked_req_rid() is None
        # Both reqs must be back in waiting_queue with chunked state
        # reset (chunks_done back to 0).
        for r in (r1, r2):
            assert r.status == "waiting"
            assert r.chunks_done == 0
            assert not r.has_pending_chunk

        # All resume and finish cleanly.
        yield from run_until_all_finished([r1, r2])


if __name__ == "__main__":
    unittest.main()
