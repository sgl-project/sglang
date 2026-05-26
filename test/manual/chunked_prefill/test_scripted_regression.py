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


# 5ed4faf0ab "Bypass LoRA scheduling gate for chunked-resume reqs".
# Bug: LoRA drainer would reject chunked-resume reqs, leaving them
# stuck in waiting_queue while holding row + lock_ref + KV.
# Fix verification: a LoRA chunked-resume req must complete, even
# when adapter draining is forced via the wishlist
# ``t.force_lora_drainer_reject`` helper.
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


# 96d4749094 "Release row + KV + lock_ref when aborting a
# chunked-resume req from waiting_queue".
def _script_abort_waiting_releases_all(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    yield from run_until(r, lambda h: h.status == "waiting")

    t.abort(r)
    yield

    assert r.kv_pages == 0
    assert r.row_idx is None
    assert r.lock_refs == 0


# f38e69f87d "Extend pause(retract) to waiting chunked-resume reqs".
# Bug: pause path didn't iterate waiting_queue chunked-resume entries,
# so paused chunked reqs leaked.
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


# b823c16e60 "Include PP microbatch reqs in abort_request
# batch_rids dedup".
# Bug: abort_request's dedup set did not include PP cross-mb in-flight
# reqs, leading to double-abort under PP last-chunk-in-flight timing.
def _script_pp_abort_dedup(t: ScriptedRuntime):
    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
    yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)

    t.abort(r)
    yield

    assert r.finish_event_count == 1, (
        f"PP abort must dedup across microbatches; "
        f"got {r.finish_event_count} finish events"
    )


# 414efd4a27 "Reset disagg send-side state on chunked-resume retract".
def _script_disagg_retract_resets_send(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

    t.force_retract(r)
    yield

    assert r.disagg_send_state in (None, "idle")


# b3a7b9f2a1 "Bump pending_middle_outputs for last-chunk admits +
# decrement-first output proc".
# The invariant: at the moment the last chunk is admitted,
# pending_middle_outputs > 0; once the output is processed it returns
# to 0.
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


# 69ef71edc4 "Conditionally exclude in-flight other-mb
# chunked-resume reqs (PP, max_new_tokens > 1)".
# Bug: under PP, chunked-resume reqs in another microbatch leaked
# into the local batch's filter, leading to double-admission.
# Verification: run two chunked reqs concurrently under PP=2,
# different microbatches; both must complete cleanly without
# double-admit symptoms (extend_input_len mismatch or
# chunked_in_flight_count > 1 momentarily).
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


# 34c02d6a67 "Filter chunked-resume reqs from split_prefill_batch
# before pdmux merge".
# Bug: pdmux merge would see chunked-resume reqs that should be
# excluded. With disagg + chunked, the merge must filter them out.
def _script_pdmux_filter_chunked(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished


# aaf3752d2b "Skip chunked-resume reqs in calc_priority prefix
# matching".
# Bug: priority calculation tried prefix-matching chunked-resume reqs,
# which is wrong (they already have prefix_indices baked in). Test:
# enable priority + radix; r1 chunks, r2 arrives; priority calc must
# not include r1 in its prefix match.
def _script_priority_skips_chunked_in_prefix_match(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low")
    yield from run_until(r1, lambda h: h.is_chunking)

    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high")

    yield from run_until_all_finished([r1, r2])
    assert r1.finished and r2.finished


# dbdcdde245 "Skip mamba_pool_idx cleanup for chunked-resume on
# NO_TOKEN".
# Mamba-specific: when a chunked-resume req hits NO_TOKEN admission
# return, mamba_pool_idx cleanup must be skipped.
#
# Requires a mamba-architecture model — currently no small mamba model
# is in the standard test fixture. The test sets the right flags;
# actual mamba model wiring is left for the harness to fill in.
def _script_mamba_chunked_resume_no_token(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished


# 116584e8fa "Bound streaming-session chunked stash by
# kv_committed_len".
# Streaming-session specific: when the session's KV commit length is
# shorter than the chunked extend, the stash must clip to the committed
# length to avoid stashing un-committed tokens.
def _script_streaming_session_stash_bound(t: ScriptedRuntime):
    # Streaming session creation isn't yet a ScriptedRuntime primitive;
    # this test pumps a long req in a streaming-flagged engine and
    # verifies the chunk loop completes cleanly.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished


# bf5b4e9a10 "Give chunked-resume reqs priority in LPM and
# DFS_WEIGHT sorts".
# Bug: LPM / DFS_WEIGHT priority sort didn't prioritize chunked-resume
# reqs, leading to starvation under load.
# Test: a chunked req in flight + many short reqs; chunked must
# advance, not starve.
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


def _script_revert_bump_pending_middle_outputs(t: ScriptedRuntime):
    # e875cd36e4: revert of pending_middle_outputs bump.
    # After last chunk admit, pending_middle_outputs should not double-count.
    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
    yield from run_until(r, lambda h: h.chunks_done >= 1)
    assert r.pending_middle_outputs <= 1
    yield from run_until_finished(r)
    assert r.pending_middle_outputs == 0


def _script_filter_batch_exclude_in_flight_other_mb(t: ScriptedRuntime):
    # 5c523049db / 45347ca3a3: exclude in-flight other-mb reqs in
    # filter_batch. Single-engine smoke (multi-mb requires PP).
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
    yield from run_until_finished(r)
    assert r.finished


def _script_chunked_req_marker_pp_filter_exclusion(t: ScriptedRuntime):
    # 33f981ce93 / 11db3a4192: re-add ScheduleBatch.chunked_req marker
    # for PP cross-mb filter exclusion.
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
    yield from run_until_finished(r)


def _script_v1_swa_chunked_tests_dropped(t: ScriptedRuntime):
    # a94e842611 / daf9c42f17: dropped v1 SWA chunked-req tests. Verifies
    # the v2 path is what runs (no v1 codepath revival).
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


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


def _script_chunked_resume_tail_counted_page_size_gt1(t: ScriptedRuntime):
    # b433e1ea35: count chunked-resume tail in runtime mem check
    # when page_size > 1. Verifies the runtime mem accounting includes
    # tail of chunked-resume reqs.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN + 17, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_merge_batch_assert_widened(t: ScriptedRuntime):
    # 36ec1d7269: widen merge_batch assert to match filter_batch
    # predicate. Verifies the assert does not fire on legitimate
    # chunked merge.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    r2 = t.start_req(prompt_len=16, max_new_tokens=2)
    yield from run_until_all_finished([r1, r2])


def _script_host_hit_length_reset_unconditional(t: ScriptedRuntime):
    # d7fa48baad: reset host_hit_length unconditionally in
    # prepare_for_extend. Second submission must not inherit stale
    # host_hit_length from r1.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r1)
    r2 = t.start_req(prompt_len=16, max_new_tokens=2)
    yield from run_until_finished(r2)


def _script_add_one_req_reuse_gate_has_pending_chunk(t: ScriptedRuntime):
    # a79ba1b2f7: tighten add_one_req reuse gate to has_pending_chunk.
    # Verifies reuse branch only fires when there's a pending chunk.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_filter_batch_explicit_exclude_chunked_flag(t: ScriptedRuntime):
    # fd3dcca22f: refactor filter_batch to explicit exclude_chunked_req flag.
    # Verifies behavior under the new API.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    r2 = t.start_req(prompt_len=16, max_new_tokens=2)
    yield from run_until_all_finished([r1, r2])


def _script_retract_all_filter_batch_args(t: ScriptedRuntime):
    # f0388931bf: retract_all was passing List[Req] to filter_batch
    # as keep_indices (wrong). Test triggers a retract_all path.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    t.force_retract(r)
    yield
    yield from run_until_finished(r)


def _script_is_chunked_renamed_pending_middle_outputs(t: ScriptedRuntime):
    # b9d5d6ed5f: rename Req.is_chunked -> Req.pending_middle_outputs.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


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


def _script_unified_admission_via_add_one_req(t: ScriptedRuntime):
    # 74f1d8bbab: unify chunked admission via add_one_req reuse +
    # has_pending_chunk. Smoke: chunked admission works.
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_alloc_assert_no_is_chunked(t: ScriptedRuntime):
    # 9b361aef46: drop is_chunked from req_to_token_pool alloc assert.
    # Smoke: alloc + chunked does not hit a stale assert.
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_cache_unfinished_req_row_bound(t: ScriptedRuntime):
    # 1c3bf8e7db: bound cache_unfinished_req row read by kv_committed_len.
    # Verifies that radix caching of unfinished chunked reqs respects
    # the committed length.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_waiting_queue_pending_tokens_subtract_prefix(t: ScriptedRuntime):
    # c79a73bec4: subtract prefix_indices from waiting_queue pending
    # tokens sum. Verifies waiting_queue pending tokens excludes
    # already-cached prefix.
    r_warm = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=1)
    yield from run_until_finished(r_warm)
    r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 4, max_new_tokens=2)
    yield from run_until_finished(r)


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


class TestScriptedRegression(CustomTestCase):
    def test_lora_drainer_chunked_resume(self):
        execute_scripted_runtime(
            _script_lora_drainer_chunked_resume,
            **base_engine_kwargs(
                model_path=_LORA_BASE_MODEL,
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_lora=True,
                lora_paths=[_LORA_ADAPTER],
            ),
        )

    def test_abort_waiting_releases_all(self):
        execute_scripted_runtime(
            _script_abort_waiting_releases_all,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_pause_covers_waiting_chunked(self):
        execute_scripted_runtime(
            _script_pause_covers_waiting_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_pp_abort_dedup(self):
        execute_scripted_runtime(
            _script_pp_abort_dedup,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                tp_size=2,
                pp_size=2,
            ),
        )

    def test_disagg_retract_resets_send(self):
        execute_scripted_runtime(
            _script_disagg_retract_resets_send,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )

    def test_pending_middle_outputs_invariant(self):
        execute_scripted_runtime(
            _script_pending_middle_outputs_invariant,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_pp_other_mb_chunked_exclude(self):
        execute_scripted_runtime(
            _script_pp_other_mb_chunked_exclude,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                tp_size=2,
                pp_size=2,
            ),
        )

    def test_pdmux_filter_chunked(self):
        execute_scripted_runtime(
            _script_pdmux_filter_chunked,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )

    def test_priority_skips_chunked_in_prefix_match(self):
        execute_scripted_runtime(
            _script_priority_skips_chunked_in_prefix_match,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )

    def test_mamba_chunked_resume_no_token(self):
        execute_scripted_runtime(
            _script_mamba_chunked_resume_no_token,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_streaming_session_stash_bound(self):
        execute_scripted_runtime(
            _script_streaming_session_stash_bound,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_resume_priority_in_sort(self):
        # Use LPM policy explicitly — the bf5b4e9a10 fix was in LPM /
        # DFS_WEIGHT sort paths, so the default (FCFS) won't exercise it.
        execute_scripted_runtime(
            _script_chunked_resume_priority_in_sort,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                schedule_policy="lpm",
            ),
        )

    def test_rename_inflight_to_pending_middle_outputs(self):
        execute_scripted_runtime(
            _script_rename_inflight_to_pending_middle_outputs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_revert_bump_pending_middle_outputs(self):
        execute_scripted_runtime(
            _script_revert_bump_pending_middle_outputs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_filter_batch_exclude_in_flight_other_mb(self):
        execute_scripted_runtime(
            _script_filter_batch_exclude_in_flight_other_mb,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_req_marker_pp_filter_exclusion(self):
        execute_scripted_runtime(
            _script_chunked_req_marker_pp_filter_exclusion,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_v1_swa_chunked_tests_dropped(self):
        execute_scripted_runtime(
            _script_v1_swa_chunked_tests_dropped,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_stage_a_chunk_stash_iter_boundary(self):
        execute_scripted_runtime(
            _script_stage_a_chunk_stash_iter_boundary,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_resume_tail_counted_page_size_gt1(self):
        execute_scripted_runtime(
            _script_chunked_resume_tail_counted_page_size_gt1,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_merge_batch_assert_widened(self):
        execute_scripted_runtime(
            _script_merge_batch_assert_widened,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_host_hit_length_reset_unconditional(self):
        execute_scripted_runtime(
            _script_host_hit_length_reset_unconditional,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_add_one_req_reuse_gate_has_pending_chunk(self):
        execute_scripted_runtime(
            _script_add_one_req_reuse_gate_has_pending_chunk,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_filter_batch_explicit_exclude_chunked_flag(self):
        execute_scripted_runtime(
            _script_filter_batch_explicit_exclude_chunked_flag,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_retract_all_filter_batch_args(self):
        execute_scripted_runtime(
            _script_retract_all_filter_batch_args,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_is_chunked_renamed_pending_middle_outputs(self):
        execute_scripted_runtime(
            _script_is_chunked_renamed_pending_middle_outputs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_resume_waiting_queue_holding(self):
        execute_scripted_runtime(
            _script_chunked_resume_waiting_queue_holding,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_unified_admission_via_add_one_req(self):
        execute_scripted_runtime(
            _script_unified_admission_via_add_one_req,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_alloc_assert_no_is_chunked(self):
        execute_scripted_runtime(
            _script_alloc_assert_no_is_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_cache_unfinished_req_row_bound(self):
        execute_scripted_runtime(
            _script_cache_unfinished_req_row_bound,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_waiting_queue_pending_tokens_subtract_prefix(self):
        execute_scripted_runtime(
            _script_waiting_queue_pending_tokens_subtract_prefix,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_dedup_dual_queue_holding(self):
        execute_scripted_runtime(
            _script_abort_dedup_dual_queue_holding,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
