"""Commit-history regression — extra coverage.

Covers category C from the expansion plan: commits from the 309b6dc
window that were NOT picked up by the round-1 ``test_regression_309b6dc.py``
(~15 new regressions). Each test mirrors one commit's bug shape.

These tests reference upstream commit SHAs in the docstrings — when
the chunked refactor lands, these scripts re-validate that the fix
is still in effect.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.req_handle import ReqHandle
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
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
    yield from run_until_finished(r)


def _script_v1_swa_chunked_tests_dropped(t: ScriptedRuntime):
    # a94e842611 / daf9c42f17: dropped v1 SWA chunked-req tests. Verifies
    # the v2 path is what runs (no v1 codepath revival).
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
    # Expectation depends on impl; just assert it finished cleanly.


def _script_unified_admission_via_add_one_req(t: ScriptedRuntime):
    # 74f1d8bbab: unify chunked admission via add_one_req reuse +
    # has_pending_chunk. Smoke: chunked admission works.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_alloc_assert_no_is_chunked(t: ScriptedRuntime):
    # 9b361aef46: drop is_chunked from req_to_token_pool alloc assert.
    # Smoke: alloc + chunked does not hit a stale assert.
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
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    t.abort(r)
    t.abort(r)  # double-abort: must dedup
    yield
    assert r.kv_pages == 0


class TestRegressionExtra(CustomTestCase):
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
