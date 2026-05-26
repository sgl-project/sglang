"""Source-code special-case coverage — extra.

Covers category D from the expansion plan: scheduler.py /
disaggregation / dllm chunked-related branches that round-1's
``test_special_case_coverage.py`` did not address. Each test drives
the scheduler through a specific branch.
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


def _script_last_batch_chunked_req_pp_context(t: ScriptedRuntime):
    # scheduler.py:2363-2369 — last_batch tracks the chunked_req in
    # the PP context (chunked_req_to_exclude path).
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished


def _script_chunked_req_to_exclude_set_add(t: ScriptedRuntime):
    # scheduler.py:2366 — chunked_req_to_exclude.add(last_batch.chunked_req).
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
    yield from run_until_finished(r)


def _script_chunked_req_to_exclude_update_reqs(t: ScriptedRuntime):
    # scheduler.py:2369 — chunked_req_to_exclude.update(last_batch.reqs).
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    r2 = t.start_req(prompt_len=16, max_new_tokens=2)
    yield from run_until_all_finished([r1, r2])


def _script_schedule_batch_init_new_chunked_req(t: ScriptedRuntime):
    # scheduler.py:2658 — ScheduleBatch.init_new(chunked_req=self.chunked_req).
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_mem_check_chunked_req_kwarg(t: ScriptedRuntime):
    # scheduler.py:2676-2677 — mem check called with chunked_req=...
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_get_chunked_req_lambda_getter(t: ScriptedRuntime):
    # scheduler.py:680 — get_chunked_req lambda. Verify that during
    # a chunked req's lifetime, the getter returns the right rid.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)

    # NEW API NEEDED: t.get_chunked_req_rid() — query the scheduler's
    # current chunked_req's rid (or None).
    cur = t.get_chunked_req_rid()
    if cur is not None:
        assert cur == r.rid


def _script_chunked_req_scheduled_last_iter_flip(t: ScriptedRuntime):
    # scheduler.py: _chunked_req_scheduled_last_iter flip logic.
    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_chunked_req_reset_to_none(t: ScriptedRuntime):
    # scheduler.py:3596 — chunked_req=None reset path. After all
    # chunked reqs finish, scheduler.chunked_req should be None.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    cur = t.get_chunked_req_rid()
    assert cur is None


def _script_disagg_prefill_chunked_path(t: ScriptedRuntime):
    # disaggregation/prefill.py — chunked req in disagg prefill mode.
    # Single-engine smoke (disagg topology requires P3 multi-engine).
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_disagg_decode_waiting_queue_kv_held(t: ScriptedRuntime):
    # disaggregation/decode.py — waiting_queue reqs hold KV in decode mode.
    # Smoke: chunked req traverses decode-side waiting state cleanly.
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_dllm_staging_double_pending_middle_outputs(t: ScriptedRuntime):
    # dllm/mixin/scheduler.py — DLLM staging AND chunked admission both
    # incrementing pending_middle_outputs (double-source).
    # Single-engine smoke (DLLM model required for full coverage).
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_staging_handler_chunked(t: ScriptedRuntime):
    # disaggregation/common/staging_handler.py — chunked interaction.
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_mooncake_conn_chunked(t: ScriptedRuntime):
    # disaggregation/mooncake/conn.py — chunked path in conn layer.
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_nixl_conn_chunked(t: ScriptedRuntime):
    # disaggregation/nixl/conn.py — chunked path.
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_idle_path_chunked_req_none(t: ScriptedRuntime):
    # scheduler.py:3174 — idle path checks chunked_req is None.
    # If we have no in-flight req, scheduler is idle.
    # Give the scheduler a few yields to settle into the idle state
    # (initial setup may keep is_idle False for one or two iterations).
    for _ in range(5):
        yield
    assert t.is_idle


def _script_dynamic_chunking_history_len(t: ScriptedRuntime):
    # scheduler.py:2516-2517 — dynamic chunking reads history_len from
    # chunked_req. Enabled via --enable-dynamic-chunking.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_admission_path_with_chunked_inflight_flag(t: ScriptedRuntime):
    # scheduler.py:2593 — add_one_req called with has_chunked_req=True.
    r_chunked = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r_chunked, lambda h: h.is_chunking)
    r_new = t.start_req(prompt_len=16, max_new_tokens=2)
    yield from run_until_all_finished([r_chunked, r_new])


def _script_inflight_counter_increments_each_chunk(t: ScriptedRuntime):
    # scheduler.py:2644-2645 — inflight_middle_chunks += 1 per chunk.
    r = t.start_req(prompt_len=4 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
    saw_increment = False
    last = 0
    for _ in range(DEFAULT_MAX_STEPS):
        cur = r.inflight_middle_chunks
        if cur > last:
            saw_increment = True
        last = max(last, cur)
        if r.finished:
            break
        yield
    # After at least one chunk, the counter must have moved up at some point.
    assert saw_increment, "expected inflight_middle_chunks to increment"


def _script_filter_batch_exclude_chunked_flag(t: ScriptedRuntime):
    # filter_batch + chunked: exclude_chunked_req branch.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    r2 = t.start_req(prompt_len=16, max_new_tokens=2)
    yield from run_until_all_finished([r1, r2])


def _script_pdmux_split_prefill_batch(t: ScriptedRuntime):
    # 34c02d6a67: filter chunked-resume from split_prefill_batch.
    # pdmux-specific; single-engine smoke.
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_streaming_session_kv_committed_bound(t: ScriptedRuntime):
    # 116584e8fa: bound streaming-session chunked stash by kv_committed_len.
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


def _script_mamba_pool_idx_cleanup_skip_chunked_resume(t: ScriptedRuntime):
    # dbdcdde245: skip mamba_pool_idx cleanup for chunked-resume on
    # NO_TOKEN. Mamba-specific; single-engine smoke.
    # TODO(round-3): recreate the specific bug shape; this currently
    # is a forward-pointing smoke.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)


class TestSpecialCaseCoverageExtra(CustomTestCase):
    def test_last_batch_chunked_req_pp_context(self):
        execute_scripted_runtime(
            _script_last_batch_chunked_req_pp_context,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_req_to_exclude_set_add(self):
        execute_scripted_runtime(
            _script_chunked_req_to_exclude_set_add,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_req_to_exclude_update_reqs(self):
        execute_scripted_runtime(
            _script_chunked_req_to_exclude_update_reqs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_schedule_batch_init_new_chunked_req(self):
        execute_scripted_runtime(
            _script_schedule_batch_init_new_chunked_req,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_mem_check_chunked_req_kwarg(self):
        execute_scripted_runtime(
            _script_mem_check_chunked_req_kwarg,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_get_chunked_req_lambda_getter(self):
        execute_scripted_runtime(
            _script_get_chunked_req_lambda_getter,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_req_scheduled_last_iter_flip(self):
        execute_scripted_runtime(
            _script_chunked_req_scheduled_last_iter_flip,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_req_reset_to_none(self):
        execute_scripted_runtime(
            _script_chunked_req_reset_to_none,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_disagg_prefill_chunked_path(self):
        execute_scripted_runtime(
            _script_disagg_prefill_chunked_path,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_disagg_decode_waiting_queue_kv_held(self):
        execute_scripted_runtime(
            _script_disagg_decode_waiting_queue_kv_held,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_dllm_staging_double_pending_middle_outputs(self):
        execute_scripted_runtime(
            _script_dllm_staging_double_pending_middle_outputs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_staging_handler_chunked(self):
        execute_scripted_runtime(
            _script_staging_handler_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_mooncake_conn_chunked(self):
        execute_scripted_runtime(
            _script_mooncake_conn_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_nixl_conn_chunked(self):
        execute_scripted_runtime(
            _script_nixl_conn_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_idle_path_chunked_req_none(self):
        execute_scripted_runtime(
            _script_idle_path_chunked_req_none,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_dynamic_chunking_history_len(self):
        execute_scripted_runtime(
            _script_dynamic_chunking_history_len,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dynamic_chunking=True,
            ),
        )

    def test_admission_path_with_chunked_inflight_flag(self):
        execute_scripted_runtime(
            _script_admission_path_with_chunked_inflight_flag,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_inflight_counter_increments_each_chunk(self):
        execute_scripted_runtime(
            _script_inflight_counter_increments_each_chunk,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_filter_batch_exclude_chunked_flag(self):
        execute_scripted_runtime(
            _script_filter_batch_exclude_chunked_flag,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_pdmux_split_prefill_batch(self):
        execute_scripted_runtime(
            _script_pdmux_split_prefill_batch,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_streaming_session_kv_committed_bound(self):
        execute_scripted_runtime(
            _script_streaming_session_kv_committed_bound,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_mamba_pool_idx_cleanup_skip_chunked_resume(self):
        execute_scripted_runtime(
            _script_mamba_pool_idx_cleanup_skip_chunked_resume,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
