"""Functional tests for ScriptedRuntime.

Script functions are top-level (spawn-mode mp imports them by name)
and underscore-prefixed (so unittest discovery skips them).
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.req_handle import ReqHandle
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

_COMMON_ENGINE_KWARGS = dict(
    model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    tp_size=1,
    dp_size=1,
    pp_size=1,
    disable_overlap_schedule=True,
    disable_cuda_graph=True,
)


def _script_start_req_returns_req_handle(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=8, max_new_tokens=2)
    assert isinstance(r1, ReqHandle)
    assert r1.rid == "scripted-0"
    r2 = t.start_req(prompt_len=8, max_new_tokens=2)
    assert r2.rid == "scripted-1"
    yield


def _script_multiple_yields_advance_scheduler(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=8, max_new_tokens=4)
    assert r1.status == "unknown"  # not yet pulled from the queue
    yield
    yield
    assert r1.status in ("waiting", "running", "unknown")


def _script_multiple_reqs_in_one_script(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=8, max_new_tokens=2)
    r2 = t.start_req(prompt_len=8, max_new_tokens=2)
    r3 = t.start_req(prompt_len=8, max_new_tokens=2)
    assert r1.rid != r2.rid != r3.rid
    yield
    yield
    for r in (r1, r2, r3):
        assert r.status in ("waiting", "running", "unknown")


def _script_empty_return(t: ScriptedRuntime):
    # Unreachable yield keeps this a generator function, but the body
    # returns without ever yielding.
    if False:
        yield
    return


def _script_assertion_failure(t: ScriptedRuntime):
    yield
    assert False, "boom"


def _script_runtime_error(t: ScriptedRuntime):
    yield
    raise RuntimeError("simulated runtime error")


def _script_not_a_generator(t: ScriptedRuntime):
    # No yield => regular function => calling returns None, not a generator.
    return None


def _script_yield_before_start_req(t: ScriptedRuntime):
    yield
    r1 = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    yield
    assert r1.status in ("waiting", "running", "unknown")


def _script_status_for_unknown_rid(t: ScriptedRuntime):
    bogus = ReqHandle(rid="never-submitted-rid", runtime=t)
    assert bogus.status == "unknown"
    yield


def _script_assertion_with_status(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    assert r1.status == "definitely-not-a-real-status", "status mismatch"


# ============================================================
# API surface smokes for round-2 expansion. Each function calls
# one new API attribute / method and verifies the call returns
# without exception (semantics tested in manual suite).
# ============================================================


def _script_api_smoke_r_finished(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    for _ in range(200):
        if r.finished:
            return
        yield


def _script_api_smoke_r_chunks_done(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    _ = r.chunks_done
    assert isinstance(r.chunks_done, int)


def _script_api_smoke_r_is_chunking(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    _ = r.is_chunking
    assert isinstance(r.is_chunking, bool)


def _script_api_smoke_r_kv_pages(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    _ = r.kv_pages
    assert isinstance(r.kv_pages, int) and r.kv_pages >= 0


def _script_api_smoke_r_row_idx(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    # row_idx may be None when not yet assigned.
    _ = r.row_idx


def _script_api_smoke_r_lock_refs(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    _ = r.lock_refs
    assert isinstance(r.lock_refs, int) and r.lock_refs >= 0


def _script_api_smoke_r_pending_middle_outputs(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    _ = r.pending_middle_outputs
    assert isinstance(r.pending_middle_outputs, int)


def _script_api_smoke_r_inflight_middle_chunks(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    _ = r.inflight_middle_chunks
    assert isinstance(r.inflight_middle_chunks, int)


def _script_api_smoke_r_finish_event_count(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    _ = r.finish_event_count
    assert isinstance(r.finish_event_count, int)


def _script_api_smoke_r_disagg_send_state(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    _ = r.disagg_send_state  # may be None outside disagg


def _script_api_smoke_r_output_tokens(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    for _ in range(200):
        if r.finished:
            break
        yield
    assert isinstance(r.output_tokens, list)


def _script_api_smoke_r_logprobs(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2, return_logprob=True)
    for _ in range(200):
        if r.finished:
            break
        yield
    _ = r.logprobs


def _script_api_smoke_r_cumulative_kv_alloc_bytes(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    _ = r.cumulative_kv_alloc_bytes
    assert r.cumulative_kv_alloc_bytes >= 0


def _script_api_smoke_t_is_idle(t: ScriptedRuntime):
    _ = t.is_idle
    assert isinstance(t.is_idle, bool)
    yield


def _script_api_smoke_t_abort(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    t.abort(r)
    yield


def _script_api_smoke_t_force_retract(t: ScriptedRuntime):
    r = t.start_req(prompt_len=128, max_new_tokens=2)
    yield
    t.force_retract(r)
    yield


def _script_api_smoke_t_exhaust_kv(t: ScriptedRuntime):
    t.exhaust_kv(leave_pages=4)
    yield


def _script_api_smoke_t_exhaust_row_pool(t: ScriptedRuntime):
    t.exhaust_row_pool(leave_rows=2)
    yield


def _script_api_smoke_t_exhaust_lock_refs(t: ScriptedRuntime):
    t.exhaust_lock_refs(leave_refs=2)
    yield


def _script_api_smoke_t_force_lora_drainer_reject(t: ScriptedRuntime):
    t.force_lora_drainer_reject(adapter="some-adapter")
    yield


def _script_api_smoke_t_batch_composition(t: ScriptedRuntime):
    comp = t.batch_composition()
    assert isinstance(comp, dict)
    yield


def _script_api_smoke_t_chunked_in_flight_count(t: ScriptedRuntime):
    cnt = t.chunked_in_flight_count()
    assert isinstance(cnt, int) and cnt >= 0
    yield


def _script_api_smoke_t_list_active_reqs(t: ScriptedRuntime):
    reqs = t.list_active_reqs()
    assert isinstance(reqs, list)
    yield


def _script_api_smoke_t_force_preempt(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=128, max_new_tokens=2, priority="low")
    r2 = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")
    yield
    t.force_preempt(victim_rid=r1.rid, by_rid=r2.rid)
    yield


def _script_api_smoke_t_last_admission_path(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    path = t.last_admission_path()
    assert path is None or isinstance(path, str)


def _script_api_smoke_t_last_scheduler_path(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    path = t.last_scheduler_path()
    assert path is None or isinstance(path, str)


def _script_api_smoke_t_engine_stats(t: ScriptedRuntime):
    stats = t.engine_stats()
    assert isinstance(stats, dict)
    yield


def _script_api_smoke_t_warmup_radix(t: ScriptedRuntime):
    t.warmup_radix(prompt_tokens=[1, 1, 1, 1])
    yield


def _script_api_smoke_t_evict_radix(t: ScriptedRuntime):
    t.evict_radix(prefix_tokens=None)
    yield


def _script_api_smoke_t_trigger_abort_on_waiting_timeout(t: ScriptedRuntime):
    t.trigger_abort_on_waiting_timeout()
    yield


def _script_api_smoke_t_get_chunked_req_rid(t: ScriptedRuntime):
    rid = t.get_chunked_req_rid()
    assert rid is None or isinstance(rid, str)
    yield


def _script_api_smoke_start_req_priority(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")
    assert isinstance(r, ReqHandle)
    yield


def _script_api_smoke_start_req_lora_path(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2, lora_path=None)
    assert isinstance(r, ReqHandle)
    yield


def _script_api_smoke_start_req_temperature(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2, temperature=0.0)
    assert isinstance(r, ReqHandle)
    yield


def _script_api_smoke_start_req_top_p_top_k(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2, top_p=0.9, top_k=40)
    assert isinstance(r, ReqHandle)
    yield


def _script_api_smoke_start_req_stop(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=4, stop=["xyz"])
    assert isinstance(r, ReqHandle)
    yield


def _script_api_smoke_start_req_stop_token_ids(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=4, stop_token_ids=[2])
    assert isinstance(r, ReqHandle)
    yield


def _script_api_smoke_start_req_ignore_eos(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=4, ignore_eos=True)
    assert isinstance(r, ReqHandle)
    yield


def _script_api_smoke_start_req_return_logprob(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2, return_logprob=True)
    assert isinstance(r, ReqHandle)
    yield


def _script_api_smoke_start_req_top_logprobs_num(t: ScriptedRuntime):
    r = t.start_req(
        prompt_len=8, max_new_tokens=2, return_logprob=True, top_logprobs_num=3
    )
    assert isinstance(r, ReqHandle)
    yield


def _script_api_smoke_start_req_min_new_tokens(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=8, min_new_tokens=2)
    assert isinstance(r, ReqHandle)
    yield


def _script_api_smoke_start_req_penalties(t: ScriptedRuntime):
    r = t.start_req(
        prompt_len=8,
        max_new_tokens=4,
        repetition_penalty=1.1,
        frequency_penalty=0.1,
        presence_penalty=0.1,
    )
    assert isinstance(r, ReqHandle)
    yield


def _script_api_smoke_start_req_explicit_rid(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2, rid="explicit-test-rid")
    assert r.rid == "explicit-test-rid"
    yield


class TestScriptedRuntimeFunctional(CustomTestCase):
    def test_start_req_returns_req_handle(self):
        execute_scripted_runtime(
            _script_start_req_returns_req_handle, **_COMMON_ENGINE_KWARGS
        )

    def test_multiple_yields_advance_scheduler(self):
        execute_scripted_runtime(
            _script_multiple_yields_advance_scheduler, **_COMMON_ENGINE_KWARGS
        )

    def test_multiple_reqs_in_one_script(self):
        execute_scripted_runtime(
            _script_multiple_reqs_in_one_script, **_COMMON_ENGINE_KWARGS
        )

    def test_empty_script_returns_immediately(self):
        execute_scripted_runtime(_script_empty_return, **_COMMON_ENGINE_KWARGS)

    def test_script_raises_assertion_surfaces_to_caller(self):
        with self.assertRaises(AssertionError) as ctx:
            execute_scripted_runtime(_script_assertion_failure, **_COMMON_ENGINE_KWARGS)
        self.assertIn("boom", str(ctx.exception))

    def test_script_raises_runtime_error_surfaces_to_caller(self):
        with self.assertRaises(AssertionError) as ctx:
            execute_scripted_runtime(_script_runtime_error, **_COMMON_ENGINE_KWARGS)
        err_text = str(ctx.exception)
        self.assertIn("RuntimeError", err_text)
        self.assertIn("simulated runtime error", err_text)

    def test_non_generator_script_function_errors_cleanly(self):
        with self.assertRaises(AssertionError) as ctx:
            execute_scripted_runtime(_script_not_a_generator, **_COMMON_ENGINE_KWARGS)
        self.assertIn("must be a generator", str(ctx.exception))

    def test_invalid_qualified_name_errors_before_engine(self):
        with self.assertRaises((ValueError, TypeError, AttributeError)):
            execute_scripted_runtime(lambda t: None, **_COMMON_ENGINE_KWARGS)

    def test_script_imported_from_pytest_file(self):
        # Exercises spawn-mode sys.path forwarding: this file's directory
        # is not normally on the subprocess's sys.path.
        execute_scripted_runtime(
            _script_start_req_returns_req_handle, **_COMMON_ENGINE_KWARGS
        )

    def test_yield_before_start_req(self):
        execute_scripted_runtime(
            _script_yield_before_start_req, **_COMMON_ENGINE_KWARGS
        )

    def test_status_for_unknown_rid(self):
        execute_scripted_runtime(
            _script_status_for_unknown_rid, **_COMMON_ENGINE_KWARGS
        )

    def test_assertion_failure_traceback_points_to_script_line(self):
        with self.assertRaises(AssertionError) as ctx:
            execute_scripted_runtime(
                _script_assertion_with_status, **_COMMON_ENGINE_KWARGS
            )
        err_text = str(ctx.exception)
        # Traceback should name the failing script function so a
        # developer can find the assert.
        self.assertIn("_script_assertion_with_status", err_text)
        self.assertIn("status mismatch", err_text)

    # ============================================================
    # Round-2 API surface smokes — one test per new attribute /
    # method introduced by the chunked expansion plan.
    # ============================================================

    def test_api_smoke_r_finished(self):
        execute_scripted_runtime(_script_api_smoke_r_finished, **_COMMON_ENGINE_KWARGS)

    def test_api_smoke_r_chunks_done(self):
        execute_scripted_runtime(
            _script_api_smoke_r_chunks_done, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_r_is_chunking(self):
        execute_scripted_runtime(
            _script_api_smoke_r_is_chunking, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_r_kv_pages(self):
        execute_scripted_runtime(_script_api_smoke_r_kv_pages, **_COMMON_ENGINE_KWARGS)

    def test_api_smoke_r_row_idx(self):
        execute_scripted_runtime(_script_api_smoke_r_row_idx, **_COMMON_ENGINE_KWARGS)

    def test_api_smoke_r_lock_refs(self):
        execute_scripted_runtime(_script_api_smoke_r_lock_refs, **_COMMON_ENGINE_KWARGS)

    def test_api_smoke_r_pending_middle_outputs(self):
        execute_scripted_runtime(
            _script_api_smoke_r_pending_middle_outputs, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_r_inflight_middle_chunks(self):
        execute_scripted_runtime(
            _script_api_smoke_r_inflight_middle_chunks, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_r_finish_event_count(self):
        execute_scripted_runtime(
            _script_api_smoke_r_finish_event_count, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_r_disagg_send_state(self):
        execute_scripted_runtime(
            _script_api_smoke_r_disagg_send_state, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_r_output_tokens(self):
        execute_scripted_runtime(
            _script_api_smoke_r_output_tokens, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_r_logprobs(self):
        execute_scripted_runtime(_script_api_smoke_r_logprobs, **_COMMON_ENGINE_KWARGS)

    def test_api_smoke_r_cumulative_kv_alloc_bytes(self):
        execute_scripted_runtime(
            _script_api_smoke_r_cumulative_kv_alloc_bytes, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_is_idle(self):
        execute_scripted_runtime(_script_api_smoke_t_is_idle, **_COMMON_ENGINE_KWARGS)

    def test_api_smoke_t_abort(self):
        execute_scripted_runtime(_script_api_smoke_t_abort, **_COMMON_ENGINE_KWARGS)

    def test_api_smoke_t_force_retract(self):
        execute_scripted_runtime(
            _script_api_smoke_t_force_retract, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_exhaust_kv(self):
        execute_scripted_runtime(
            _script_api_smoke_t_exhaust_kv, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_exhaust_row_pool(self):
        execute_scripted_runtime(
            _script_api_smoke_t_exhaust_row_pool, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_exhaust_lock_refs(self):
        execute_scripted_runtime(
            _script_api_smoke_t_exhaust_lock_refs, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_force_lora_drainer_reject(self):
        execute_scripted_runtime(
            _script_api_smoke_t_force_lora_drainer_reject, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_batch_composition(self):
        execute_scripted_runtime(
            _script_api_smoke_t_batch_composition, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_chunked_in_flight_count(self):
        execute_scripted_runtime(
            _script_api_smoke_t_chunked_in_flight_count, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_list_active_reqs(self):
        execute_scripted_runtime(
            _script_api_smoke_t_list_active_reqs, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_force_preempt(self):
        execute_scripted_runtime(
            _script_api_smoke_t_force_preempt, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_last_admission_path(self):
        execute_scripted_runtime(
            _script_api_smoke_t_last_admission_path, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_last_scheduler_path(self):
        execute_scripted_runtime(
            _script_api_smoke_t_last_scheduler_path, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_engine_stats(self):
        execute_scripted_runtime(
            _script_api_smoke_t_engine_stats, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_warmup_radix(self):
        execute_scripted_runtime(
            _script_api_smoke_t_warmup_radix, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_evict_radix(self):
        execute_scripted_runtime(
            _script_api_smoke_t_evict_radix, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_t_trigger_abort_on_waiting_timeout(self):
        execute_scripted_runtime(
            _script_api_smoke_t_trigger_abort_on_waiting_timeout,
            **_COMMON_ENGINE_KWARGS,
        )

    def test_api_smoke_t_get_chunked_req_rid(self):
        execute_scripted_runtime(
            _script_api_smoke_t_get_chunked_req_rid, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_start_req_priority(self):
        execute_scripted_runtime(
            _script_api_smoke_start_req_priority, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_start_req_lora_path(self):
        execute_scripted_runtime(
            _script_api_smoke_start_req_lora_path, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_start_req_temperature(self):
        execute_scripted_runtime(
            _script_api_smoke_start_req_temperature, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_start_req_top_p_top_k(self):
        execute_scripted_runtime(
            _script_api_smoke_start_req_top_p_top_k, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_start_req_stop(self):
        execute_scripted_runtime(
            _script_api_smoke_start_req_stop, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_start_req_stop_token_ids(self):
        execute_scripted_runtime(
            _script_api_smoke_start_req_stop_token_ids, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_start_req_ignore_eos(self):
        execute_scripted_runtime(
            _script_api_smoke_start_req_ignore_eos, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_start_req_return_logprob(self):
        execute_scripted_runtime(
            _script_api_smoke_start_req_return_logprob, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_start_req_top_logprobs_num(self):
        execute_scripted_runtime(
            _script_api_smoke_start_req_top_logprobs_num, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_start_req_min_new_tokens(self):
        execute_scripted_runtime(
            _script_api_smoke_start_req_min_new_tokens, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_start_req_penalties(self):
        execute_scripted_runtime(
            _script_api_smoke_start_req_penalties, **_COMMON_ENGINE_KWARGS
        )

    def test_api_smoke_start_req_explicit_rid(self):
        execute_scripted_runtime(
            _script_api_smoke_start_req_explicit_rid, **_COMMON_ENGINE_KWARGS
        )


if __name__ == "__main__":
    unittest.main()
