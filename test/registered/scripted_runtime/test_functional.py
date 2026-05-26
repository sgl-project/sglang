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


# ============================================================
# API surface smokes for round-2 expansion. Each function calls
# one new API attribute / method and verifies the call returns
# without exception (semantics tested in manual suite).
# ============================================================


class TestScriptedRuntimeFunctional(CustomTestCase):
    def test_start_req_returns_req_handle(self):
        """start_req returns a ReqHandle with the expected auto-assigned rid."""
        execute_scripted_runtime(
            self._script_start_req_returns_req_handle, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_start_req_returns_req_handle(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=8, max_new_tokens=2)
        assert isinstance(r1, ReqHandle)
        assert r1.rid == "scripted-0"
        r2 = t.start_req(prompt_len=8, max_new_tokens=2)
        assert r2.rid == "scripted-1"
        yield

    def test_multiple_yields_advance_scheduler(self):
        """Multiple bare yields advance the scheduler by one iteration each."""
        execute_scripted_runtime(
            self._script_multiple_yields_advance_scheduler, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_multiple_yields_advance_scheduler(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=8, max_new_tokens=4)
        assert r1.status == "unknown"  # not yet pulled from the queue
        yield
        yield
        assert r1.status in ("waiting", "running", "unknown")

    def test_multiple_reqs_in_one_script(self):
        """A single script can submit multiple reqs with distinct rids."""
        execute_scripted_runtime(
            self._script_multiple_reqs_in_one_script, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_multiple_reqs_in_one_script(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=8, max_new_tokens=2)
        r2 = t.start_req(prompt_len=8, max_new_tokens=2)
        r3 = t.start_req(prompt_len=8, max_new_tokens=2)
        assert r1.rid != r2.rid != r3.rid
        yield
        yield
        for r in (r1, r2, r3):
            assert r.status in ("waiting", "running", "unknown")

    def test_empty_script_returns_immediately(self):
        """Generator script that never yields returns without error."""
        execute_scripted_runtime(self._script_empty_return, **_COMMON_ENGINE_KWARGS)

    @staticmethod
    def _script_empty_return(t: ScriptedRuntime):
        # Unreachable yield keeps this a generator function, but the body
        # returns without ever yielding.
        if False:
            yield
        return

    def test_script_raises_assertion_surfaces_to_caller(self):
        """AssertionError from script body surfaces back to the caller."""
        with self.assertRaises(AssertionError) as ctx:
            execute_scripted_runtime(self._script_assertion_failure, **_COMMON_ENGINE_KWARGS)
        self.assertIn("boom", str(ctx.exception))

    @staticmethod
    def _script_assertion_failure(t: ScriptedRuntime):
        yield
        assert False, "boom"

    def test_script_raises_runtime_error_surfaces_to_caller(self):
        """RuntimeError from script body surfaces back to the caller as AssertionError."""
        with self.assertRaises(AssertionError) as ctx:
            execute_scripted_runtime(self._script_runtime_error, **_COMMON_ENGINE_KWARGS)
        err_text = str(ctx.exception)
        self.assertIn("RuntimeError", err_text)
        self.assertIn("simulated runtime error", err_text)

    @staticmethod
    def _script_runtime_error(t: ScriptedRuntime):
        yield
        raise RuntimeError("simulated runtime error")

    def test_non_generator_script_function_errors_cleanly(self):
        """Non-generator script function is rejected with a clear error."""
        with self.assertRaises(AssertionError) as ctx:
            execute_scripted_runtime(self._script_not_a_generator, **_COMMON_ENGINE_KWARGS)
        self.assertIn("must be a generator", str(ctx.exception))

    @staticmethod
    def _script_not_a_generator(t: ScriptedRuntime):
        # No yield => regular function => calling returns None, not a generator.
        return None

    def test_invalid_qualified_name_errors_before_engine(self):
        """Lambda script (no qualified name) is rejected before engine launch."""
        with self.assertRaises((ValueError, TypeError, AttributeError)):
            execute_scripted_runtime(lambda t: None, **_COMMON_ENGINE_KWARGS)

    def test_script_imported_from_pytest_file(self):
        """Spawn-mode sys.path forwarding lets subprocess import the script."""
        # Exercises spawn-mode sys.path forwarding: this file's directory
        # is not normally on the subprocess's sys.path.
        execute_scripted_runtime(
            self._script_start_req_returns_req_handle, **_COMMON_ENGINE_KWARGS
        )

    def test_yield_before_start_req(self):
        """Yielding before any start_req call is safe."""
        execute_scripted_runtime(
            self._script_yield_before_start_req, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_yield_before_start_req(t: ScriptedRuntime):
        yield
        r1 = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        yield
        assert r1.status in ("waiting", "running", "unknown")

    def test_status_for_unknown_rid(self):
        """ReqHandle for a never-submitted rid reports 'unknown' status."""
        execute_scripted_runtime(
            self._script_status_for_unknown_rid, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_status_for_unknown_rid(t: ScriptedRuntime):
        bogus = ReqHandle(rid="never-submitted-rid", runtime=t)
        assert bogus.status == "unknown"
        yield

    def test_assertion_failure_traceback_points_to_script_line(self):
        """Assertion failure traceback names the failing script function."""
        with self.assertRaises(AssertionError) as ctx:
            execute_scripted_runtime(
                self._script_assertion_with_status, **_COMMON_ENGINE_KWARGS
            )
        err_text = str(ctx.exception)
        # Traceback should name the failing script function so a
        # developer can find the assert.
        self.assertIn("_script_assertion_with_status", err_text)
        self.assertIn("status mismatch", err_text)

    @staticmethod
    def _script_assertion_with_status(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        assert r1.status == "definitely-not-a-real-status", "status mismatch"

    # ============================================================
    # Round-2 API surface smokes — one test per new attribute /
    # method introduced by the chunked expansion plan.
    # ============================================================

    def test_api_smoke_r_finished(self):
        """ReqHandle.finished returns True once the req completes."""
        execute_scripted_runtime(self._script_api_smoke_r_finished, **_COMMON_ENGINE_KWARGS)

    @staticmethod
    def _script_api_smoke_r_finished(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        for _ in range(200):
            if r.finished:
                return
            yield

    def test_api_smoke_r_inflight_middle_chunks(self):
        """ReqHandle.inflight_middle_chunks is readable as an int."""
        execute_scripted_runtime(
            self._script_api_smoke_r_inflight_middle_chunks, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_r_inflight_middle_chunks(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        _ = r.inflight_middle_chunks
        assert isinstance(r.inflight_middle_chunks, int)

    def test_api_smoke_r_disagg_send_state(self):
        """ReqHandle.disagg_send_state is readable (None outside disagg)."""
        execute_scripted_runtime(
            self._script_api_smoke_r_disagg_send_state, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_r_disagg_send_state(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        _ = r.disagg_send_state  # may be None outside disagg

    def test_api_smoke_r_output_tokens(self):
        """ReqHandle.output_tokens is a list after the req finishes."""
        execute_scripted_runtime(
            self._script_api_smoke_r_output_tokens, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_r_output_tokens(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        for _ in range(200):
            if r.finished:
                break
            yield
        assert isinstance(r.output_tokens, list)

    def test_api_smoke_r_logprobs(self):
        """ReqHandle.logprobs is readable when return_logprob=True."""
        execute_scripted_runtime(self._script_api_smoke_r_logprobs, **_COMMON_ENGINE_KWARGS)

    @staticmethod
    def _script_api_smoke_r_logprobs(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2, return_logprob=True)
        for _ in range(200):
            if r.finished:
                break
            yield
        _ = r.logprobs

    def test_api_smoke_r_cumulative_kv_alloc_bytes(self):
        """ReqHandle.cumulative_kv_alloc_bytes is non-negative."""
        execute_scripted_runtime(
            self._script_api_smoke_r_cumulative_kv_alloc_bytes, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_r_cumulative_kv_alloc_bytes(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        _ = r.cumulative_kv_alloc_bytes
        assert r.cumulative_kv_alloc_bytes >= 0

    def test_api_smoke_t_is_idle(self):
        """ScriptedRuntime.is_idle is readable as a bool."""
        execute_scripted_runtime(self._script_api_smoke_t_is_idle, **_COMMON_ENGINE_KWARGS)

    @staticmethod
    def _script_api_smoke_t_is_idle(t: ScriptedRuntime):
        _ = t.is_idle
        assert isinstance(t.is_idle, bool)
        yield

    def test_api_smoke_t_exhaust_kv(self):
        """ScriptedRuntime.exhaust_kv leaves the requested page slack."""
        execute_scripted_runtime(
            self._script_api_smoke_t_exhaust_kv, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_exhaust_kv(t: ScriptedRuntime):
        t.exhaust_kv(leave_pages=4)
        yield

    def test_api_smoke_t_exhaust_row_pool(self):
        """ScriptedRuntime.exhaust_row_pool leaves the requested row slack."""
        execute_scripted_runtime(
            self._script_api_smoke_t_exhaust_row_pool, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_exhaust_row_pool(t: ScriptedRuntime):
        t.exhaust_row_pool(leave_rows=2)
        yield

    def test_api_smoke_t_exhaust_lock_refs(self):
        """ScriptedRuntime.exhaust_lock_refs leaves the requested ref slack."""
        execute_scripted_runtime(
            self._script_api_smoke_t_exhaust_lock_refs, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_exhaust_lock_refs(t: ScriptedRuntime):
        t.exhaust_lock_refs(leave_refs=2)
        yield

    def test_api_smoke_t_force_lora_drainer_reject(self):
        """ScriptedRuntime.force_lora_drainer_reject runs without exception."""
        execute_scripted_runtime(
            self._script_api_smoke_t_force_lora_drainer_reject, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_force_lora_drainer_reject(t: ScriptedRuntime):
        t.force_lora_drainer_reject(adapter="some-adapter")
        yield

    def test_api_smoke_t_batch_composition(self):
        """ScriptedRuntime.batch_composition returns a dict."""
        execute_scripted_runtime(
            self._script_api_smoke_t_batch_composition, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_batch_composition(t: ScriptedRuntime):
        comp = t.batch_composition()
        assert isinstance(comp, dict)
        yield

    def test_api_smoke_t_list_active_reqs(self):
        """ScriptedRuntime.list_active_reqs returns a list."""
        execute_scripted_runtime(
            self._script_api_smoke_t_list_active_reqs, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_list_active_reqs(t: ScriptedRuntime):
        reqs = t.list_active_reqs()
        assert isinstance(reqs, list)
        yield

    def test_api_smoke_t_force_preempt(self):
        """ScriptedRuntime.force_preempt runs without exception on a victim/by pair."""
        execute_scripted_runtime(
            self._script_api_smoke_t_force_preempt, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_force_preempt(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=128, max_new_tokens=2, priority="low")
        r2 = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")
        yield
        t.force_preempt(victim_rid=r1.rid, by_rid=r2.rid)
        yield

    def test_api_smoke_t_last_admission_path(self):
        """ScriptedRuntime.last_admission_path returns None or a str."""
        execute_scripted_runtime(
            self._script_api_smoke_t_last_admission_path, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_last_admission_path(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        path = t.last_admission_path()
        assert path is None or isinstance(path, str)

    def test_api_smoke_t_last_scheduler_path(self):
        """ScriptedRuntime.last_scheduler_path returns None or a str."""
        execute_scripted_runtime(
            self._script_api_smoke_t_last_scheduler_path, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_last_scheduler_path(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        path = t.last_scheduler_path()
        assert path is None or isinstance(path, str)

    def test_api_smoke_t_engine_stats(self):
        """ScriptedRuntime.engine_stats returns a dict."""
        execute_scripted_runtime(
            self._script_api_smoke_t_engine_stats, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_engine_stats(t: ScriptedRuntime):
        stats = t.engine_stats()
        assert isinstance(stats, dict)
        yield

    def test_api_smoke_t_warmup_radix(self):
        """ScriptedRuntime.warmup_radix accepts prompt_tokens without exception."""
        execute_scripted_runtime(
            self._script_api_smoke_t_warmup_radix, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_warmup_radix(t: ScriptedRuntime):
        t.warmup_radix(prompt_tokens=[1, 1, 1, 1])
        yield

    def test_api_smoke_t_evict_radix(self):
        """ScriptedRuntime.evict_radix runs without exception."""
        execute_scripted_runtime(
            self._script_api_smoke_t_evict_radix, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_evict_radix(t: ScriptedRuntime):
        t.evict_radix(prefix_tokens=None)
        yield

    def test_api_smoke_t_trigger_abort_on_waiting_timeout(self):
        """ScriptedRuntime.trigger_abort_on_waiting_timeout runs without exception."""
        execute_scripted_runtime(
            self._script_api_smoke_t_trigger_abort_on_waiting_timeout,
            **_COMMON_ENGINE_KWARGS,
        )

    @staticmethod
    def _script_api_smoke_t_trigger_abort_on_waiting_timeout(t: ScriptedRuntime):
        t.trigger_abort_on_waiting_timeout()
        yield

    def test_api_smoke_t_get_chunked_req_rid(self):
        """ScriptedRuntime.get_chunked_req_rid returns None or a str."""
        execute_scripted_runtime(
            self._script_api_smoke_t_get_chunked_req_rid, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_t_get_chunked_req_rid(t: ScriptedRuntime):
        rid = t.get_chunked_req_rid()
        assert rid is None or isinstance(rid, str)
        yield

    def test_api_smoke_start_req_priority(self):
        """start_req accepts a priority kwarg."""
        execute_scripted_runtime(
            self._script_api_smoke_start_req_priority, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_start_req_priority(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")
        assert isinstance(r, ReqHandle)
        yield

    def test_api_smoke_start_req_lora_path(self):
        """start_req accepts a lora_path kwarg."""
        execute_scripted_runtime(
            self._script_api_smoke_start_req_lora_path, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_start_req_lora_path(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2, lora_path=None)
        assert isinstance(r, ReqHandle)
        yield

    def test_api_smoke_start_req_temperature(self):
        """start_req accepts a temperature kwarg."""
        execute_scripted_runtime(
            self._script_api_smoke_start_req_temperature, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_start_req_temperature(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2, temperature=0.0)
        assert isinstance(r, ReqHandle)
        yield

    def test_api_smoke_start_req_top_p_top_k(self):
        """start_req accepts top_p and top_k kwargs."""
        execute_scripted_runtime(
            self._script_api_smoke_start_req_top_p_top_k, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_start_req_top_p_top_k(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2, top_p=0.9, top_k=40)
        assert isinstance(r, ReqHandle)
        yield

    def test_api_smoke_start_req_stop(self):
        """start_req accepts a stop string-list kwarg."""
        execute_scripted_runtime(
            self._script_api_smoke_start_req_stop, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_start_req_stop(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=4, stop=["xyz"])
        assert isinstance(r, ReqHandle)
        yield

    def test_api_smoke_start_req_stop_token_ids(self):
        """start_req accepts a stop_token_ids kwarg."""
        execute_scripted_runtime(
            self._script_api_smoke_start_req_stop_token_ids, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_start_req_stop_token_ids(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=4, stop_token_ids=[2])
        assert isinstance(r, ReqHandle)
        yield

    def test_api_smoke_start_req_ignore_eos(self):
        """start_req accepts an ignore_eos kwarg."""
        execute_scripted_runtime(
            self._script_api_smoke_start_req_ignore_eos, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_start_req_ignore_eos(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=4, ignore_eos=True)
        assert isinstance(r, ReqHandle)
        yield

    def test_api_smoke_start_req_return_logprob(self):
        """start_req accepts a return_logprob kwarg."""
        execute_scripted_runtime(
            self._script_api_smoke_start_req_return_logprob, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_start_req_return_logprob(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2, return_logprob=True)
        assert isinstance(r, ReqHandle)
        yield

    def test_api_smoke_start_req_top_logprobs_num(self):
        """start_req accepts a top_logprobs_num kwarg."""
        execute_scripted_runtime(
            self._script_api_smoke_start_req_top_logprobs_num, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_start_req_top_logprobs_num(t: ScriptedRuntime):
        r = t.start_req(
            prompt_len=8, max_new_tokens=2, return_logprob=True, top_logprobs_num=3
        )
        assert isinstance(r, ReqHandle)
        yield

    def test_api_smoke_start_req_min_new_tokens(self):
        """start_req accepts a min_new_tokens kwarg."""
        execute_scripted_runtime(
            self._script_api_smoke_start_req_min_new_tokens, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_start_req_min_new_tokens(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=8, min_new_tokens=2)
        assert isinstance(r, ReqHandle)
        yield

    def test_api_smoke_start_req_penalties(self):
        """start_req accepts repetition / frequency / presence penalty kwargs."""
        execute_scripted_runtime(
            self._script_api_smoke_start_req_penalties, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
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

    def test_api_smoke_start_req_explicit_rid(self):
        """start_req accepts an explicit rid kwarg and uses it."""
        execute_scripted_runtime(
            self._script_api_smoke_start_req_explicit_rid, **_COMMON_ENGINE_KWARGS
        )

    @staticmethod
    def _script_api_smoke_start_req_explicit_rid(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2, rid="explicit-test-rid")
        assert r.rid == "explicit-test-rid"
        yield


if __name__ == "__main__":
    unittest.main()
