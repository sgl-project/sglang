"""Functional tests for ScriptedContext.

Script functions are top-level (spawn-mode mp imports them by name)
and underscore-prefixed (so unittest discovery skips them).
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


# ============================================================
# API surface smokes for round-2 expansion. Each function calls
# one new API attribute / method and verifies the call returns
# without exception (semantics tested in manual suite).
# ============================================================


class TestFunctional(ScriptedTestCase):
    ENGINE_KWARGS = dict(
        model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        tp_size=1,
        dp_size=1,
        pp_size=1,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
    )

    def test_start_req_returns_req_handle(self):
        """start_req returns a ScriptedReqHandle with the expected auto-assigned rid."""
        self.server.execute_script(self._script_start_req_returns_req_handle)

    @staticmethod
    def _script_start_req_returns_req_handle(t: ScriptedContext):
        r1 = t.start_req(prompt_len=8, max_new_tokens=2)
        assert isinstance(r1, ScriptedReqHandle)
        assert r1.rid == "scripted-0"
        r2 = t.start_req(prompt_len=8, max_new_tokens=2)
        assert r2.rid == "scripted-1"
        yield

    def test_multiple_yields_advance_scheduler(self):
        """Multiple bare yields advance the scheduler by one iteration each."""
        self.server.execute_script(self._script_multiple_yields_advance_scheduler)

    @staticmethod
    def _script_multiple_yields_advance_scheduler(t: ScriptedContext):
        r1 = t.start_req(prompt_len=8, max_new_tokens=4)
        assert r1.status == "unknown"  # not yet pulled from the queue
        yield
        yield
        assert r1.status in ("waiting", "running", "unknown")

    def test_multiple_reqs_in_one_script(self):
        """A single script can submit multiple reqs with distinct rids."""
        self.server.execute_script(self._script_multiple_reqs_in_one_script)

    @staticmethod
    def _script_multiple_reqs_in_one_script(t: ScriptedContext):
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
        self.server.execute_script(self._script_empty_return)

    @staticmethod
    def _script_empty_return(t: ScriptedContext):
        # Unreachable yield keeps this a generator function, but the body
        # returns without ever yielding.
        if False:
            yield
        return

    def test_script_raises_assertion_surfaces_to_caller(self):
        """AssertionError from script body surfaces back to the caller."""
        with self.assertRaises(AssertionError) as ctx:
            self.server.execute_script(self._script_assertion_failure)
        self.assertIn("boom", str(ctx.exception))

    @staticmethod
    def _script_assertion_failure(t: ScriptedContext):
        yield
        assert False, "boom"

    def test_script_raises_runtime_error_surfaces_to_caller(self):
        """RuntimeError from script body surfaces back to the caller as AssertionError."""
        with self.assertRaises(AssertionError) as ctx:
            self.server.execute_script(self._script_runtime_error)
        err_text = str(ctx.exception)
        self.assertIn("RuntimeError", err_text)
        self.assertIn("simulated runtime error", err_text)

    @staticmethod
    def _script_runtime_error(t: ScriptedContext):
        yield
        raise RuntimeError("simulated runtime error")

    def test_non_generator_script_function_errors_cleanly(self):
        """Non-generator script function is rejected with a clear error."""
        with self.assertRaises(AssertionError) as ctx:
            self.server.execute_script(self._script_not_a_generator)
        err_text = str(ctx.exception)
        # The router does ``yield from sub_gen`` where ``sub_gen`` is None,
        # which raises TypeError caught by the router and surfaced here.
        self.assertIn("TypeError", err_text)
        self.assertIn("NoneType", err_text)

    @staticmethod
    def _script_not_a_generator(t: ScriptedContext):
        # No yield => regular function => calling returns None, not a generator.
        return None

    def test_script_imported_from_pytest_file(self):
        """Spawn-mode sys.path forwarding lets subprocess import the script."""
        # Exercises spawn-mode sys.path forwarding: this file's directory
        # is not normally on the subprocess's sys.path.
        self.server.execute_script(self._script_start_req_returns_req_handle)

    def test_yield_before_start_req(self):
        """Yielding before any start_req call is safe."""
        self.server.execute_script(self._script_yield_before_start_req)

    @staticmethod
    def _script_yield_before_start_req(t: ScriptedContext):
        yield
        r1 = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        yield
        assert r1.status in ("waiting", "running", "unknown")

    def test_status_for_unknown_rid(self):
        """ScriptedReqHandle for a never-submitted rid reports 'unknown' status."""
        self.server.execute_script(self._script_status_for_unknown_rid)

    @staticmethod
    def _script_status_for_unknown_rid(t: ScriptedContext):
        bogus = ScriptedReqHandle(
            rid="never-submitted-rid", scheduler_hook=t._scheduler_hook
        )
        assert bogus.status == "unknown"
        yield

    def test_assertion_failure_traceback_points_to_script_line(self):
        """Assertion failure traceback names the failing script function."""
        with self.assertRaises(AssertionError) as ctx:
            self.server.execute_script(self._script_assertion_with_status)
        err_text = str(ctx.exception)
        # Traceback should name the failing script function so a
        # developer can find the assert.
        self.assertIn("_script_assertion_with_status", err_text)
        self.assertIn("status mismatch", err_text)

    @staticmethod
    def _script_assertion_with_status(t: ScriptedContext):
        r1 = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        assert r1.status == "definitely-not-a-real-status", "status mismatch"

    # ============================================================
    # Round-2 API surface smokes — one test per new attribute /
    # method introduced by the chunked expansion plan.
    # ============================================================

    def test_api_smoke_r_finished(self):
        """ScriptedReqHandle.finished returns True once the req completes."""
        self.server.execute_script(self._script_api_smoke_r_finished)

    @staticmethod
    def _script_api_smoke_r_finished(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        for _ in range(200):
            if r.finished:
                return
            yield

    def test_api_smoke_r_inflight_middle_chunks(self):
        """ScriptedReqHandle.inflight_middle_chunks is readable as an int."""
        self.server.execute_script(self._script_api_smoke_r_inflight_middle_chunks)

    @staticmethod
    def _script_api_smoke_r_inflight_middle_chunks(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        _ = r.inflight_middle_chunks
        assert isinstance(r.inflight_middle_chunks, int)

    def test_api_smoke_r_disagg_send_state(self):
        """ScriptedReqHandle.disagg_send_state is readable (None outside disagg)."""
        self.server.execute_script(self._script_api_smoke_r_disagg_send_state)

    @staticmethod
    def _script_api_smoke_r_disagg_send_state(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        _ = r.disagg_send_state  # may be None outside disagg

    def test_api_smoke_r_output_tokens(self):
        """ScriptedReqHandle.output_tokens is a list after the req finishes."""
        self.server.execute_script(self._script_api_smoke_r_output_tokens)

    @staticmethod
    def _script_api_smoke_r_output_tokens(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        for _ in range(200):
            if r.finished:
                break
            yield
        assert isinstance(r.output_tokens, list)

    def test_api_smoke_r_logprobs(self):
        """ScriptedReqHandle.logprobs is readable when return_logprob=True."""
        self.server.execute_script(self._script_api_smoke_r_logprobs)

    @staticmethod
    def _script_api_smoke_r_logprobs(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2, return_logprob=True)
        for _ in range(200):
            if r.finished:
                break
            yield
        _ = r.logprobs

    def test_api_smoke_r_cumulative_kv_alloc_bytes(self):
        """ScriptedReqHandle.cumulative_kv_alloc_bytes is non-negative."""
        self.server.execute_script(self._script_api_smoke_r_cumulative_kv_alloc_bytes)

    @staticmethod
    def _script_api_smoke_r_cumulative_kv_alloc_bytes(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        _ = r.cumulative_kv_alloc_bytes
        assert r.cumulative_kv_alloc_bytes >= 0

    def test_api_smoke_t_is_idle(self):
        """ScriptedContext.is_idle is readable as a bool."""
        self.server.execute_script(self._script_api_smoke_t_is_idle)

    @staticmethod
    def _script_api_smoke_t_is_idle(t: ScriptedContext):
        _ = t.is_idle
        assert isinstance(t.is_idle, bool)
        yield

    def test_api_smoke_t_exhaust_kv(self):
        """ScriptedContext.exhaust_kv leaves the requested page slack."""
        self.server.execute_script(self._script_api_smoke_t_exhaust_kv)

    @staticmethod
    def _script_api_smoke_t_exhaust_kv(t: ScriptedContext):
        t.exhaust_kv(leave_pages=4)
        yield

    def test_api_smoke_t_exhaust_row_pool(self):
        """ScriptedContext.exhaust_row_pool leaves the requested row slack."""
        self.server.execute_script(self._script_api_smoke_t_exhaust_row_pool)

    @staticmethod
    def _script_api_smoke_t_exhaust_row_pool(t: ScriptedContext):
        t.exhaust_row_pool(leave_rows=2)
        yield

    def test_api_smoke_t_exhaust_lock_refs(self):
        """ScriptedContext.exhaust_lock_refs leaves the requested ref slack."""
        self.server.execute_script(self._script_api_smoke_t_exhaust_lock_refs)

    @staticmethod
    def _script_api_smoke_t_exhaust_lock_refs(t: ScriptedContext):
        t.exhaust_lock_refs(leave_refs=2)
        yield

    def test_api_smoke_t_force_lora_drainer_reject(self):
        """ScriptedContext.force_lora_drainer_reject runs without exception."""
        self.server.execute_script(self._script_api_smoke_t_force_lora_drainer_reject)

    @staticmethod
    def _script_api_smoke_t_force_lora_drainer_reject(t: ScriptedContext):
        t.force_lora_drainer_reject(adapter="some-adapter")
        yield

    def test_api_smoke_t_batch_composition(self):
        """ScriptedContext.batch_composition returns a dict."""
        self.server.execute_script(self._script_api_smoke_t_batch_composition)

    @staticmethod
    def _script_api_smoke_t_batch_composition(t: ScriptedContext):
        comp = t.batch_composition()
        assert isinstance(comp, dict)
        yield

    def test_api_smoke_t_list_active_reqs(self):
        """ScriptedContext.list_active_reqs returns a list."""
        self.server.execute_script(self._script_api_smoke_t_list_active_reqs)

    @staticmethod
    def _script_api_smoke_t_list_active_reqs(t: ScriptedContext):
        reqs = t.list_active_reqs()
        assert isinstance(reqs, list)
        yield

    def test_api_smoke_t_force_preempt(self):
        """ScriptedContext.force_preempt runs without exception on a victim/by pair."""
        self.server.execute_script(self._script_api_smoke_t_force_preempt)

    @staticmethod
    def _script_api_smoke_t_force_preempt(t: ScriptedContext):
        r1 = t.start_req(prompt_len=128, max_new_tokens=2, priority="low")
        r2 = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")
        yield
        t.force_preempt(victim_rid=r1.rid, by_rid=r2.rid)
        yield

    def test_api_smoke_t_last_admission_path(self):
        """ScriptedContext.last_admission_path returns None or a str."""
        self.server.execute_script(self._script_api_smoke_t_last_admission_path)

    @staticmethod
    def _script_api_smoke_t_last_admission_path(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        path = t.last_admission_path()
        assert path is None or isinstance(path, str)

    def test_api_smoke_t_last_scheduler_path(self):
        """ScriptedContext.last_scheduler_path returns None or a str."""
        self.server.execute_script(self._script_api_smoke_t_last_scheduler_path)

    @staticmethod
    def _script_api_smoke_t_last_scheduler_path(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        yield
        path = t.last_scheduler_path()
        assert path is None or isinstance(path, str)

    def test_api_smoke_t_engine_stats(self):
        """ScriptedContext.engine_stats returns a dict."""
        self.server.execute_script(self._script_api_smoke_t_engine_stats)

    @staticmethod
    def _script_api_smoke_t_engine_stats(t: ScriptedContext):
        stats = t.engine_stats()
        assert isinstance(stats, dict)
        yield

    def test_api_smoke_t_warmup_radix(self):
        """ScriptedContext.warmup_radix accepts prompt_tokens without exception."""
        self.server.execute_script(self._script_api_smoke_t_warmup_radix)

    @staticmethod
    def _script_api_smoke_t_warmup_radix(t: ScriptedContext):
        t.warmup_radix(prompt_tokens=[1, 1, 1, 1])
        yield

    def test_api_smoke_t_evict_radix(self):
        """ScriptedContext.evict_radix runs without exception."""
        self.server.execute_script(self._script_api_smoke_t_evict_radix)

    @staticmethod
    def _script_api_smoke_t_evict_radix(t: ScriptedContext):
        t.evict_radix(prefix_tokens=None)
        yield

    def test_api_smoke_t_trigger_abort_on_waiting_timeout(self):
        """ScriptedContext.trigger_abort_on_waiting_timeout runs without exception."""
        self.server.execute_script(
            self._script_api_smoke_t_trigger_abort_on_waiting_timeout
        )

    @staticmethod
    def _script_api_smoke_t_trigger_abort_on_waiting_timeout(t: ScriptedContext):
        t.trigger_abort_on_waiting_timeout()
        yield

    def test_api_smoke_t_get_chunked_req_rid(self):
        """ScriptedContext.get_chunked_req_rid returns None or a str."""
        self.server.execute_script(self._script_api_smoke_t_get_chunked_req_rid)

    @staticmethod
    def _script_api_smoke_t_get_chunked_req_rid(t: ScriptedContext):
        rid = t.get_chunked_req_rid()
        assert rid is None or isinstance(rid, str)
        yield

    def test_api_smoke_start_req_priority(self):
        """start_req accepts a priority kwarg."""
        self.server.execute_script(self._script_api_smoke_start_req_priority)

    @staticmethod
    def _script_api_smoke_start_req_priority(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")
        assert isinstance(r, ScriptedReqHandle)
        yield

    def test_api_smoke_start_req_lora_path(self):
        """start_req accepts a lora_path kwarg."""
        self.server.execute_script(self._script_api_smoke_start_req_lora_path)

    @staticmethod
    def _script_api_smoke_start_req_lora_path(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2, lora_path=None)
        assert isinstance(r, ScriptedReqHandle)
        yield

    def test_api_smoke_start_req_temperature(self):
        """start_req accepts a temperature kwarg."""
        self.server.execute_script(self._script_api_smoke_start_req_temperature)

    @staticmethod
    def _script_api_smoke_start_req_temperature(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2, temperature=0.0)
        assert isinstance(r, ScriptedReqHandle)
        yield

    def test_api_smoke_start_req_top_p_top_k(self):
        """start_req accepts top_p and top_k kwargs."""
        self.server.execute_script(self._script_api_smoke_start_req_top_p_top_k)

    @staticmethod
    def _script_api_smoke_start_req_top_p_top_k(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2, top_p=0.9, top_k=40)
        assert isinstance(r, ScriptedReqHandle)
        yield

    def test_api_smoke_start_req_stop(self):
        """start_req accepts a stop string-list kwarg."""
        self.server.execute_script(self._script_api_smoke_start_req_stop)

    @staticmethod
    def _script_api_smoke_start_req_stop(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=4, stop=["xyz"])
        assert isinstance(r, ScriptedReqHandle)
        yield

    def test_api_smoke_start_req_stop_token_ids(self):
        """start_req accepts a stop_token_ids kwarg."""
        self.server.execute_script(self._script_api_smoke_start_req_stop_token_ids)

    @staticmethod
    def _script_api_smoke_start_req_stop_token_ids(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=4, stop_token_ids=[2])
        assert isinstance(r, ScriptedReqHandle)
        yield

    def test_api_smoke_start_req_ignore_eos(self):
        """start_req accepts an ignore_eos kwarg."""
        self.server.execute_script(self._script_api_smoke_start_req_ignore_eos)

    @staticmethod
    def _script_api_smoke_start_req_ignore_eos(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=4, ignore_eos=True)
        assert isinstance(r, ScriptedReqHandle)
        yield

    def test_api_smoke_start_req_return_logprob(self):
        """start_req accepts a return_logprob kwarg."""
        self.server.execute_script(self._script_api_smoke_start_req_return_logprob)

    @staticmethod
    def _script_api_smoke_start_req_return_logprob(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2, return_logprob=True)
        assert isinstance(r, ScriptedReqHandle)
        yield

    def test_api_smoke_start_req_top_logprobs_num(self):
        """start_req accepts a top_logprobs_num kwarg."""
        self.server.execute_script(self._script_api_smoke_start_req_top_logprobs_num)

    @staticmethod
    def _script_api_smoke_start_req_top_logprobs_num(t: ScriptedContext):
        r = t.start_req(
            prompt_len=8, max_new_tokens=2, return_logprob=True, top_logprobs_num=3
        )
        assert isinstance(r, ScriptedReqHandle)
        yield

    def test_api_smoke_start_req_min_new_tokens(self):
        """start_req accepts a min_new_tokens kwarg."""
        self.server.execute_script(self._script_api_smoke_start_req_min_new_tokens)

    @staticmethod
    def _script_api_smoke_start_req_min_new_tokens(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=8, min_new_tokens=2)
        assert isinstance(r, ScriptedReqHandle)
        yield

    def test_api_smoke_start_req_penalties(self):
        """start_req accepts repetition / frequency / presence penalty kwargs."""
        self.server.execute_script(self._script_api_smoke_start_req_penalties)

    @staticmethod
    def _script_api_smoke_start_req_penalties(t: ScriptedContext):
        r = t.start_req(
            prompt_len=8,
            max_new_tokens=4,
            repetition_penalty=1.1,
            frequency_penalty=0.1,
            presence_penalty=0.1,
        )
        assert isinstance(r, ScriptedReqHandle)
        yield

    def test_api_smoke_start_req_explicit_rid(self):
        """start_req accepts an explicit rid kwarg and uses it."""
        self.server.execute_script(self._script_api_smoke_start_req_explicit_rid)

    @staticmethod
    def _script_api_smoke_start_req_explicit_rid(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2, rid="explicit-test-rid")
        assert r.rid == "explicit-test-rid"
        yield


if __name__ == "__main__":
    unittest.main()
