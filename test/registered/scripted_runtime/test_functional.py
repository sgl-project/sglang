"""Functional tests for ScriptedRuntime.

Script functions are top-level (spawn-mode mp imports them by name)
and underscore-prefixed (so unittest discovery skips them).
"""

import unittest

from sglang.test.scripted_runtime import (
    ReqHandle,
    ScriptedRuntime,
    execute_scripted_runtime,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

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


if __name__ == "__main__":
    unittest.main()
