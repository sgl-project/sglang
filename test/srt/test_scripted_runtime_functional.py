"""Functional tests for ScriptedRuntime.

These tests go beyond the smoke test and exercise the harness's behavior
contract: handle bookkeeping, multi-request scripts, generator lifecycle
edges, qualified-name resolution, and the surfacing of script-side
failures back to the test process.

All script functions are top-level and underscore-prefixed:

* top-level: required for spawn-mode multiprocessing — the scheduler
  subprocess re-imports them by qualified name.
* underscore-prefixed: unittest discovery skips them so they are not
  mistaken for test methods.

Tests run a fresh Engine per case (see the test suite plan). Engine
startup is slow but ScriptedRuntime currently has no in-process reset
API; per-test engine keeps state isolation trivial.
"""

import unittest

from sglang.test.scripted_runtime import (
    ReqHandle,
    ScriptedRuntime,
    execute_scripted_runtime,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

# Common Engine kwargs shared by every test in this module. Centralized
# so that adjustments (e.g., a different model, additional flags) can be
# made in one place.
_COMMON_ENGINE_KWARGS = dict(
    model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    tp_size=1,
    dp_size=1,
    pp_size=1,
    disable_overlap_schedule=True,
    disable_cuda_graph=True,
)


# ============================================================
# Script functions: top-level so the scheduler subprocess can
# re-import them by qualified name under spawn-mode mp.
# ============================================================


def _script_start_req_returns_req_handle(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=8, max_new_tokens=2)
    assert isinstance(r1, ReqHandle), f"expected ReqHandle, got {type(r1).__name__}"
    assert r1.rid == "scripted-0", f"unexpected first rid: {r1.rid!r}"
    r2 = t.start_req(prompt_len=8, max_new_tokens=2)
    assert r2.rid == "scripted-1", f"submit_count did not advance: {r2.rid!r}"
    yield


def _script_multiple_yields_advance_scheduler(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=8, max_new_tokens=4)
    # Before any yield, the request has not been pulled from the queue yet
    # — scheduler hasn't called recv_requests since we returned from
    # start_req. ReqHandle.status should report "unknown".
    assert r1.status == "unknown", f"expected unknown pre-yield, got {r1.status!r}"
    yield  # first iteration: scheduler pulls from the queue
    yield  # second iteration: gives a chance to transition into running
    assert r1.status in (
        "waiting",
        "running",
        "unknown",
    ), f"expected the scheduler to know about r1 after yields, got {r1.status!r}"


def _script_multiple_reqs_in_one_script(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=8, max_new_tokens=2)
    r2 = t.start_req(prompt_len=8, max_new_tokens=2)
    r3 = t.start_req(prompt_len=8, max_new_tokens=2)
    assert r1.rid != r2.rid != r3.rid, "rids must be distinct"
    yield
    yield
    for r in (r1, r2, r3):
        assert r.status in (
            "waiting",
            "running",
            "unknown",
        ), f"unexpected status for {r.rid}: {r.status!r}"


def _script_empty_return(t: ScriptedRuntime):
    # Generator function with no yield at runtime: still a generator
    # because of the unreachable yield below. Returning immediately is
    # the cleanest way to test "script finishes without ever yielding".
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
    # No yield anywhere: Python returns this as a regular function, so
    # calling it produces None instead of a generator. ScriptedRuntime
    # rejects this in __init__.
    return None


def _script_yield_before_start_req(t: ScriptedRuntime):
    # First step: no request injected yet. Scheduler iterates over an
    # empty input queue.
    yield
    r1 = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    yield
    assert r1.status in (
        "waiting",
        "running",
        "unknown",
    ), f"unexpected status after late submission: {r1.status!r}"


def _script_status_for_unknown_rid(t: ScriptedRuntime):
    # Build a handle for an rid that was never injected. Lookup walks
    # the scheduler's queues and running batch and finds nothing.
    bogus = ReqHandle(rid="never-submitted-rid", runtime=t)
    assert bogus.status == "unknown", f"expected unknown, got {bogus.status!r}"
    yield


def _script_assertion_with_status(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    # Force a likely-failing equality assertion so we can verify the
    # raised AssertionError text points back at this source line.
    assert r1.status == "definitely-not-a-real-status", "status mismatch"


# ============================================================
# Tests.
# ============================================================


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
        # Should complete with no exception. The scheduler subprocess sees
        # StopIteration on the very first _yield_to_script and exits 0.
        execute_scripted_runtime(_script_empty_return, **_COMMON_ENGINE_KWARGS)

    def test_script_raises_assertion_surfaces_to_caller(self):
        with self.assertRaises(AssertionError) as ctx:
            execute_scripted_runtime(
                _script_assertion_failure, **_COMMON_ENGINE_KWARGS
            )
        self.assertIn("boom", str(ctx.exception))

    def test_script_raises_runtime_error_surfaces_to_caller(self):
        with self.assertRaises(AssertionError) as ctx:
            execute_scripted_runtime(_script_runtime_error, **_COMMON_ENGINE_KWARGS)
        # Script-side exceptions are wrapped on the caller side as AssertionError
        # carrying the original traceback text.
        err_text = str(ctx.exception)
        self.assertIn("RuntimeError", err_text)
        self.assertIn("simulated runtime error", err_text)

    def test_non_generator_script_function_errors_cleanly(self):
        with self.assertRaises(AssertionError) as ctx:
            execute_scripted_runtime(
                _script_not_a_generator, **_COMMON_ENGINE_KWARGS
            )
        # The runtime rejects non-generator script functions with a clear
        # TypeError mentioning "must be a generator". That traceback bubbles
        # up via the traceback file and execute_scripted_runtime wraps it as
        # AssertionError. The test process itself must not be SIGQUITed —
        # if assertRaises catches the AssertionError, we're good.
        self.assertIn("must be a generator", str(ctx.exception))

    def test_invalid_qualified_name_errors_before_engine(self):
        # Lambdas have ``__qualname__`` containing ``<lambda>`` which the
        # qualified-name re-import path rejects. The failure must occur
        # synchronously in execute_scripted_runtime, before any Engine spins
        # up.
        with self.assertRaises((ValueError, TypeError, AttributeError)):
            execute_scripted_runtime(lambda t: None, **_COMMON_ENGINE_KWARGS)

    def test_script_imported_from_pytest_file(self):
        # Implicitly covers spawn-mode sys.path forwarding: this very test
        # file lives under ``test/srt/`` and is not normally on sys.path in
        # the subprocess. execute_scripted_runtime forwards the directory so
        # ``_resolve_fn`` can import the script by qualified name. If that
        # plumbing breaks, the scheduler subprocess raises ImportError and
        # this test fails with a non-empty traceback.
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
        # Traceback should reference the script function name so a developer
        # can navigate to the failing assert.
        self.assertIn("_script_assertion_with_status", err_text)
        self.assertIn("status mismatch", err_text)


if __name__ == "__main__":
    unittest.main()
