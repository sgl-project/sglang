"""Self-tests for ScriptedHttpServer + ScriptedTestCase machinery.

The first four tests exercise the Session lifecycle (sequential dispatch,
error isolation, idempotent shutdown, dirty-state gating) directly via
``unittest.TestCase`` because they manage their own session. The fifth
test verifies that :class:`ScriptedTestCase` itself wires up
setUpClass / tearDownClass correctly with a minimal end-to-end run.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.http_server import ScriptedHttpServer
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


_ENGINE_KWARGS = dict(
    model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    tp_size=1,
    dp_size=1,
    pp_size=1,
    disable_overlap_schedule=True,
    disable_cuda_graph=True,
)


def _script_records_started(t: ScriptedContext):
    """Submit one req and confirm the scheduler sees it after a single yield."""
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    assert r.status in ("waiting", "running", "unknown")


def _script_followup_clean(t: ScriptedContext):
    """A second sub-script that should observe no leaked state from the first."""
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    assert r.rid.startswith("scripted-")
    yield
    assert r.status in ("waiting", "running", "unknown")


def _script_raises_assertion(t: ScriptedContext):
    """Sub-script that fails partway through so we can verify error surfacing."""
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    assert r.rid == "this-rid-will-never-match"


def _script_minimal_after_error(t: ScriptedContext):
    """Tiny no-op sub-script run after an error to confirm the session lives on."""
    yield
    yield


def _script_minimal_ok(t: ScriptedContext):
    """Smallest possible successful sub-script for the testcase smoke."""
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    yield
    assert r.status in ("waiting", "running", "unknown")


class TestScriptedHttpServerSequentialDispatch(CustomTestCase):
    """Two sub-scripts run back-to-back on one session must both succeed."""

    def test_two_sub_scripts_sequential(self):
        """Run two sub-scripts in order; both finish without cross-pollution."""
        session = ScriptedHttpServer.start(**_ENGINE_KWARGS)
        try:
            session.execute_script(_script_records_started)
            session.execute_script(_script_followup_clean)
        finally:
            session.shutdown()


class TestScriptedHttpServerErrorRecovery(CustomTestCase):
    """A failing sub-script must be re-raised and not poison the session."""

    def test_assertion_reraised_session_survives(self):
        """AssertionError surfaces with traceback; next run() still works."""
        session = ScriptedHttpServer.start(**_ENGINE_KWARGS)
        try:
            with self.assertRaises(AssertionError) as ctx:
                session.execute_script(_script_raises_assertion)
            assert "this-rid-will-never-match" in str(ctx.exception)
            assert "Traceback" in str(ctx.exception) or "assert" in str(ctx.exception)
            session.execute_script(_script_minimal_after_error)
        finally:
            session.shutdown()


class TestScriptedHttpServerShutdownIdempotent(CustomTestCase):
    """Calling shutdown() twice in a row must not raise."""

    def test_shutdown_twice(self):
        """shutdown() is idempotent — second call is a no-op."""
        session = ScriptedHttpServer.start(**_ENGINE_KWARGS)
        session.shutdown()
        session.shutdown()
        assert session._shutdown_done is True


class TestScriptedHttpServerDirtyRefusesRun(CustomTestCase):
    """A session marked dirty must reject further run() calls."""

    def test_dirty_state_blocks_run(self):
        """Setting _dirty makes run() raise RuntimeError without dispatching."""
        session = ScriptedHttpServer.start(**_ENGINE_KWARGS)
        try:
            session._dirty = "test dirty"
            with self.assertRaises(RuntimeError) as ctx:
                session.execute_script(_script_minimal_ok)
            assert "test dirty" in str(ctx.exception)
        finally:
            session.shutdown()


class TestScriptedRuntimeTestCaseSmoke(ScriptedTestCase):
    """Smallest possible class using ScriptedTestCase end-to-end."""

    ENGINE_KWARGS = _ENGINE_KWARGS

    def test_minimal_start_req_and_yield(self):
        """One req, two yields, status is observable — the testcase wires up."""
        self.server.execute_script(_script_minimal_ok)


if __name__ == "__main__":
    unittest.main()
