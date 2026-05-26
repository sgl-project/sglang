"""Self-tests for ScriptedRuntimeSession + ScriptedRuntimeTestCase machinery.

The first four tests exercise the Session lifecycle (sequential dispatch,
error isolation, idempotent shutdown, dirty-state gating) directly via
``unittest.TestCase`` because they manage their own session. The fifth
test verifies that :class:`ScriptedRuntimeTestCase` itself wires up
setUpClass / tearDownClass correctly with a minimal end-to-end run.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.session import ScriptedRuntimeSession
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
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


def _script_records_started(t: ScriptedRuntime):
    """Submit one req and confirm the scheduler sees it after a single yield."""
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    assert r.status in ("waiting", "running", "unknown")


def _script_followup_clean(t: ScriptedRuntime):
    """A second sub-script that should observe no leaked state from the first."""
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    assert r.rid.startswith("scripted-")
    yield
    assert r.status in ("waiting", "running", "unknown")


def _script_raises_assertion(t: ScriptedRuntime):
    """Sub-script that fails partway through so we can verify error surfacing."""
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    assert r.rid == "this-rid-will-never-match"


def _script_minimal_after_error(t: ScriptedRuntime):
    """Tiny no-op sub-script run after an error to confirm the session lives on."""
    yield
    yield


def _script_minimal_ok(t: ScriptedRuntime):
    """Smallest possible successful sub-script for the testcase smoke."""
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    yield
    assert r.status in ("waiting", "running", "unknown")


class TestScriptedRuntimeSessionSequentialDispatch(CustomTestCase):
    """Two sub-scripts run back-to-back on one session must both succeed."""

    def test_two_sub_scripts_sequential(self):
        """Run two sub-scripts in order; both finish without cross-pollution."""
        session = ScriptedRuntimeSession.start(**_ENGINE_KWARGS)
        try:
            session.run(_script_records_started)
            session.run(_script_followup_clean)
        finally:
            session.shutdown()


class TestScriptedRuntimeSessionErrorRecovery(CustomTestCase):
    """A failing sub-script must be re-raised and not poison the session."""

    def test_assertion_reraised_session_survives(self):
        """AssertionError surfaces with traceback; next run() still works."""
        session = ScriptedRuntimeSession.start(**_ENGINE_KWARGS)
        try:
            with self.assertRaises(AssertionError) as ctx:
                session.run(_script_raises_assertion)
            assert "this-rid-will-never-match" in str(ctx.exception)
            assert "Traceback" in str(ctx.exception) or "assert" in str(ctx.exception)
            session.run(_script_minimal_after_error)
        finally:
            session.shutdown()


class TestScriptedRuntimeSessionShutdownIdempotent(CustomTestCase):
    """Calling shutdown() twice in a row must not raise."""

    def test_shutdown_twice(self):
        """shutdown() is idempotent — second call is a no-op."""
        session = ScriptedRuntimeSession.start(**_ENGINE_KWARGS)
        session.shutdown()
        session.shutdown()
        assert session._shutdown_done is True


class TestScriptedRuntimeSessionDirtyRefusesRun(CustomTestCase):
    """A session marked dirty must reject further run() calls."""

    def test_dirty_state_blocks_run(self):
        """Setting _dirty makes run() raise RuntimeError without dispatching."""
        session = ScriptedRuntimeSession.start(**_ENGINE_KWARGS)
        try:
            session._dirty = "test dirty"
            with self.assertRaises(RuntimeError) as ctx:
                session.run(_script_minimal_ok)
            assert "test dirty" in str(ctx.exception)
        finally:
            session.shutdown()


class TestScriptedRuntimeTestCaseSmoke(ScriptedRuntimeTestCase):
    """Smallest possible class using ScriptedRuntimeTestCase end-to-end."""

    ENGINE_KWARGS = _ENGINE_KWARGS

    def test_minimal_start_req_and_yield(self):
        """One req, two yields, status is observable — the testcase wires up."""
        self.runtime.run(_script_minimal_ok)


if __name__ == "__main__":
    unittest.main()
