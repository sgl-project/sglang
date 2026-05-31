
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
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield


def _script_followup_clean(t: ScriptedContext):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    assert r.rid.startswith("scripted-")
    yield


def _script_raises_assertion(t: ScriptedContext):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    assert r.rid == "this-rid-will-never-match"


def _script_minimal_after_error(t: ScriptedContext):
    yield
    yield


def _script_minimal_ok(t: ScriptedContext):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield
    yield


class TestScriptedHttpServerSequentialDispatch(CustomTestCase):

    def test_two_sub_scripts_sequential(self):
        session = ScriptedHttpServer.start(**_ENGINE_KWARGS)
        try:
            session.execute_script(_script_records_started)
            session.execute_script(_script_followup_clean)
        finally:
            session.shutdown()


class TestScriptedHttpServerErrorRecovery(CustomTestCase):

    def test_assertion_reraised_session_survives(self):
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

    def test_shutdown_twice(self):
        session = ScriptedHttpServer.start(**_ENGINE_KWARGS)
        session.shutdown()
        session.shutdown()
        assert session._shutdown_done is True


class TestScriptedHttpServerDirtyRefusesRun(CustomTestCase):

    def test_dirty_state_blocks_run(self):
        session = ScriptedHttpServer.start(**_ENGINE_KWARGS)
        try:
            session._dirty = "test dirty"
            with self.assertRaises(RuntimeError) as ctx:
                session.execute_script(_script_minimal_ok)
            assert "test dirty" in str(ctx.exception)
        finally:
            session.shutdown()


class TestScriptedTestCaseSmoke(ScriptedTestCase):

    ENGINE_KWARGS = _ENGINE_KWARGS

    def test_minimal_start_req_and_yield(self):
        self.server.execute_script(_script_minimal_ok)


if __name__ == "__main__":
    unittest.main()
