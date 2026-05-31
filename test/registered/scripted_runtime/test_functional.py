import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


class TestFunctional(ScriptedTestCase):
    ENGINE_KWARGS = dict(
        model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        tp_size=1,
        dp_size=1,
        pp_size=1,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
    )

    def test_empty_script_returns_immediately(self):
        self.server.execute_script(self._script_empty_return)

    @staticmethod
    def _script_empty_return(t: ScriptedContext):
        if False:
            yield
        return

    def test_script_raises_assertion_surfaces_to_caller(self):
        with self.assertRaises(AssertionError) as ctx:
            self.server.execute_script(self._script_assertion_failure)
        self.assertIn("boom", str(ctx.exception))

    @staticmethod
    def _script_assertion_failure(t: ScriptedContext):
        yield
        assert False, "boom"

    def test_script_raises_runtime_error_surfaces_to_caller(self):
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
        with self.assertRaises(AssertionError) as ctx:
            self.server.execute_script(self._script_not_a_generator)
        err_text = str(ctx.exception)
        self.assertIn("TypeError", err_text)
        self.assertIn("NoneType", err_text)

    @staticmethod
    def _script_not_a_generator(t: ScriptedContext):
        return None

    def test_api_smoke_r_finished(self):
        self.server.execute_script(self._script_api_smoke_r_finished)

    @staticmethod
    def _script_api_smoke_r_finished(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2)
        for _ in range(200):
            if r.finished:
                return
            yield

    def test_api_smoke_start_req_explicit_rid(self):
        self.server.execute_script(self._script_api_smoke_start_req_explicit_rid)

    @staticmethod
    def _script_api_smoke_start_req_explicit_rid(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2, rid="explicit-test-rid")
        assert r.rid == "explicit-test-rid"
        yield


if __name__ == "__main__":
    unittest.main()
