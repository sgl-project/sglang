"""Minimal end-to-end smoke test for ScriptedContext."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class TestSmoke(ScriptedTestCase):
    ENGINE_KWARGS = dict(
        model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        tp_size=1,
        dp_size=1,
        pp_size=1,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
    )

    def test_smoke(self):
        """Minimal end-to-end smoke for ScriptedContext launch and one req."""
        self.server.execute_script(self._smoke_script)

    @staticmethod
    def _smoke_script(t: ScriptedContext):
        r1 = t.start_req(prompt_len=10, max_new_tokens=4)
        yield
        yield
        assert r1.status in ("waiting", "running", "unknown")


if __name__ == "__main__":
    unittest.main()
